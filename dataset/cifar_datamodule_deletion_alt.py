import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import pytorch_lightning as pl

from utils.dataset_deletion import (
    QuerySupportTrainDataset,
    QuerySupportTestDataset,
    query_support_train_collate_fn_deletion,
    query_support_test_collate_fn,
)
import random
from torch.utils.data import Subset
from functools import partial


class CIFARDataModuleDeletionAlt(pl.LightningDataModule):
    """Alternate CIFAR DataModule that correctly uses the deletion collate_fn.

    This mirrors `CIFARDataModuleDeletion` but explicitly imports and
    wires the deletion collate function so downstream alternate models
    can receive the 6-tuple batch.
    """

    def __init__(self, support_size_train, support_size_eval, batch_size, num_classes=10, query_digits=None, support_digits=None, dataset_name="cifar10", model_type='edm', deletion_fraction=0.2, **kwargs):
        super().__init__()
        self.support_size_train = support_size_train
        self.support_size_eval = support_size_eval
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.query_digits = query_digits
        self.support_digits = support_digits
        self.dataset_name = dataset_name.lower()
        self.deletion_fraction = deletion_fraction

        if model_type == 'classifier' or model_type == 'ablation':
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif model_type == 'edm' or model_type == 'ddpm':
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        if self.dataset_name == "cifar100":
            self.dataset = torchvision.datasets.CIFAR100
        else:
            self.dataset = torchvision.datasets.CIFAR10

    def prepare_data(self):
        self.dataset(root='./data', train=True, download=True)
        self.dataset(root='./data', train=False, download=True)

    def setup(self, stage=None):
        if stage in ("fit", None):
            cifar_train = self.dataset(root='./data', train=True, transform=self.train_transform)
            self.train_dataset = QuerySupportTrainDataset(cifar_train, self.support_size_train, self.num_classes)
            cifar_val = self.dataset(root='./data', train=False, transform=self.test_transform)
            self.val_dataset = QuerySupportTestDataset(cifar_val, cifar_train, self.support_size_eval, self.query_digits, self.support_digits, self.num_classes)
            self.test_dataset = self.dataset(root='./data', train=False, transform=self.test_transform)
        if stage == "test" or stage == "finetune":
            self.train_dataset = self.dataset(root='./data', train=True, transform=self.train_transform)
            self.test_dataset = self.dataset(root='./data', train=False, transform=self.test_transform)

    def train_dataloader(self):
        # Use the deletion-aware collate only when the train_dataset is
        # a QuerySupportTrainDataset (it returns (img,label,idx) tuples).
        # In 'test' or 'finetune' stages the datamodule may replace
        # train_dataset with the raw dataset (img,label) — in that case
        # we must not set the collate_fn that expects 3-tuples.
        from utils.dataset_deletion import QuerySupportTrainDataset
        if isinstance(self.train_dataset, QuerySupportTrainDataset):
            collate = partial(
                query_support_train_collate_fn_deletion,
                support_dataset=self.train_dataset,
                support_size=self.support_size_train,
                num_classes=self.num_classes,
                deletion_fraction=self.deletion_fraction,
            )
        else:
            collate = None

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
            collate_fn=collate,
        )

    def val_dataloader(self):
        collate = partial(
            query_support_test_collate_fn,
            support_dataset=self.val_dataset,
            support_size=self.support_size_eval,
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
            collate_fn=collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
        )

    def set_no_train_digits(self, query_digits):
        self.query_digits = list(query_digits)

    def set_random_unlearn_portion(self, portion):
        self.random_unlearn_portion = portion

    def set_indexes_to_replace(self, indexes):
        self.indexes_to_replace = list(indexes)

    def _train_unlearn_indices(self):
        if hasattr(self, 'random_unlearn_portion'):
            if not hasattr(self, 'exclude_idx'):
                num_exclude = int(len(self.train_dataset) * self.random_unlearn_portion)
                self.exclude_idx = set(random.sample(range(len(self.train_dataset)), num_exclude))
            return sorted(self.exclude_idx)
        if hasattr(self, 'indexes_to_replace'):
            return list(self.indexes_to_replace)
        query_digits = set(self.query_digits or range(self.num_classes))
        return [i for i, (_, label) in enumerate(self.train_dataset) if label in query_digits]

    def support_loader(self):
        excluded = set(self._train_unlearn_indices())
        if self.support_digits is None:
            query_digits = set(self.query_digits or [])
            support_indices = [
                i for i, (_, label) in enumerate(self.train_dataset)
                if i not in excluded and label not in query_digits
            ]
        else:
            support_digits = set(self.support_digits)
            support_indices = [
                i for i, (_, label) in enumerate(self.train_dataset)
                if i not in excluded and label in support_digits
            ]
        return DataLoader(
            Subset(self.train_dataset, support_indices),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
        )

    def unlearn_train_dataloader(self):
        return DataLoader(
            Subset(self.train_dataset, self._train_unlearn_indices()),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
        )

    def rest_train_dataloader(self):
        unlearn_indices = set(self._train_unlearn_indices())
        rest_indices = [i for i in range(len(self.train_dataset)) if i not in unlearn_indices]
        return DataLoader(
            Subset(self.train_dataset, rest_indices),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
        )

    def unlearn_test_dataloader(self):
        query_digits = set(self.query_digits or range(self.num_classes))
        indices = [
            i for i, (_, label) in enumerate(self.test_dataset)
            if label in query_digits
        ]
        return DataLoader(
            Subset(self.test_dataset, indices),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
        )

    def rest_test_dataloader(self):
        query_digits = set(self.query_digits or range(self.num_classes))
        if hasattr(self, 'random_unlearn_portion') or hasattr(self, 'indexes_to_replace'):
            indices = list(range(len(self.test_dataset)))
        else:
            indices = [
                i for i, (_, label) in enumerate(self.test_dataset)
                if label not in query_digits
            ]
        return DataLoader(
            Subset(self.test_dataset, indices),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
        )
