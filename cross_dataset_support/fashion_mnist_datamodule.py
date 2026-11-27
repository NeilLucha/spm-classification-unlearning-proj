"""Fashion-MNIST data module for the cross-dataset SPM experiments."""

from __future__ import annotations

import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from utils.dataset import (
    QuerySupportTestDataset,
    QuerySupportTrainDataset,
    query_support_test_collate_fn,
    query_support_train_collate_fn,
)


class FashionMNISTDataModule(pl.LightningDataModule):
    """Class-balanced episodic Fashion-MNIST data module."""

    def __init__(
        self,
        support_size_train=256,
        support_size_eval=1000,
        batch_size=128,
        num_classes=10,
        query_digits=None,
        support_digits=None,
        model_type="classifier",
        data_dir="./data",
        selection_artifact=None,
        reduction_enabled=False,
        **kwargs,
    ):
        super().__init__()
        if num_classes != 10:
            raise ValueError("FashionMNISTDataModule requires num_classes=10")
        if model_type not in ("classifier", "ablation"):
            raise ValueError(
                "FashionMNISTDataModule supports classifier and ablation models"
            )
        self.support_size_train = support_size_train
        self.support_size_eval = support_size_eval
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.query_digits = query_digits
        self.support_digits = support_digits
        self.data_dir = data_dir
        self.dataset_name = "fashion-mnist"
        self.selection_artifact = selection_artifact
        self.reduction_enabled = reduction_enabled
        normalize = transforms.Normalize((0.2860,) * 3, (0.3530,) * 3)
        self.train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ])

    def prepare_data(self):
        torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=True, download=True
        )
        torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=False, download=True
        )

    def raw_training_dataset(self):
        return torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=True, transform=self.test_transform
        )

    def setup(self, stage=None):
        if stage in ("fit", None):
            train_set = torchvision.datasets.FashionMNIST(
                root=self.data_dir, train=True, transform=self.train_transform
            )
            if self.reduction_enabled or self.selection_artifact:
                from utils.dataset_reduction import load_selection_artifact

                if not self.selection_artifact:
                    raise ValueError(
                        "Dataset reduction is enabled but selection_artifact is missing"
                    )
                artifact = load_selection_artifact(
                    self.selection_artifact,
                    len(train_set),
                    [int(target) for target in train_set.targets],
                )
                original_size = len(train_set)
                train_set = Subset(train_set, artifact["selected_indices"])
                self.selection_metadata = artifact
                print(
                    f"[FashionMNIST] Using {len(train_set)}/{original_size} "
                    f"selected training samples from {self.selection_artifact}",
                    flush=True,
                )
            self.train_dataset = QuerySupportTrainDataset(
                train_set, self.support_size_train, self.num_classes
            )
            test_set = torchvision.datasets.FashionMNIST(
                root=self.data_dir, train=False, transform=self.test_transform
            )
            self.val_dataset = QuerySupportTestDataset(
                test_set,
                train_set,
                self.support_size_eval,
                self.query_digits,
                self.support_digits,
                self.num_classes,
            )
            self.test_dataset = test_set
        elif stage in ("test", "finetune"):
            self.train_dataset = torchvision.datasets.FashionMNIST(
                root=self.data_dir, train=True, transform=self.train_transform
            )
            self.test_dataset = torchvision.datasets.FashionMNIST(
                root=self.data_dir, train=False, transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
            collate_fn=lambda batch: query_support_train_collate_fn(
                batch,
                self.train_dataset,
                self.support_size_train,
                self.num_classes,
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
            collate_fn=lambda batch: query_support_test_collate_fn(
                batch, self.val_dataset, self.support_size_eval
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
        )
