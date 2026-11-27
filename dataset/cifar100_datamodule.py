import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import pytorch_lightning as pl

from utils.dataset import (
    QuerySupportTrainDataset,
    QuerySupportTestDataset,
)


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _sample_balanced_indices(
    label_to_indices, support_size, num_classes, excluded=None, labels=None
):
    """Sample approximately the same number of examples for every class."""
    excluded = set() if excluded is None else set(excluded)
    labels = list(range(num_classes)) if labels is None else sorted(set(labels))
    if not labels:
        return []
    base, extra = divmod(support_size, len(labels))
    sampled = []

    for position, label in enumerate(labels):
        candidates = [index for index in label_to_indices.get(label, []) if index not in excluded]
        if not candidates:
            continue
        count = base + (position < extra)
        replace = len(candidates) < count
        sampled.extend(np.random.choice(candidates, count, replace=replace).tolist())

    return [int(index) for index in sampled]


def query_support_train_collate_fn_cifar100(batch, support_dataset, support_size, num_classes):
    """Build a training episode with a balanced support set over all classes."""
    query_imgs, query_labels, query_indices = zip(*batch)
    query_imgs = torch.stack(query_imgs, dim=0)
    query_labels = torch.tensor(query_labels, dtype=torch.long)

    query_raw_indices = [support_dataset.indices[index] for index in query_indices]
    support_indices = _sample_balanced_indices(
        support_dataset.label_to_indices,
        support_size,
        num_classes,
        excluded=query_raw_indices,
        # Match the working CIFAR/ImageNet episodic sampler: support classes
        # are the classes represented by the current query episode.
        labels=torch.unique(query_labels).tolist(),
    )
    support_data = [support_dataset.dataset[index] for index in support_indices]
    support_imgs = torch.stack([
        image if image.dim() == 3 else image.unsqueeze(0)
        for image, _ in support_data
    ], dim=0)
    support_labels = torch.tensor([label for _, label in support_data], dtype=torch.long)

    return query_imgs, support_imgs, query_labels, support_labels


def query_support_test_collate_fn_cifar100(batch, support_dataset, support_size):
    """Build a validation episode with a balanced support set over all classes."""
    query_imgs, query_labels, _ = zip(*batch)
    query_imgs = torch.stack(query_imgs, dim=0)
    query_labels = torch.tensor(query_labels, dtype=torch.long)

    support_label_to_indices = {}
    support_targets = support_dataset.support_dataset.targets
    for index in support_dataset.support_indices:
        label = support_targets[index]
        support_label_to_indices.setdefault(int(label), []).append(index)

    support_indices = _sample_balanced_indices(
        support_label_to_indices,
        support_size,
        support_dataset.num_classes,
    )
    support_data = [support_dataset.support_dataset[index] for index in support_indices]
    support_imgs = torch.stack([
        image if image.dim() == 3 else image.unsqueeze(0)
        for image, _ in support_data
    ], dim=0)
    support_labels = torch.tensor([label for _, label in support_data], dtype=torch.long)

    return query_imgs, support_imgs, query_labels, support_labels


class CIFAR100DataModule(pl.LightningDataModule):
    """CIFAR-100 data module with class-balanced SPM support sampling."""

    def __init__(
        self,
        support_size_train=1000,
        support_size_eval=5000,
        # Previously 256. CIFAR-100 episodes encode query and support images
        # together, so keep the default aligned with the base config.
        batch_size=64,
        num_classes=100,
        query_digits=None,
        support_digits=None,
        model_type="classifier",
        selection_artifact=None,
        reduction_enabled=False,
        **kwargs,
    ):
        super().__init__()
        if num_classes != 100:
            raise ValueError("CIFAR100DataModule requires num_classes=100")
        if model_type not in ("classifier", "ablation"):
            raise ValueError("CIFAR100DataModule supports classifier and ablation models")

        self.support_size_train = support_size_train
        self.support_size_eval = support_size_eval
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.query_digits = query_digits
        self.support_digits = support_digits
        self.selection_artifact = selection_artifact
        self.reduction_enabled = reduction_enabled
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])

    def prepare_data(self):
        torchvision.datasets.CIFAR100(root="./data", train=True, download=True)
        torchvision.datasets.CIFAR100(root="./data", train=False, download=True)

    def raw_training_dataset(self):
        return torchvision.datasets.CIFAR100(
            root="./data", train=True, transform=self.test_transform
        )

    def setup(self, stage=None):
        if stage in ("fit", None):
            cifar_train = torchvision.datasets.CIFAR100(
                root="./data", train=True, transform=self.train_transform
            )
            if self.reduction_enabled or self.selection_artifact:
                from torch.utils.data import Subset
                from utils.dataset_reduction import load_selection_artifact
                targets = [int(target) for target in cifar_train.targets]
                if not self.selection_artifact:
                    raise ValueError("Dataset reduction is enabled but selection_artifact is missing")
                artifact = load_selection_artifact(
                    self.selection_artifact, len(cifar_train), targets
                )
                cifar_train = Subset(cifar_train, artifact["selected_indices"])
                self.selection_metadata = artifact
                print(
                    f"[DataModule] Using {len(artifact['selected_indices'])}/{len(targets)} "
                    f"CIFAR-100 training samples from {self.selection_artifact}"
                )
            self.train_dataset = QuerySupportTrainDataset(
                cifar_train, self.support_size_train, self.num_classes
            )
            cifar_test = torchvision.datasets.CIFAR100(
                root="./data", train=False, transform=self.test_transform
            )
            self.val_dataset = QuerySupportTestDataset(
                cifar_test,
                cifar_train,
                self.support_size_eval,
                self.query_digits,
                self.support_digits,
                self.num_classes,
            )
            self.test_dataset = cifar_test
        elif stage in ("test", "finetune"):
            self.train_dataset = torchvision.datasets.CIFAR100(
                root="./data", train=True, transform=self.train_transform
            )
            self.test_dataset = torchvision.datasets.CIFAR100(
                root="./data", train=False, transform=self.test_transform
            )

    def train_dataloader(self):
        if not isinstance(self.train_dataset, QuerySupportTrainDataset):
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=6,
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
            collate_fn=lambda batch: query_support_train_collate_fn_cifar100(
                batch, self.train_dataset, self.support_size_train, self.num_classes
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
            collate_fn=lambda batch: query_support_test_collate_fn_cifar100(
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
