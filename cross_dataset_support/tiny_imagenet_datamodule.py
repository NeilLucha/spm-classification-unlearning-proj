"""Tiny ImageNet data module kept separate from the existing datamodules."""

import os
import urllib.request
import zipfile
from pathlib import Path

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from utils.dataset import (
    QuerySupportTestDataset,
    QuerySupportTrainDataset,
    query_support_test_collate_fn,
    query_support_train_collate_fn,
)


TINY_IMAGENET_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_MEAN = (0.4802, 0.4481, 0.3975)
TINY_STD = (0.2302, 0.2265, 0.2262)


class TinyImageNetValDataset(Dataset):
    def __init__(self, root, wnid_to_index, transform=None):
        self.root = Path(root)
        annotation = self.root / "val_annotations.txt"
        self.samples = []
        with annotation.open("r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.strip().split("\t")
                if len(fields) < 2:
                    continue
                image_name, wnid = fields[:2]
                if wnid in wnid_to_index:
                    self.samples.append((
                        str(self.root / "images" / image_name),
                        wnid_to_index[wnid],
                    ))
        self.targets = [label for _, label in self.samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class TinyImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        support_size_train=256,
        support_size_eval=4096,
        batch_size=128,
        num_classes=200,
        query_digits=None,
        support_digits=None,
        model_type="classifier",
        data_dir="./data/tiny-imagenet-200",
        selection_artifact=None,
        reduction_enabled=False,
        **kwargs,
    ):
        super().__init__()
        self.support_size_train = support_size_train
        self.support_size_eval = support_size_eval
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.query_digits = query_digits
        self.support_digits = support_digits
        self.data_dir = os.path.abspath(data_dir)
        self.dataset_name = "tinyimagenet"
        self.selection_artifact = selection_artifact
        self.reduction_enabled = reduction_enabled
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TINY_MEAN, TINY_STD),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TINY_MEAN, TINY_STD),
        ])

    def prepare_data(self):
        if Path(self.data_dir).exists():
            return
        parent = Path(self.data_dir).parent
        parent.mkdir(parents=True, exist_ok=True)
        archive = parent / "tiny-imagenet-200.zip"
        if not archive.exists():
            print(f"[TinyImageNet] Downloading {TINY_IMAGENET_URL}", flush=True)
            urllib.request.urlretrieve(TINY_IMAGENET_URL, archive)
        print(f"[TinyImageNet] Extracting {archive}", flush=True)
        with zipfile.ZipFile(archive, "r") as handle:
            handle.extractall(parent)
        if not Path(self.data_dir).exists():
            raise RuntimeError(
                f"Tiny ImageNet extraction did not create {self.data_dir}"
            )

    def raw_training_dataset(self):
        return datasets.ImageFolder(
            os.path.join(self.data_dir, "train"),
            transform=self.test_transform,
        )

    def setup(self, stage=None):
        train_set = datasets.ImageFolder(
            os.path.join(self.data_dir, "train"),
            transform=self.train_transform,
        )
        wnids = sorted(os.path.basename(path) for path in
                       os.listdir(os.path.join(self.data_dir, "train"))
                       if os.path.isdir(os.path.join(self.data_dir, "train", path)))
        wnid_to_index = {wnid: index for index, wnid in enumerate(wnids)}
        val_set = TinyImageNetValDataset(
            os.path.join(self.data_dir, "val"),
            wnid_to_index,
            transform=self.test_transform,
        )
        if stage in ("fit", None):
            if self.reduction_enabled or self.selection_artifact:
                from utils.dataset_reduction import load_selection_artifact
                targets = [int(target) for target in train_set.targets]
                if not self.selection_artifact:
                    raise ValueError("Dataset reduction is enabled but selection_artifact is missing")
                artifact = load_selection_artifact(
                    self.selection_artifact, len(train_set), targets
                )
                train_set = Subset(train_set, artifact["selected_indices"])
                self.selection_metadata = artifact
                print(
                    f"[TinyImageNet] Using {len(artifact['selected_indices'])}/{len(targets)} "
                    f"training samples"
                )
            self.train_dataset = QuerySupportTrainDataset(
                train_set, self.support_size_train, self.num_classes
            )
            self.val_dataset = QuerySupportTestDataset(
                val_set,
                train_set,
                self.support_size_eval,
                self.query_digits,
                self.support_digits,
                self.num_classes,
            )
        elif stage in ("test", "finetune"):
            self.train_dataset = train_set
            self.test_dataset = val_set

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
            drop_last=True,
            collate_fn=lambda batch: query_support_train_collate_fn(
                batch, self.train_dataset, self.support_size_train, self.num_classes
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
            drop_last=True,
            collate_fn=lambda batch: query_support_test_collate_fn(
                batch, self.val_dataset, self.support_size_eval
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            getattr(self, "test_dataset"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
        )
