"""Dataset preparation helper for cross-dataset experiments."""

import argparse
from pathlib import Path

import torchvision

if __package__:
    from .tiny_imagenet_datamodule import TinyImageNetDataModule
else:
    # Support the documented `python cross_dataset_support/download_datasets.py`
    # invocation as well as `python -m cross_dataset_support.download_datasets`.
    from cross_dataset_support.tiny_imagenet_datamodule import TinyImageNetDataModule


def prepare(args):
    if args.dataset == "cifar100":
        torchvision.datasets.CIFAR100(root=args.cifar_dir, train=True, download=True)
        torchvision.datasets.CIFAR100(root=args.cifar_dir, train=False, download=True)
        print(f"CIFAR-100 ready under {args.cifar_dir}")
    elif args.dataset == "fashion-mnist":
        torchvision.datasets.FashionMNIST(root=args.cifar_dir, train=True, download=True)
        torchvision.datasets.FashionMNIST(root=args.cifar_dir, train=False, download=True)
        print(f"Fashion-MNIST ready under {args.cifar_dir}")
    elif args.dataset == "tiny-imagenet":
        module = TinyImageNetDataModule(data_dir=args.tiny_imagenet_dir)
        module.prepare_data()
        print(f"Tiny ImageNet ready under {args.tiny_imagenet_dir}")
    else:
        path = Path(args.imagenet_dir)
        required = (path / "train", path / "val")
        missing = [str(item) for item in required if not item.exists()]
        if missing:
            raise RuntimeError(
                "ImageNet must be downloaded manually. Missing: "
                + ", ".join(missing)
                + ". Use ILSVRC/CLS-LOC train and val directories."
            )
        print(f"ImageNet directory structure found under {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=("cifar100", "fashion-mnist", "tiny-imagenet", "imagenet"),
    )
    parser.add_argument("--cifar-dir", default="./data")
    parser.add_argument("--tiny-imagenet-dir", default="./data/tiny-imagenet-200")
    parser.add_argument("--imagenet-dir", default="./data/imagenet")
    prepare(parser.parse_args())
