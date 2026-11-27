# Cross-dataset support

This folder contains isolated support for CIFAR-100, Fashion-MNIST, and Tiny ImageNet.
Existing CIFAR-10 and standard ImageNet configs/datamodules remain unchanged.

## Download and preparation

```bash
PYTHONPATH=. python cross_dataset_support/download_datasets.py cifar100
PYTHONPATH=. python cross_dataset_support/download_datasets.py fashion-mnist
PYTHONPATH=. python cross_dataset_support/download_datasets.py tiny-imagenet
PYTHONPATH=. python cross_dataset_support/download_datasets.py imagenet \
  --imagenet-dir ./data/imagenet
```

CIFAR-100 is downloaded automatically by torchvision. Tiny ImageNet is
downloaded and extracted automatically from the standard Stanford Tiny
ImageNet archive. Full ILSVRC ImageNet cannot be obtained through this helper
and must be downloaded manually because of its Kaggle/license requirements.

Expected ImageNet layout:

```text
data/imagenet/train/<synset>/<images>
data/imagenet/val/<synset>/<images>
```

## Configurations

- [`configs/cifar100_resnet18.yaml`](./configs/cifar100_resnet18.yaml)
- [`configs/cifar100_resnet18_kmeans_k1.yaml`](./configs/cifar100_resnet18_kmeans_k1.yaml)
- [`configs/fashion_mnist_resnet18.yaml`](./configs/fashion_mnist_resnet18.yaml)
- [`configs/fashion_mnist_resnet18_kmeans_k1.yaml`](./configs/fashion_mnist_resnet18_kmeans_k1.yaml)
- [`configs/tiny_imagenet_resnet18.yaml`](./configs/tiny_imagenet_resnet18.yaml)
- [`configs/tiny_imagenet_resnet18_kmeans_k3.yaml`](./configs/tiny_imagenet_resnet18_kmeans_k3.yaml)
- [`configs/imagenet_resnet18.yaml`](./configs/imagenet_resnet18.yaml)
- [`configs/imagenet_resnet18_kmeans_k3.yaml`](./configs/imagenet_resnet18_kmeans_k3.yaml)

Existing `config/classification/imagenet_resnet18.yaml` and
`imagenet_resnet18_kmeans_k3.yaml` remain the ImageNet configurations.

## Training

```bash
PYTHONPATH=. python main.py \
  --config cross_dataset_support/configs/cifar100_resnet18_kmeans_k1.yaml \
  --seed 7 --gpu-limit-pct 45

PYTHONPATH=. python main.py \
  --config cross_dataset_support/configs/fashion_mnist_resnet18.yaml \
  --seed 7 --gpu-limit-pct 45

PYTHONPATH=. python main.py \
  --config cross_dataset_support/configs/fashion_mnist_resnet18_kmeans_k1.yaml \
  --seed 7 --gpu-limit-pct 45

PYTHONPATH=. python main.py \
  --config cross_dataset_support/configs/tiny_imagenet_resnet18_kmeans_k3.yaml \
  --seed 7 --gpu-limit-pct 45

PYTHONPATH=. python main.py \
  --config config/classification/imagenet_resnet18_kmeans_k3.yaml \
  --seed 7 --gpu-limit-pct 45
```

## Random or EL2N/K-center selection

```bash
PYTHONPATH=. python scripts/select_random.py \
  --config cross_dataset_support/configs/cifar100_resnet18_kmeans_k1.yaml \
  --selection-pct 50 --class-wise --seed 7 --gpu-limit-pct 45

PYTHONPATH=. python scripts/select_random.py \
  --config cross_dataset_support/configs/fashion_mnist_resnet18.yaml \
  --selection-pct 50 --class-wise --seed 7 --gpu-limit-pct 45

PYTHONPATH=. python scripts/select_random.py \
  --config cross_dataset_support/configs/tiny_imagenet_resnet18_kmeans_k3.yaml \
  --selection-pct 50 --class-wise --seed 7 --gpu-limit-pct 45

PYTHONPATH=. python scripts/select_dataset.py \
  --config cross_dataset_support/configs/tiny_imagenet_resnet18_kmeans_k3.yaml \
  --train-pilot --pilot-epochs 10 \
  --kcenter-retention-pct 50 --seed 7 --gpu-limit-pct 45
```

Train with an artifact:

```bash
PYTHONPATH=. python -m overfitting_mitigation.train \
  --config cross_dataset_support/configs/cifar100_resnet18_kmeans_k1.yaml \
  --mitigation-config cross_dataset_support/examples/cifar100_mitigation.yaml \
  --dataset-reduction --selection-artifact <artifact.json> \
  --seed 7 --gpu-limit-pct 45

PYTHONPATH=. python -m overfitting_mitigation.train \
  --config cross_dataset_support/configs/fashion_mnist_resnet18.yaml \
  --mitigation-config overfitting_mitigation/examples/cifar_exp1_300epochs_v2_less_mitigation.yaml \
  --seed 7 --gpu-limit-pct 45

PYTHONPATH=. python -m overfitting_mitigation.train \
  --config cross_dataset_support/configs/tiny_imagenet_resnet18_kmeans_k3.yaml \
  --mitigation-config cross_dataset_support/examples/tiny_imagenet_mitigation.yaml \
  --dataset-reduction --selection-artifact <artifact.json> \
  --seed 7 --gpu-limit-pct 45
```

Full ImageNet uses the existing `ImageNetDataModule` and
`config/classification/imagenet_resnet18*.yaml`; pass the data directory with
`data_dir` in YAML and use `--gpu-limit-pct` for an RTX A4000.

## Fashion-MNIST and overfitting mitigation

Fashion-MNIST is converted from grayscale 28x28 images to normalized
three-channel 32x32 tensors so it can use the existing SPM ResNet encoder.
It is fully compatible with `overfitting_mitigation.train`; the command above
uses the existing mitigation configuration because mitigation is model/data
module agnostic. The validation and test sets remain untouched by optional
training-set reduction.
