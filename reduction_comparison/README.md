# Dataset reduction comparison

This folder contains isolated implementations of:

- **Herding**: class-wise greedy selection of examples closest to matching the
  class feature mean.
- **Moderate Coreset** (Xia et al., ICLR 2023): class-wise retention of samples
  whose embedding distance from the class mean is nearest the class median.

Artifacts use the existing `spm_dataset_reduction` JSON format, so they can be
passed directly to `main.py` or `overfitting_mitigation.train` with
`--dataset-reduction --selection-artifact <json>`.

## Output locations

```text
reduction_comparison/
├── select_reduction.py
├── methods.py
├── subsets/
└── README.md
```

## Herding

```bash
PYTHONPATH=. python reduction_comparison/select_reduction.py \
  --config config/classification/cifar_resnet18.yaml \
  --method herding \
  --retention-pct 50 \
  --checkpoint results/classification/cifar_resnet18/last.ckpt \
  --seed 7 \
  --batch-size 64 \
  --gpu-limit-pct 45
```

## Moderate Coreset

```bash
PYTHONPATH=. python reduction_comparison/select_reduction.py \
  --config config/classification/cifar_resnet18.yaml \
  --method moderate \
  --retention-pct 50 \
  --checkpoint results/classification/cifar_resnet18/last.ckpt \
  --seed 7 \
  --batch-size 64 \
  --gpu-limit-pct 45
```

The checkpoint is optional but strongly recommended for meaningful feature
representations. Without it, the encoder is initialized from the model config
and is not a trained representation.

## CIFAR-100 and Tiny ImageNet

Use the dedicated config paths:

```bash
PYTHONPATH=. python reduction_comparison/select_reduction.py \
  --config cross_dataset_support/configs/cifar100_resnet18_kmeans_k1.yaml \
  --method moderate --retention-pct 50 --seed 7 --gpu-limit-pct 45

PYTHONPATH=. python reduction_comparison/select_reduction.py \
  --config cross_dataset_support/configs/tiny_imagenet_resnet18_kmeans_k3.yaml \
  --method herding --retention-pct 50 --seed 7 --gpu-limit-pct 45
```

## Train with the generated subset

```bash
PYTHONPATH=. python main.py \
  --config config/classification/cifar_resnet18.yaml \
  --dataset-reduction \
  --selection-artifact reduction_comparison/subsets/herding-selection-<id>.json \
  --seed 7 \
  --gpu-limit-pct 45

PYTHONPATH=. python -m overfitting_mitigation.train \
  --config config/classification/cifar_resnet18.yaml \
  --mitigation-config overfitting_mitigation/examples/cifar.yaml \
  --dataset-reduction \
  --selection-artifact reduction_comparison/subsets/moderate-selection-<id>.json \
  --seed 7 \
  --gpu-limit-pct 45
```

## Method notes

Moderate Coreset is implemented in encoder feature space: each class mean is
computed, sample distances to that mean are calculated, and samples nearest
the median distance are retained. This differs from EL2N/K-center and is
reproducible with `--seed`. Herding is a class-wise greedy feature-matching
method. Both retain a class-balanced fraction approximately equal to
`--retention-pct`.
