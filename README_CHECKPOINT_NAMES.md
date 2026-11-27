# Checkpoint Names

Checkpoints created through `main.py` include the main experiment settings in their filename.

Example:

```text
best-cifar10-c10-k1-lr0p1-wd1e-3-s256x4096-nols-plateau-scratch-seed0-epoch66-val_acc0.94.ckpt
```

## Name Fields

| Field | Meaning | Example |
|---|---|---|
| `best` / `last` | Checkpoint type | `best`, `last` |
| `cifar10` | Dataset | `cifar10`, `cifar100` |
| `c10` | Number of classes | `c10`, `c100` |
| `k1` | Prototypes per class | `k1`, `k3`, `k5` |
| `lr0p1` | Learning rate; `p` represents a decimal point | `lr0p1` = `0.1` |
| `wd1e-3` | Weight decay | `wd1e-3` = `0.001` |
| `s256x4096` | Train and evaluation support sizes | `256` train, `4096` evaluation |
| `nols` / `ls` | Label smoothing disabled/enabled | `nols`, `ls` |
| `plateau` | Adaptive `ReduceLROnPlateau` scheduler | `plateau` |
| `scratch` / `imgnet` | Encoder initialization | Randomly initialized / ImageNet-pretrained |
| `seed0` | Random seed | `seed0` |
| `epoch66` | Epoch at checkpoint save | `epoch66` |
| `val_acc0.94` | Validation accuracy at save time | `0.94` = `94%` |

## Examples

Default K=1 run without label smoothing:

```text
best-cifar10-c10-k1-lr0p1-wd1e-3-s256x4096-nols-plateau-scratch-seed0-epoch66-val_acc0.94.ckpt
```

K=3 run with label smoothing:

```text
best-cifar10-c10-k3-lr0p1-wd1e-3-s256x4096-ls-plateau-scratch-seed0-epoch42-val_acc0.93.ckpt
```

The corresponding `last-...ckpt` file uses the same experiment fields but does not include epoch or validation accuracy.

For standard `main.py` runs, the checkpoint directory also contains
`training_configuration_curves/last-<experiment-tag>.png` and the matching
`.csv` file. Their basename matches the stable `last-<experiment-tag>.ckpt`
checkpoint basename for full-dataset runs.

For reduced-dataset runs, curve basenames append the selection artifact
identifier:

```text
last-<experiment-tag>-sel<subset-id>.png
last-<experiment-tag>-sel<subset-id>.csv
```

This prevents two subsets with identical reduction percentages from
overwriting each other's curve files. The best checkpoint's epoch/accuracy
suffix remains dynamic.

## Optional overfitting-mitigation checkpoints

The isolated [`overfitting_mitigation/`](./overfitting_mitigation/) entry point
appends concise enabled-technique fields without changing ordinary names:

```text
aug-randaugment9-erase0p25-mixup0p2-wd0p001-drop0p1-ls0p1-ema-swa-best-epoch20-0.84.ckpt
```

The complete configuration is stored under the `overfitting_mitigation`
checkpoint key and is authoritative for reproducing the run.