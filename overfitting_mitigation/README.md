# Optional SPM overfitting mitigation

This package is disabled unless `overfitting_mitigation.train` is used. It
patches the instantiated SPM run instead of changing the original model,
datamodules, evaluation scripts, or checkpoint loading format.

## Techniques

* **Weight decay** adds an L2 penalty through existing optimizer parameter
  groups. It is unset by default.
* **Feature dropout** applies inverted dropout to encoder embeddings.
* **Label smoothing** retains the SPM NLL log-probability objective while
  replacing one-hot targets with a uniform mixture.
* **RandAugment/AugMix and random erasing** affect training images only.
* **Mixup/CutMix** affect query images only. Support images and labels remain
  valid for the support-conditioned episode.
* **EMA/SWA** average weights after a configurable epoch. Validation uses the
  averaged weights temporarily. SWA recalculates BatchNorm statistics from up
  to 64 training episodes before validation/checkpointing.
* **Early stopping and gradient clipping** use Lightning's standard behavior.

## Configuration reference

All keys belong under `overfitting_mitigation:` in YAML. Values shown in the
last column are the defaults. `null` disables an optional feature.

| Parameter | Type and allowed values | Meaning | Default |
|---|---|---|---|
| `weight_decay` | `null` or float `>= 0` | Override optimizer weight decay | `null` |
| `decoupled_weight_decay` | `true` or `false` | Request decoupled decay; currently rejected for the base SGD optimizer | `false` |
| `dropout` | float in `[0, 1)` | Encoder feature-dropout probability | `0.0` |
| `label_smoothing` | float in `[0, 1)` | Uniform-target smoothing factor | `0.0` |
| `randaugment` | `true` or `false` | Enable RandAugment | `false` |
| `randaugment_num_ops` | Integer `>= 0` | RandAugment operations per image | `2` |
| `randaugment_magnitude` | Integer (torchvision-supported range, normally `0-31`) | RandAugment operation strength | `9` |
| `augmix` | `true` or `false` | Enable AugMix instead of RandAugment | `false` |
| `random_erasing` | float `>= 0` (recommended `[0, 1]`) | Random Erasing probability | `0.0` |
| `mixup` | float `>= 0` | Mixup Beta-distribution alpha; `0` disables | `0.0` |
| `cutmix` | float `>= 0` | CutMix Beta-distribution alpha; `0` disables | `0.0` |
| `averaging` | `none`, `ema`, `swa`, or `ema+swa` | Weight averaging method | `none` |
| `averaging_start_epoch` | Integer `>= 0` | Epoch at which averaging starts | `0` |
| `averaging_frequency` | Integer `>= 1` | Update interval in optimizer steps | `1` |
| `ema_decay` | Float strictly in `(0, 1)` | EMA decay coefficient | `0.999` |
| `swa_bn_update` | `true` or `false` | Recalculate BatchNorm statistics for SWA | `true` |
| `early_stopping_patience` | `null` or integer `>= 0` | Number of unimproved validation checks before stopping | `null` |
| `early_stopping_monitor` | String metric name, typically `val_loss` or `val_acc` | Lightning metric to monitor | `val_loss` |
| `gradient_clip_val` | `null` or float `>= 0` | Maximum gradient norm/value configured in Lightning | `null` |
| `max_epochs` | `null` or integer `>= 1` | Override the base Lightning trainer's maximum epoch count | `null` |
| `seed` | `null` or integer | Reproducibility seed | `null` |

Boolean switches are independent except that `randaugment` and `augmix`
should not normally be enabled together; RandAugment takes precedence in the
current transform builder. `mixup` and `cutmix` may be enabled together, in
which case each mixed batch randomly selects one method. Positive Mixup and
CutMix values are Beta distribution concentration parameters, not direct
probabilities.

## Commands

```bash
# Individual techniques
PYTHONPATH=. python -m overfitting_mitigation.train --config config/classification/cifar_resnet18.yaml --weight-decay 0.001
PYTHONPATH=. python -m overfitting_mitigation.train --config config/classification/cifar_resnet18.yaml --dropout 0.1
PYTHONPATH=. python -m overfitting_mitigation.train --config config/classification/cifar_resnet18.yaml --label-smoothing 0.1
PYTHONPATH=. python -m overfitting_mitigation.train --config config/classification/cifar_resnet18.yaml --mixup 0.2

# CutMix or AugMix/SWA (put the latter options in a YAML mitigation config)
PYTHONPATH=. python -m overfitting_mitigation.train --config config/classification/cifar_resnet18.yaml --cutmix 1.0

# Combination
PYTHONPATH=. python -m overfitting_mitigation.train \
  --config config/classification/cifar_resnet18.yaml \
  --mitigation-config overfitting_mitigation/examples/cifar.yaml

# Train using a previously generated dataset-selection artifact
PYTHONPATH=. python -m overfitting_mitigation.train \
  --config config/classification/cifar_resnet18.yaml \
  --mitigation-config overfitting_mitigation/examples/cifar.yaml \
  --dataset-reduction \
  --selection-artifact results/dataset_selection/cifar10/selection-<id>.json \
  --seed 7
```

The same entry point accepts the ImageNet config. Use `--seed N` or the YAML
`seed` for reproducibility. Begin with weight decay `1e-3`, dropout `0.1`,
label smoothing `0.1`, RandAugment magnitude `9`, or EMA decay `0.999`.
The selection artifact is validated and its metadata is stored in the
checkpoint under `dataset_reduction`, matching `main.py`. If the artifact
path and reduction settings are present in the YAML, `--dataset-reduction`
and `--selection-artifact` are not required.

Reduced-dataset runs append a distinct suffix to checkpoint, PNG, and CSV
names, including the retention percentage and selection-artifact identifier,
for example:

```text
last-aug-randaugment7-...-maxep300-red-r50-selselection-0ae315d97782
```

Non-reduced runs retain their existing names, so reduced and full-dataset
outputs cannot overwrite one another.

For K-means CIFAR configs such as
`config/classification/cifar_resnet18_kmeans_k3.yaml`, the base model uses
`ReduceLROnPlateau`: it monitors `val_loss` once per epoch, reduces each
optimizer parameter group's learning rate by `0.1` after 5 unimproved
validation epochs, and stops at `1e-5`. The mitigation wrapper preserves this
scheduler and its monitor; configured weight decay is applied to the existing
optimizer groups without replacing the scheduler.

## Evaluation and compatibility

The normal Lightning `state_dict` is retained, so averaged checkpoints can be
loaded by the existing evaluator:

```bash
PYTHONPATH=. python evaluate.py --ckpt_path results/overfitting_mitigation/<checkpoint>.ckpt \
  --config config/classification/cifar_resnet18.yaml --model_type classifier
```

The full configuration is stored under the `overfitting_mitigation`
checkpoint key. Ordinary `main.py` runs, checkpoint names, and evaluation
remain unchanged. Decoupled decay is rejected for the current SGD optimizer
rather than silently changing optimizer semantics. EMA and SWA together are
supported and their shadows are averaged equally.

If `max_epochs` is set in the mitigation YAML, it overrides
`lightning.trainer.max_epochs` from the model config for this entry point only.
The original config file and ordinary `main.py` training are unchanged. The
value is also included in mitigation checkpoint and curve filenames.

## Live training curves and metric interpretation

Each mitigation run updates a four-panel PNG after the training epoch and
after validation completes:

```text
overfitting_mitigation/training_configuration_curves/<model-config-and-parameters>.png
```

The same updates also write a CSV with the identical stem:

```text
overfitting_mitigation/training_configuration_curves/<model-config-and-parameters>.csv
```

Its columns are `epoch`, `train_acc`, `val_acc`, `train_loss`, and
`val_loss`. Values are left blank until their corresponding training or
validation event has completed.

The filename starts with the YAML model configuration stem and then includes
learning rate, support sizes, seed, and mitigation suffix, for example:

```text
cifar_resnet18_kmeans_k3-lr0p1-wdbase-s256x4096-seed7-aug-randaugment9-mixup0p2-ema-swa.png
```

Reduced-dataset curve names append the actual selection-artifact identifier
instead of relying only on percentages:

```text
cifar_resnet18_kmeans_k3-lr0p1-wdbase-s256x4096-seed7-aug-randaugment9-red-r50-sel4ac713aad149.png
```

The matching CSV has the same basename with a `.csv` extension.

Training accuracy and loss are updated after every training epoch, while
validation accuracy and loss are only read and updated after each completed
validation run. The progress bar reports the running, example-weighted aggregates
`train_acc_epoch` and `train_loss_epoch` after every training batch; no
`train_acc_step` or `train_loss_step` values are emitted by the mitigation
entry point. Thus the displayed values evolve throughout each epoch while
converging to the epoch aggregate.

Training and validation accuracy are not expected to be identical. Training
accuracy is measured while the model is in training mode, with stochastic
support/query episode sampling and any enabled RandAugment, erasing, Mixup,
CutMix, or dropout. Validation uses evaluation mode and deterministic query
transforms, so it is an easier and cleaner measurement. This difference is
especially visible for K-means SPMs because each training episode samples
support examples and recomputes prototypes, whereas validation evaluates
different query/support episodes. A lower training value therefore does not
by itself indicate a logging error; persistent divergence after disabling
stochastic techniques is the more useful overfitting diagnostic.
