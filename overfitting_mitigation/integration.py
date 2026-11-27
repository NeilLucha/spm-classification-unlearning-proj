"""Runtime integration that leaves the ordinary SPM entry point untouched."""

from types import MethodType
import torch
import torch.nn.functional as F
from torch import nn

from .augmentations import build_image_transform, mix_query_batch
from .losses import LabelSmoothingNLLLoss, mixed_nll


def _replace_dataset_transform(dataset, transform, seen=None):
    seen = seen or set()
    if id(dataset) in seen:
        return
    seen.add(id(dataset))
    if hasattr(dataset, "transform"):
        dataset.transform = transform
    for name in ("dataset", "query_dataset", "support_dataset"):
        child = getattr(dataset, name, None)
        if child is not None and child is not dataset:
            _replace_dataset_transform(child, transform, seen)


def configure_datamodule(datamodule, config):
    """Install strong transforms after the original datamodule setup."""
    if not (config.randaugment or config.augmix or config.random_erasing):
        return datamodule
    original_setup = datamodule.setup

    def setup(self, stage=None):
        original_setup(stage)
        base = getattr(self, "train_transform", None)
        if base is None:
            return
        transform = build_image_transform(
            base,
            randaugment=config.randaugment,
            randaugment_num_ops=config.randaugment_num_ops,
            randaugment_magnitude=config.randaugment_magnitude,
            augmix=config.augmix,
            random_erasing=config.random_erasing,
        )
        self.train_transform = transform
        train_dataset = getattr(self, "train_dataset", None)
        if train_dataset is not None:
            _replace_dataset_transform(train_dataset, transform)
        # Validation queries must retain their deterministic test transform.
        # Its support set is drawn from the training set and may use training
        # augmentation independently.
        val_dataset = getattr(self, "val_dataset", None)
        support_dataset = getattr(val_dataset, "support_dataset", None)
        if support_dataset is not None:
            _replace_dataset_transform(support_dataset, transform)

    datamodule.setup = MethodType(setup, datamodule)
    return datamodule


def configure_model(model, config):
    """Add opt-in loss, feature dropout, mixup/cutmix, and optimizer behavior."""
    if config.dropout:
        def dropout_hook(_module, _inputs, output):
            return F.dropout(output, p=config.dropout, training=model.training)
        model.encoder.register_forward_hook(dropout_hook)
    if config.label_smoothing:
        model.loss_fn = LabelSmoothingNLLLoss(config.label_smoothing)

    original_epoch_start = getattr(model, "on_train_epoch_start", None)

    def on_train_epoch_start(self):
        self._mitigation_train_correct = 0
        self._mitigation_train_examples = 0
        self._mitigation_train_loss = 0.0
        if original_epoch_start is not None:
            original_epoch_start()

    def training_step(self, batch, batch_idx):
        query, support, labels, support_labels = batch
        if config.mixup or config.cutmix:
            query, target_a, target_b, coefficient = mix_query_batch(
                query, labels, mixup_alpha=config.mixup, cutmix_alpha=config.cutmix
            )
            log_probs = self(query, support, support_labels)
            loss = mixed_nll(self.loss_fn, log_probs, target_a, target_b, coefficient)
        else:
            log_probs = self(query, support, support_labels)
            loss = self.loss_fn(log_probs, labels)
        batch_size = labels.size(0)
        self._mitigation_train_correct += int((log_probs.argmax(-1) == labels).sum().item())
        self._mitigation_train_examples += batch_size
        self._mitigation_train_loss += float(loss.detach().item()) * batch_size
        running_accuracy = self._mitigation_train_correct / self._mitigation_train_examples
        running_loss = self._mitigation_train_loss / self._mitigation_train_examples
        self.log(
            "train_loss_epoch",
            running_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        self.log(
            "train_acc_epoch",
            running_accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        return loss

    # All optional runs use epoch aggregation, including no-mixup runs. This
    # prevents the base K-means running metric from appearing as train_acc_step.
    model.on_train_epoch_start = MethodType(on_train_epoch_start, model)
    model.training_step = MethodType(training_step, model)

    if config.weight_decay is not None or config.decoupled_weight_decay:
        original_configure = model.configure_optimizers

        def configure_optimizers(self):
            configured = original_configure()
            if isinstance(configured, dict):
                optimizer = configured["optimizer"]
            elif isinstance(configured, (list, tuple)):
                optimizer = configured[0]
                if isinstance(optimizer, (list, tuple)):
                    optimizer = optimizer[0]
            else:
                optimizer = configured
            if config.weight_decay is not None:
                for group in optimizer.param_groups:
                    group["weight_decay"] = config.weight_decay
            if config.decoupled_weight_decay and optimizer.__class__.__name__ == "Adam":
                raise ValueError(
                    "decoupled_weight_decay requires an AdamW optimizer in the base model; "
                    "the current SPM optimizer is SGD"
                )
            return configured
        model.configure_optimizers = MethodType(configure_optimizers, model)
    return model
