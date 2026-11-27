import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models.spm_classifier_deletion import SemiParametricClassifier


class SemiParametricClassifierAlt(SemiParametricClassifier):
    """Alternate deletion-aware classifier.

    Extends `SemiParametricClassifier` to add an auxiliary loss computed
    on a support set with a fraction of samples removed. The training
    dataloader should supply the 6-tuple returned by
    `query_support_train_collate_fn_deletion` from
    `utils/dataset_deletion.py`.
    """

    def __init__(self, *args, deletion_loss_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.deletion_loss_weight = deletion_loss_weight

    def training_step(self, batch, batch_idx):
        # Support both standard (4-tuple) and deletion-aware (6-tuple) batches
        if not isinstance(batch, (list, tuple)):
            return super().training_step(batch, batch_idx)

        if len(batch) == 4:
            return super().training_step(batch, batch_idx)

        # Unpack deletion-aware batch: expected 6 items
        try:
            (query_img, support_imgs, query_label, support_labels,
             deleted_support_imgs, deleted_support_labels) = batch
        except Exception:
            # Fallback to original behavior
            return super().training_step(batch, batch_idx)

        # Full-support forward + loss
        logits_full = self(query_img, support_imgs, support_labels)
        loss_full = self.loss_fn(logits_full, query_label)

        # Deleted-support forward + loss
        logits_deleted = self(query_img, deleted_support_imgs, deleted_support_labels)
        loss_deleted = self.loss_fn(logits_deleted, query_label)

        # Combined loss
        loss = loss_full + self.deletion_loss_weight * loss_deleted

        # Logging
        self.log("train_loss_full", loss_full, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train_loss_deleted", loss_deleted, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)

        preds = torch.argmax(logits_full, dim=-1)
        acc = (preds == query_label).float().mean()
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=False)

        # Deleted-support accuracy (useful to track robustness)
        preds_del = torch.argmax(logits_deleted, dim=-1)
        acc_del = (preds_del == query_label).float().mean()
        self.log("train_acc_deleted", acc_del, prog_bar=False, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        # Accept 4- or 6-tuple; use the full support for validation metrics
        if len(batch) == 4:
            return super().validation_step(batch, batch_idx)

        query_img, support_imgs, query_label, support_labels, *_ = batch
        logits = self(query_img, support_imgs, support_labels)
        loss = self.loss_fn(logits, query_label)
        self.log("val_loss", loss, prog_bar=True)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == query_label).float().mean()
        self.log("val_acc", acc, prog_bar=True)
