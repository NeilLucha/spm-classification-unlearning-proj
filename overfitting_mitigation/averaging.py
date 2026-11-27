"""EMA/SWA callbacks with evaluation-compatible averaged checkpoints."""

from collections import OrderedDict
from typing import Dict

import torch
import pytorch_lightning as pl


class WeightAveragingCallback(pl.Callback):
    def __init__(self, config):
        self.config = config
        self.ema = {} if "ema" in config.averaging else None
        self.swa = {} if "swa" in config.averaging else None
        self.swa_count = 0
        self._raw_state = None
        self._bn_state = {}

    def _averaged(self, model):
        result = OrderedDict(model.state_dict())
        if self.ema is not None and self.swa is not None and self.swa_count:
            for key in result:
                result[key] = (self.ema[key] + self.swa[key]) / 2
        elif self.ema is not None:
            result.update(self.ema)
        elif self.swa is not None and self.swa_count:
            result.update(self.swa)
        result.update(self._bn_state)
        return result

    @torch.no_grad()
    def _update_batch_norm(self, trainer, model):
        if not self.config.swa_bn_update or self.swa is None or not self.swa_count:
            return
        was_training = model.training
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()
                module.momentum = None
        loader = trainer.datamodule.train_dataloader()
        for batch_index, batch in enumerate(loader):
            if batch_index >= 64:
                break
            query, support, _, support_labels = batch
            model(query.to(model.device), support.to(model.device), support_labels.to(model.device))
        self._bn_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if "running_" in key or key.endswith("num_batches_tracked")
        }
        model.train(was_training)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        epoch = trainer.current_epoch
        if epoch < self.config.averaging_start_epoch:
            return
        if trainer.global_step % max(1, self.config.averaging_frequency):
            return
        current = pl_module.state_dict()
        if self.ema is not None:
            if not self.ema:
                self.ema = {k: v.detach().clone() for k, v in current.items()}
            else:
                decay = self.config.ema_decay
                for key, value in current.items():
                    if torch.is_floating_point(value):
                        self.ema[key].mul_(decay).add_(value.detach(), alpha=1 - decay)
                    else:
                        self.ema[key] = value.detach().clone()
        if self.swa is not None:
            if not self.swa_count:
                self.swa = {k: v.detach().clone() for k, v in current.items()}
            else:
                for key, value in current.items():
                    if torch.is_floating_point(value):
                        self.swa[key].add_((value.detach() - self.swa[key]) / (self.swa_count + 1))
                    else:
                        self.swa[key] = value.detach().clone()
            self.swa_count += 1

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.ema is None and not self.swa_count:
            return
        self._raw_state = OrderedDict(pl_module.state_dict())
        pl_module.load_state_dict(self._averaged(pl_module), strict=True)
        self._update_batch_norm(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._raw_state is not None:
            pl_module.load_state_dict(self._raw_state, strict=True)
            self._raw_state = None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is None and not self.swa_count:
            return
        checkpoint["state_dict"] = self._averaged(pl_module)
        checkpoint["averaging"] = {
            "method": self.config.averaging,
            "swa_count": self.swa_count,
            "config": self.config.as_dict(),
        }
