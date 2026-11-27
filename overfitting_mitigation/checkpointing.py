"""Checkpoint metadata callback for reproducible mitigation experiments."""

import pytorch_lightning as pl


class MitigationMetadataCallback(pl.Callback):
    def __init__(self, config):
        self.config = config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["overfitting_mitigation"] = self.config.as_dict()
