"""Optional Lightning callbacks used by the mitigation entry point."""

from pathlib import Path
import csv

import pytorch_lightning as pl


def build_callbacks(config):
    from .checkpointing import MitigationMetadataCallback
    callbacks = [MitigationMetadataCallback(config)]
    if config.early_stopping_patience is not None:
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor=config.early_stopping_monitor,
            patience=config.early_stopping_patience,
            mode="min" if "loss" in config.early_stopping_monitor else "max",
        ))
    if config.averaging != "none":
        from .averaging import WeightAveragingCallback
        callbacks.append(WeightAveragingCallback(config))
    return callbacks


class TrainingCurvesCallback(pl.Callback):
    """Persist live train/validation metric curves after training events."""

    def __init__(self, path):
        self.path = Path(path)
        self.csv_path = self.path.with_suffix(".csv")
        self.history = {
            "train_acc": [],
            "val_acc": [],
            "train_loss": [],
            "val_loss": [],
        }

    @staticmethod
    def _value(metrics, *names):
        for name in names:
            value = metrics.get(name)
            if value is not None:
                return float(
                    value.detach().cpu().item() if hasattr(value, "detach") else value
                )
        return None

    def _record(self, trainer, names):
        metrics = dict(trainer.callback_metrics)
        metrics.update(getattr(trainer, "logged_metrics", {}))
        metrics.update(getattr(trainer, "progress_bar_metrics", {}))
        for name in names:
            if name == "train_acc":
                value = self._value(metrics, "train_acc_epoch", "train_acc")
            elif name == "train_loss":
                value = self._value(metrics, "train_loss_epoch", "train_loss")
            else:
                value = self._value(metrics, name)
            if value is not None:
                values = self.history[name]
                if len(values) <= trainer.current_epoch:
                    values.extend([None] * (trainer.current_epoch + 1 - len(values)))
                values[trainer.current_epoch] = value
        self._save()

    def _save(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.path.parent.mkdir(parents=True, exist_ok=True)
        epochs = range(1, max((len(values) for values in self.history.values()), default=0) + 1)
        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        plots = (
            (axes[0, 0], "train_acc", "Training accuracy"),
            (axes[0, 1], "val_acc", "Validation accuracy"),
            (axes[1, 0], "train_loss", "Training loss"),
            (axes[1, 1], "val_loss", "Validation loss"),
        )
        for axis, name, title in plots:
            values = self.history[name]
            axis.plot(list(epochs)[:len(values)], values, marker="o", label=name)
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(True, alpha=0.3)
            axis.legend()
        figure.suptitle("SPM training curves")
        figure.tight_layout()
        figure.savefig(self.path, dpi=140)
        plt.close(figure)
        self._save_csv()

    def _save_csv(self):
        max_epochs = max((len(values) for values in self.history.values()), default=0)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", *self.history])
            for epoch in range(max_epochs):
                writer.writerow([
                    epoch + 1,
                    *(self.history[name][epoch] if epoch < len(self.history[name])
                      else None for name in self.history),
                ])

    def on_train_epoch_end(self, trainer, pl_module):
        self._record(trainer, ("train_acc", "train_loss"))

    def on_validation_epoch_end(self, trainer, pl_module):
        self._record(trainer, ("val_acc", "val_loss"))
