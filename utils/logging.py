import torch
import pytorch_lightning as pl

class GPUMemoryCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self, trainer, pl_module):
        import gc
        gc.collect()
        torch.cuda.empty_cache()


class EpochMetricsConsoleCallback(pl.Callback):
    """Print the current learning rate each epoch and metrics periodically."""

    def __init__(self, interval=10):
        super().__init__()
        self.interval = interval

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.interval != 0:
            return

        metrics = trainer.callback_metrics

        def metric_value(name):
            value = metrics.get(name)
            return "n/a" if value is None else f"{value.detach().cpu().item():.4f}"

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss: {metric_value('train_loss')} | "
            f"train_acc: {metric_value('train_acc')} | "
            f"val_loss: {metric_value('val_loss')} | "
            f"val_acc: {metric_value('val_acc')}"
        )

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        learning_rates = [group["lr"] for group in trainer.optimizers[0].param_groups]
        formatted_lrs = ", ".join(f"{learning_rate:.6g}" for learning_rate in learning_rates)
        print(f"Epoch {epoch:03d} | learning_rate: {formatted_lrs}")