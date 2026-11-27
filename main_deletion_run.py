import argparse
import json
import os
import time
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from utils.config import load_config, instantiate_from_config
from utils.gpu_limit import apply_gpu_limit


def main(args):
    apply_gpu_limit(getattr(args, "gpu_limit_pct", 45.0))
    config = load_config(args.config)
    model = instantiate_from_config(config['model'])
    datamodule = instantiate_from_config(config['data'])

    lightning_config = config.get('lightning', {})
    logger = instantiate_from_config(lightning_config['logger']) if 'logger' in lightning_config else None
    callbacks = []
    checkpoint_dir = None
    if 'callbacks' in lightning_config and lightning_config['callbacks'] is not None:
        for name, cb in lightning_config['callbacks'].items():
            if cb is not None:
                callbacks.append(instantiate_from_config(cb))
                # detect checkpoint callback dir if provided in config
                if name == 'checkpoint_callback' and 'params' in cb and 'dirpath' in cb['params']:
                    checkpoint_dir = cb['params']['dirpath']

    trainer_config = lightning_config.get('trainer', {})
    trainer_config['logger'] = logger
    # Determine output directory for metrics (prefer checkpoint callback dir if set)
    out_dir = checkpoint_dir or os.path.join("results", "classification", "deletion_run")
    os.makedirs(out_dir, exist_ok=True)

    # Add monitoring callback to save metrics each epoch
    class TrainingMonitorCallback(Callback):
        def __init__(self, out_dir):
            super().__init__()
            self.out_dir = out_dir
            self.csv_path = os.path.join(out_dir, 'training_metrics.csv')
            self.json_path = os.path.join(out_dir, 'training_metrics.json')
            self.records = []

        def on_train_epoch_end(self, trainer, pl_module):
            # Collect relevant metrics from trainer.callback_metrics
            cm = trainer.callback_metrics
            epoch = int(trainer.current_epoch)
            rec = {
                'epoch': epoch,
                'timestamp': time.time(),
                'train_loss': float(cm.get('train_loss', float('nan'))),
                'train_loss_full': float(cm.get('train_loss_full', float('nan'))),
                'train_loss_deleted': float(cm.get('train_loss_deleted', float('nan'))),
                'train_acc': float(cm.get('train_acc', float('nan'))),
                'train_acc_deleted': float(cm.get('train_acc_deleted', float('nan'))),
                'val_loss': float(cm.get('val_loss', float('nan'))),
                'val_acc': float(cm.get('val_acc', float('nan'))),
            }
            self.records.append(rec)
            # Append to CSV (write header if new)
            write_header = not os.path.exists(self.csv_path)
            with open(self.csv_path, 'a') as f:
                if write_header:
                    f.write(','.join(rec.keys()) + '\n')
                f.write(','.join(str(rec[k]) for k in rec.keys()) + '\n')
            # Save JSON snapshot
            with open(self.json_path, 'w') as jf:
                json.dump(self.records, jf, indent=2)

    monitor = TrainingMonitorCallback(out_dir)
    callbacks.append(monitor)

    trainer_config['callbacks'] = callbacks
    trainer = pl.Trainer(**trainer_config)

    t0 = time.perf_counter()
    trainer.fit(model, datamodule=datamodule)
    t1 = time.perf_counter()
    print(f"Training completed in {t1 - t0:.2f} seconds")

    # Run test and capture metrics
    test_metrics = trainer.test(model, datamodule=datamodule)

    # Determine output directory for results
    out_dir = checkpoint_dir or os.path.join("results", "classification", "deletion_run")
    os.makedirs(out_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(out_dir, f"test_metrics_seed{args.seed}.json")
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # Additional deleted-support robustness evaluation on validation set
    def delete_support_episode(support_imgs, support_labels, deletion_fraction=0.2):
        # support_imgs: (N, C, H, W), support_labels: (N,)
        N = support_labels.size(0)
        num_delete = int(N * deletion_fraction)
        num_delete = max(1, num_delete)
        num_delete = min(num_delete, N - 1)

        deletion_candidates = []
        for lab in torch.unique(support_labels):
            class_indices = (support_labels == lab).nonzero(as_tuple=True)[0].tolist()
            if len(class_indices) > 1:
                deletion_candidates.extend(class_indices[:-1])
        if len(deletion_candidates) == 0:
            # nothing to delete
            mask = torch.ones(N, dtype=torch.bool)
            return support_imgs[mask], support_labels[mask]

        import numpy as np
        deletion_candidates = np.array(deletion_candidates)
        if len(deletion_candidates) >= num_delete:
            delete_indices = np.random.choice(deletion_candidates, num_delete, replace=False)
        else:
            delete_indices = deletion_candidates
        mask = torch.ones(N, dtype=torch.bool)
        mask[list(delete_indices)] = False
        return support_imgs[mask], support_labels[mask]

    # Evaluate robustness on validation set (a subset to save time)
    val_loader = datamodule.val_dataloader()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    n_batches = 0
    acc_full_total = 0.0
    acc_deleted_total = 0.0
    for batch in val_loader:
        q_imgs, supp_imgs, q_labels, supp_labels = batch
        q_imgs = q_imgs.to(device)
        supp_imgs = supp_imgs.to(device)
        q_labels = q_labels.to(device)
        supp_labels = supp_labels.to(device)
        with torch.no_grad():
            logits_full = model(q_imgs, supp_imgs, supp_labels)
            preds_full = torch.argmax(logits_full, dim=-1)
            acc_full = (preds_full == q_labels).float().mean().item()
            # build deleted support
            del_supp_imgs, del_supp_labels = delete_support_episode(supp_imgs, supp_labels, deletion_fraction= getattr(datamodule, 'deletion_fraction', 0.2))
            if del_supp_imgs.size(0) == 0:
                continue
            del_supp_imgs = del_supp_imgs.to(device)
            del_supp_labels = del_supp_labels.to(device)
            logits_del = model(q_imgs, del_supp_imgs, del_supp_labels)
            preds_del = torch.argmax(logits_del, dim=-1)
            acc_del = (preds_del == q_labels).float().mean().item()
        acc_full_total += acc_full
        acc_deleted_total += acc_del
        n_batches += 1
        if n_batches >= 50:
            break

    if n_batches > 0:
        robustness = {
            'val_acc_full_avg': acc_full_total / n_batches,
            'val_acc_deleted_avg': acc_deleted_total / n_batches,
            'val_deleted_gap': (acc_full_total - acc_deleted_total) / n_batches,
        }
        rob_path = os.path.join(out_dir, f'val_deleted_robustness_seed{args.seed}.json')
        with open(rob_path, 'w') as rf:
            json.dump(robustness, rf, indent=2)
        print('Saved deleted-support robustness metrics to', rob_path)

    print(f"Test metrics saved to {metrics_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument("--gpu-limit-pct", type=float, default=45.0,
                        help="Per-process CUDA memory limit percentage (1-100)")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    main(args)
