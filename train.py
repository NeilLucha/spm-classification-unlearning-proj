"""Opt-in training entry point.

Example:
  PYTHONPATH=. python -m overfitting_mitigation.train --config config/classification/cifar_resnet18.yaml --mitigation-config overfitting_mitigation/examples/cifar.yaml
"""

import argparse
import copy
import random
import time

import numpy as np
import pytorch_lightning as pl
import torch

from utils.config import instantiate_from_config, load_config
from utils.gpu_limit import apply_gpu_limit
from .callbacks import build_callbacks
from .config import load_mitigation_config
from .integration import configure_datamodule, configure_model


def main(args):
    apply_gpu_limit(getattr(args, "gpu_limit_pct", 45.0))
    config = load_config(args.config)
    mitigation = load_mitigation_config(args.mitigation_config, {
        "seed": args.seed,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "mixup": args.mixup,
        "cutmix": args.cutmix,
    })
    if mitigation.seed is not None:
        pl.seed_everything(mitigation.seed, workers=True)
        random.seed(mitigation.seed)
        np.random.seed(mitigation.seed)
        torch.manual_seed(mitigation.seed)
    model = configure_model(instantiate_from_config(config["model"]), mitigation)
    datamodule = configure_datamodule(instantiate_from_config(config["data"]), mitigation)
    lightning = copy.deepcopy(config.get("lightning", {}))
    trainer_config = dict(lightning.get("trainer", {}))
    trainer_config["logger"] = None
    trainer_config["callbacks"] = build_callbacks(mitigation)
    if mitigation.gradient_clip_val is not None:
        trainer_config["gradient_clip_val"] = mitigation.gradient_clip_val
    suffix = mitigation.filename_suffix()
    for callback in lightning.get("callbacks", {}).values():
        if callback and callback.get("target", "").endswith("ModelCheckpoint"):
            params = callback.setdefault("params", {})
            params["filename"] = f"{suffix}-best-{{epoch:02d}}-{{val_acc:.2f}}"
            params["dirpath"] = params.get("dirpath", "./results/overfitting_mitigation")
            trainer_config["callbacks"].append(instantiate_from_config(callback))
    trainer = pl.Trainer(**trainer_config)
    started = time.perf_counter()
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)
    print(f"Training completed in {time.perf_counter() - started:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-limit-pct", type=float, default=45.0,
                        help="Per-process CUDA memory limit percentage (1-100)")
    parser.add_argument("--mitigation-config", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--mixup", type=float, default=None)
    parser.add_argument("--cutmix", type=float, default=None)
    main(parser.parse_args())
