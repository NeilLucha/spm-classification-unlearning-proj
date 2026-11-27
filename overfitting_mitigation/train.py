"""Opt-in training entry point that leaves the ordinary SPM pipeline untouched."""

import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from utils.config import instantiate_from_config, load_config
from utils.dataset_reduction import load_selection_artifact
from utils.gpu_limit import apply_gpu_limit
from .callbacks import TrainingCurvesCallback, build_callbacks
from .config import load_mitigation_config
from .integration import configure_datamodule, configure_model


def main(args):
    apply_gpu_limit(getattr(args, "gpu_limit_pct", 45.0))
    config = load_config(args.config)
    data_params = config.setdefault("data", {}).setdefault("params", {})
    reduction_tag = ""
    curve_reduction_tag = ""
    if args.selection_artifact:
        data_params["selection_artifact"] = args.selection_artifact
        data_params["reduction_enabled"] = True
    if args.dataset_reduction:
        data_params["reduction_enabled"] = True
    if data_params.get("reduction_enabled", False) or data_params.get("selection_artifact"):
        artifact = load_selection_artifact(data_params.get("selection_artifact"))
        if args.kcenter_retention_pct is not None and args.kcenter_exclusion_pct is not None:
            raise ValueError("Use either --kcenter-retention-pct or --kcenter-exclusion-pct, not both")
        configured = {
            "el2n_low_exclusion_pct": args.el2n_low_exclusion_pct,
            "el2n_high_exclusion_pct": args.el2n_high_exclusion_pct,
            "kcenter_exclusion_pct": args.kcenter_exclusion_pct,
            "kcenter_retention_pct": args.kcenter_retention_pct,
        }
        selection = artifact["selection"]
        for key, value in configured.items():
            if value is None:
                continue
            artifact_value = selection.get(key)
            if key == "kcenter_exclusion_pct":
                artifact_value = selection.get(
                    key, 100 - selection.get("kcenter_retention_pct", 100)
                )
            elif key == "kcenter_retention_pct":
                artifact_value = selection.get(key)
            if artifact_value is None or abs(float(value) - float(artifact_value)) > 1e-8:
                raise ValueError(
                    f"{key} does not match the selection artifact "
                    f"({value} != {artifact_value})"
                )
        data_params["selection_artifact"] = artifact["_artifact_path"]
        data_params["selection_metadata"] = artifact["selection"]
        data_params["reduction_metadata"] = artifact
        retention = artifact.get("overall_retention_pct")
        retention_text = (
            f"r{retention:g}".replace(".", "p")
            if isinstance(retention, (int, float))
            else "runknown"
        )
        artifact_id = Path(artifact["_artifact_path"]).stem
        reduction_tag = f"-red-{retention_text}-sel{artifact_id}"
        subset_identifier = artifact.get("identifier") or artifact_id
        curve_reduction_tag = f"-red-{retention_text}-sel{subset_identifier}"
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
    model.hparams["dataset_reduction"] = data_params.get("reduction_metadata") or {
        "enabled": False
    }
    datamodule = configure_datamodule(instantiate_from_config(config["data"]), mitigation)
    lightning = copy.deepcopy(config.get("lightning", {}))
    trainer_config = dict(lightning.get("trainer", {}))
    if mitigation.max_epochs is not None:
        trainer_config["max_epochs"] = mitigation.max_epochs
    trainer_config["logger"] = None
    trainer_config["callbacks"] = build_callbacks(mitigation)
    model_name = Path(args.config).stem
    model_params = config.get("model", {}).get("params", {})
    data_params = config.get("data", {}).get("params", {})
    model_tag = (
        f"{model_name}-lr{model_params.get('learning_rate', 'na')}"
        f"-wd{model_params.get('weight_decay', 'base')}"
        f"-s{data_params.get('support_size_train', 'na')}x"
        f"{data_params.get('support_size_eval', 'na')}"
        f"-seed{mitigation.seed if mitigation.seed is not None else 'na'}"
        f"-{mitigation.filename_suffix()}"
        f"{curve_reduction_tag}"
    ).replace(".", "p")
    curves_path = Path("overfitting_mitigation") / "training_configuration_curves" / (
        f"{model_tag}.png"
    )
    trainer_config["callbacks"].append(TrainingCurvesCallback(curves_path))
    if mitigation.gradient_clip_val is not None:
        trainer_config["gradient_clip_val"] = mitigation.gradient_clip_val
    for callback in lightning.get("callbacks", {}).values():
        if callback and callback.get("target", "").endswith("ModelCheckpoint"):
            callback = copy.deepcopy(callback)
            params = callback.setdefault("params", {})
            checkpoint_suffix = f"{mitigation.filename_suffix()}{reduction_tag}"
            params["filename"] = f"{checkpoint_suffix}-best-{{epoch:02d}}-{{val_acc:.2f}}"
            params["dirpath"] = params.get("dirpath", "./results/overfitting_mitigation")
            checkpoint_callback = instantiate_from_config(callback)
            checkpoint_callback.CHECKPOINT_NAME_LAST = (
                f"last-{checkpoint_suffix}"
            )
            trainer_config["callbacks"].append(checkpoint_callback)

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
    parser.add_argument("--dataset-reduction", action="store_true",
                        help="Enable the selection artifact configured in the YAML")
    parser.add_argument("--selection-artifact", default=None,
                        help="Selection JSON produced by scripts/select_dataset.py")
    parser.add_argument("--el2n-low-exclusion-pct", type=float, default=None)
    parser.add_argument("--el2n-high-exclusion-pct", type=float, default=None)
    parser.add_argument("--kcenter-retention-pct", type=float, default=None)
    parser.add_argument("--kcenter-exclusion-pct", type=float, default=None)
    main(parser.parse_args())
