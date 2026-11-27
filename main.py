import argparse
import torch
import random
import numpy as np
import pytorch_lightning as pl
from utils.config import load_config, instantiate_from_config

import time
from pathlib import Path
from utils.dataset_reduction import load_selection_artifact
from utils.gpu_limit import apply_gpu_limit

torch.set_float32_matmul_precision("high")


def build_experiment_tag(config, args):
    """Build a filesystem-safe tag containing the run-defining settings."""
    model_params = config.get("model", {}).get("params", {})
    data_params = config.get("data", {}).get("params", {})
    model_target = config.get("model", {}).get("target", "model")
    encoder_type = model_params.get("encoder_type", "unknown")
    data_target = config.get("data", {}).get("target", "").lower()
    dataset_name = data_params.get(
        "dataset_name",
        "cifar100" if "cifar100" in data_target else "dataset",
    )
    num_classes = model_params.get("num_classes", data_params.get("num_classes", "na"))
    k = model_params.get("K", "na")
    learning_rate = model_params.get("learning_rate", "na")
    weight_decay = model_params.get("weight_decay", "1e-3" if encoder_type == "resnet18" else "default")
    smoothing = "ls" if model_params.get("use_label_smoothing", False) or args.label_smoothing else "nols"
    if encoder_type == "resnet18" and not model_params.get("pretrained_imagenet", True):
        scheduler = "plateau" if "spm_classifier_kmeans" in model_target else "multistep"
    else:
        scheduler = "configured"
    support_train = data_params.get("support_size_train", "na")
    support_eval = data_params.get("support_size_eval", "na")
    pretrained = "imgnet" if model_params.get("pretrained_imagenet", False) else "scratch"

    tag = (
        f"{dataset_name}-c{num_classes}-k{k}-lr{learning_rate}"
        f"-wd{weight_decay}-s{support_train}x{support_eval}"
        f"-{smoothing}-{scheduler}-{pretrained}-seed{args.seed}"
    ).replace(".", "p")
    reduction = config.get("data", {}).get("params", {})
    if reduction.get("reduction_enabled", False) or reduction.get("selection_artifact"):
        selection = reduction.get("reduction_metadata", {})
        def tag_number(value):
            return f"{value:g}" if isinstance(value, (int, float)) else str(value)
        tag += (
            f"-red-el{tag_number(selection.get('selection', {}).get('el2n_low_exclusion_pct', args.el2n_low_exclusion_pct or 0))}"
            f"-eh{tag_number(selection.get('selection', {}).get('el2n_high_exclusion_pct', args.el2n_high_exclusion_pct or 0))}"
            f"-kc{tag_number(selection.get('selection', {}).get('kcenter_retention_pct', 100 - (args.kcenter_exclusion_pct or 0)))}"
            f"-r{tag_number(selection.get('overall_retention_pct', 'na'))}"
        ).replace(".", "p")
    return tag


def main(args):
    apply_gpu_limit(getattr(args, "gpu_limit_pct", 45.0))

    config = load_config(args.config)
    data_params = config.setdefault("data", {}).setdefault("params", {})
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
            if value is not None:
                artifact_value = selection.get(key)
                if key == "kcenter_exclusion_pct":
                    artifact_value = selection.get(
                        key, 100 - selection.get("kcenter_retention_pct", 100)
                    )
                elif key == "kcenter_retention_pct":
                    artifact_value = selection.get(key)
                if abs(float(value) - float(artifact_value)) > 1e-8:
                    raise ValueError(
                        f"{key} does not match the selection artifact "
                        f"({value} != {artifact_value})"
                    )
        data_params["selection_artifact"] = artifact["_artifact_path"]
        data_params["selection_metadata"] = artifact["selection"]
        data_params["reduction_metadata"] = artifact
    if args.label_smoothing:
        model_target = config["model"].get("target", "")
        if model_target == "models.spm_classifier_kmeans.SemiParametricClassifier_KMeans":
            config["model"].setdefault("params", {})["use_label_smoothing"] = True
        else:
            raise ValueError("--label-smoothing is only supported for SemiParametricClassifier_KMeans")
    model = instantiate_from_config(config['model'])
    reduction_metadata = data_params.get("reduction_metadata")
    model.hparams["dataset_reduction"] = reduction_metadata or {
        "enabled": False
    }
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    datamodule = instantiate_from_config(config['data'])
    lightning_config = config.get('lightning', {})
    experiment_tag = build_experiment_tag(config, args)
    logger = instantiate_from_config(lightning_config['logger']) if 'logger' in lightning_config else None
    callbacks = []
    from utils.logging import EpochMetricsConsoleCallback
    callbacks.append(EpochMetricsConsoleCallback(interval=10))
    checkpoint_dir = None
    checkpoint_basename = f"last-{experiment_tag}"
    curve_basename = checkpoint_basename
    if reduction_metadata:
        selection_identifier = reduction_metadata.get("identifier")
        if not selection_identifier:
            selection_identifier = Path(
                reduction_metadata["_artifact_path"]
            ).stem
        curve_basename = f"{checkpoint_basename}-sel{selection_identifier}"
    if 'callbacks' in lightning_config and lightning_config['callbacks'] is not None:
        for cb in lightning_config['callbacks'].values():
            if cb is not None:
                if cb.get("target") == "pytorch_lightning.callbacks.ModelCheckpoint":
                    checkpoint_params = cb.setdefault("params", {})
                    checkpoint_dir = checkpoint_params.get(
                        "dirpath", "./results/classification"
                    )
                    cb.setdefault("params", {})["filename"] = (
                        f"best-{experiment_tag}-best_epoch{{epoch:02d}}-"
                        f"best_val_acc{{val_acc:.2f}}"
                    )
                callback = instantiate_from_config(cb)
                if cb.get("target") == "pytorch_lightning.callbacks.ModelCheckpoint":
                    callback.CHECKPOINT_NAME_LAST = f"last-{experiment_tag}"
                callbacks.append(callback)

    if checkpoint_dir is not None:
        from overfitting_mitigation.callbacks import TrainingCurvesCallback

        curves_dir = Path(checkpoint_dir) / "training_configuration_curves"
        callbacks.append(
            TrainingCurvesCallback(curves_dir / curve_basename)
        )
    
    from pytorch_lightning.loggers import WandbLogger
    if logger is not None and isinstance(logger, WandbLogger):
        logger.log_hyperparams(config)
    
    trainer_config = lightning_config.get('trainer', {})
    trainer_config['logger'] = logger
    trainer_config['callbacks'] = callbacks
    trainer = pl.Trainer(**trainer_config)
    t0 = time.perf_counter()
    trainer.fit(model, datamodule=datamodule)
    t1 = time.perf_counter()
    print(f"Training completed in {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--gpu-limit-pct", type=float, default=45.0,
                        help="Per-process CUDA memory limit percentage (1-100)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--dataset-reduction", action="store_true", help="Enable precomputed training-set reduction")
    parser.add_argument("--selection-artifact", type=str, default=None, help="Selection JSON produced by scripts/select_dataset.py")
    parser.add_argument("--el2n-low-exclusion-pct", type=float, default=None)
    parser.add_argument("--el2n-high-exclusion-pct", type=float, default=None)
    parser.add_argument("--kcenter-retention-pct", type=float, default=None)
    parser.add_argument("--kcenter-exclusion-pct", type=float, default=None)
    parser.add_argument(
        "--label-smoothing",
        action="store_true",
        help="Enable label-smoothed NLL loss for the K-means SPM classifier",
    )


    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    main(args)