"""Create compatible Herding or Moderate Coreset selection artifacts."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from utils.config import instantiate_from_config, load_config
from utils.dataset_reduction import ARTIFACT_VERSION, write_selection_artifact
from utils.gpu_limit import apply_gpu_limit
try:
    from .methods import herding_select, moderate_select
except ImportError:
    from methods import herding_select, moderate_select


def _make_raw_training_dataset(datamodule: Any):
    if hasattr(datamodule, "raw_training_dataset"):
        return datamodule.raw_training_dataset()
    if getattr(datamodule, "dataset_name", "").lower() in ("cifar10", "cifar100"):
        import torchvision
        name = getattr(datamodule, "dataset_name").lower()
        dataset = (
            torchvision.datasets.CIFAR100
            if name == "cifar100"
            else torchvision.datasets.CIFAR10
        )
        return dataset(
            root="./data",
            train=True,
            transform=datamodule.test_transform,
        )
    if hasattr(datamodule, "data_dir"):
        import torchvision
        return torchvision.datasets.ImageNet(
            root=datamodule.data_dir,
            split="train",
            transform=datamodule.test_transform,
        )
    raise RuntimeError("Could not construct the raw training dataset")


def _targets(dataset: Any) -> list[int]:
    if hasattr(dataset, "targets"):
        return [int(value) for value in dataset.targets]
    if hasattr(dataset, "samples"):
        return [int(sample[1]) for sample in dataset.samples]
    return [int(dataset[index][1]) for index in range(len(dataset))]


def _extract_embeddings(model, dataset, indices, batch_size, device):
    model.eval()
    embeddings = []
    batches = range(0, len(indices), batch_size)
    if tqdm is not None:
        batches = tqdm(
            batches,
            total=(len(indices) + batch_size - 1) // batch_size,
            desc="[reduction] extracting embeddings",
            unit="batch",
        )
    with torch.no_grad():
        for start in batches:
            batch_indices = indices[start:start + batch_size]
            images = torch.stack([dataset[index][0] for index in batch_indices]).to(device)
            features = model.encoder(images)
            embeddings.append(features.detach().cpu().numpy())
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def select_reduction(args):
    apply_gpu_limit(getattr(args, "gpu_limit_pct", 45.0))
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.retention_pct < 0 or args.retention_pct > 100:
        raise ValueError("--retention-pct must be between 0 and 100")

    config = load_config(args.config)
    datamodule = instantiate_from_config(config["data"])
    datamodule.prepare_data()
    dataset = _make_raw_training_dataset(datamodule)
    labels = _targets(dataset)
    num_classes = config["model"]["params"].get(
        "num_classes",
        config["data"]["params"].get("num_classes"),
    )
    valid_indices = [
        index
        for index, label in enumerate(labels)
        if num_classes is None or label < num_classes
    ]
    valid_labels = np.asarray([labels[index] for index in valid_indices], dtype=np.int64)

    model = instantiate_from_config(config["model"])
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state, strict=False)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model.to(device)
    embeddings = _extract_embeddings(
        model,
        dataset,
        valid_indices,
        args.batch_size,
        device,
    )

    if args.method == "herding":
        selected = herding_select(
            embeddings,
            valid_labels,
            valid_indices,
            args.retention_pct,
            args.seed,
        )
        method_name = "herding"
    else:
        selected = moderate_select(
            embeddings,
            valid_labels,
            valid_indices,
            args.retention_pct,
            args.seed,
        )
        method_name = "moderate"

    identifier = hashlib.sha256(
        repr((
            os.path.abspath(args.config),
            args.checkpoint and os.path.abspath(args.checkpoint),
            args.method,
            args.retention_pct,
            args.seed,
        )).encode()
    ).hexdigest()[:12]
    per_class = {
        str(label): count
        for label, count in sorted(
            Counter(labels[index] for index in selected).items()
        )
    }
    artifact = {
        "artifact_type": "spm_dataset_reduction",
        "artifact_version": ARTIFACT_VERSION,
        "selected_indices": selected,
        "selection": {
            "method": method_name,
            "retention_pct": args.retention_pct,
            "seed": args.seed,
            "config": os.path.abspath(args.config),
            "checkpoint": (
                os.path.abspath(args.checkpoint) if args.checkpoint else None
            ),
            "el2n_low_exclusion_pct": 0.0,
            "el2n_high_exclusion_pct": 0.0,
            "kcenter_retention_pct": 100.0,
            "kcenter_exclusion_pct": 0.0,
        },
        "original_training_size": len(valid_indices),
        "retained_count": len(selected),
        "excluded_count": len(valid_indices) - len(selected),
        "overall_retention_pct": (
            100 * len(selected) / len(valid_indices)
            if valid_indices
            else 0.0
        ),
        "overall_exclusion_pct": (
            100 - 100 * len(selected) / len(valid_indices)
            if valid_indices
            else 0.0
        ),
        "per_class_retained": per_class,
        "identifier": identifier,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or str(
        output_dir / f"{method_name}-selection-{identifier}.json"
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_selection_artifact(output, artifact)
    print(
        f"[reduction-comparison] Saved {method_name} artifact: {output}",
        flush=True,
    )
    print(
        f"[reduction-comparison] Retained {len(selected)}/{len(valid_indices)} "
        f"training examples ({time.perf_counter() - started:.1f}s)",
        flush=True,
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create Herding or Moderate Coreset selection artifacts"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--method",
        choices=("herding", "moderate"),
        required=True,
    )
    parser.add_argument("--retention-pct", type=float, required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-limit-pct", type=float, default=45.0)
    parser.add_argument(
        "--output-dir",
        default="reduction_comparison/subsets",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    select_reduction(args)
