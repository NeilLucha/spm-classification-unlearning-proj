"""Create a reproducible random training-subset selection artifact."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import time
from collections import Counter
from typing import Any

import numpy as np
import torch
from utils.gpu_limit import apply_gpu_limit

from utils.config import instantiate_from_config, load_config
from utils.dataset_reduction import ARTIFACT_VERSION, write_selection_artifact
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        total = kwargs.get("total")
        desc = kwargs.get("desc", "progress")
        for index, item in enumerate(iterable, 1):
            if total and (index == 1 or index == total or index % max(1, total // 10) == 0):
                print(f"{desc}: {index}/{total}", flush=True)
            yield item


def _make_raw_training_dataset(datamodule: Any):
    if hasattr(datamodule, "raw_training_dataset"):
        return datamodule.raw_training_dataset()
    if getattr(datamodule, "dataset_name", "").lower() in ("cifar10", "cifar100"):
        import torchvision
        dataset_name = getattr(datamodule, "dataset_name").lower()
        dataset = torchvision.datasets.CIFAR100 if dataset_name == "cifar100" else torchvision.datasets.CIFAR10
        return dataset(root="./data", train=True, transform=datamodule.test_transform)
    if hasattr(datamodule, "dataset_name"):
        return datamodule.dataset(
            root="./data", train=True, transform=datamodule.test_transform
        )
    return datamodule.dataset(
        root=datamodule.data_dir, split="train", transform=datamodule.test_transform
    )


def _targets(dataset: Any) -> list[int]:
    if hasattr(dataset, "targets"):
        return [int(target) for target in dataset.targets]
    if hasattr(dataset, "samples"):
        return [int(sample[1]) for sample in dataset.samples]
    return [int(dataset[index][1]) for index in range(len(dataset))]


def _validate_percentage(value: float) -> float:
    value = float(value)
    if not 0 <= value <= 100:
        raise ValueError(f"--selection-pct must be between 0 and 100, got {value}")
    return value


def _select_indices(
    indices: list[int],
    labels: list[int],
    selection_pct: float,
    class_wise: bool,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    if class_wise:
        selected: list[int] = []
        class_labels = sorted(set(labels[index] for index in indices))
        for label in tqdm(
            class_labels,
            total=len(class_labels),
            desc="[random-selection] selecting classes",
            unit="class",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            class_indices = [index for index in indices if labels[index] == label]
            count = int(round(len(class_indices) * selection_pct / 100))
            if selection_pct > 0:
                count = max(1, count)
            selected.extend(rng.choice(class_indices, size=count, replace=False).tolist())
        return sorted(selected)

    count = int(round(len(indices) * selection_pct / 100))
    if selection_pct > 0:
        count = max(1, count)
    return sorted(rng.choice(indices, size=count, replace=False).tolist())


def select_random(args: argparse.Namespace) -> str:
    apply_gpu_limit(getattr(args, "gpu_limit_pct", 45.0))
    started = time.perf_counter()
    selection_pct = _validate_percentage(args.selection_pct)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(
        f"[random-selection] Starting: selection={selection_pct}%, "
        f"mode={'class-wise' if args.class_wise else 'global'}, seed={args.seed}",
        flush=True,
    )
    config = load_config(args.config)
    print(f"[random-selection] Loading config: {args.config}", flush=True)
    datamodule = instantiate_from_config(config["data"])
    print("[random-selection] Preparing dataset", flush=True)
    datamodule.prepare_data()
    print(
        f"[random-selection] Dataset preparation complete "
        f"(elapsed={time.perf_counter() - started:.1f}s)",
        flush=True,
    )
    dataset = _make_raw_training_dataset(datamodule)
    labels = _targets(dataset)
    num_classes = config["model"]["params"].get(
        "num_classes", config["data"]["params"].get("num_classes")
    )
    valid_indices = [
        index for index, label in enumerate(labels)
        if num_classes is None or label < num_classes
    ]
    print(
        f"[random-selection] Training dataset loaded: {len(valid_indices)} eligible "
        f"examples across {len(set(labels[index] for index in valid_indices))} classes",
        flush=True,
    )

    selected = _select_indices(
        valid_indices, labels, selection_pct, args.class_wise, args.seed
    )
    print(
        f"[random-selection] Random selection complete: "
        f"{len(selected)}/{len(valid_indices)} retained "
        f"(elapsed={time.perf_counter() - started:.1f}s)",
        flush=True,
    )
    per_class = {
        str(label): count
        for label, count in sorted(Counter(labels[index] for index in selected).items())
    }
    identifier = hashlib.sha256(
        repr((
            os.path.abspath(args.config),
            args.seed,
            selection_pct,
            args.class_wise,
        )).encode()
    ).hexdigest()[:12]
    artifact = {
        "artifact_type": "spm_dataset_reduction",
        "artifact_version": ARTIFACT_VERSION,
        "selected_indices": selected,
        "selection": {
            "method": "random",
            "random_selection_pct": selection_pct,
            "class_wise": args.class_wise,
            "seed": args.seed,
            "config": os.path.abspath(args.config),
            "el2n_low_exclusion_pct": 0.0,
            "el2n_high_exclusion_pct": 0.0,
            "kcenter_retention_pct": 100.0,
            "kcenter_exclusion_pct": 0.0,
        },
        "original_training_size": len(valid_indices),
        "retained_count": len(selected),
        "excluded_count": len(valid_indices) - len(selected),
        "overall_retention_pct": (
            100 * len(selected) / len(valid_indices) if valid_indices else 0.0
        ),
        "overall_exclusion_pct": (
            100 - 100 * len(selected) / len(valid_indices) if valid_indices else 0.0
        ),
        "per_class_retained": per_class,
        "identifier": identifier,
    }
    output = args.output or os.path.join(
        args.output_dir, f"random-selection-{identifier}.json"
    )
    write_selection_artifact(output, artifact)
    print(
        f"[random-selection] Saved artifact to {output}: "
        f"{len(selected)}/{len(valid_indices)} retained "
        f"({time.perf_counter() - started:.1f}s)",
        flush=True,
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select a reproducible random subset of the training data"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--selection-pct",
        type=float,
        required=True,
        help="Percentage of eligible training examples to retain",
    )
    parser.add_argument(
        "--class-wise",
        action="store_true",
        help="Sample the percentage independently within each class",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/dataset_selection/random")
    parser.add_argument("--gpu-limit-pct", type=float, default=45.0,
                        help="Per-process CUDA memory limit percentage (1-100)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    select_random(args)
