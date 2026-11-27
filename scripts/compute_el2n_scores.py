"""Precompute and cache EL2N scores for a pilot checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
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

from utils.config import instantiate_from_config, load_config
from utils.dataset_reduction import EL2N_SCORE_FORMAT_VERSION
from utils.gpu_limit import apply_gpu_limit


def _make_raw_training_dataset(datamodule: Any):
    if hasattr(datamodule, "raw_training_dataset"):
        return datamodule.raw_training_dataset()
    if getattr(datamodule, "dataset_name", "").lower() in ("cifar10", "cifar100"):
        import torchvision
        name = getattr(datamodule, "dataset_name").lower()
        dataset = torchvision.datasets.CIFAR100 if name == "cifar100" else torchvision.datasets.CIFAR10
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


def compute_scores(args: argparse.Namespace) -> str:
    apply_gpu_limit(getattr(args, "gpu_limit_pct", 45.0))
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(
        f"[el2n-cache] Starting pilot={args.pilot_checkpoint}, "
        f"support_size={args.support_size}, seed={args.seed}",
        flush=True,
    )
    config = load_config(args.config)
    datamodule = instantiate_from_config(config["data"])
    datamodule.prepare_data()
    print(
        f"[el2n-cache] Dataset preparation complete "
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
        f"[el2n-cache] Training dataset: {len(valid_indices)} eligible examples "
        f"across {len(set(labels[index] for index in valid_indices))} classes",
        flush=True,
    )

    model = instantiate_from_config(config["model"])
    print(
        f"[el2n-cache] Loading pilot checkpoint: {args.pilot_checkpoint}",
        flush=True,
    )
    checkpoint = torch.load(args.pilot_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(
        f"[el2n-cache] Pilot checkpoint loaded "
        f"(elapsed={time.perf_counter() - started:.1f}s)",
        flush=True,
    )
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    by_class = {
        label: [index for index in valid_indices if labels[index] == label]
        for label in range(num_classes)
    }
    support_indices: list[int] = []
    for offset in range(max(1, args.support_size)):
        for label in range(num_classes):
            candidates = by_class[label]
            if candidates:
                support_indices.append(candidates[offset % len(candidates)])
            if len(support_indices) >= args.support_size:
                break
        if len(support_indices) >= args.support_size:
            break
    support_indices = list(dict.fromkeys(support_indices))
    print(
        f"[el2n-cache] Using consistent reference support with "
        f"{len(support_indices)} examples",
        flush=True,
    )

    scores = np.zeros(len(valid_indices), dtype=np.float64)
    valid_position = {
        original_index: position
        for position, original_index in enumerate(valid_indices)
    }
    batch_size = args.batch_size
    batch_count = (len(valid_indices) + batch_size - 1) // batch_size
    score_started = time.perf_counter()
    with torch.no_grad():
        for start in tqdm(
            range(0, len(valid_indices), batch_size),
            total=batch_count,
            desc="[el2n-cache] computing EL2N",
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            batch_indices = valid_indices[start:start + batch_size]
            batch_images = torch.stack(
                [dataset[index][0] for index in batch_indices]
            ).to(device)
            for row, original_index in tqdm(
                enumerate(batch_indices),
                total=len(batch_indices),
                desc=(
                    f"[el2n-cache] scoring examples "
                    f"{start + 1}-{start + len(batch_indices)}"
                ),
                leave=False,
                unit="example",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ):
                allowed = [
                    support_index
                    for support_index in support_indices
                    if support_index != original_index
                ]
                support = torch.stack(
                    [dataset[index][0] for index in allowed]
                ).to(device)
                support_labels = torch.tensor(
                    [labels[index] for index in allowed],
                    device=device,
                )
                probabilities = model(
                    batch_images[row:row + 1],
                    support,
                    support_labels,
                ).exp()
                target = torch.zeros(num_classes, device=device)
                target[labels[original_index]] = 1
                scores[valid_position[original_index]] = (
                    torch.linalg.vector_norm(
                        probabilities[0] - target
                    ).item()
                )
    print(
        f"[el2n-cache] EL2N computation complete: "
        f"{len(valid_indices)} examples in {time.perf_counter() - score_started:.1f}s",
        flush=True,
    )

    pilot_path = os.path.abspath(args.pilot_checkpoint)
    pilot_stem = Path(pilot_path).stem
    output = args.output
    if output is None:
        os.makedirs(args.output_dir, exist_ok=True)
        output = os.path.join(
            args.output_dir,
            f"{pilot_stem}-el2n_scores.pt",
        )
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    payload = {
        "format_version": EL2N_SCORE_FORMAT_VERSION,
        "pilot_checkpoint": pilot_path,
        "pilot_checkpoint_name": os.path.basename(pilot_path),
        "config": os.path.abspath(args.config),
        "dataset_name": args.config,
        "original_training_size": len(dataset),
        "num_classes": num_classes,
        "support_size": args.support_size,
        "seed": args.seed,
        "indices": valid_indices,
        "labels": [int(labels[index]) for index in valid_indices],
        "scores": scores.tolist(),
    }
    payload_checksum = hashlib.sha256(
        repr(payload).encode()
    ).hexdigest()[:16]
    payload["checksum"] = payload_checksum
    torch.save(payload, output)

    metadata_output = os.path.splitext(output)[0] + "_metadata.json"
    with open(metadata_output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print(
        f"[el2n-cache] Saved {len(valid_indices)} aligned scores to {output}",
        flush=True,
    )
    print(
        f"[el2n-cache] Saved metadata to {metadata_output} "
        f"(checksum={payload_checksum}, elapsed={time.perf_counter() - started:.1f}s)",
        flush=True,
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute EL2N scores for an existing pilot checkpoint"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--pilot-checkpoint", required=True)
    parser.add_argument("--output-dir", default="results/dataset_selection/el2n_scores")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--support-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-limit-pct", type=float, default=45.0,
                        help="Per-process CUDA memory limit percentage (1-100)")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    compute_scores(parser.parse_args())
