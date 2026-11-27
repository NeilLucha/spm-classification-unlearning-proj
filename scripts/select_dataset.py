"""Precompute an SPM EL2N + class-wise K-center training subset."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import time
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
from utils.gpu_limit import apply_gpu_limit
from utils.dataset_reduction import (
    ARTIFACT_VERSION,
    _classwise_filter,
    _global_filter,
    class_counts,
    kcenter_greedy,
    load_el2n_scores,
    write_selection_artifact,
)


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
        return [int(x) for x in dataset.targets]
    if hasattr(dataset, "samples"):
        return [int(x[1]) for x in dataset.samples]
    return [int(dataset[i][1]) for i in range(len(dataset))]


def _score_and_embed(model, dataset, labels, num_classes, support_size, batch_size, device):
    model.eval()
    valid = [i for i, label in enumerate(labels) if label < num_classes]
    print(
        f"[selection] Scoring {len(valid)} training examples with "
        f"support size {support_size} on {device}",
        flush=True,
    )
    by_class = {label: [i for i in valid if labels[i] == label] for label in range(num_classes)}
    support_indices = []
    for offset in range(max(1, support_size)):
        for label in range(num_classes):
            candidates = by_class[label]
            if candidates:
                support_indices.append(candidates[offset % len(candidates)])
            if len(support_indices) >= support_size:
                break
        if len(support_indices) >= support_size:
            break
    support_indices = list(dict.fromkeys(support_indices))
    scores = np.zeros(len(labels), dtype=np.float64)
    sample = dataset[valid[0]][0].unsqueeze(0).to(device)
    embedding_dim = model.encoder(sample).shape[-1]
    embeddings = np.zeros((len(valid), embedding_dim), dtype=np.float32)
    with torch.no_grad():
        batches = range(0, len(valid), batch_size)
        for start in tqdm(
            batches,
            total=(len(valid) + batch_size - 1) // batch_size,
            desc="[selection] EL2N + embeddings",
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            batch_indices = valid[start:start + batch_size]
            batch_images = torch.stack([dataset[i][0] for i in batch_indices]).to(device)
            batch_embeddings = model.encoder(batch_images)
            embeddings[start:start + len(batch_indices)] = (
                batch_embeddings.cpu().numpy()
            )
            for row, original_index in tqdm(
                enumerate(batch_indices),
                total=len(batch_indices),
                desc=f"[selection] scoring examples {start + 1}-{start + len(batch_indices)}",
                leave=False,
                unit="example",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ):
                allowed = [j for j in support_indices if j != original_index]
                support = torch.stack([dataset[j][0] for j in allowed]).to(device)
                support_y = torch.tensor([labels[j] for j in allowed], device=device)
                probabilities = model(
                    batch_images[row:row + 1], support, support_y
                ).exp()
                target = torch.zeros(num_classes, device=device)
                target[labels[original_index]] = 1
                scores[original_index] = torch.linalg.vector_norm(
                    probabilities[0] - target
                ).item()
    return valid, scores, embeddings


def _extract_embeddings(model, dataset, valid_indices, batch_size, device):
    model.eval()
    sample = dataset[valid_indices[0]][0].unsqueeze(0).to(device)
    embedding_dim = model.encoder(sample).shape[-1]
    embeddings = np.zeros(
        (len(valid_indices), embedding_dim),
        dtype=np.float32,
    )
    with torch.no_grad():
        batches = range(0, len(valid_indices), batch_size)
        for start in tqdm(
            batches,
            total=(len(valid_indices) + batch_size - 1) // batch_size,
            desc="[selection] extracting embeddings",
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            batch_indices = valid_indices[start:start + batch_size]
            batch_images = torch.stack(
                [dataset[index][0] for index in batch_indices]
            ).to(device)
            batch_embeddings = model.encoder(batch_images)
            embeddings[start:start + len(batch_indices)] = (
                batch_embeddings.cpu().numpy()
            )
    return embeddings


def select_dataset(args: argparse.Namespace) -> str:
    apply_gpu_limit(getattr(args, "gpu_limit_pct", 45.0))
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(
        f"[selection] Starting with seed={args.seed}, "
        f"EL2N low/high exclusions={args.el2n_low_exclusion_pct}%/"
        f"{args.el2n_high_exclusion_pct}%, "
        f"EL2N scope={'class-wise' if args.el2n_class_wise else 'global'}, "
        f"K-center retention={args.kcenter_retention_pct}%",
        flush=True,
    )
    config = load_config(args.config)
    print(f"[selection] Loading config: {args.config}", flush=True)
    datamodule = instantiate_from_config(config["data"])
    print("[selection] Preparing dataset", flush=True)
    datamodule.prepare_data()
    print(
        f"[selection] Dataset preparation complete "
        f"(elapsed={time.perf_counter() - started:.1f}s)",
        flush=True,
    )
    raw = _make_raw_training_dataset(datamodule)
    labels = _targets(raw)
    print(
        f"[selection] Training dataset loaded: {len(raw)} examples, "
        f"{len(set(labels))} classes",
        flush=True,
    )
    model = instantiate_from_config(config["model"])
    if args.train_pilot:
        try:
            import pytorch_lightning as pl
        except ImportError as exc:
            raise RuntimeError(
                "Pilot training requires pytorch_lightning; install the project requirements"
            ) from exc
        print(
            f"[selection] Training pilot model for {args.pilot_epochs} epochs",
            flush=True,
        )
        trainer = pl.Trainer(
            max_epochs=args.pilot_epochs,
            accelerator="auto",
            devices=1,
            logger=False,
            enable_progress_bar=True,
            log_every_n_steps=1,
        )
        trainer.fit(model, datamodule=datamodule)
        pilot_checkpoint = args.pilot_checkpoint or os.path.join(
            args.output_dir, "pilot.ckpt"
        )
        os.makedirs(os.path.dirname(os.path.abspath(pilot_checkpoint)), exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, pilot_checkpoint)
        print(f"[selection] Saved trained pilot checkpoint: {pilot_checkpoint}", flush=True)
    else:
        if not args.pilot_checkpoint:
            raise ValueError("--pilot-checkpoint is required unless --train-pilot is used")
        print(f"[selection] Loading pilot checkpoint: {args.pilot_checkpoint}", flush=True)
        checkpoint = torch.load(args.pilot_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(
            f"[selection] Pilot checkpoint loaded "
            f"(elapsed={time.perf_counter() - started:.1f}s)",
            flush=True,
        )
    device = torch.device(args.device)
    model.to(device)
    pilot_checkpoint_path = os.path.abspath(
        pilot_checkpoint if args.train_pilot else args.pilot_checkpoint
    )
    score_stage_started = time.perf_counter()
    if args.el2n_scores:
        print(
            f"[selection] Loading precomputed EL2N scores: {args.el2n_scores}",
            flush=True,
        )
        valid_indices = [
            index for index, label in enumerate(labels)
            if label < config["model"]["params"]["num_classes"]
        ]
        valid, scores = load_el2n_scores(
            args.el2n_scores,
            dataset_size=len(raw),
            valid_indices=valid_indices,
            labels=labels,
            num_classes=config["model"]["params"]["num_classes"],
            support_size=args.support_size,
            pilot_checkpoint=pilot_checkpoint_path,
            config_path=args.config,
        )
        embeddings = _extract_embeddings(
            model,
            raw,
            valid,
            args.batch_size,
            device,
        )
        print(
            f"[selection] Reusing {len(valid)} aligned EL2N scores; "
            f"skipping EL2N score computation "
            f"(loaded in {time.perf_counter() - score_stage_started:.1f}s)",
            flush=True,
        )
    else:
        valid, scores, embeddings = _score_and_embed(
            model, raw, labels, config["model"]["params"]["num_classes"],
            args.support_size, args.batch_size, device,
        )
        print(
            f"[selection] EL2N and embedding computation complete "
            f"(elapsed={time.perf_counter() - score_stage_started:.1f}s)",
            flush=True,
        )
    print(
        f"[selection] Applying {'class-wise' if args.el2n_class_wise else 'global'} "
        "EL2N filtering",
        flush=True,
    )
    if args.el2n_class_wise:
        filtered = _classwise_filter(
            valid, labels, scores, args.el2n_low_exclusion_pct,
            args.el2n_high_exclusion_pct,
        )
    else:
        filtered = _global_filter(
            valid, scores, args.el2n_low_exclusion_pct,
            args.el2n_high_exclusion_pct,
        )
    print(
        f"[selection] EL2N stage retained {len(filtered)}/{len(valid)} examples",
        flush=True,
    )
    print("[selection] Applying class-wise K-center Greedy", flush=True)
    filtered_embeddings = embeddings[
        [valid.index(index) for index in filtered]
    ]
    selected = kcenter_greedy(
        filtered, labels, filtered_embeddings, args.kcenter_retention_pct, args.seed
    )
    print(
        f"[selection] K-center stage retained {len(selected)}/{len(filtered)} "
        f"EL2N-filtered examples",
        flush=True,
    )
    el2n_scores_path = os.path.abspath(args.el2n_scores) if args.el2n_scores else None
    el2n_scores_checksum = None
    if el2n_scores_path:
        with open(el2n_scores_path, "rb") as handle:
            el2n_scores_checksum = hashlib.sha256(handle.read()).hexdigest()[:16]
    identifier = hashlib.sha256(
        repr((pilot_checkpoint_path, args.seed,
              args.el2n_low_exclusion_pct, args.el2n_high_exclusion_pct,
              args.kcenter_retention_pct, args.el2n_class_wise,
              el2n_scores_checksum)).encode()
    ).hexdigest()[:12]
    artifact = {
        "artifact_type": "spm_dataset_reduction",
        "artifact_version": ARTIFACT_VERSION,
        "selected_indices": selected,
        "selection": {
            "el2n_low_exclusion_pct": args.el2n_low_exclusion_pct,
            "el2n_high_exclusion_pct": args.el2n_high_exclusion_pct,
            "el2n_scope": "class-wise" if args.el2n_class_wise else "global",
            "kcenter_retention_pct": args.kcenter_retention_pct,
            "kcenter_exclusion_pct": 100 - args.kcenter_retention_pct,
            "seed": args.seed,
            "pilot_checkpoint": pilot_checkpoint_path,
            "el2n_scores_path": el2n_scores_path,
            "el2n_scores_checksum": el2n_scores_checksum,
        },
        "original_training_size": len(valid),
        "retained_count": len(selected),
        "excluded_count": len(valid) - len(selected),
        "overall_retention_pct": 100 * len(selected) / len(valid),
        "overall_exclusion_pct": 100 - 100 * len(selected) / len(valid),
        "per_class_retained": class_counts(selected, labels),
        "identifier": identifier,
    }
    output = args.output or os.path.join(
        args.output_dir, f"selection-{identifier}.json"
    )
    write_selection_artifact(output, artifact)
    print(
        f"[selection] Saved selection artifact to {output} "
        f"({len(selected)}/{len(valid)} overall, "
        f"{time.perf_counter() - started:.1f}s)",
        flush=True,
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pilot-checkpoint", default=None)
    parser.add_argument(
        "--train-pilot",
        action="store_true",
        help="Train a pilot with the configured SPM model before selection",
    )
    parser.add_argument("--pilot-epochs", type=int, default=1)
    parser.add_argument("--output-dir", default="results/dataset_selection")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--support-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-limit-pct", type=float, default=45.0,
                        help="Per-process CUDA memory limit percentage (1-100)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--el2n-low-exclusion-pct", type=float, default=0)
    parser.add_argument("--el2n-high-exclusion-pct", type=float, default=0)
    parser.add_argument(
        "--el2n-scores",
        type=str,
        default=None,
        help="Optional precomputed EL2N .pt file; skips EL2N score computation",
    )
    el2n_scope = parser.add_mutually_exclusive_group()
    el2n_scope.add_argument(
        "--el2n-class-wise",
        dest="el2n_class_wise",
        action="store_true",
        help="Apply EL2N exclusions independently within each class (default)",
    )
    el2n_scope.add_argument(
        "--el2n-global",
        dest="el2n_class_wise",
        action="store_false",
        help="Apply EL2N exclusions globally across all training examples",
    )
    parser.set_defaults(el2n_class_wise=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--kcenter-retention-pct", type=float, default=None)
    group.add_argument("--kcenter-exclusion-pct", type=float, default=None)
    parser.set_defaults(kcenter_retention_pct=100)
    args = parser.parse_args()
    if args.kcenter_exclusion_pct is not None:
        args.kcenter_retention_pct = 100 - args.kcenter_exclusion_pct
    select_dataset(args)
