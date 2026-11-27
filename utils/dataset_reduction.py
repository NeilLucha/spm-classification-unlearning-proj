"""Dataset reduction artifacts and deterministic EL2N/K-center selection."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Any, Iterable

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


ARTIFACT_VERSION = 1
EL2N_SCORE_FORMAT_VERSION = 1


def _validate_percentage(value: float, name: str) -> float:
    value = float(value)
    if not 0 <= value <= 100:
        raise ValueError(f"{name} must be between 0 and 100, got {value}")
    return value


def validate_selection_artifact(
    artifact: dict[str, Any],
    dataset_size: int | None = None,
    targets: Iterable[int] | None = None,
) -> dict[str, Any]:
    if artifact.get("artifact_type") != "spm_dataset_reduction":
        raise ValueError("Selection artifact has an unsupported artifact_type")
    indices = artifact.get("selected_indices")
    if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
        raise ValueError("Selection artifact selected_indices must be a list of integers")
    if len(indices) != len(set(indices)):
        raise ValueError("Selection artifact contains duplicate indices")
    if dataset_size is not None and any(i < 0 or i >= dataset_size for i in indices):
        raise ValueError("Selection artifact contains an out-of-range training index")
    params = artifact.get("selection", {})
    for key in ("el2n_low_exclusion_pct", "el2n_high_exclusion_pct"):
        _validate_percentage(params.get(key, 0), key)
    if "kcenter_retention_pct" in params:
        _validate_percentage(params["kcenter_retention_pct"], "kcenter_retention_pct")
    if "kcenter_exclusion_pct" in params:
        _validate_percentage(params["kcenter_exclusion_pct"], "kcenter_exclusion_pct")
    if targets is not None:
        target_list = list(targets)
        expected = Counter(int(target_list[i]) for i in indices)
        recorded = {
            int(k): int(v)
            for k, v in artifact.get("per_class_retained", {}).items()
        }
        if recorded and dict(expected) != recorded:
            raise ValueError("Selection artifact per-class counts do not match selected_indices")
    return artifact


def load_selection_artifact(
    path: str,
    dataset_size: int | None = None,
    targets: Iterable[int] | None = None,
) -> dict[str, Any]:
    if not path:
        raise ValueError("A selection artifact path is required when reduction is enabled")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Selection artifact not found: {path}")
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    artifact["_artifact_path"] = os.path.abspath(path)
    return validate_selection_artifact(artifact, dataset_size, targets)


def load_el2n_scores(
    path: str,
    dataset_size: int,
    valid_indices: list[int],
    labels: list[int],
    num_classes: int,
    support_size: int,
    pilot_checkpoint: str,
    config_path: str,
) -> tuple[list[int], np.ndarray]:
    if not path:
        raise ValueError("EL2N scores path is required")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"EL2N scores file not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("EL2N scores file must contain a metadata dictionary")
    if payload.get("format_version") != EL2N_SCORE_FORMAT_VERSION:
        raise ValueError(
            f"Incompatible EL2N scores format: {payload.get('format_version')}"
        )

    cached_indices = payload.get("indices")
    cached_labels = payload.get("labels")
    cached_scores = payload.get("scores")
    if not isinstance(cached_indices, list) or not isinstance(cached_labels, list):
        raise ValueError("EL2N scores file has missing or invalid indices/labels")
    if len(cached_indices) != len(set(cached_indices)):
        raise ValueError("EL2N scores file contains duplicate indices")
    if len(cached_indices) != len(cached_labels):
        raise ValueError("EL2N scores indices and labels are not aligned")
    if cached_indices != valid_indices:
        raise ValueError(
            "EL2N scores indices do not match the current training dataset "
            "or class eligibility"
        )
    if cached_labels != [int(labels[index]) for index in valid_indices]:
        raise ValueError("EL2N scores labels do not match the current training dataset")
    if len(cached_scores) != len(cached_indices):
        raise ValueError("EL2N scores count does not match the cached indices")
    if payload.get("original_training_size") != dataset_size:
        raise ValueError("EL2N scores file was computed for a different training-set size")
    if payload.get("num_classes") != num_classes:
        raise ValueError("EL2N scores file was computed for a different num_classes")
    if payload.get("support_size") != support_size:
        raise ValueError(
            "EL2N scores file support_size does not match --support-size "
            f"({payload.get('support_size')} != {support_size})"
        )

    cached_pilot = os.path.abspath(payload.get("pilot_checkpoint", ""))
    if cached_pilot != os.path.abspath(pilot_checkpoint):
        raise ValueError(
            "EL2N scores file was computed with a different pilot checkpoint "
            f"({cached_pilot} != {os.path.abspath(pilot_checkpoint)})"
        )
    cached_config = os.path.abspath(payload.get("config", ""))
    if cached_config != os.path.abspath(config_path):
        raise ValueError(
            "EL2N scores file was computed with a different config "
            f"({cached_config} != {os.path.abspath(config_path)})"
        )
    return list(cached_indices), np.asarray(cached_scores, dtype=np.float64)


def _classwise_filter(
    indices: list[int],
    labels: list[int],
    scores: np.ndarray,
    low_pct: float,
    high_pct: float,
) -> list[int]:
    low_pct = _validate_percentage(low_pct, "el2n_low_exclusion_pct")
    high_pct = _validate_percentage(high_pct, "el2n_high_exclusion_pct")
    kept: list[int] = []
    for label in sorted(set(labels)):
        class_indices = [i for i in indices if labels[i] == label]
        class_indices.sort(key=lambda i: (float(scores[i]), i))
        low_count = int(len(class_indices) * low_pct / 100)
        high_count = int(len(class_indices) * high_pct / 100)
        excluded = set(class_indices[:low_count])
        excluded.update(class_indices[len(class_indices) - high_count:] if high_count else [])
        kept.extend(i for i in class_indices if i not in excluded)
    return sorted(kept)


def _global_filter(
    indices: list[int],
    scores: np.ndarray,
    low_pct: float,
    high_pct: float,
) -> list[int]:
    low_pct = _validate_percentage(low_pct, "el2n_low_exclusion_pct")
    high_pct = _validate_percentage(high_pct, "el2n_high_exclusion_pct")
    ordered = sorted(indices, key=lambda i: (float(scores[i]), i))
    low_count = int(len(ordered) * low_pct / 100)
    high_count = int(len(ordered) * high_pct / 100)
    excluded = set(ordered[:low_count])
    excluded.update(ordered[len(ordered) - high_count:] if high_count else [])
    return sorted(i for i in ordered if i not in excluded)


def _stratified_retention_counts(
    indices: list[int],
    labels: list[int],
    retention_pct: float,
) -> dict[int, int]:
    """Allocate a rounded total retention target proportionally across classes."""
    retention_pct = _validate_percentage(retention_pct, "retention_pct")
    class_labels = sorted({labels[index] for index in indices})
    class_sizes = {
        label: sum(labels[index] == label for index in indices)
        for label in class_labels
    }
    target_total = int(np.floor(len(indices) * retention_pct / 100 + 0.5))
    exact_targets = {
        label: class_sizes[label] * retention_pct / 100
        for label in class_labels
    }
    retained = {
        label: int(np.floor(exact_targets[label]))
        for label in class_labels
    }
    remaining = target_total - sum(retained.values())
    ranked_classes = sorted(
        class_labels,
        key=lambda label: (-(exact_targets[label] - retained[label]), label),
    )
    for label in ranked_classes[:remaining]:
        retained[label] += 1
    return retained


def _allocate_stratified_counts(
    class_labels: list[int],
    class_sizes: dict[int, int],
    capacities: dict[int, int],
    target_count: int,
) -> dict[int, int]:
    allocation = {label: 0 for label in class_labels}
    remaining = min(target_count, sum(capacities.values()))

    while remaining:
        available = [
            label for label in class_labels
            if allocation[label] < capacities[label]
        ]
        if not available:
            break
        weight_total = sum(class_sizes[label] for label in available)
        exact = {
            label: remaining * class_sizes[label] / weight_total
            for label in available
        }
        added = 0
        for label in available:
            count = min(
                capacities[label] - allocation[label],
                int(np.floor(exact[label])),
            )
            allocation[label] += count
            added += count
        remaining -= added
        if not remaining:
            break

        ranked = sorted(
            available,
            key=lambda label: (-(exact[label] - np.floor(exact[label])), label),
        )
        for label in ranked:
            if not remaining:
                break
            if allocation[label] < capacities[label]:
                allocation[label] += 1
                remaining -= 1
                added += 1
        if not added:
            break
    return allocation


def classwise_el2n_filter_for_retention(
    indices: list[int],
    labels: list[int],
    scores: np.ndarray,
    low_pct: float,
    high_pct: float,
    retention_pct: float,
) -> tuple[list[int], dict[int, int], dict[str, int | bool]]:
    """Filter EL2N-ranked examples without exceeding each class retention target.

    If configured EL2N exclusions exceed the requested total exclusion, divide
    each class's exclusion budget into a 2:1 low-score/high-score split.
    """
    low_pct = _validate_percentage(low_pct, "el2n_low_exclusion_pct")
    high_pct = _validate_percentage(high_pct, "el2n_high_exclusion_pct")
    retention_pct = _validate_percentage(retention_pct, "retention_pct")
    retained_targets = _stratified_retention_counts(
        indices, labels, retention_pct
    )
    requested_total = low_pct + high_pct
    requested_exclusion_budget = 100 - retention_pct
    constrained = requested_total > requested_exclusion_budget
    class_labels = sorted(retained_targets)
    class_sizes = {
        label: sum(labels[index] == label for index in indices)
        for label in class_labels
    }
    exclusion_capacities = {
        label: class_sizes[label] - retained_targets[label]
        for label in class_labels
    }

    if constrained:
        exclusion_total = sum(exclusion_capacities.values())
        low_total = int(np.floor(exclusion_total * 2 / 3 + 0.5))
        high_total = exclusion_total - low_total
    else:
        low_total = int(np.floor(len(indices) * low_pct / 100 + 0.5))
        high_total = int(np.floor(len(indices) * high_pct / 100 + 0.5))

    low_by_class = _allocate_stratified_counts(
        class_labels, class_sizes, exclusion_capacities, low_total
    )
    remaining_capacities = {
        label: exclusion_capacities[label] - low_by_class[label]
        for label in class_labels
    }
    high_by_class = _allocate_stratified_counts(
        class_labels, class_sizes, remaining_capacities, high_total
    )

    filtered: list[int] = []
    for label in class_labels:
        class_indices = [index for index in indices if labels[index] == label]
        class_indices.sort(key=lambda index: (float(scores[index]), index))
        low_count = low_by_class[label]
        high_count = high_by_class[label]
        high_start = len(class_indices) - high_count
        filtered.extend(
            class_indices[low_count:high_start]
            if high_count
            else class_indices[low_count:]
        )

    total = len(indices)
    stats: dict[str, int | bool] = {
        "low_excluded_count": sum(low_by_class.values()),
        "high_excluded_count": sum(high_by_class.values()),
        "el2n_exclusion_constrained": constrained,
        "requested_retained_count": sum(retained_targets.values()),
        "el2n_filtered_count": len(filtered),
        "total_count": total,
    }
    return sorted(filtered), retained_targets, stats


def kcenter_greedy(
    indices: list[int],
    labels: list[int],
    embeddings: np.ndarray,
    retention_pct: float,
    seed: int,
    progress: bool = True,
    target_counts: dict[int, int] | None = None,
) -> list[int]:
    retention_pct = _validate_percentage(retention_pct, "kcenter_retention_pct")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    position = {index: pos for pos, index in enumerate(indices)}
    class_labels = sorted(set(labels[i] for i in indices))
    started = time.perf_counter()
    print(
        f"[selection] K-center: processing {len(indices)} examples across "
        f"{len(class_labels)} classes (retention={retention_pct}%, seed={seed})",
        flush=True,
    ) if progress else None
    for class_number, label in enumerate(class_labels, 1):
        class_indices = sorted(i for i in indices if labels[i] == label)
        if target_counts is None:
            count = int(np.ceil(len(class_indices) * retention_pct / 100)) if retention_pct else 0
            count = min(len(class_indices), max(1, count)) if class_indices else 0
        else:
            count = int(target_counts.get(label, 0))
            if count < 0 or count > len(class_indices):
                raise ValueError(
                    f"Invalid K-center target for class {label}: "
                    f"{count} of {len(class_indices)}"
                )
        if count == 0:
            if progress:
                print(
                    f"[selection] K-center class {class_number}/{len(class_labels)} "
                    f"(label={label}): empty selection",
                    flush=True,
                )
            continue
        if progress:
            class_started = time.perf_counter()
            print(
                f"[selection] K-center class {class_number}/{len(class_labels)} "
                f"(label={label}): selecting {count}/{len(class_indices)}",
                flush=True,
            )
        # The seed controls the first point, while all subsequent choices use
        # farthest-first distances with index-based tie breaking.
        first = class_indices[int(rng.integers(len(class_indices)))]
        chosen = [first]
        min_dist = np.full(len(class_indices), np.inf, dtype=np.float64)
        iterations = range(1, count)
        if progress and tqdm is not None:
            iterations = tqdm(
                iterations,
                total=max(0, count - 1),
                desc=f"[selection] K-center class {label}",
                unit="point",
            )
        for iteration in iterations:
            last = embeddings[position[chosen[-1]]]
            distances = np.sum((embeddings[[position[i] for i in class_indices]] - last) ** 2, axis=1)
            min_dist = np.minimum(min_dist, distances)
            candidates = [i for i in class_indices if i not in chosen]
            next_index = max(candidates, key=lambda i: (min_dist[class_indices.index(i)], -i))
            chosen.append(next_index)
            if progress and tqdm is None and (
                iteration == 1
                or iteration == count - 1
                or iteration % max(1, (count - 1) // 10) == 0
            ):
                print(
                    f"[selection] K-center class {label}: "
                    f"{iteration + 1}/{count} selected",
                    flush=True,
                )
        selected.extend(chosen)
        if progress:
            print(
                f"[selection] K-center class {class_number}/{len(class_labels)} "
                f"complete: {len(chosen)} selected "
                f"({time.perf_counter() - class_started:.1f}s; "
                f"{len(selected)}/{len(indices)} overall)",
                flush=True,
            )
    if progress:
        print(
            f"[selection] K-center complete: {len(selected)}/{len(indices)} selected "
            f"({time.perf_counter() - started:.1f}s)",
            flush=True,
        )
    return sorted(selected)


def write_selection_artifact(path: str, artifact: dict[str, Any]) -> None:
    validate_selection_artifact(artifact)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)


def class_counts(indices: Iterable[int], labels: Iterable[int]) -> dict[str, int]:
    labels_list = list(labels)
    return {str(k): v for k, v in sorted(Counter(labels_list[i] for i in indices).items())}
