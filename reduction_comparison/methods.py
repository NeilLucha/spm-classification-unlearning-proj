"""Class-wise Herding and Moderate Coreset selection methods."""

from collections import defaultdict

import numpy as np


def herding_select(
    embeddings: np.ndarray,
    labels: np.ndarray,
    valid_indices: list[int],
    retention_pct: float,
    seed: int = 0,
) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in sorted(set(int(value) for value in labels)):
        positions = [
            position
            for position, sample_label in enumerate(labels)
            if int(sample_label) == label
        ]
        if not positions:
            continue
        class_embeddings = embeddings[positions]
        count = int(round(len(positions) * retention_pct / 100))
        if retention_pct > 0:
            count = max(1, count)
        count = min(count, len(positions))
        if count == 0:
            continue

        class_mean = class_embeddings.mean(axis=0)
        remaining = list(range(len(positions)))
        first = int(rng.choice(remaining))
        remaining.remove(first)
        chosen_positions = [first]
        running_sum = class_embeddings[first].copy()

        while len(chosen_positions) < count:
            target = len(chosen_positions) + 1
            mean_target = class_mean * target - running_sum
            distances = np.linalg.norm(
                class_embeddings[remaining] - mean_target,
                axis=1,
            )
            order = np.argsort(distances, kind="stable")
            chosen_local = int(remaining[int(order[0])])
            chosen_positions.append(chosen_local)
            remaining.remove(chosen_local)
            running_sum += class_embeddings[chosen_local]

        selected.extend(valid_indices[position] for position in
                       (positions[index] for index in chosen_positions))
    return sorted(set(selected))


def moderate_select(
    embeddings: np.ndarray,
    labels: np.ndarray,
    valid_indices: list[int],
    retention_pct: float,
    seed: int = 0,
) -> list[int]:
    """Xia et al. Moderate Coreset: retain samples near each class median distance."""
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in sorted(set(int(value) for value in labels)):
        positions = [
            position
            for position, sample_label in enumerate(labels)
            if int(sample_label) == label
        ]
        if not positions:
            continue
        class_embeddings = embeddings[positions]
        count = int(round(len(positions) * retention_pct / 100))
        if retention_pct > 0:
            count = max(1, count)
        count = min(count, len(positions))
        if count == 0:
            continue

        class_mean = class_embeddings.mean(axis=0)
        distances = np.linalg.norm(class_embeddings - class_mean, axis=1)
        median = float(np.median(distances))
        order = np.argsort(
            np.abs(distances - median),
            kind="stable",
        )
        if count > 1:
            nearest = order[:count].tolist()
            jitter = rng.permutation(nearest)
            nearest = jitter.tolist()
        else:
            nearest = order[:count].tolist()
        selected.extend(valid_indices[positions[index]] for index in nearest)
    return sorted(set(selected))
