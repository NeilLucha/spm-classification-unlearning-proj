"""Episode-safe image augmentation utilities.

Mixup and CutMix are deliberately applied only to query images. Mixing support
images would change the support-label relation that defines an SPM episode.
"""

from typing import Optional, Tuple

import torch


def build_image_transform(
    base_transform,
    *,
    randaugment: bool = False,
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 9,
    augmix: bool = False,
    random_erasing: float = 0.0,
):
    from torchvision import transforms

    ops = list(getattr(base_transform, "transforms", [base_transform]))
    if randaugment:
        ops.insert(max(0, len(ops) - 2), transforms.RandAugment(
            num_ops=randaugment_num_ops, magnitude=randaugment_magnitude
        ))
    elif augmix:
        ops.insert(max(0, len(ops) - 2), transforms.AugMix())
    if random_erasing:
        ops.append(transforms.RandomErasing(p=random_erasing, value="random"))
    return transforms.Compose(ops)


def mix_query_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    probability: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return mixed queries and two target tensors, or an unchanged batch."""
    if not (mixup_alpha > 0 or cutmix_alpha > 0) or torch.rand(()) > probability:
        return images, labels, labels, 1.0
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or torch.rand(()) < 0.5)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    coefficient = float(torch.distributions.Beta(alpha, alpha).sample().item())
    permutation = torch.randperm(images.size(0), device=images.device)
    target_b = labels[permutation]
    if not use_cutmix:
        return (
            coefficient * images + (1 - coefficient) * images[permutation],
            labels,
            target_b,
            coefficient,
        )

    height, width = images.shape[-2:]
    cut_ratio = (1 - coefficient) ** 0.5
    cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
    cy, cx = torch.randint(height, (1,)).item(), torch.randint(width, (1,)).item()
    y1, y2 = max(0, cy - cut_h // 2), min(height, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(width, cx + cut_w // 2)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    coefficient = 1 - ((y2 - y1) * (x2 - x1) / float(height * width))
    return mixed, labels, target_b, coefficient
