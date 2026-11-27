"""Configuration and checkpoint-name helpers for the optional framework."""

from dataclasses import asdict, dataclass
import os
from typing import Any, Dict, Optional

import yaml


@dataclass
class MitigationConfig:
    weight_decay: Optional[float] = None
    decoupled_weight_decay: bool = False
    dropout: float = 0.0
    label_smoothing: float = 0.0
    randaugment: bool = False
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 9
    augmix: bool = False
    random_erasing: float = 0.0
    mixup: float = 0.0
    cutmix: float = 0.0
    averaging: str = "none"
    averaging_start_epoch: int = 0
    averaging_frequency: int = 1
    ema_decay: float = 0.999
    swa_bn_update: bool = True
    early_stopping_patience: Optional[int] = None
    early_stopping_monitor: str = "val_loss"
    gradient_clip_val: Optional[float] = None
    max_epochs: Optional[int] = None
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]]) -> "MitigationConfig":
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown mitigation options: {sorted(unknown)}")
        result = cls(**values)
        if result.averaging not in {"none", "ema", "swa", "ema+swa"}:
            raise ValueError("averaging must be one of none, ema, swa, ema+swa")
        if not 0 <= result.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 <= result.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if result.mixup < 0 or result.cutmix < 0 or result.random_erasing < 0:
            raise ValueError("augmentation probabilities/strengths cannot be negative")
        if result.averaging_frequency < 1 or result.averaging_start_epoch < 0:
            raise ValueError("averaging frequency must be positive and start epoch non-negative")
        if not 0 < result.ema_decay < 1:
            raise ValueError("ema_decay must be in (0, 1)")
        if result.max_epochs is not None and result.max_epochs < 1:
            raise ValueError("max_epochs must be a positive integer")
        return result

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def filename_suffix(self) -> str:
        parts = []
        if self.randaugment:
            parts.append(f"aug-randaugment{self.randaugment_magnitude}")
        if self.augmix:
            parts.append("aug-augmix")
        if self.random_erasing:
            parts.append(f"erase{self.random_erasing:g}".replace(".", "p"))
        if self.mixup:
            parts.append(f"mixup{self.mixup:g}".replace(".", "p"))
        if self.cutmix:
            parts.append(f"cutmix{self.cutmix:g}".replace(".", "p"))
        if self.weight_decay is not None:
            parts.append(f"wd{self.weight_decay:g}".replace(".", "p"))
        if self.dropout:
            parts.append(f"drop{self.dropout:g}".replace(".", "p"))
        if self.label_smoothing:
            parts.append(f"ls{self.label_smoothing:g}".replace(".", "p"))
        if self.averaging != "none":
            parts.append(self.averaging.replace("+", "-"))
        if self.max_epochs is not None:
            parts.append(f"maxep{self.max_epochs}")
        return "-".join(parts) or "baseline"


def load_mitigation_config(path: Optional[str], overrides=None) -> MitigationConfig:
    values: Dict[str, Any] = {}
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        values = loaded.get("overfitting_mitigation", loaded)
    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})
    return MitigationConfig.from_dict(values)
