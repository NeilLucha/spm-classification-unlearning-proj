"""Optional per-process CUDA memory limits."""

import torch


def apply_gpu_limit(gpu_limit_pct: float) -> None:
    """Limit visible CUDA devices to the requested percentage of their memory."""
    if gpu_limit_pct is None:
        gpu_limit_pct = 45.0
    if not 1 <= float(gpu_limit_pct) <= 100:
        raise ValueError("--gpu-limit-pct must be between 1 and 100")
    if not torch.cuda.is_available():
        return
    fraction = float(gpu_limit_pct) / 100.0
    for device in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(fraction, device=device)
        name = torch.cuda.get_device_name(device)
        print(
            f"[GPU] Limiting CUDA device {device} ({name}) "
            f"to {gpu_limit_pct:g}% of device memory"
        )
