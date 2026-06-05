"""Device and dtype resolution shared by every model-bearing pipeline step.

Centralized so that local runs (Apple Silicon / MPS), pod runs (CUDA), and CPU-only
unit tests all resolve identically and the choice is recorded in provenance. Never
re-implement device selection inside a runner -- call :func:`resolve_device`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing torch at module import time (keeps light tests fast)
    import torch

_DTYPES = ("float16", "bfloat16", "float32")


def resolve_device(device: str = "auto") -> str:
    """Resolve a requested device to a concrete one.

    Args:
        device: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``. ``"auto"`` prefers
            CUDA (pod), then MPS (local Apple Silicon), then CPU.

    Returns:
        The concrete device string actually available.

    Raises:
        ValueError: if an explicit device is requested but unavailable, so a run never
            silently falls back to a slower device and mislabels its provenance.
    """
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("device='mps' requested but MPS is not available.")
    if device not in ("cuda", "mps", "cpu"):
        raise ValueError(f"Unknown device {device!r}; expected auto/cuda/mps/cpu.")
    return device


def resolve_dtype(dtype: str, device: str) -> "torch.dtype":
    """Map a dtype name to a torch dtype, applying device-specific safety rules.

    MPS + bfloat16 is numerically unreliable in current PyTorch; this downgrades it to
    float16 and the caller is expected to record the effective dtype in provenance.

    Args:
        dtype: one of ``"float16"``, ``"bfloat16"``, ``"float32"``.
        device: the concrete device (from :func:`resolve_device`).

    Returns:
        The corresponding ``torch.dtype``.
    """
    import torch

    if dtype not in _DTYPES:
        raise ValueError(f"Unknown dtype {dtype!r}; expected one of {_DTYPES}.")
    if device == "mps" and dtype == "bfloat16":
        dtype = "float16"
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]


def effective_dtype_name(dtype: str, device: str) -> str:
    """Return the dtype name actually used after device safety rules, for provenance."""
    if device == "mps" and dtype == "bfloat16":
        return "float16"
    if dtype not in _DTYPES:
        raise ValueError(f"Unknown dtype {dtype!r}; expected one of {_DTYPES}.")
    return dtype
