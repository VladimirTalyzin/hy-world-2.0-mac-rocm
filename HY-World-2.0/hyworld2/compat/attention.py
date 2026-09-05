"""Portable scaled-dot-product attention for the ROCm / MPS ports.

Upstream imports FlashAttention (``flash_attn_interface`` v3, else
``flash_attn`` v2) at module scope and hard-fails without it. Neither builds
against ROCm on RDNA3.5 (gfx115x) or against Metal, so this module provides a
single entry point that picks the best kernel actually available:

  1. FlashAttention-3 / -2, when the package is importable and we are on a
     CUDA (not HIP) build — keeps NVIDIA behaviour bit-identical to upstream.
  2. ``torch.nn.functional.scaled_dot_product_attention`` — on ROCm this maps
     to the AOTriton flash kernel (see ``backend.configure_rocm_env``), on MPS
     to the Metal kernels, and it is the fastest correct option on both.

All entry points use the **(B, H, N, D)** layout that the model already works
in; the ``(B, N, H, D)`` transposes upstream performs for FlashAttention are
skipped when they are not needed.
"""

from __future__ import annotations

import functools
import os

import torch
import torch.nn.functional as F

from .backend import is_rocm

__all__ = ["attention", "attention_backend_name"]


@functools.lru_cache(maxsize=1)
def _flash_attn():
    """Return ``(fn, version)`` for an available FlashAttention, else ``(None, None)``.

    ROCm ships a HIP port of flash-attn for CDNA parts only, and it is
    routinely slower than (or broken on) RDNA. Opt in explicitly with
    ``HYWORLD_ALLOW_FLASH_ATTN=1`` if you know your build has it.
    """
    if is_rocm() and os.environ.get("HYWORLD_ALLOW_FLASH_ATTN", "0") != "1":
        return None, None
    try:
        from flash_attn_interface import flash_attn_func  # FA3
        return flash_attn_func, 3
    except ImportError:
        pass
    try:
        from flash_attn.flash_attn_interface import flash_attn_func  # FA2
        return flash_attn_func, 2
    except ImportError:
        pass
    return None, None


def attention_backend_name() -> str:
    fn, ver = _flash_attn()
    if fn is not None:
        return f"flash-attn-v{ver}"
    return "torch-sdpa"


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float) -> torch.Tensor:
    """SDPA with a MATH fallback for shapes the fused kernels reject.

    AOTriton (ROCm) and the Metal kernels only cover a subset of head
    dimensions / dtypes; when they bail out PyTorch raises instead of falling
    back, so we do it here.
    """
    try:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    except RuntimeError:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              dropout_p: float = 0.0) -> torch.Tensor:
    """Multi-head attention over ``(B, H, N, D)`` tensors, returning the same layout."""
    flash_fn, ver = _flash_attn()

    if flash_fn is not None and q.dtype in (torch.float16, torch.bfloat16):
        # FlashAttention wants (B, N, H, D) and requires contiguous inputs.
        def _to_bnhd(t):
            t = t.transpose(1, 2)
            return t if t.is_contiguous() else t.contiguous()

        q, k, v = _to_bnhd(q), _to_bnhd(k), _to_bnhd(v)
        out = flash_fn(q, k, v) if ver == 3 else flash_fn(q, k, v, dropout_p=dropout_p)
        if isinstance(out, tuple):          # FA3 returns (out, lse)
            out = out[0]
        return out.transpose(1, 2)

    return _sdpa(q, k, v, dropout_p)
