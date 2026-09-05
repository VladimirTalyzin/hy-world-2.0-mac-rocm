"""Device/runtime abstraction shared by the ROCm and MPS ports.

Upstream HY-World 2.0 assumes a CUDA build of PyTorch throughout: it hardcodes
``torch.device("cuda")``, ``torch.amp.autocast("cuda", ...)`` and calls
``torch.cuda.*`` unconditionally.

* On **ROCm** most of that happens to work, because PyTorch's HIP build keeps
  the ``torch.cuda`` namespace. What does *not* work out of the box is
  attention (FlashAttention-2/3 are CUDA-only) and the fused SDPA kernels,
  which ROCm gates behind an env flag on consumer/APU parts.
* On **MPS** none of it works: there is no ``torch.cuda`` device, no bf16
  autocast on older releases, and no fused attention kernel.

This module centralises those decisions so the rest of the port can stay
device-agnostic.
"""

from __future__ import annotations

import functools
import os
import warnings

import torch

__all__ = [
    "configure_rocm_env",
    "can_compile",
    "maybe_compile",
    "device_type",
    "get_device",
    "autocast",
    "supports_bf16",
    "preferred_dtype",
    "empty_cache",
    "synchronize",
    "memory_allocated",
    "max_memory_allocated",
    "reset_peak_memory_stats",
    "describe",
]


# ---------------------------------------------------------------------------
# ROCm environment
# ---------------------------------------------------------------------------

def configure_rocm_env() -> None:
    """Enable the ROCm knobs the model needs, unless the user set them already.

    ``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL`` unlocks the AOTriton flash /
    memory-efficient SDPA kernels on architectures ROCm still marks as
    experimental (gfx1100/gfx1151/... — i.e. RDNA3 and RDNA3.5). Without it,
    ``scaled_dot_product_attention`` silently falls back to the MATH backend,
    which materialises the full ``N x N`` score matrix. That is both ~13x
    slower and quadratic in memory, which OOMs on WorldMirror's global
    attention over multi-view token sequences.

    The flag is read lazily by ATen when SDPA dispatches, so setting it here
    is effective even if ``torch`` was imported before this module.
    """
    if not is_rocm():
        return
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    # Sub-optimal tile heuristics on gfx1151 otherwise; harmless elsewhere.
    os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "0")

    # Expandable segments reduce caching-allocator fragmentation, which matters
    # when a large model and a large activation both want contiguous blocks.
    # NOTE: the ROCm runtime on *Windows* does not implement this and warns
    # "expandable_segments not supported on this platform" -- it is a no-op
    # there, kept for Linux ROCm. PYTORCH_ALLOC_CONF is the current name;
    # the CUDA/HIP-specific spellings are deprecated in torch 2.9.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # MIOpen benchmarks candidate convolution solvers the first time it meets a
    # problem shape and remembers the winner. That search is *expensive* -- the
    # Wan VAE's shapes took minutes apiece here -- so where the answers are kept
    # matters a great deal.
    #
    # scripts/rocm_env.sh already points MIOpen at ~/.cache/miopen, but nothing
    # did so for code entered directly through Python, which left those runs on
    # MIOpen's own default (~/.miopen). The two caches then accumulated
    # *disjoint* entries -- measured at 792 versus 184 problems with zero
    # overlap -- so the same searches were paid for twice depending on how the
    # pipeline happened to be started. Setting them here makes both routes agree.
    cache = os.path.join(os.path.expanduser("~"), ".cache", "miopen")
    os.environ.setdefault("MIOPEN_USER_DB_PATH", cache)
    os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", os.environ["MIOPEN_USER_DB_PATH"])
    try:
        os.makedirs(os.environ["MIOPEN_USER_DB_PATH"], exist_ok=True)
    except OSError:
        pass  # a read-only home is not worth failing a whole run over


@functools.lru_cache(maxsize=1)
def is_rocm() -> bool:
    return getattr(torch.version, "hip", None) is not None


@functools.lru_cache(maxsize=1)
def is_mps() -> bool:
    return (
        not is_rocm()
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        and not torch.cuda.is_available()
    )


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def device_type() -> str:
    """``"cuda"`` for both NVIDIA and ROCm builds, ``"mps"``, or ``"cpu"``."""
    forced = os.environ.get("HYWORLD_DEVICE")
    if forced:
        return torch.device(forced).type
    if torch.cuda.is_available():   # also true for ROCm/HIP builds
        return "cuda"
    if is_mps():
        return "mps"
    return "cpu"


def get_device(index: int | None = None) -> torch.device:
    dt = device_type()
    if dt == "cuda" and index is not None:
        return torch.device("cuda", index)
    return torch.device(dt)


def autocast(enabled: bool = True, dtype: torch.dtype | None = None):
    """``torch.amp.autocast`` bound to the active device type.

    MPS autocast only supports fp16 in current PyTorch releases, so a bf16
    request is downgraded there rather than raising.
    """
    dt = device_type()
    if dtype is None:
        dtype = preferred_dtype()
    if dt == "mps" and dtype is torch.bfloat16:
        dtype = torch.float16
    if dt == "cpu":
        return torch.amp.autocast("cpu", enabled=enabled, dtype=torch.bfloat16)
    return torch.amp.autocast(dt, enabled=enabled, dtype=dtype)


@functools.lru_cache(maxsize=1)
def supports_bf16() -> bool:
    dt = device_type()
    if dt == "cuda":
        try:
            return torch.cuda.is_bf16_supported()
        except Exception:
            return False
    if dt == "mps":
        # bf16 matmul support on MPS is incomplete; fp16 is the safe choice.
        return False
    return True


def preferred_dtype() -> torch.dtype:
    if device_type() == "mps":
        return torch.float16
    return torch.bfloat16 if supports_bf16() else torch.float16


# ---------------------------------------------------------------------------
# Memory / sync helpers (no-ops where the backend has no equivalent)
# ---------------------------------------------------------------------------

def empty_cache() -> None:
    dt = device_type()
    if dt == "cuda":
        torch.cuda.empty_cache()
    elif dt == "mps":
        torch.mps.empty_cache()


def synchronize(device: torch.device | None = None) -> None:
    dt = device_type()
    if dt == "cuda":
        torch.cuda.synchronize(device)
    elif dt == "mps":
        torch.mps.synchronize()


def memory_allocated(device: torch.device | None = None) -> int:
    dt = device_type()
    if dt == "cuda":
        return torch.cuda.memory_allocated(device)
    if dt == "mps":
        return torch.mps.current_allocated_memory()
    return 0


def max_memory_allocated(device: torch.device | None = None) -> int:
    dt = device_type()
    if dt == "cuda":
        return torch.cuda.max_memory_allocated(device)
    if dt == "mps":
        return torch.mps.current_allocated_memory()
    return 0


def reset_peak_memory_stats(device: torch.device | None = None) -> None:
    if device_type() == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


@functools.lru_cache(maxsize=1)
def can_compile() -> bool:
    """Whether ``torch.compile`` can actually produce GPU code here.

    Inductor lowers GPU kernels through Triton, and there is no Triton for
    ROCm on Windows. ``torch.compile`` still *returns* happily -- compilation
    is lazy -- and then raises ``InductorError: No module named 'triton'`` at
    the first forward pass, which is a long way from the call that caused it.

    Checking up front lets callers skip the wrapper entirely and run eager.
    """
    if device_type() == "cpu":
        # The C++ backend needs no Triton, but a compile of a large model buys
        # little on CPU and costs minutes; treat it as unavailable.
        return False
    if device_type() == "mps":
        return False
    import importlib.util
    return importlib.util.find_spec("triton") is not None


def maybe_compile(module, **kwargs):
    """``torch.compile(module)`` where that works, the module itself otherwise.

    Upstream compiles the text encoder and the VAE unconditionally. Eager mode
    is slower but correct, and both are small parts of this pipeline's cost
    next to the 17 B transformer, which upstream leaves uncompiled anyway.
    """
    if not can_compile():
        return module
    return torch.compile(module, **kwargs)


def describe() -> str:
    dt = device_type()
    if dt == "cuda":
        p = torch.cuda.get_device_properties(0)
        arch = getattr(p, "gcnArchName", None)
        flavour = f"ROCm {torch.version.hip}" if is_rocm() else f"CUDA {torch.version.cuda}"
        return (f"{p.name} ({arch or 'sm_%d%d' % (p.major, p.minor)}), "
                f"{p.total_memory / 2**30:.1f} GiB, {flavour}, torch {torch.__version__}")
    if dt == "mps":
        return f"Apple MPS, torch {torch.__version__}"
    return f"CPU, torch {torch.__version__}"


configure_rocm_env()
