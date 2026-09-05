# -*- coding: utf-8 -*-
"""Single-process stand-ins for the ``torch.distributed`` API.

PyTorch's ROCm-on-Windows build ships without a distributed backend:
``torch._C._distributed_c10d`` is missing, so ``torch.distributed`` imports as a
hollow module. ``dist.is_available()`` returns ``False`` and the collective
functions are simply absent::

    >>> import torch.distributed as dist
    >>> dist.is_available()
    False
    >>> dist.get_rank()
    AttributeError: module 'torch.distributed' has no attribute 'get_rank'

The worldgen stages were written for 8 GPUs and call those functions
unconditionally -- 27 ``barrier()`` calls alone -- so on this platform they die
on the first collective rather than degrading to one rank.

Every *guarded* call site upstream already spells the check the same way::

    if dist.is_available() and dist.is_initialized():
        ...distributed path...
    else:
        ...single-process path...

so this module deliberately leaves ``is_available()`` reporting ``False``. Those
sites keep choosing the correct branch on their own. What it fills in are the
**unguarded** calls, with the semantics a world of size one has anyway: a
barrier over one rank is a no-op, an all-gather over one rank is a copy, a
broadcast from rank 0 to itself is a copy.

Collectives that cannot mean anything on one rank (the sequence-parallel
all-to-alls) are installed as stubs that raise a descriptive error instead of
silently returning wrong data. They are only reachable with ``sp > 1``, which
this platform cannot request.

Nothing here is installed on a build that has a real distributed backend, and
existing attributes are never overwritten.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as _dist

__all__ = [
    "distributed_available",
    "install_single_process_shims",
    "single_process",
    "world_size",
    "rank",
]


def distributed_available() -> bool:
    """True when torch was built with a working c10d backend."""
    return bool(getattr(_dist, "is_available", lambda: False)())


def single_process() -> bool:
    """True when this run has no distributed backend to speak of."""
    return not distributed_available()


def world_size() -> int:
    if distributed_available() and _dist.is_initialized():
        return _dist.get_world_size()
    return int(os.getenv("WORLD_SIZE", "1"))


def rank() -> int:
    if distributed_available() and _dist.is_initialized():
        return _dist.get_rank()
    return int(os.getenv("RANK", "0"))


# ---------------------------------------------------------------------------
# The stand-ins themselves. Signatures mirror torch.distributed closely enough
# that upstream call sites pass their arguments positionally or by keyword
# without adaptation; unused parameters are accepted and ignored on purpose.
# ---------------------------------------------------------------------------

def _is_initialized() -> bool:
    # There is no process group, and saying otherwise would send code down a
    # multi-GPU path that cannot work here.
    return False


def _get_rank(group: Any = None) -> int:
    return 0


def _get_world_size(group: Any = None) -> int:
    return 1


def _get_group_rank(group: Any = None, rank: int = 0) -> int:
    return 0


def _get_global_rank(group: Any = None, group_rank: int = 0) -> int:
    return 0


def _init_process_group(*args: Any, **kwargs: Any) -> None:
    """Accepted and ignored: a group of one needs no rendezvous."""
    return None


def _destroy_process_group(group: Any = None) -> None:
    return None


def _barrier(*args: Any, **kwargs: Any) -> None:
    """A barrier over a single rank is reached the moment it is called."""
    return None


def _broadcast(tensor: torch.Tensor, src: int = 0, group: Any = None,
               async_op: bool = False) -> None:
    """Rank 0 broadcasting to itself: the tensor already holds the value."""
    return None


def _all_reduce(tensor: torch.Tensor, op: Any = None, group: Any = None,
                async_op: bool = False) -> None:
    """Reducing one contribution leaves it unchanged, for sum, max, min or product."""
    return None


def _reduce(tensor: torch.Tensor, dst: int = 0, op: Any = None, group: Any = None,
            async_op: bool = False) -> None:
    return None


def _all_gather(tensor_list: list, tensor: torch.Tensor, group: Any = None,
                async_op: bool = False) -> None:
    if len(tensor_list) != 1:
        raise RuntimeError(
            f"all_gather wants {len(tensor_list)} contributions but this build "
            "has no distributed backend, so there is only one rank."
        )
    tensor_list[0].copy_(tensor)
    return None


def _all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor,
                            group: Any = None, async_op: bool = False) -> None:
    output.copy_(input.reshape(output.shape))
    return None


def _all_gather_object(object_list: list, obj: Any, group: Any = None) -> None:
    if len(object_list) != 1:
        raise RuntimeError(
            f"all_gather_object wants {len(object_list)} contributions but this "
            "build has no distributed backend, so there is only one rank."
        )
    object_list[0] = obj
    return None


def _gather(tensor: torch.Tensor, gather_list: list | None = None, dst: int = 0,
            group: Any = None, async_op: bool = False) -> None:
    if gather_list:
        gather_list[0].copy_(tensor)
    return None


def _gather_object(obj: Any, object_gather_list: list | None = None, dst: int = 0,
                   group: Any = None) -> None:
    if object_gather_list:
        object_gather_list[0] = obj
    return None


def _new_group(*args: Any, **kwargs: Any) -> None:
    """``None`` is what every collective here treats as "the default group"."""
    return None


def _unsupported(name: str):
    def _raise(*args: Any, **kwargs: Any):
        raise RuntimeError(
            f"torch.distributed.{name} was called, but this PyTorch build has no "
            "distributed backend (torch._C._distributed_c10d is missing, which is "
            "normal for ROCm on Windows). This collective only has meaning across "
            "several ranks; run with sequence-parallel size 1."
        )
    return _raise


class _ReduceOp:
    """Placeholder for the enum, so ``op=dist.ReduceOp.SUM`` still parses."""
    SUM = "sum"
    AVG = "avg"
    PRODUCT = "product"
    MIN = "min"
    MAX = "max"
    BAND = "band"
    BOR = "bor"
    BXOR = "bxor"


_SHIMS = {
    "is_initialized": _is_initialized,
    "get_rank": _get_rank,
    "get_world_size": _get_world_size,
    "get_group_rank": _get_group_rank,
    "get_global_rank": _get_global_rank,
    "init_process_group": _init_process_group,
    "destroy_process_group": _destroy_process_group,
    "barrier": _barrier,
    "broadcast": _broadcast,
    "all_reduce": _all_reduce,
    "reduce": _reduce,
    "all_gather": _all_gather,
    "all_gather_into_tensor": _all_gather_into_tensor,
    "all_gather_object": _all_gather_object,
    "gather": _gather,
    "gather_object": _gather_object,
    "new_group": _new_group,
    "ReduceOp": _ReduceOp,
    # Meaningful only across ranks; fail loudly rather than return nonsense.
    "all_to_all": _unsupported("all_to_all"),
    "all_to_all_single": _unsupported("all_to_all_single"),
    "scatter": _unsupported("scatter"),
    "reduce_scatter": _unsupported("reduce_scatter"),
    "reduce_scatter_tensor": _unsupported("reduce_scatter_tensor"),
}

_installed = False


def install_single_process_shims(verbose: bool = False) -> bool:
    """Fill in the missing ``torch.distributed`` API for a one-rank run.

    Returns True when shims were installed, False when the build already has a
    real distributed backend and nothing was touched. Idempotent.
    """
    global _installed
    if distributed_available():
        return False
    if _installed:
        return True

    added = []
    for name, impl in _SHIMS.items():
        if not hasattr(_dist, name):
            setattr(_dist, name, impl)
            added.append(name)

    # Environment defaults that upstream reads directly, e.g. rank0_log().
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    _installed = True
    if verbose and added:
        print(f"[compat] torch.distributed has no backend; installed "
              f"single-process stand-ins for {len(added)} entry points.")
    return True
