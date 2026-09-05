# -*- coding: utf-8 -*-
"""Load a large checkpoint without ever holding two copies of it.

The usual ``load_state_dict(load_file(path))`` pattern needs the model *and* the
whole checkpoint resident at once. For WorldStereo 2.0 that is 32.5 GiB plus
34.9 GiB against 64 GB of system RAM, so it cannot complete on this machine --
and weights are staged through host RAM before they reach the GPU, which makes
host RAM the binding budget rather than anything the GPU reports.

The alternative here builds the module tree on the ``meta`` device (shapes, no
storage) and streams the checkpoint into it one tensor at a time, so the
transient cost is a single tensor rather than a second model. ``safetensors``
memory-maps the file, so each ``get_tensor`` reads just that slice.

Peak host RAM becomes the model itself plus the largest tensor in the file.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterable

import torch

__all__ = [
    "checkpoint_keys",
    "load_safetensors_into_model",
    "remaining_meta_tensors",
]


def checkpoint_keys(path: str | Path) -> set[str]:
    """Tensor names in a safetensors file, read from its header alone.

    Cheap enough to call before deciding how to load: the header is a JSON
    blob at the start of the file, so this touches a few hundred KB of a
    35 GB checkpoint.
    """
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return set(header)


def _assign(model: torch.nn.Module, name: str, tensor: torch.Tensor,
            requires_grad: bool) -> None:
    """Put ``tensor`` at dotted path ``name``, replacing a meta placeholder.

    ``accelerate`` ships a well-tested version of this; fall back to doing it
    by hand so the compat layer keeps working if accelerate is absent.
    """
    parts = name.split(".")
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    leaf = parts[-1]

    # A meta placeholder carries the shape the architecture expects. Replacing
    # it with a differently shaped tensor would install a model that only fails
    # much later, somewhere unrelated, so check here where the name is known.
    existing = module._parameters.get(leaf)
    if existing is None:
        existing = module._buffers.get(leaf)
    if existing is not None and tuple(existing.shape) != tuple(tensor.shape):
        raise RuntimeError(
            f"shape mismatch for {name}: the model expects "
            f"{tuple(existing.shape)}, the checkpoint has {tuple(tensor.shape)}"
        )

    if leaf in module._parameters:
        module._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=requires_grad)
    elif leaf in module._buffers:
        module._buffers[leaf] = tensor
    else:  # pragma: no cover - a name neither parameter nor buffer
        setattr(module, leaf, tensor)


def load_safetensors_into_model(
    model: torch.nn.Module,
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    requires_grad: bool = False,
    strict: bool = False,
    progress: bool = False,
) -> tuple[list[str], list[str]]:
    """Stream a safetensors checkpoint into ``model``, tensor by tensor.

    Args:
        model: usually built under ``accelerate.init_empty_weights()``, so its
            parameters are on ``meta`` and its non-persistent buffers (RoPE
            tables and the like) are already real.
        path: the checkpoint.
        device: where each tensor should land. ``"cpu"`` keeps the staging
            behaviour; naming the GPU writes straight there.
        dtype: cast floating-point tensors on the way in. ``None`` keeps the
            checkpoint's own dtype, which for a bf16 checkpoint is what you
            want.
        requires_grad: ``False`` for inference, which also avoids allocating
            gradient bookkeeping for 17 B parameters.
        strict: raise if the checkpoint does not cover every parameter.
        progress: print a line every 10% of tensors.

    Returns:
        ``(missing, unexpected)`` -- names the model wanted but the checkpoint
        lacked, and names the checkpoint carried that the model has no slot
        for. Both are lists so they read like ``load_state_dict``'s result.
    """
    from safetensors import safe_open

    wanted = dict(model.state_dict())
    missing: list[str] = []
    unexpected: list[str] = []
    n_done = 0

    with safe_open(str(path), framework="pt", device="cpu") as f:
        have = set(f.keys())
        missing = sorted(k for k in wanted if k not in have)
        unexpected = sorted(k for k in have if k not in wanted)

        if strict and missing:
            raise RuntimeError(
                f"{len(missing)} tensor(s) are missing from {path}, "
                f"first few: {missing[:5]}"
            )

        loadable = [k for k in have if k in wanted]
        step = max(1, len(loadable) // 10)
        for name in loadable:
            tensor = f.get_tensor(name)
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype)
            _assign(model, name, tensor.to(device), requires_grad)
            n_done += 1
            if progress and n_done % step == 0:
                print(f"    {n_done}/{len(loadable)} tensors", flush=True)

    return missing, unexpected


def remaining_meta_tensors(model: torch.nn.Module) -> list[str]:
    """Names still on the ``meta`` device -- i.e. shapes with no storage.

    Anything listed here would fail at the first forward pass with a confusing
    error, so check it right after loading rather than at inference time.
    """
    stranded: Iterable[tuple[str, torch.Tensor]] = list(model.named_parameters())
    stranded = list(stranded) + list(model.named_buffers())
    return sorted(name for name, t in stranded if t.is_meta)
