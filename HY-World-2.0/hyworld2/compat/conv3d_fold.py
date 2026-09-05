"""Fold Qwen-Image's causal Conv3d to Conv2d for single-frame input.

MIOpen's Conv3d solver choice on gfx1151 is a lottery decided by the exact
spatial shape, and losing costs two orders of magnitude. Measured on the real
``QwenImageCausalConv3d`` (128 channels, 3x3x3, bf16, one frame):

    VAE input     Conv3d            Conv2d (folded)
    1376x768       0.03 TFLOP/s      10.76 TFLOP/s     <-- cliff
    1184x896       0.03 TFLOP/s      10.82 TFLOP/s     <-- cliff
     896x1184      0.03 TFLOP/s      10.71 TFLOP/s     <-- cliff
    1280x800       1.83 TFLOP/s      10.83 TFLOP/s
    1024x1024      1.87 TFLOP/s      10.30 TFLOP/s
     800x1280      1.84 TFLOP/s      10.81 TFLOP/s

The cliff shapes are not exotic: 1376x768 is what a 16:9 photo becomes, and
1184x896 what a 4:3 one does. A 1800x1127 input (VAE 1280x800) wedged a real
run for 15 minutes even off the cliff, because the whole encoder pays the 6x
gap at every layer.

Why the folding is exact, not an approximation. ``QwenImageCausalConv3d`` pads
the temporal axis on the *left* only, by ``2 * padding[0]``, and then convolves
with no padding of its own. With a single frame and the usual 3x3x3 / padding=1
the padded input is ``[0, 0, x]``, so the kernel window sees real data at the
last temporal position alone: every other temporal weight slice is multiplied
by zero. ``conv2d`` with ``weight[:, :, -1]`` therefore computes the same thing,
and the check in ``tools/bench_conv3d_shapes.py`` confirms it to bf16 rounding
(relative error ~4e-3, which is 2^-8).

An earlier attempt at this measured *slower* and was abandoned. The difference
here is ``.contiguous()``: a strided view of the weight sends MIOpen down a
generic path and gives the speedup straight back.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["fold_causal_conv3d", "unfold_causal_conv3d"]

_PATCHED = "_hyworld_folded_forward"


def _foldable(module) -> bool:
    """True if this layer's temporal window collapses to one weight slice.

    Requires the left-only causal padding to be exactly ``kernel_d - 1`` (so a
    single frame yields a single output frame) and no temporal dilation.
    """
    pad = getattr(module, "_padding", None)
    if pad is None or len(pad) != 6 or pad[5] != 0:
        return False
    kernel_d = module.kernel_size[0]
    return (
        pad[4] == kernel_d - 1
        and module.dilation[0] == 1
        and module.groups == 1
    )


def _make_forward(module, original_forward):
    def forward(x, cache_x=None):
        # cache_x concatenates previous frames (video streaming), and anything
        # with more than one frame needs the real temporal convolution.
        if cache_x is not None or x.shape[2] != 1:
            return original_forward(x, cache_x)

        # Cache the folded weight. The key has to catch both kinds of change:
        # an in-place edit (LoRA fusing) bumps `_version`, while `.to()` or a
        # dtype cast rebinds `weight` to a *new* tensor whose version counter
        # starts at zero again -- hence the storage address as well, or a moved
        # model would keep using a weight left on the old device.
        weight = module.weight
        key = (weight.data_ptr(), weight._version)
        cached = getattr(module, _PATCHED, None)
        if cached is None or cached[0] != key:
            cached = (key, weight[:, :, -1].contiguous())
            setattr(module, _PATCHED, cached)

        pad_w, pad_h = module._padding[0], module._padding[2]
        return F.conv2d(
            x.squeeze(2), cached[1], module.bias,
            stride=module.stride[1:], padding=(pad_h, pad_w),
            dilation=module.dilation[1:],
        ).unsqueeze(2)

    return forward


def fold_causal_conv3d(model: torch.nn.Module) -> int:
    """Patch every foldable causal Conv3d under ``model``. Returns how many.

    Idempotent, and reversible with :func:`unfold_causal_conv3d`. Layers that
    are not foldable, and calls that carry more than one frame, keep the
    original path, so a video input still works.
    """
    try:
        from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
            QwenImageCausalConv3d,
        )
    except ImportError:
        return 0

    patched = 0
    for module in model.modules():
        if not isinstance(module, QwenImageCausalConv3d) or not _foldable(module):
            continue
        if hasattr(module, "_hyworld_original_forward"):
            patched += 1
            continue
        module._hyworld_original_forward = module.forward
        module.forward = _make_forward(module, module._hyworld_original_forward)
        patched += 1
    return patched


def unfold_causal_conv3d(model: torch.nn.Module) -> int:
    """Undo :func:`fold_causal_conv3d`. Returns how many layers were restored."""
    restored = 0
    for module in model.modules():
        original = getattr(module, "_hyworld_original_forward", None)
        if original is None:
            continue
        module.forward = original
        del module._hyworld_original_forward
        if hasattr(module, _PATCHED):
            delattr(module, _PATCHED)
        restored += 1
    return restored
