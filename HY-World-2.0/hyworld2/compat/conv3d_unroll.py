"""Compute a causal Conv3d as a sum of Conv2d passes, for any clip length.

``conv3d_fold`` handles the single-frame case by noticing that causal padding
leaves only the last temporal weight slice multiplying real data. The Wan VAE
does not fit that: ``AutoencoderKLWan._encode`` walks a clip one frame first and
then **four at a time**, so most of its convolutions see four frames and the
single-frame shortcut does not apply.

They land on the same MIOpen cliff regardless. Measured on the real Wan VAE at
480x832, bf16, a 21-frame clip -- *after* MIOpen had finished its exhaustive
solver search, so these are the best kernels it could find, not a first-run cost:

    encode, 21 frames (6 passes)    177.6 s      29.6 s per pass
    the same, unrolled                5.5 s       0.92 s per pass    32x

One encoder pass is 10.63 TFLOP of convolution (``tools/t_wan_vae_flops.py``),
so those are 0.36 and 11.6 TFLOP/s respectively -- the cliff, and the rate this
part manages on Conv2d.

Almost all of it is **one layer shape**. Per-shape, 3x3x3, bf16, 4 frames
(``tools/bench_wan_conv_shapes.py``):

    channels  spatial     conv3d              conv3d1            conv2d
      96      480x832     8635.8 ms 0.09 T/s   204.7 ms 3.88     109.1 ms  7.29
     192      240x416       99.7 ms 7.98 T/s   120.5 ms 6.60      71.3 ms 11.16
     384      120x208       55.0 ms 14.5 T/s    57.6 ms 13.8      33.0 ms 24.09
     384       60x104       13.3 ms 14.9 T/s    14.4 ms 13.8       9.2 ms 21.70

A single convolution at 96x480x832 takes **8.6 seconds**; the encoder runs
several at that resolution, which is the whole story. Note that Conv2d wins even
where Conv3d is nowhere near the cliff, permute and copy included.

The general identity is the same one every convolution obeys, just written out
along the temporal axis. For a Conv3d with temporal kernel ``kt``, stride
``st`` and dilation ``dt``, over an input already padded as the causal layer
wants it:

    out[:, :, o] = sum_k  conv2d( x[:, :, o*st + k*dt],  weight[:, :, k] )

So ``kt`` plain 2D convolutions, each over all output frames at once (the
temporal axis folded into the batch), summed. Identical arithmetic, identical
result up to floating-point summation order -- and Conv2d does not have the
cliff.

``.contiguous()`` on each weight slice is load-bearing, exactly as in
``conv3d_fold``: handing MIOpen a strided view sends it down a generic path and
gives the win straight back.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["conv3d_as_conv2d", "unroll_wan_conv3d", "restore_wan_conv3d"]

_CACHE = "_hyworld_unrolled_weights"
_ORIGINAL = "_hyworld_original_forward_unroll"


def conv3d_as_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
    slices: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Conv3d with no padding, computed as ``kt`` Conv2d passes.

    Args:
        x: ``[B, C, T, H, W]``, already padded however the caller wants.
        weight: ``[O, C, kt, kh, kw]``.
        bias: added once at the end, as a Conv3d would.
        stride, dilation: 3-tuples in (t, h, w) order.
        slices: pre-made contiguous ``weight[:, :, k]`` slices, to avoid
            redoing that work on every call.

    Returns ``[B, O, T_out, H_out, W_out]``.
    """
    batch, _, frames, height, width = x.shape
    kt, _, _ = weight.shape[2:]
    st, sh, sw = stride
    dt, dh, dw = dilation

    t_out = (frames - dt * (kt - 1) - 1) // st + 1
    if t_out < 1:
        raise ValueError(
            f"input has {frames} padded frames, too few for a temporal kernel "
            f"of {kt} with dilation {dt}"
        )

    out = None
    for k in range(kt):
        start = k * dt
        # Frames this weight slice multiplies: one per output frame.
        taps = x[:, :, start : start + st * t_out : st][:, :, :t_out]
        w = slices[k] if slices is not None else weight[:, :, k : k + 1].contiguous()
        if w.dim() == 5:
            # Keep the 5D layout and convolve with a temporal kernel of one.
            # Same arithmetic, and it avoids the permute+reshape copy the 2D
            # route needs -- which is not free when an early VAE layer is
            # hundreds of megabytes per tap.
            y = F.conv3d(taps, w, None, stride=(1, sh, sw), dilation=(1, dh, dw))
        else:
            taps = taps.permute(0, 2, 1, 3, 4).reshape(batch * t_out, -1, height, width)
            y = F.conv2d(taps, w, None, stride=(sh, sw), dilation=(dh, dw))
            y = y.reshape(batch, t_out, y.shape[1], y.shape[2], y.shape[3])
            y = y.permute(0, 2, 1, 3, 4)
        out = y if out is None else out.add_(y)

    if bias is not None:
        out = out + bias.view(1, -1, 1, 1, 1)
    return out


def _weight_slices(module, mode: str) -> list[torch.Tensor]:
    """Contiguous per-tap weight slices, cached until the weight changes.

    ``mode="conv2d"`` yields 4D slices ``[O, C, kh, kw]``; ``mode="conv3d1"``
    yields 5D ones ``[O, C, 1, kh, kw]``, which keeps the 5D layout and skips
    the permute. ``conv3d_as_conv2d`` picks its route from the slice's rank.

    The cache key has to notice both an in-place edit (``_version``) and a
    rebind to a new tensor from ``.to()`` or a dtype cast (``data_ptr``), or a
    moved model would keep convolving with weights left on the old device.
    """
    weight = module.weight
    key = (weight.data_ptr(), weight._version, weight.dtype, mode)
    cached = getattr(module, _CACHE, None)
    if cached is None or cached[0] != key:
        if mode == "conv3d1":
            slices = [weight[:, :, k : k + 1].contiguous()
                      for k in range(weight.shape[2])]
        else:
            slices = [weight[:, :, k].contiguous() for k in range(weight.shape[2])]
        cached = (key, slices)
        setattr(module, _CACHE, cached)
    return cached[1]


def _make_forward(module, mode: str):
    """A ``WanCausalConv3d.forward`` that pads as usual, then unrolls."""

    def forward(x, cache_x=None):
        padding = list(module._padding)
        if cache_x is not None and module._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return conv3d_as_conv2d(
            x, module.weight, module.bias,
            tuple(module.stride), tuple(module.dilation),
            slices=_weight_slices(module, mode),
        )

    return forward


def _eligible(module) -> bool:
    """Grouped convolutions would need per-group slicing; none appear here."""
    return getattr(module, "groups", 1) == 1 and hasattr(module, "_padding")


def unroll_wan_conv3d(model: torch.nn.Module, mode: str = "conv2d") -> int:
    """Replace every ``WanCausalConv3d`` under ``model``. Returns how many.

    Args:
        mode: ``"conv2d"`` folds the temporal axis into the batch and calls
            ``conv2d``; ``"conv3d1"`` instead convolves each tap with a
            temporal kernel of one, keeping the 5D layout and skipping the
            permute. ``"conv2d"`` is the default because it measured faster at
            every shape the Wan VAE uses, permute included -- see the table in
            this module's docstring.

    Idempotent, and reversible with :func:`restore_wan_conv3d`.
    """
    if mode not in ("conv3d1", "conv2d"):
        raise ValueError(f"mode must be 'conv3d1' or 'conv2d', not {mode!r}")
    try:
        from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d
    except ImportError:
        return 0

    patched = 0
    for module in model.modules():
        if not isinstance(module, WanCausalConv3d) or not _eligible(module):
            continue
        if not hasattr(module, _ORIGINAL):
            setattr(module, _ORIGINAL, module.forward)
            module.forward = _make_forward(module, mode)
        patched += 1
    return patched


def restore_wan_conv3d(model: torch.nn.Module) -> int:
    """Undo :func:`unroll_wan_conv3d`. Returns how many layers were restored."""
    restored = 0
    for module in model.modules():
        original = getattr(module, _ORIGINAL, None)
        if original is None:
            continue
        module.forward = original
        delattr(module, _ORIGINAL)
        if hasattr(module, _CACHE):
            delattr(module, _CACHE)
        restored += 1
    return restored
