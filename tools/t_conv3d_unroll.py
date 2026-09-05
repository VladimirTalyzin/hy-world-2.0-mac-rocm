#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check that unrolling a causal Conv3d into Conv2d passes changes nothing.

The rewrite in ``hyworld2/compat/conv3d_unroll.py`` is an identity, not an
approximation, so the bar is floating-point rounding rather than "close
enough". This exercises the real ``WanCausalConv3d`` across the shapes the Wan
VAE actually uses -- one frame and four, temporal stride 1 and 2, with and
without the streaming cache -- and compares against the unmodified forward.

fp32 is the reference. bf16 is reported too, but judged against the fp32 answer
rather than against the bf16 original, because two bf16 paths can agree with
each other while both drift from the truth: summing in a different order is
exactly the kind of change that shows up here.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / "HY-World-2.0"
sys.path.insert(0, str(REPO))

from hyworld2 import compat  # noqa: E402  (ROCm env before any torch use)

import torch  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d  # noqa: E402

from hyworld2.compat.conv3d_unroll import unroll_wan_conv3d  # noqa: E402


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    scale = a.abs().max().clamp_min(1e-12)
    return ((a - b).abs().max() / scale).item()


def main() -> int:
    device = compat.get_device()
    torch.manual_seed(0)

    cases = [
        # (channels, kernel, stride, frames, use_cache) -- the Wan encoder's mix
        (16, (3, 3, 3), (1, 1, 1), 1, False),
        (16, (3, 3, 3), (1, 1, 1), 4, False),
        (16, (3, 3, 3), (1, 1, 1), 4, True),
        (32, (3, 3, 3), (1, 2, 2), 4, False),
        (32, (3, 3, 3), (2, 2, 2), 5, False),
        (8, (1, 1, 1), (1, 1, 1), 4, False),
        (8, (3, 1, 1), (1, 1, 1), 6, False),
    ]

    print(f"{'mode':>9} {'channels':>8} {'kernel':>10} {'stride':>10} "
          f"{'frames':>7} {'cache':>6} {'fp32 rel':>10} {'bf16 vs fp32':>14} "
          f"{'shape ok':>9}")
    print("-" * 92)

    worst_fp32 = 0.0
    for mode in ("conv3d1", "conv2d"):
        for channels, kernel, stride, frames, use_cache in cases:
            pad = (kernel[0] - 1, kernel[1] // 2, kernel[2] // 2)
            conv = WanCausalConv3d(channels, channels, kernel, stride=stride,
                                   padding=pad).to(device).eval()

            x = torch.randn(1, channels, frames, 24, 32, device=device)
            cache = (torch.randn(1, channels, 2, 24, 32, device=device)
                     if use_cache else None)

            with torch.no_grad():
                reference = conv(x, cache).clone()

            assert unroll_wan_conv3d(conv, mode=mode) == 1
            with torch.no_grad():
                unrolled = conv(x, cache).clone()

            err32 = relative_error(reference, unrolled)
            worst_fp32 = max(worst_fp32, err32)
            shape_ok = tuple(reference.shape) == tuple(unrolled.shape)

            # bf16, judged against the fp32 reference.
            conv_bf16 = conv.to(torch.bfloat16)
            with torch.no_grad():
                bf16_out = conv_bf16(
                    x.to(torch.bfloat16),
                    None if cache is None else cache.to(torch.bfloat16))
            err16 = relative_error(reference, bf16_out.float())

            print(f"{mode:>9} {channels:>8} {str(kernel):>10} {str(stride):>10} "
                  f"{frames:>7} {str(use_cache):>6} {err32:>10.2e} "
                  f"{err16:>14.2e} {str(shape_ok):>9}")

    print(f"\nworst fp32 relative error: {worst_fp32:.2e}")
    ok = worst_fp32 < 1e-5
    print("PASS -- the unrolled path is the same convolution" if ok
          else "FAIL -- the unrolled path does not match")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
