#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Time the Wan VAE's heaviest convolutions, one route at a time.

Benchmarking through the whole VAE costs minutes per data point, which is a poor
way to answer "which formulation is fast here". These are the four shapes
`tools/t_wan_vae_flops.py` names as the bulk of an encoder pass -- together they
are most of its 10.63 TFLOP -- measured directly:

* **conv3d**    what the model does today.
* **conv3d1**   the same arithmetic as `kt` convolutions with a temporal kernel
                of one, keeping the 5D layout.
* **conv2d**    the same arithmetic again, with the temporal axis folded into
                the batch, which costs a permute + copy per tap.

Results are reported in TFLOP/s so they compare directly against the numbers in
ROCM_PORT.md: 0.24 TFLOP/s on the Conv3d cliff, ~7 off it, ~10.8 for Conv2d, and
34.5 for dense bf16 matmul.

Only one HIP process at a time on this box.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / "HY-World-2.0"
sys.path.insert(0, str(REPO))

from hyworld2 import compat  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# (channels, height, width) at the encoder's working resolutions, from
# t_wan_vae_flops.py's "heaviest layers".
SHAPES = [
    (96, 480, 832),
    (192, 240, 416),
    (384, 120, 208),
    (384, 60, 104),
]


def timed(fn, repeats: int) -> float:
    fn()                       # warm: MIOpen picks and caches a solver here
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.time() - t0)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda")
    t_out = args.frames

    print(f"3x3x3 causal convolutions, {args.dtype}, {t_out} output frames\n")
    print(f"{'channels':>8} {'spatial':>12} {'TFLOP':>7} "
          f"{'conv3d':>18} {'conv3d1':>18} {'conv2d':>18}")
    print("-" * 88)

    for channels, height, width in SHAPES:
        # Causal padding: 2 extra frames in front, 1 pixel each spatial side.
        x = torch.randn(1, channels, t_out + 2, height + 2, width + 2,
                        device=device, dtype=dtype)
        weight = torch.randn(channels, channels, 3, 3, 3, device=device, dtype=dtype)
        taps3d = [weight[:, :, k:k + 1].contiguous() for k in range(3)]
        taps2d = [weight[:, :, k].contiguous() for k in range(3)]

        tflop = 2 * (channels * t_out * height * width) * channels * 27 / 1e12

        def run_conv3d():
            F.conv3d(x, weight)

        def run_conv3d1():
            out = None
            for k in range(3):
                y = F.conv3d(x[:, :, k:k + t_out], taps3d[k])
                out = y if out is None else out.add_(y)

        def run_conv2d():
            out = None
            for k in range(3):
                taps = x[:, :, k:k + t_out].permute(0, 2, 1, 3, 4)
                taps = taps.reshape(t_out, channels, height + 2, width + 2)
                y = F.conv2d(taps, taps2d[k])
                y = y.reshape(1, t_out, channels, height, width).permute(0, 2, 1, 3, 4)
                out = y if out is None else out.add_(y)

        cells = []
        for fn in (run_conv3d, run_conv3d1, run_conv2d):
            try:
                dt = timed(fn, args.repeats)
                cells.append(f"{dt * 1000:8.1f}ms {tflop / dt:5.2f}T/s")
            except RuntimeError as exc:
                cells.append(f"{type(exc).__name__:>17}")

        print(f"{channels:>8} {height:>5}x{width:<6} {tflop:>7.2f} "
              + " ".join(f"{c:>18}" for c in cells), flush=True)
        del x, weight, taps3d, taps2d
        torch.cuda.empty_cache()

    print("\nROCM_PORT.md for reference: 0.24 T/s on the Conv3d cliff, ~7 off it,"
          "\n~10.8 folded to Conv2d, 34.5 for dense bf16 matmul.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
