#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Time the Wan 3D VAE encode, and find out whether it is on a MIOpen cliff.

A first WorldStereo clip parked inside a single `WanCausalConv3d` for minutes,
with one CPU core busy and no visible progress -- the same signature the Qwen
VAE showed on this part (see ROCM_PORT.md, "the Qwen VAE is a 3D causal-conv
video VAE, and it hits a size cliff"). That earlier case turned out to be
MIOpen picking a 60x slower Conv3d solver for particular *spatial shapes*,
independent of area.

This isolates the question. The VAE is 0.24 GiB, so it loads in seconds and
leaves the GPU free, unlike the 46 GiB the full pipeline needs.

`WanAutoencoderKL._encode` walks the clip in chunks: one frame first, then four
at a time, so a 21-frame clip is 6 encoder passes. Timing is reported per pass
as well as in total, because a per-pass figure is what can be compared against
the Conv3d numbers already in ROCM_PORT.md.

Only one HIP process at a time on this box.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / "HY-World-2.0"
sys.path.insert(0, str(REPO))

from hyworld2 import compat  # noqa: E402  (ROCm env must be set before torch use)

import torch  # noqa: E402
from diffusers.models import AutoencoderKLWan  # noqa: E402

WAN_REPO = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"


def timed_encode(vae, x: torch.Tensor, *, warmup: bool) -> float:
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        vae.encode(x)
    torch.cuda.synchronize()
    dt = time.time() - t0
    if warmup:
        print(f"      (first call, includes MIOpen kernel selection: {dt:.1f}s)")
    return dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=21)
    ap.add_argument("--shapes", default="480x832,480x768,512x720,448x832,480x640",
                    help="Comma-separated HxW to try.")
    ap.add_argument("--tiling", action="store_true",
                    help="Also time each shape with the VAE's own tiling on.")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--unroll", default="",
                    help="Comma-separated unroll routes to time as well: "
                         "'conv3d1', 'conv2d', or both. Empty means none.")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda")

    print(f"loading the Wan VAE ({args.dtype})…", flush=True)
    vae = AutoencoderKLWan.from_pretrained(
        WAN_REPO, subfolder="vae", torch_dtype=dtype, local_files_only=True
    ).eval().to(device)
    print(f"  {sum(p.numel() for p in vae.parameters()) / 1e6:.1f} M params, "
          f"{torch.cuda.memory_allocated() / 2**20:.0f} MiB\n", flush=True)

    passes = 1 + (args.frames - 1) // 4
    print(f"{args.frames} frames -> {passes} encoder passes "
          f"(1 frame, then 4 at a time)\n")
    print(f"{'shape':>12} {'tiling':>7} {'unroll':>7} {'total':>9} {'per pass':>10}")
    print("-" * 60)

    warmed = False
    for shape in args.shapes.split(","):
        h, w = (int(v) for v in shape.strip().split("x"))
        x = torch.randn(1, 3, args.frames, h, w, device=device, dtype=dtype)
        for tiling in ([False, True] if args.tiling else [False]):
            vae.enable_tiling() if tiling else vae.disable_tiling()
            compat.restore_wan_conv3d(vae)
            if not warmed:
                timed_encode(vae, x, warmup=True)
                warmed = True
            best = min(timed_encode(vae, x, warmup=False)
                       for _ in range(args.repeats))
            print(f"{h:>5}x{w:<6} {str(tiling):>7} {'no':>7} {best:>8.2f}s "
                  f"{best / passes:>9.2f}s", flush=True)

            for mode in [m.strip() for m in args.unroll.split(",") if m.strip()]:
                n = compat.unroll_wan_conv3d(vae, mode=mode)
                unrolled = min(timed_encode(vae, x, warmup=False)
                               for _ in range(args.repeats))
                print(f"{h:>5}x{w:<6} {str(tiling):>7} {mode:>7} "
                      f"{unrolled:>8.2f}s {unrolled / passes:>9.2f}s"
                      f"   ({best / unrolled:.1f}x faster, {n} layers)", flush=True)
                compat.restore_wan_conv3d(vae)
        del x
        torch.cuda.empty_cache()

    print("\nFor scale, ROCM_PORT.md measured the Qwen VAE's Conv3d at "
          "0.24 TFLOP/s on the cliff\nversus ~7 TFLOP/s off it, and ~10.8 "
          "TFLOP/s once folded to Conv2d.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
