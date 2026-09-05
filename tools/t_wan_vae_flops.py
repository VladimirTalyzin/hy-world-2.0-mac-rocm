#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Count the arithmetic in one Wan VAE encoder pass, to know what "slow" means.

A wall-clock number alone cannot say whether a kernel is bad or the work is
simply large. This walks the encoder on the ``meta`` device -- shapes only, no
storage, no GPU -- and adds up the multiply-accumulates every convolution
performs. Dividing the measured time by this gives an achieved TFLOP/s, which
is directly comparable to the 34.5 TFLOP/s of bf16 matmul this part sustains
and to the Conv3d figures in ROCM_PORT.md.

Counts convolutions only. They are essentially all of it here: the attention
block sits at the bottleneck resolution and the norms are memory-bound trivia.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / "HY-World-2.0"
sys.path.insert(0, str(REPO))

import hyworld2  # noqa: E402,F401
import torch  # noqa: E402
from diffusers.models import AutoencoderKLWan  # noqa: E402

WAN_REPO = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--frames", type=int, default=4,
                    help="Frames in one encoder pass (the Wan encoder does 1, "
                         "then 4 at a time).")
    ap.add_argument("--seconds", type=float, default=None,
                    help="Measured seconds for one pass, to convert into an "
                         "achieved TFLOP/s.")
    args = ap.parse_args()

    with torch.device("meta"):
        vae = AutoencoderKLWan.from_config(
            AutoencoderKLWan.load_config(WAN_REPO, subfolder="vae")
        )

    total = {"flops": 0, "calls": 0}
    biggest: list[tuple[int, str, tuple]] = []

    def hook(name):
        def fn(module, inputs, output):
            # 2 FLOP per multiply-accumulate, one MAC per output element per
            # input channel per kernel position.
            k = 1
            for d in module.kernel_size:
                k *= d
            out_elems = output.numel()
            flops = 2 * out_elems * (module.in_channels // module.groups) * k
            total["flops"] += flops
            total["calls"] += 1
            biggest.append((flops, name, tuple(output.shape)))
        return fn

    handles = []
    for name, module in vae.encoder.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
            handles.append(module.register_forward_hook(hook(name)))
    handles.append(vae.quant_conv.register_forward_hook(hook("quant_conv")))

    x = torch.zeros(1, 3, args.frames, args.height, args.width, device="meta")
    with torch.no_grad():
        out = vae.encoder(x)
    for h in handles:
        h.remove()

    tflop = total["flops"] / 1e12
    print(f"input        : [1, 3, {args.frames}, {args.height}, {args.width}]")
    print(f"latent       : {tuple(out.shape)}")
    print(f"convolutions : {total['calls']}")
    print(f"arithmetic   : {tflop:.2f} TFLOP per pass")

    biggest.sort(reverse=True)
    print("\nheaviest layers:")
    for flops, name, shape in biggest[:6]:
        print(f"  {flops / 1e12:6.2f} TFLOP  {name:<44} -> {shape}")

    print("\nat a given throughput, one pass would take:")
    for rate in (0.24, 1.9, 7.0, 10.8, 34.5):
        print(f"  {rate:5.1f} TFLOP/s -> {tflop / rate:7.2f} s")
    if args.seconds:
        print(f"\nmeasured {args.seconds:.2f} s per pass "
              f"-> {tflop / args.seconds:.2f} TFLOP/s achieved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
