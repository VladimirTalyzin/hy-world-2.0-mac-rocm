#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Time one WorldStereo transformer block against sequence length.

A denoising step at 32760 tokens measured ~360 s, which works out at about
5.4 TFLOP/s against the 34.5 this part sustains on dense bf16 matmul. Two
explanations fit that equally well from the outside:

* the workload is simply memory-bound at this sequence length, or
* the model plus its activations sit at the edge of the 64 GiB carve-out and
  some of the traffic is being served from GTT.

They are easy to tell apart by sweeping the sequence length. Memory-bound work
gives throughput that is flat, or improves gently as the matmuls get bigger.
Spilling gives a *cliff*: fine while it fits, catastrophic once it does not.

One block rather than the whole model, so a data point costs seconds instead of
minutes, and randomly initialised rather than loaded, since timing does not care
what the weights are. Multiply by 40 for a whole-model estimate.

Only one HIP process at a time on this box.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / "HY-World-2.0"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "hyworld2" / "worldgen"))

import hyworld2  # noqa: E402,F401

import torch  # noqa: E402

# Matches the checkpoint: 40 heads x 128, ffn 13824, text context 512.
DIM, HEADS, HEAD_DIM, FFN, TEXT_LEN = 5120, 40, 128, 13824, 512


def block_tflop(tokens: int, layers: int = 1) -> float:
    """Dense-matmul and attention FLOPs for `layers` blocks at `tokens` tokens."""
    qkv_out = 4 * DIM * DIM                 # q, k, v, out projections
    cross = 2 * DIM * DIM                   # k, v from the text context
    ffn = 2 * DIM * FFN                     # up and down
    dense = 2 * tokens * (qkv_out + cross + ffn)
    attn = 4 * tokens * tokens * DIM        # QK^T and AV
    return layers * (dense + attn) / 1e12


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", default="4680,9360,16380,23400,32760",
                    help="Sequence lengths to sweep. 32760 is a 21-keyframe "
                         "clip at 480x832; 9360 is what 6 latent frames give.")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    from models.controlnet import WanTransformerSparseSpatialBlock

    block = WanTransformerSparseSpatialBlock(
        DIM, FFN, HEADS, "rms_norm_across_heads", True, 1e-6, DIM,
    ).to(device=device, dtype=dtype).eval()
    params = sum(p.numel() for p in block.parameters())
    print(f"one block: {params / 1e6:.1f} M params, "
          f"{torch.cuda.memory_allocated() / 2**20:.0f} MiB\n")

    print(f"{'tokens':>8} {'TFLOP':>8} {'time':>10} {'TFLOP/s':>9} "
          f"{'x40 layers':>12} {'peak GiB':>9}")
    print("-" * 64)

    for spec in args.tokens.split(","):
        tokens = int(spec.strip())
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        hidden = torch.randn(1, tokens, DIM, device=device, dtype=dtype)
        ref = torch.randn(1, tokens // 8, DIM, device=device, dtype=dtype)
        text = torch.randn(1, TEXT_LEN, DIM, device=device, dtype=dtype)
        temb = torch.randn(1, 6, DIM, device=device, dtype=dtype)
        rot = (torch.randn(1, tokens, 1, HEAD_DIM, device=device, dtype=dtype),
               torch.randn(1, tokens, 1, HEAD_DIM, device=device, dtype=dtype))
        rot_ref = (torch.randn(1, tokens // 8, 1, HEAD_DIM, device=device, dtype=dtype),
                   torch.randn(1, tokens // 8, 1, HEAD_DIM, device=device, dtype=dtype))

        def once():
            with torch.no_grad():
                block(hidden_states=hidden, ref_states=ref,
                      encoder_hidden_states=text, temb=temb,
                      rotary_emb=rot, rotary_emb_ref=rot_ref)

        try:
            once()
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(args.repeats):
                t0 = time.time()
                once()
                torch.cuda.synchronize()
                best = min(best, time.time() - t0)
        except RuntimeError as exc:
            print(f"{tokens:>8} {'':>8} {type(exc).__name__:>10}  {str(exc)[:30]}")
            del hidden, ref, text, temb, rot, rot_ref
            continue

        tflop = block_tflop(tokens)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"{tokens:>8} {tflop:>8.2f} {best * 1000:>9.1f}ms "
              f"{tflop / best:>9.2f} {best * 40:>11.1f}s {peak:>9.2f}", flush=True)

        del hidden, ref, text, temb, rot, rot_ref

    print("\nFlat or gently rising TFLOP/s means memory-bound work.\n"
          "A collapse at the largest size means the carve-out is the problem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
