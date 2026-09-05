#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate one panorama in a process of its own, for the web UI to drive.

Why a subprocess rather than an in-process call. The same generation completes
reliably standalone and wedges inside the Gradio server, and the difference was
narrowed by elimination rather than guessed at -- output size, step count,
allocator cache, worker thread, the UI's stage wrappers and its stdout capture
were each replicated in `tools/t_pano_memory.py` and each completed
(1952x960 at 10 steps: 382 s in a thread, 386 s with the wrappers, 386 s with
status polling on top). What is left is the Gradio/uvicorn process itself, and
chasing that further costs more than it buys.

So the UI spawns this instead. Three things fall out of it beyond working at
all: the carve-out is fully released when the process exits, which matters when
the model is 54 GB of a 64 GB budget; a stuck run can be killed without taking
the server with it; and one HIP process at a time on this GPU stops being a
rule the UI has to remember.

The cost is honest: the model is reloaded per generation, about 80 s.

Every stage is printed on stdout in the same format the in-process path used,
so `gradio_app`'s stage tracker keeps working unchanged.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR / "HY-World-2.0"
sys.path.insert(0, str(REPO_DIR))
sys.path.append(str(REPO_DIR / "hyworld2" / "panogen"))

from hyworld2 import compat  # noqa: E402  (sets the ROCm environment before torch)


def announce(pipe) -> None:
    """Print stage markers around the components that are otherwise silent.

    Between "start generating" and the first denoising step the pipeline says
    nothing, and that gap holds the text encoder and the VAE encode -- which is
    exactly where a bad convolution shape used to sit for a quarter of an hour.
    """
    def wrap(owner, name, tag):
        original = getattr(owner, name)

        def announced(*args, **kwargs):
            print(f"[Stage] {tag}", flush=True)
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                print(f"[Stage] {tag} done in {time.perf_counter() - t0:.1f}s", flush=True)

        setattr(owner, name, announced)

    if hasattr(pipe, "vae"):
        wrap(pipe.vae, "encode", "vae-encode")
        wrap(pipe.vae, "decode", "vae-decode")
    if hasattr(pipe, "encode_prompt"):
        wrap(pipe, "encode_prompt", "text")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate one panorama and exit.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=str(PROJECT_DIR / "weights"))
    ap.add_argument("--prompt", default="")
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=1952)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--true-cfg-scale", type=float, default=7.5)
    ap.add_argument("--blend-width", type=int, default=32)
    ap.add_argument("--crop-border", type=float, default=0.0)
    ap.add_argument("--cpu-offload", action="store_true")
    args = ap.parse_args()

    from pipeline_with_qwen_image import HunyuanPanoPipeline  # noqa: E402

    print(f"[Worker] {compat.describe()}", flush=True)
    weights = Path(args.weights)
    base = weights / "Qwen-Image-Edit-2509"
    base = str(base) if base.is_dir() else HunyuanPanoPipeline.DEFAULT_MODEL_ID
    lora = str(weights) if (weights / "HY-Pano-2.0").is_dir() else HunyuanPanoPipeline.DEFAULT_LORA_PATH

    t0 = time.perf_counter()
    pipeline = HunyuanPanoPipeline.from_pretrained(
        base, lora_path=lora, lora_subfolder="HY-Pano-2.0", cpu_offload=args.cpu_offload)
    print(f"[Worker] model ready in {time.perf_counter() - t0:.1f} s", flush=True)
    announce(pipeline.pipe)

    t0 = time.perf_counter()
    image = pipeline(
        args.image, prompt=args.prompt, negative_prompt=args.negative_prompt,
        seed=args.seed, height=args.height, width=args.width,
        num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
        true_cfg_scale=args.true_cfg_scale, blend_width=args.blend_width,
        crop_border=args.crop_border,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"[Worker] panorama generated in {time.perf_counter() - t0:.1f} s "
          f"(seed={args.seed}) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - the parent reads this off stdout
        traceback.print_exc()
        raise SystemExit(1)
