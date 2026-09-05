#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download the weights stage 3 (World Expansion / WorldStereo 2.0) needs.

Two economies over a plain ``snapshot_download`` of both repos, worth 66 GB:

* The WorldStereo checkpoint already contains the whole transformer (17.43 B
  params, bf16 -- verified from its safetensors header), so the 65.9 GB of
  ``transformer/*.safetensors`` in the Wan repo are redundant. Only
  ``transformer/config.json`` is kept, to instantiate the architecture.
* Everything else in the Wan repo (UMT5 text encoder, CLIP image encoder, VAE,
  tokenizer, scheduler) is genuinely needed and is fetched in full.

Files go to the standard HF cache, because the repo ids are hardcoded upstream
(``video_gen.py`` asks for ``hanshanxue/WorldStereo`` and the checkpoint config
names ``Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`` as its base). Caching them under
their real ids keeps ``local_files_only=True`` working without path patching.

Resumable: re-running skips what is already complete.
"""
from __future__ import annotations

import argparse
import sys
import time

from huggingface_hub import snapshot_download

WORLDSTEREO_REPO = "hanshanxue/WorldStereo"
WAN_REPO = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

# The Wan transformer weights are the 65.9 GB we are deliberately not fetching.
# Its config.json survives this pattern; the shards and their index do not.
WAN_IGNORE = ["transformer/*.safetensors", "transformer/*.index.json"]


def fetch(repo: str, *, allow=None, ignore=None, workers: int = 8) -> str:
    label = repo + (f" [{allow[0]}]" if allow else "")
    print(f"==> {label}", flush=True)
    t0 = time.time()
    path = snapshot_download(
        repo,
        allow_patterns=allow,
        ignore_patterns=ignore,
        max_workers=workers,
        resume_download=True,
    )
    print(f"    {path}  ({time.time() - t0:.0f}s)", flush=True)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-type", default="worldstereo-memory-dmd",
                    choices=["worldstereo-memory-dmd", "worldstereo-memory",
                             "worldstereo-camera"],
                    help="Which WorldStereo variant to fetch (default: the "
                         "four-step distilled one upstream recommends).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-wan", action="store_true",
                    help="Only fetch the WorldStereo checkpoint.")
    ap.add_argument("--skip-worldstereo", action="store_true",
                    help="Only fetch the Wan auxiliary stack.")
    args = ap.parse_args()

    if not args.skip_worldstereo:
        fetch(WORLDSTEREO_REPO, allow=[f"{args.model_type}/*"],
              workers=args.workers)
    if not args.skip_wan:
        fetch(WAN_REPO, ignore=WAN_IGNORE, workers=args.workers)

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
