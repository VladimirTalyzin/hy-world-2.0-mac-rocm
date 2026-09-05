#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch the model weights (and, optionally, upstream's demo inputs).

    python download_weights.py                 # WorldMirror 2.0 + HY-Pano LoRA + Qwen-Image-Edit base
    python download_weights.py --recon-only    # just WorldMirror 2.0 (3D reconstruction, ~4.8 GB)
    python download_weights.py --examples      # also restore HY-World-2.0/examples/worldrecon (~110 MB)
    python download_weights.py --worldgen      # also the WorldStereo 2.0 stack (~59 GB, CLI only)

What goes where, and why:

  weights/HY-WorldMirror-2.0/      4.8 GB   feed-forward reconstruction (the "3D scene" and
                                            "panorama -> 3D" tabs)
  weights/HY-Pano-2.0/             0.8 GB   the HY-Pano 2.0 LoRA
  weights/Qwen-Image-Edit-2509/     54 GB   the base model the LoRA sits on. Panorama generation
                                            needs the whole thing resident: budget ~60 GB of GPU
                                            memory (or unified memory) and ~64 GB of host RAM to
                                            stage it through.
  ~/.cache/huggingface/            59 GB    WorldStereo 2.0 + the Wan 2.1 auxiliary stack, for the
                                            world-expansion experiment (tools/t_worldstereo_clip.py).
                                            Fetched by tools/dl_worldgen.py; see WORLDGEN_PORT.md.

Everything is resumable: re-running skips what is already complete. Set
HYWORLD_WEIGHTS to keep the weights somewhere other than ./weights.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = Path(os.environ.get("HYWORLD_WEIGHTS", PROJECT_DIR / "weights")).resolve()
REPO_DIR = PROJECT_DIR / "HY-World-2.0"
UPSTREAM_GIT = "https://github.com/Tencent-Hunyuan/HY-World-2.0.git"

HY_WORLD_REPO = "tencent/HY-World-2.0"
QWEN_REPO = "Qwen/Qwen-Image-Edit-2509"


def fetch(repo: str, *, local_dir: Path, allow=None, ignore=None, workers: int = 8) -> Path:
    from huggingface_hub import snapshot_download

    label = repo + (f"  [{', '.join(allow)}]" if allow else "")
    print(f"==> {label}\n    -> {local_dir}", flush=True)
    t0 = time.time()
    path = snapshot_download(repo, local_dir=str(local_dir), allow_patterns=allow,
                             ignore_patterns=ignore, max_workers=workers)
    print(f"    done in {time.time() - t0:.0f} s", flush=True)
    return Path(path)


def fetch_examples() -> None:
    """Restore HY-World-2.0/examples/worldrecon from upstream with a sparse,
    blob-less clone, so only those 110 MB are transferred."""
    dest = REPO_DIR / "examples" / "worldrecon"
    if dest.is_dir() and any(dest.iterdir()):
        print(f"==> examples already present in {dest}")
        return
    if shutil.which("git") is None:
        raise SystemExit("git is needed to fetch the examples (or copy examples/worldrecon from upstream by hand)")
    print(f"==> upstream examples/worldrecon -> {dest}", flush=True)
    with tempfile.TemporaryDirectory(prefix="hyworld_examples_") as td:
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", UPSTREAM_GIT, td],
                       check=True)
        subprocess.run(["git", "-C", td, "sparse-checkout", "set", "examples/worldrecon"], check=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(td) / "examples" / "worldrecon", dest, dirs_exist_ok=True)
    print("    done", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon-only", action="store_true",
                    help="only WorldMirror 2.0 (skip the panorama models, ~55 GB)")
    ap.add_argument("--no-qwen", action="store_true",
                    help="WorldMirror + the HY-Pano LoRA, but not the 54 GB Qwen-Image-Edit base")
    ap.add_argument("--examples", action="store_true", help="also restore upstream's demo inputs")
    ap.add_argument("--worldgen", action="store_true",
                    help="also the WorldStereo 2.0 stack for world expansion (~59 GB into the HF cache)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"weights directory: {WEIGHTS_DIR}")

    fetch(HY_WORLD_REPO, local_dir=WEIGHTS_DIR, allow=["HY-WorldMirror-2.0/*"], workers=args.workers)
    if not args.recon_only:
        fetch(HY_WORLD_REPO, local_dir=WEIGHTS_DIR, allow=["HY-Pano-2.0/*"], workers=args.workers)
        if not args.no_qwen:
            fetch(QWEN_REPO, local_dir=WEIGHTS_DIR / "Qwen-Image-Edit-2509",
                  ignore=["*.pth", "*.onnx", "*.msgpack"], workers=args.workers)
    if args.examples:
        fetch_examples()
    if args.worldgen:
        subprocess.run([sys.executable, str(PROJECT_DIR / "tools" / "dl_worldgen.py"),
                        "--workers", str(args.workers)], check=True)

    print("\nAll done. Start the UI with launch.ps1 / launch.sh (or double-click 'HY-World 2.0.cmd').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
