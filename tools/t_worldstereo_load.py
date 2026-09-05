#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load the WorldStereo 2.0 transformer and report what it cost.

This is the memory half of the stage-3 go/no-go gate. It exercises the lean
path end to end -- build on ``meta`` from the architecture config, stream the
34.9 GB checkpoint in tensor by tensor, straight to the GPU -- and reports peak
host RAM and peak device memory, which is what decides whether stage 3 can run
on one part at all.

It needs only the WorldStereo checkpoint and the base model's
``transformer/config.json`` (466 bytes), so it can run before the auxiliary
stack (UMT5, CLIP, VAE) has finished downloading.

Only one HIP process at a time on this box -- stop the UI server first.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / "HY-World-2.0"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "hyworld2" / "worldgen"))

import hyworld2  # noqa: E402,F401  (compat: ROCm flags + distributed stand-ins)
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from models.worldstereo_wrapper import WorldStereo  # noqa: E402

WS_REPO = "hanshanxue/WorldStereo"


def host_rss_gib() -> float | None:
    """Resident set size of this process, or None without psutil."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / 2**30


def report(label: str, t0: float) -> None:
    rss = host_rss_gib()
    dev = torch.cuda.memory_allocated() / 2**30
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    line = (f"{label:<28} {time.time() - t0:7.1f}s | device {dev:6.2f} GiB "
            f"(peak {peak:6.2f}, reserved {reserved:6.2f})")
    if rss is not None:
        line += f" | host RSS {rss:6.2f} GiB"
    print(line, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-type", default="worldstereo-memory-dmd")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--forward", action="store_true",
                    help="Also run one forward pass with random latents, to "
                         "measure activation memory and a step's cost.")
    ap.add_argument("--frames", type=int, default=21)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg_path = hf_hub_download(WS_REPO, "config.json", subfolder=args.model_type,
                               local_files_only=True)
    weights = hf_hub_download(WS_REPO, "model.safetensors", subfolder=args.model_type,
                              local_files_only=True)
    size_gib = os.path.getsize(weights) / 2**30
    print(f"checkpoint : {weights}")
    print(f"             {size_gib:.2f} GiB")

    cfg = OmegaConf.create(WorldStereo._load_hf_config(cfg_path))

    t0 = time.time()
    transformer = WorldStereo._load_transformer(
        cfg,
        args.model_type,
        weights,
        sp_world_size=1,
        fsdp=False,
        device_mesh=None,
        device=device,
    )
    report("transformer loaded", t0)

    n = sum(p.numel() for p in transformer.parameters())
    dtypes = {str(p.dtype) for p in transformer.parameters()}
    print(f"parameters : {n / 1e9:.3f} B, dtypes {sorted(dtypes)}")
    print(f"on device  : {next(transformer.parameters()).device}")

    if args.forward:
        # Latent geometry follows the Wan VAE: 8x spatial, 4x temporal with a
        # causal first frame, and 36 input channels for the I2V variant.
        lat_f = (args.frames - 1) // 4 + 1
        lat_h, lat_w = args.height // 8, args.width // 8
        ch = int(cfg.get("in_channels", 36)) if hasattr(cfg, "get") else 36
        dt = next(transformer.parameters()).dtype
        print(f"\nforward    : latents [1, {ch}, {lat_f}, {lat_h}, {lat_w}] dtype={dt}")
        hidden = torch.randn(1, ch, lat_f, lat_h, lat_w, device=device, dtype=dt)
        timestep = torch.tensor([999], device=device)
        text = torch.randn(1, 512, int(cfg.get("text_dim", 4096)), device=device, dtype=dt)
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        with torch.no_grad():
            transformer(hidden_states=hidden, timestep=timestep,
                        encoder_hidden_states=text, return_dict=False)
        torch.cuda.synchronize()
        report("one forward pass", t1)

    del transformer
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
