#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe: can the WorldStereo transformer be built without materialising weights?

Upstream's loader builds the 14 B backbone from the Wan checkpoint (~33 GB of
host RAM), then loads the 34.9 GB WorldStereo checkpoint as a *second* full copy
before copying it in. That peaks around 70 GB against 64 GB of system RAM on
this box, so it cannot work as written.

The alternative is to build the module tree on the ``meta`` device -- shapes
only, zero bytes -- and stream the checkpoint into it. This script checks the
part that can fail silently: which parameters and buffers are left on ``meta``
once the checkpoint's keys are accounted for. Anything still on ``meta``
afterwards would surface much later as a confusing runtime error, and
non-persistent buffers (RoPE frequency tables, for instance) are absent from
every checkpoint by construction.

Reads only config files and the checkpoint *header*. Downloads nothing beyond
those, and needs no GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / "HY-World-2.0"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "hyworld2" / "worldgen"))

import hyworld2  # noqa: E402,F401  (compat layer: distributed shims, ROCm flags)
import torch  # noqa: E402
from accelerate import init_empty_weights  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from models.worldstereo import WorldStereoModel, WorldStereoRefSModel  # noqa: E402

WAN_REPO = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
WS_REPO = "hanshanxue/WorldStereo"


def checkpoint_keys(path: Path) -> set[str]:
    """Key set of a safetensors file, read from its header alone."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return set(header)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-type", default="worldstereo-memory-dmd")
    ap.add_argument("--wan-config", type=Path,
                    help="transformer/config.json of the Wan base; fetched from "
                         "the Hub when omitted.")
    args = ap.parse_args()

    ws_cfg_path = hf_hub_download(WS_REPO, "config.json", subfolder=args.model_type)
    ws_cfg = json.load(open(ws_cfg_path, encoding="utf-8"))
    wan_cfg_path = args.wan_config or hf_hub_download(WAN_REPO, "transformer/config.json")
    wan_cfg = json.load(open(wan_cfg_path, encoding="utf-8"))

    print(f"base_model       : {ws_cfg['base_model']}")
    print(f"model_type       : {args.model_type}")

    init_kwargs = {k: v for k, v in wan_cfg.items() if not k.startswith("_")}
    # The model mutates controlnet_cfg attribute-style, so it wants an
    # OmegaConf node, which is what the wrapper hands it in the real loader.
    init_kwargs["controlnet_cfg"] = OmegaConf.create(ws_cfg["controlnet_cfg"])
    init_kwargs["base_model"] = ws_cfg["base_model"]

    cls = WorldStereoModel if args.model_type == "worldstereo-camera" else WorldStereoRefSModel
    print(f"class            : {cls.__name__}")

    with init_empty_weights():
        model = cls(**init_kwargs)
        model.build_controlnet(
            load_uni3c=False,
            freeze_backbone=ws_cfg.get("freeze_backbone", True),
        )

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    n_param = sum(p.numel() for p in params.values())
    print(f"parameters       : {len(params)} tensors, {n_param / 1e9:.3f} B")
    print(f"buffers          : {len(buffers)} tensors, "
          f"{sum(b.numel() for b in buffers.values()) / 1e6:.3f} M")
    print(f"bf16 footprint   : {n_param * 2 / 2**30:.1f} GiB")

    sd_keys = set(model.state_dict())
    print(f"state_dict keys  : {len(sd_keys)}")

    # Non-persistent buffers never appear in any checkpoint, so streaming
    # weights in cannot possibly materialise them.
    non_persistent = {n for n, _ in model.named_buffers() if n not in sd_keys}
    print(f"non-persistent buffers (absent from every checkpoint): {len(non_persistent)}")
    for n in sorted(non_persistent)[:10]:
        print(f"    {n}  shape={tuple(buffers[n].shape)} dtype={buffers[n].dtype}")

    try:
        ckpt = Path(hf_hub_download(WS_REPO, "model.safetensors",
                                    subfolder=args.model_type,
                                    local_files_only=True))
    except Exception:
        print("\ncheckpoint not downloaded yet; skipping the key comparison")
        return 0

    have = checkpoint_keys(ckpt)
    missing = sorted(sd_keys - have)
    unexpected = sorted(have - sd_keys)
    print(f"\ncheckpoint keys  : {len(have)}")
    print(f"in model, not in checkpoint : {len(missing)}")
    for n in missing[:15]:
        print(f"    {n}")
    print(f"in checkpoint, not in model : {len(unexpected)}")
    for n in unexpected[:15]:
        print(f"    {n}")

    stranded = sorted(set(missing) | non_persistent)
    print(f"\nWOULD REMAIN ON META: {len(stranded)} tensor(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
