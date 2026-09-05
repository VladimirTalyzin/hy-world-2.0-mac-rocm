#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch the handful of Wan backbone tensors the WorldStereo checkpoint omits.

The WorldStereo 2.0 checkpoint carries the complete transformer except for
``blocks.{0..39}.attn2.norm_added_q.weight`` -- 40 RMSNorm vectors of a few KB
each. Downloading a 4.9 GB shard for those would be absurd, so this pulls them
straight out of the remote safetensors files with HTTP range requests:

    header length   bytes 0..7          (little-endian u64)
    header JSON     bytes 8..8+n        (key -> dtype, shape, byte range)
    tensor payload  bytes (8+n+start)..(8+n+end)

Total transfer is well under a megabyte. The result is written as one small
safetensors file that the stage-3 loader merges over the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download, hf_hub_url
from safetensors.torch import save_file

WAN_REPO = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
INDEX_FILE = "transformer/diffusion_pytorch_model.safetensors.index.json"

# safetensors dtype strings -> torch dtypes, for the ones Wan actually uses.
DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}


def _get_range(session: requests.Session, url: str, start: int, end: int) -> bytes:
    """Inclusive byte range ``start..end``, following HF's redirect to the CDN."""
    r = session.get(url, headers={"Range": f"bytes={start}-{end}"},
                    allow_redirects=True, timeout=60)
    r.raise_for_status()
    data = r.content
    want = end - start + 1
    if len(data) != want:
        raise RuntimeError(f"range {start}-{end} returned {len(data)} bytes, wanted {want}")
    return data


def read_header(session: requests.Session, url: str) -> tuple[dict, int]:
    """Return the safetensors header dict and the offset its data starts at."""
    n = struct.unpack("<Q", _get_range(session, url, 0, 7))[0]
    header = json.loads(_get_range(session, url, 8, 8 + n - 1))
    header.pop("__metadata__", None)
    return header, 8 + n


def missing_keys(index: dict, present: set[str]) -> list[str]:
    return sorted(set(index["weight_map"]) - present)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    help="Local WorldStereo model.safetensors, read for its key "
                         "list only (header, not weights). Omit to read the "
                         "header off the Hub instead, which works before the "
                         "34.9 GB download has finished.")
    ap.add_argument("--model-type", default="worldstereo-memory-dmd",
                    help="Variant to consult when --checkpoint is omitted.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Where to write the extracted tensors.")
    args = ap.parse_args()

    # Which keys does the checkpoint already have? Its header is enough.
    if args.checkpoint:
        with open(args.checkpoint, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            local = json.loads(f.read(n))
        local.pop("__metadata__", None)
    else:
        with requests.Session() as s:
            local, _ = read_header(
                s, hf_hub_url("hanshanxue/WorldStereo",
                              f"{args.model_type}/model.safetensors"))
    present = set(local)

    index = json.load(open(hf_hub_download(WAN_REPO, INDEX_FILE), encoding="utf-8"))
    wanted = missing_keys(index, present)
    if not wanted:
        print("Nothing missing; checkpoint covers every Wan key.")
        return 0
    print(f"{len(wanted)} key(s) missing from the checkpoint, fetching by range")

    by_shard: dict[str, list[str]] = defaultdict(list)
    for k in wanted:
        by_shard[index["weight_map"][k]].append(k)

    out: dict[str, torch.Tensor] = {}
    total = 0
    with requests.Session() as session:
        for shard, keys in sorted(by_shard.items()):
            url = hf_hub_url(WAN_REPO, f"transformer/{shard}")
            header, base = read_header(session, url)
            print(f"  {shard}: {len(keys)} tensor(s)")
            for k in keys:
                meta = header[k]
                start, end = meta["data_offsets"]
                raw = _get_range(session, url, base + start, base + end - 1)
                total += len(raw)
                t = torch.frombuffer(bytearray(raw), dtype=DTYPES[meta["dtype"]])
                out[k] = t.reshape(meta["shape"]).clone()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.out))
    print(f"Wrote {len(out)} tensors ({total / 1024:.1f} KiB transferred) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
