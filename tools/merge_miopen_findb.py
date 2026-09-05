#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge MIOpen user find-databases that drifted into two locations.

MIOpen benchmarks candidate convolution solvers the first time it meets a
problem shape and writes the winner to its *user find database*, a plain text
file of one ``<problem descriptor>=<solver>:<time>,...`` record per line. That
search is the expensive part -- minutes per shape for the Wan VAE on gfx1151 --
so losing the file means paying for it again.

This port accumulated two of them: ``scripts/rocm_env.sh`` pointed MIOpen at
``~/.cache/miopen`` while anything started directly through Python used
MIOpen's own default, ``~/.miopen/db``. The two filled up with disjoint
problems. Consolidating on one location is the fix, and this folds the other
one in rather than throwing it away.

Records are keyed by everything left of the first ``=``. Keys present in both
files keep the destination's version unless ``--prefer-source`` is given; the
compiled-kernel cache (``.ukdb``, a SQLite database) is deliberately **not**
touched, because kernels rebuild cheaply once the search result is known.

The destination is backed up next to itself before anything is written.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path


def read_records(path: Path) -> dict[str, str]:
    """Map problem descriptor -> full line, for one find-db."""
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        records[line.split("=", 1)[0]] = line
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True,
                    help="find-db to merge *from* (left unmodified).")
    ap.add_argument("--dest", type=Path, required=True,
                    help="find-db to merge *into*.")
    ap.add_argument("--prefer-source", action="store_true",
                    help="On a key present in both, keep the source's record. "
                         "Default keeps the destination's.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.source.is_file():
        print(f"source not found: {args.source}")
        return 1

    src = read_records(args.source)
    dst = read_records(args.dest) if args.dest.is_file() else {}
    shared = set(src) & set(dst)
    added = sorted(set(src) - set(dst))

    print(f"source : {args.source}  ({len(src)} records)")
    print(f"dest   : {args.dest}  ({len(dst)} records)")
    print(f"shared keys : {len(shared)}"
          + (f" (keeping {'source' if args.prefer_source else 'destination'})"
             if shared else ""))
    print(f"new to dest : {len(added)}")

    if args.dry_run:
        return 0
    if not added and not (shared and args.prefer_source):
        print("nothing to do")
        return 0

    merged = dict(dst)
    for key, line in src.items():
        if key not in merged or args.prefer_source:
            merged[key] = line

    if args.dest.is_file():
        backup = args.dest.with_suffix(args.dest.suffix + f".bak-{int(time.time())}")
        shutil.copy2(args.dest, backup)
        print(f"backed up   : {backup}")

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    # Write via a temporary file so an interrupted run cannot leave MIOpen with
    # a half-written database.
    tmp = args.dest.with_suffix(args.dest.suffix + ".tmp")
    tmp.write_text("\n".join(merged[k] for k in sorted(merged)) + "\n",
                   encoding="utf-8")
    tmp.replace(args.dest)
    print(f"wrote       : {args.dest}  ({len(merged)} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
