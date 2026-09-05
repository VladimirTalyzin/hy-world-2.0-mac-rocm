#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check the installation and say what will work.

    python scripts/doctor.py

Reports the torch backend, the attention path, the weights that are present,
optional components (gsplat for the fly-through video, the sky-segmentation
model), and runs a small GPU arithmetic check -- on ROCm a GPU that survived a
driver reset keeps running and returns wrong numbers without raising, so a
matmul whose result is off is the one symptom worth checking before trusting
any output.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PROJECT_DIR / "HY-World-2.0"
WEIGHTS_DIR = Path(os.environ.get("HYWORLD_WEIGHTS", PROJECT_DIR / "weights")).resolve()
sys.path.insert(0, str(REPO_DIR))

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"


def line(tag: str, name: str, detail: str = "") -> None:
    print(f"{tag}  {name:<22s} {detail}")


def size_of(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 2**30 if path.is_dir() else 0.0


def main() -> int:
    problems = 0
    print(f"project   {PROJECT_DIR}\nweights   {WEIGHTS_DIR}\npython    {sys.version.split()[0]}\n")

    # ---- torch and the compat layer ---------------------------------------
    try:
        from hyworld2 import compat
        import torch
    except Exception as e:  # noqa: BLE001
        line(FAIL, "torch / compat", f"{type(e).__name__}: {e}")
        return 1
    dt = compat.device_type()
    line(OK if dt != "cpu" else WARN, "device", compat.describe())
    line(OK if compat.attention_backend_name() != "math" else WARN, "attention", compat.attention_backend_name())
    line(OK, "dtype", str(compat.preferred_dtype()).replace("torch.", ""))
    if dt == "cuda" and compat.is_rocm():
        flag = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")
        line(OK if flag == "1" else WARN, "AOTriton SDPA",
             "enabled" if flag == "1" else "not set -- attention falls back to the MATH backend (~13x slower)")
        line(OK, "MIOpen cache", os.environ.get("MIOPEN_USER_DB_PATH", "(MIOpen default)"))

    # ---- GPU arithmetic -----------------------------------------------------
    if dt != "cpu":
        try:
            dev = compat.get_device()
            a = torch.ones(1024, 1024, device=dev, dtype=torch.float32)
            mean = float((a @ a).mean())
            compat.synchronize()
            if abs(mean - 1024.0) < 1e-2:
                line(OK, "gpu arithmetic", f"matmul mean {mean:.1f}, expected 1024")
            else:
                line(FAIL, "gpu arithmetic", f"WRONG RESULTS: matmul mean {mean:.1f}, expected 1024 -- reboot before trusting output")
                problems += 1
        except Exception as e:  # noqa: BLE001
            line(FAIL, "gpu arithmetic", f"{type(e).__name__}: {e}")
            problems += 1

    # ---- weights ------------------------------------------------------------
    print()
    wm = WEIGHTS_DIR / "HY-WorldMirror-2.0"
    have_wm = (wm / "model.safetensors").is_file()
    line(OK if have_wm else FAIL, "WorldMirror 2.0", f"{size_of(wm):.1f} GB" if have_wm else "missing -- python download_weights.py --recon-only")
    problems += 0 if have_wm else 1
    lora = WEIGHTS_DIR / "HY-Pano-2.0"
    have_lora = (lora / "pytorch_lora_weights.safetensors").is_file()
    line(OK if have_lora else WARN, "HY-Pano 2.0 LoRA", f"{size_of(lora):.1f} GB" if have_lora else "missing -- panorama tab unavailable")
    qwen = WEIGHTS_DIR / "Qwen-Image-Edit-2509"
    have_qwen = (qwen / "model_index.json").is_file() and (qwen / "transformer").is_dir()
    line(OK if have_qwen else WARN, "Qwen-Image-Edit base", f"{size_of(qwen):.0f} GB" if have_qwen else "missing -- panorama tab unavailable")
    ex = REPO_DIR / "examples" / "worldrecon"
    line(OK if ex.is_dir() else WARN, "bundled examples", "present" if ex.is_dir() else "python download_weights.py --examples")
    sky = REPO_DIR / "skyseg.onnx"
    line(OK if sky.is_file() else WARN, "sky segmentation", "present" if sky.is_file() else "downloaded on the first reconstruction (168 MB)")

    # ---- optional components -----------------------------------------------
    print()
    try:
        import gsplat  # noqa: F401
        from gsplat import rasterization  # noqa: F401
        line(OK, "gsplat", "built -- fly-through video available")
    except Exception as e:  # noqa: BLE001
        line(WARN, "gsplat", f"not built ({type(e).__name__}) -- reconstruction works, no fly-through video")
    for mod, why in (("gradio", "web UI"), ("trimesh", "GLB/OBJ export"), ("plyfile", "PLY reading"),
                     ("diffusers", "panorama"), ("peft", "panorama LoRA"), ("onnxruntime", "sky mask"),
                     ("psutil", "status line")):
        try:
            __import__(mod)
            line(OK, mod, why)
        except ImportError:
            line(FAIL if mod in ("gradio", "trimesh", "plyfile") else WARN, mod, f"missing ({why})")
            problems += 1 if mod in ("gradio", "trimesh", "plyfile") else 0

    # ---- memory -------------------------------------------------------------
    print()
    if dt == "cuda":
        free, total = torch.cuda.mem_get_info()
        line(OK, "gpu memory", f"{free / 2**30:.1f} / {total / 2**30:.1f} GiB free")
        if have_qwen and total < 60 * 2**30:
            line(WARN, "panorama budget", "HY-Pano peaks near 58 GiB on the GPU; this device reports less")
    elif dt == "mps":
        budget = getattr(torch.mps, "recommended_max_memory", lambda: 0)()
        if budget:
            line(OK, "gpu memory", f"{budget / 2**30:.1f} GiB of unified memory available to Metal")
            if budget < 60 * 2**30:
                line(WARN if have_qwen else OK, "panorama budget",
                     "HY-Pano needs ~58 GiB resident; reconstruction needs ~6 GiB and is fine here")
        line(OK, "low precision", f"{'bf16' if compat.supports_bf16() else 'fp16'} "
             "(HYWORLD_MPS_DTYPE=bf16|fp16 overrides)")
    try:
        import psutil
        vm = psutil.virtual_memory()
        line(OK if vm.total >= 60 * 2**30 or not have_qwen else WARN, "host RAM",
             f"{vm.total / 2**30:.0f} GB installed" + ("" if vm.total >= 60 * 2**30 or not have_qwen
                                                      else " -- the 54 GB base model is staged through host RAM"))
    except ImportError:
        pass

    print()
    if problems:
        print(f"{problems} problem(s) above need attention.")
    else:
        print("Reconstruction is usable." + (" Panorama generation is usable." if (have_lora and have_qwen) else
                                              " Panorama generation needs the HY-Pano weights."))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
