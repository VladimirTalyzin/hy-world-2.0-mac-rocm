# HY-World 2.0 — MPS (Apple Silicon) port

Status: **worldrecon (3D reconstruction) and panorama → 3D are verified on a
Mac** — Apple M4 Pro, 24 GB unified memory, macOS 26.6, PyTorch 2.14.0,
Python 3.12 — through the CLI, the web interface, the viewer, the Results tab
and every export. **Panorama generation (HY-Pano 2.0) is not**: it needs
~58 GB resident and has not been run on Metal. The shared compatibility layer
(`hyworld2/compat/`) is backend-agnostic and recognises MPS; this document
records what it covers, what was run, what the Mac run found and fixed, and
what is left.

Read `ROCM_PORT.md` first — most blockers there (FlashAttention, gsplat,
distributed, dependency pins) are *shared* between the two ports, and the fixes
were written to serve both.

## What the compat layer already handles

`hyworld2/compat/backend.py`:

| concern | behaviour on MPS |
|---|---|
| `get_device()` / `device_type()` | returns `mps` when available and no CUDA/HIP device is present; `HYWORLD_DEVICE` overrides |
| `autocast(...)` | binds autocast to `mps`; downgrades a bf16 request to fp16 only where Metal has no bf16 autocast (macOS < 14 or an old torch); an fp32 request disables autocast on every backend (the model uses one around its heads to undo the outer low-precision context) |
| `supports_bf16()` / `preferred_dtype()` | **bf16** when the Metal backend actually provides it — checked by running a small matmul under `torch.amp.autocast("mps", dtype=torch.bfloat16)` — else fp16. `HYWORLD_MPS_DTYPE=bf16\|fp16` overrides. On the M4 Pro bf16 and fp16 run at the same speed and bf16 is what the model was trained in |
| `configure_mps_env()` | `PYTORCH_ENABLE_MPS_FALLBACK=1` (ops without a Metal kernel run on the CPU instead of raising) and `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (no soft allocator cap), both `setdefault`, applied at import so the CLI entry points get them too |
| `empty_cache()` / `synchronize()` | routed to `torch.mps.*` |
| `memory_allocated()` / `max_memory_allocated()` | `torch.mps.current_allocated_memory()`; peak stats are a no-op |
| `configure_rocm_env()` | no-op off ROCm |

`hyworld2/compat/attention.py` picks `F.scaled_dot_product_attention` whenever
FlashAttention is absent, which is always the case on Metal, and falls back to
the MATH backend for shapes the Metal kernels reject.

## Shared fixes that MPS also needs (already done)

* **FlashAttention hard import** in `worldrecon/.../layers/attention.py` —
  removed; see ROCM_PORT.md §1.
* **`gsplat` imported at module scope** — softened, so reconstruction runs
  without a rasterizer; see §3. There is no Metal gsplat build either, so MPS
  depends on this same change.
* **`torch.distributed.fsdp` imported at module scope** and an unquoted
  `torch._C._distributed_c10d` annotation — both guarded; see §4. MPS builds
  have no NCCL and single-device inference is the only mode.
* **`dist.is_initialized()` on a stub `torch.distributed`** — guarded with
  `dist.is_available()` across worldrecon and worldgen.
* **`torch.amp.autocast("cuda", ...)`** in `visual_transformer.py`,
  `worldmirror.py` and `worldrecon/pipeline.py` — now `compat.autocast(...)`.

## Remaining work, by subsystem

### worldrecon (WorldMirror 2.0) — verified on a Mac

**Done.** Every hardcoded `torch.bfloat16` is gone from the reconstruction
path; the low-precision dtype now comes from `compat.preferred_dtype()`, which
returns bf16 on CUDA/ROCm and fp16 on MPS. Sites changed:

* `pipeline.py` — `_cast_noncritical_fp32_to_bf16()`, the `from_pretrained`
  cast block, both `all(p.dtype == ...)` guards, and the autocast call
  (`compat.autocast(enabled=...)` now picks its own dtype).
* `models/models/worldmirror.py` — 5 input casts (`views['img']`, `poses`,
  `depths`, `rays`, `context_imgs`) and 1 autocast.
* `models/models/visual_transformer.py` — 2 autocasts.

The `--enable_bf16` CLI flag keeps its name for compatibility; it now means
"use the backend's low-precision dtype".

**Verified on ROCm** so the change cannot regress the working backend: same
example, inference 156.55 s versus 155.94 s before the refactor, with
visually identical depth and normal maps.

**`torch.linalg` needs no work.** The only inverse in the inference path,
in `worldmirror.py`, already has a `try/except` CPU fallback upstream;
`pipeline.py` and `inference_utils.py` cast to fp32 first; the one in
`rasterization.py` is inside gsplat and is never reached at inference.

The remaining `cuda` literals in `pipeline.py` (lines 303/304 and the
process-group timeouts) are all inside `is_distributed` branches that MPS
never enters.

**Hardware validation is done** — see *What the Mac run found* below. No
Metal kernel was missing or wrong.

### panogen (HY-Pano 2.0)

`pipeline_with_qwen_image.py` now selects the device via
`compat.get_device()` and enables VAE tiling off CUDA. The base model is
Qwen-Image-Edit-2509 at ~54 GB in bf16, so a Mac needs either a very large
unified-memory configuration or `enable_model_cpu_offload()`. The 3D
causal-conv VAE that is slow on ROCm is worth measuring on Metal too.

### worldgen (WorldNav / WorldStereo 2.0 / 3DGS)

Not started for either port. Known blockers, both backends:

* `hyworld2/worldgen/src/panorama_utils.py` imports **cupy** and
  `cupyx.scipy.sparse.linalg.lsmr` for the panorama depth solve. There is no
  cupy for Metal (nor for ROCm on Windows); this needs a `scipy.sparse` CPU
  fallback or a torch implementation.
* **gsplat** is required for stage 5 (3DGS training), and has no Metal build.
* `hyworld2/worldgen/models/attention.py` already degrades to SDPA on its own.

## What the Mac run found (2026-09-06, M4 Pro 24 GB, torch 2.14.0)

Everything ran on the first attempt; the port was correct on Metal without
a single missing kernel or NaN. The problems were around the edges:

* **The inference timer was wrong on MPS.** `pipeline.py` synchronised only
  when `device.type == "cuda"`, so on Metal the timer stopped at the last
  kernel *launch* and the forward pass was charged to whatever synchronised
  next. Now any non-CPU device is synchronised.
* **CPU autocast ignored the requested dtype.** `compat.autocast(dtype=fp32)`
  — the model's way of undoing the outer low-precision context around its
  heads — became a *bf16* CPU autocast, so the CPU reference run produced
  bf16 depth maps and `numpy` refused to save them. The fp32 request now
  disables autocast on every backend, as CUDA and MPS already did.
* **`supports_bf16()` said no on MPS.** torch ≥ 2.6 on macOS 14+ has bf16
  autocast and bf16 matmuls, at the same speed as fp16 on this GPU. The
  answer is now measured rather than assumed, with an env override.
* **The mesh export read the sky mask inverted** (`export3d.py`). The
  pipeline writes the mask white where a pixel *is* sky; the export treated
  white as "not sky" and kept 44 vertices of an indoor scene. Backend-
  independent; the ROCm side had the same bug.
* **A UI handler returned 6 values for 7 outputs** on its early-return paths
  (`run_reconstruction` with nothing selected), so an empty click raised
  inside Gradio instead of showing the hint. Backend-independent.
* **The CLI scripts looked for `venv/Scripts/python.exe`** (Windows) and fell
  back to whatever `python` was on PATH — the system 3.9 here. They now try
  `venv/bin/python` first. `rocm_env.sh` sets the two MPS variables on
  Darwin and skips the ROCm ones.
* **The System tab said "no GPU"** on MPS. It now reports Metal's allocated /
  driver-held bytes and the unified-memory budget (`recommended_max_memory`),
  and the pre-job cache release works on MPS too.
* **One exit-time crash** after a 32-view run: results written, then
  `libc++abi: terminating ... recursive_mutex lock failed` while the
  interpreter tore down. Not reproducible on a second run. The CLI now
  synchronises and empties the cache before exit; the outputs were intact
  either way.

### Numbers

| what | 2 photos, 518 px | 2 photos, 952 px | 11 photos (Valley) | 32 photos (Park_Stone) | pano → 3D, 8 × 768 |
|---|---|---|---|---|---|
| inference | 2.9 s | 6.0 s | 14 s (adaptive 672) | 70 s (adaptive 504) | 28 s |
| whole case | 4.5 s | 9.0 s | 22 s | 88 s | 37 s |
| Gaussians after prune | 0.33 M | 0.80 M | 1.34 M | 1.25 M | 1.96 M |
| peak RSS | 5.5 GB | 5.5 GB | 5.5 GB | 5.5 GB | — |

Model load 6–10 s; `torch.mps.recommended_max_memory()` reports 17.8 GiB on
the 24 GB machine.

Accuracy against a CPU fp32 run of the same two photos at 518 px
(`HYWORLD_DEVICE=cpu`, 39 s inference):

| | depth, mean rel. error | depth, p99 | normals, mean angle | camera rotation | focal |
|---|---|---|---|---|---|
| MPS bf16 | 0.14 % | 0.6–1.1 % | 0.27–0.30° | 0.05° / 0.21° | 383.5 vs 383.6 |
| MPS fp16 | 0.44 % | 1.0–1.4 % | 0.21–0.22° | 0.03° / 0.12° | 383.5 vs 386.0 |

Both are well inside what a different SDPA kernel and accumulation order
produce; neither shows a failure mode. bf16 is the default.

### Not verified

* **HY-Pano 2.0 (panorama generation).** 54 GB of weights; the machine has
  24 GB. The code path selects the device through `compat`, loads in bf16
  (which Metal now supports) and can use `enable_model_cpu_offload`; the
  Conv3d fold and decode-only VAE tiling are ROCm-only defaults and were
  not tried on Metal. Someone with a 64/96/128 GB Mac: run
  `./scripts/run_panogen.sh photo.jpg --num-inference-steps 20` and report
  the per-step time and whether the VAE decode of 1952×960 completes.
* **Fly-through video** — no Metal build of gsplat.
* **WorldStereo 2.0** — not attempted on MPS (see WORLDGEN_PORT.md).

## How to validate on a Mac

```bash
pip install -r requirements-rocm.txt   # torch-free; the name is historical
python -c "from hyworld2 import compat; print(compat.describe())"
python -m hyworld2.worldrecon.pipeline --input_path examples/worldrecon/realistic/Desk
```

`compat.describe()` should report `Apple MPS`. Set `HYWORLD_DEVICE=cpu` to
bisect a suspected Metal kernel bug against a CPU reference, and
`HYWORLD_MPS_DTYPE=fp16` to rule the dtype in or out. `scripts/doctor.py`
prints the dtype in use and the unified-memory budget.

## Web UI

The Gradio + three.js UI at the wrapper root (`../gradio_app.py`, see *Web UI*
in `ROCM_PORT.md`) is backend-agnostic: it only goes through `hyworld2.compat`,
and `../launch.sh` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` and lifts the MPS
high-watermark cap on Darwin (as does `compat` itself). Driven end to end in
a browser on the M4 Pro: reconstruction of a bundled example, panorama → 3D
from an uploaded equirectangular image, the Results tab, and the exports.
