# HY-World 2.0 — MPS (Apple Silicon) port

Status: **worldrecon is code-complete and ROCm-verified; not yet run on a Mac.** The shared
compatibility layer added for the ROCm port (`hyworld2/compat/`) is already
backend-agnostic and recognises MPS; this document records exactly what that
covers, what is verified, and what still needs doing on real hardware.

Read `ROCM_PORT.md` first — most blockers there (FlashAttention, gsplat,
distributed, dependency pins) are *shared* between the two ports, and the fixes
were written to serve both.

## What the compat layer already handles

`hyworld2/compat/backend.py`:

| concern | behaviour on MPS |
|---|---|
| `get_device()` / `device_type()` | returns `mps` when available and no CUDA/HIP device is present; `HYWORLD_DEVICE` overrides |
| `autocast(...)` | binds autocast to `mps`, and **downgrades a bf16 request to fp16** — MPS autocast does not support bf16 |
| `supports_bf16()` | `False` on MPS (bf16 matmul coverage is incomplete), so `preferred_dtype()` returns fp16 |
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

### worldrecon (WorldMirror 2.0) — code-complete, awaiting hardware

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

**What is left is hardware validation**: run the reconstruction on a Mac and
find which Metal kernels are missing or wrong. Nothing further can be
determined from a ROCm machine.

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

## How to validate on a Mac

```bash
pip install -r requirements-rocm.txt   # torch-free; the name is historical
python -c "from hyworld2 import compat; print(compat.describe())"
python -m hyworld2.worldrecon.pipeline --input_path examples/worldrecon/realistic/Desk
```

`compat.describe()` should report `Apple MPS`. Set `HYWORLD_DEVICE=cpu` to
bisect a suspected Metal kernel bug against a CPU reference.

## Web UI

The Gradio + three.js UI at the wrapper root (`../gradio_app.py`, see *Web UI*
in `ROCM_PORT.md`) is backend-agnostic: it only goes through `hyworld2.compat`,
and `../launch.sh` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` and lifts the MPS
high-watermark cap on Darwin. It has not been run on a Mac yet, like the rest
of this port.
