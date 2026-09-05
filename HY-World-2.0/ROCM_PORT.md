# HY-World 2.0 — ROCm port

Port of [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) to AMD
GPUs via ROCm/HIP. Upstream targets CUDA 12.8 + FlashAttention and does not run
on ROCm as shipped.

This document records **what was changed and why**, so the diff against
`origin/main` stays reviewable and the same reasoning can be reused for the
Metal/MPS port.

## Reference machine

| | |
|---|---|
| GPU | AMD Radeon 8060S (Ryzen AI Max+ 395 "Strix Halo" iGPU), **gfx1151**, RDNA 3.5 |
| Memory | **128 GB installed**, split in BIOS between a dedicated-VRAM carve-out and system RAM. It was 96 GB VRAM / **31.6 GB RAM** for every measurement up to the *Memory split* section below; since 2026-09-02 it is **64 GB / 64 GB** (63.6 GB visible to Windows). |
| | `torch.cuda.get_device_properties` reports carve-out + GTT (107.87 GiB before, 99.7 GiB now) — what the GPU can *address*, **not** the host budget. Weights are staged through host RAM, so budget against that. |
| Stack | PyTorch 2.9.1+rocmsdk20260116, HIP 7.2, Python 3.12 |
| OS | Windows 11 |

Measured on this part: **34.5 TFLOP/s** bf16 / 37.8 TFLOP/s fp16 / 3.0 TFLOP/s
fp32 dense matmul.

## Quick start

```bash
# 1. A ROCm PyTorch build must already be installed and working.
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"

# 2. Port dependencies (does NOT touch torch -- see requirements-rocm.txt).
pip install -r requirements-rocm.txt

# 3. Optional: header-only libs needed only to build gsplat (rendering / 3DGS).
./tools/setup_rocm_thirdparty.sh
```

Reconstruction (multi-view / video -> depth, normals, cameras, point cloud, 3DGS):

```bash
./scripts/run_worldrecon.sh examples/worldrecon/realistic/Desk
```

Panorama generation (image -> 360 deg equirectangular):

```bash
./scripts/run_panogen.sh path/to/image.jpg --num-inference-steps 40
```

Web UI (panorama, reconstruction, panorama → 3D, with a three.js viewer):

```bash
../launch.sh                                  # macOS / Linux
powershell -ExecutionPolicy Bypass -File ..\launch.ps1   # Windows
```

See *Web UI* at the end of this document. All launchers source
`scripts/rocm_env.sh`, and every Python entry point
imports `hyworld2.compat`, so the ROCm runtime settings apply either way.
Weights are read from `$HYWORLD_WEIGHTS` (default `../weights`) and downloaded
from the Hub on first use otherwise.

## Blockers found, and how they were resolved

### 1. FlashAttention is a hard import (worldrecon)

`hyworld2/worldrecon/.../layers/attention.py` imported `flash_attn_interface`
(FA3) or `flash_attn` (FA2) at module scope, with no fallback. Neither builds
for RDNA 3.5.

**Fix** — new `hyworld2/compat/attention.py` exposes one `attention(q,k,v)`
entry point over the `(B, H, N, D)` layout the model already uses, and picks
the best kernel present: FlashAttention-2/3 on CUDA (unchanged upstream
behaviour), `F.scaled_dot_product_attention` everywhere else. The
`(B, N, H, D)` transposes upstream did only to satisfy FlashAttention are
skipped when they are not needed.

`hyworld2/worldgen/models/attention.py` already degraded to SDPA on its own and
needed no change.

### 2. SDPA silently falls back to the MATH backend on ROCm

This is the important one. On gfx1151 PyTorch refuses both fused SDPA backends
unless an env flag is set:

```
UserWarning: Flash Efficient attention on Current AMD GPU is still experimental.
Enable it with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1.
```

Without the flag `scaled_dot_product_attention` still *works*, so nothing
fails loudly — it just drops to the MATH backend, which materialises the full
`N x N` score matrix. Measured on `(1, 16, 4096, 64)` bf16:

| backend | time | vs MATH |
|---|---|---|
| MATH (default on ROCm) | 62.3 ms | 1.0x |
| AOTriton FLASH | 4.81 ms | **13.0x** |
| AOTriton MEM_EFFICIENT | 4.80 ms | 13.0x |

Max abs error vs MATH: 9.8e-4 (bf16), 1.2e-4 (fp16) — i.e. rounding-level.

**Fix** — `hyworld2/compat/backend.py::configure_rocm_env()` sets the flag via
`os.environ.setdefault` and is invoked from `hyworld2/__init__.py`, so any
entry point into the package gets it. ATen reads the flag lazily at SDPA
dispatch, so this is effective even when `torch` was imported first. The flag
is also documented in `scripts/rocm_env.sh` for shell-level override.

fp32 has no AOTriton flash kernel; the compat layer falls back to
mem-efficient/MATH there automatically.

### 3. `gsplat` is a CUDA extension with no ROCm wheel

`rasterization.py` imported `gsplat` at module scope, which made the whole
worldrecon pipeline unimportable.

Feed-forward reconstruction never rasterizes: at `is_inference=True` the
renderer returns `predictions["splats"]` and returns early. gsplat is only
needed for novel-view rendering, video export and 3DGS training.

**Fix** — soft import with a stub that raises a descriptive `ImportError` only
if rasterization is actually called. Reconstruction (points, depth, normals,
cameras, splat parameters) runs without gsplat.

### 4. PyTorch ROCm on Windows has no distributed backend

`torch._C._distributed_c10d` is absent, so
`from torch.distributed.fsdp import ...` raises at import time in
`worldrecon/pipeline.py`, and an *unquoted* type annotation
(`sp_group: torch._C._distributed_c10d.ProcessGroup`) in
`models/models/visual_transformer.py` raises when the `def` is evaluated.

**Fix** — FSDP imports are wrapped in `try/except` and `_wrap_model_fsdp()`
raises a clear error if multi-GPU is requested without distributed support;
the annotation is quoted. Single-GPU inference is unaffected.

### 5. Dependency pins that would destroy a working ROCm stack

Upstream `requirements.txt` pins `torch==2.7.1` and `numpy==1.26.4`; installing
it replaces the ROCm build with a CUDA/CPU one. `requirements-rocm.txt`
carries the same dependency set minus torch and minus the pins that conflict
with the ROCm wheels, and documents what is deliberately omitted.

## Notes on the platform

* **MIOpen first-run cost.** The DPT heads use many distinct convolution
  shapes; MIOpen compiles and benchmarks a kernel per shape on first use and
  caches the result in `~/.miopen/db/gfx1151_*.ufdb.txt`. The first
  reconstruction at a new resolution therefore takes minutes of mostly-CPU
  time before any progress prints. Subsequent runs reuse the cache.
* **HIP synchronisation busy-waits.** A ROCm inference process sits at ~100% of
  one CPU core while the GPU works. That is expected, not a hang.
* **Watch the Windows commit charge, not just "VRAM".** With 31.6 GiB of RAM
  and a 96 GiB page file the commit limit is 127.6 GiB, and it is easy to
  exhaust: an unrelated `llama-server` holding 31 GiB plus one leaked 10 GiB
  Python process put the system at 120.7/127.6 GiB, after which loading
  WorldMirror's 4.8 GB checkpoint **segfaulted during `.to(device)`** — no
  Python traceback, just exit 139. If a load that used to work starts
  segfaulting, check `\Memory\Committed Bytes` before suspecting the code.
  Force-killing a large ROCm process can leave its commit behind.
* **One HIP process at a time.** Running two PyTorch processes against this
  iGPU concurrently segfaulted the second one (exit 139) reproducibly. Unlike a
  discrete card, the APU shares its queues with the desktop compositor; keep
  inference serialized.
* **Windows GPU counters do not see ROCm compute queues.** `\GPU Engine(*)`
  perf counters report 0% while HIP kernels are running; only
  `\GPU Process Memory(*)` is meaningful. Use it to confirm a process is live.

## Measured performance (gfx1151, 110 GiB unified)

### Attention (`scaled_dot_product_attention`, `(1, 16, 4096, 64)`)

| dtype | MATH | AOTriton FLASH | AOTriton MEM_EFF |
|---|---|---|---|
| bf16 | 62.3 ms | 4.81 ms | 4.80 ms |
| fp16 | 62.5 ms | 4.74 ms | 4.72 ms |
| fp32 | 56.0 ms | no kernel | 39.4 ms |

### Convolution — the actual bottleneck in WorldMirror

The DPT heads run many convolutions at up to full input resolution, and this is
where the reconstruction time goes. Measured with `B=2`, NCHW:

| shape | fp32 | bf16 | fp16 |
|---|---|---|---|
| 256→256 3x3 @204x272 | 8.5 ms | 7.6 ms | **3.8 ms** |
| 256→256 3x3 @408x544 | 219.0 ms | 36.6 ms | **20.3 ms** |
| 256→128 3x3 @714x952 | 182.6 ms | 87.0 ms | **17.5 ms** |
| 3→256 3x3 @714x952 | 20.1 ms | 11.2 ms | 11.4 ms |

Two consequences for this port:

* **fp32 convolutions are pathologically slow** (2.4-4.4 TFLOP/s vs 34.5
  TFLOP/s for bf16 matmul). Upstream deliberately keeps `output_conv2` and
  `MlpFP32.fc2` in fp32 for numerical stability; on ROCm those few layers cost
  a large share of the runtime. Running `--enable_bf16` is therefore not
  optional here, it is the difference between usable and unusable.
* **`channels_last` is catastrophic on gfx1151** — 0.03-0.04 TFLOP/s, roughly
  **500x slower** than NCHW for the same convolution, in every dtype tested.
  The usual "NHWC is faster for convnets" advice is inverted on this part.
  Do not add `memory_format=torch.channels_last` anywhere in this port.

### End-to-end: WorldMirror 2.0, 2 images at 714x952, `--enable_bf16`

| stage | cold | warm |
|---|---|---|
| model load | 13.8 s | 12.5 s |
| inference | 406.7 s | **155.9 s** |
| filter mask | 287.8 s (incl. one-off `skyseg.onnx` download) | 1.6 s |
| save (depth/normal/camera/points/gaussians) | 2.6 s | 2.7 s |
| **total** | 697.1 s | **160.3 s** |

Peak GPU memory: 13.7 GB. Outputs: 1.32 M points, 799 K gaussians after voxel
pruning.

The cold/warm gap is MIOpen compiling a kernel per convolution shape; the cache
lives in `~/.miopen/db/gfx1151_*.ufdb.txt` and is reused by later runs at the
same resolution.

## Checkpoint loading note

`_load_state_dict_selective` reports `Loaded 1545/1593 keys`. The 48 keys it
skips are all `...attn.rope.periods` — RoPE frequency buffers that are
recomputed deterministically at construction time. This is identical on CUDA
and is not a sign of a broken load.

## gsplat on ROCm — **built and working**

`gsplat` is only needed to *rasterize* splats (novel-view rendering, video
export, 3DGS training). Feed-forward reconstruction does not need it, so the
soft import in §3 keeps everything else working. Porting it is a separate piece
of work; this is how far it got and what remains.

Build entry point: `tools/build_gsplat_rocm.bat`, headers fetched by
`tools/setup_rocm_thirdparty.sh`.

### Solved

1. **`ROCM_HOME` is misdetected with the `rocm-sdk` wheels.**
   `torch.utils.cpp_extension` derives it from `where hipcc`, which resolves to
   `<python-prefix>\Scripts\hipcc.exe`; parent-of-parent is the Python prefix,
   so torch then looks for a non-existent `<prefix>\bin\hipcc.exe` and every
   compile dies with `CreateProcess failed`. The usable ROCm tree (bin +
   include + lib) is the `_rocm_sdk_core` package. Note that
   `rocm-sdk path --root` reports `_rocm_sdk_devel`, whose `include/` is empty
   even after `rocm-sdk init` — it is *not* a valid `ROCM_HOME`.

2. **nvcc-only flags in gsplat's `setup.py`.** `--use_fast_math`,
   `-diag-suppress`, `-allow-unsupported-compiler` and
   `--expt-relaxed-constexpr` are rejected by clang. The HIP branch now emits
   `-ffast-math` and skips the rest.

3. **ROCm device bitcode not found.** The `rocm-sdk` wheels keep AMDGCN bitcode
   under `lib/llvm/amdgcn/bitcode`, not `<root>/amdgcn/bitcode` where clang
   probes, so the build failed with "cannot find ROCm device library". Fixed by
   passing `--rocm-device-lib-path` explicitly.

4. **No rocThrust headers anywhere.** `torch/headeronly/util/complex.h` does
   `#include <thrust/complex.h>` whenever `__HIPCC__` is defined, so *every*
   HIP extension built against PyTorch needs rocThrust — and the `rocm-sdk`
   wheels ship none. `tools/setup_rocm_thirdparty.sh` checks out rocThrust,
   rocPRIM and hipCUB and generates the three version headers that CMake's
   `configure_file()` would normally produce.

5. **glm missing entirely.** gsplat vendors glm as a git submodule upstream;
   HY-World-2.0 ships only the empty directory, so the build fails with
   "glm/gtc/type_ptr.hpp file not found" on CUDA too. It is now checked out
   *outside* the gsplat tree, because torch's hipify mirrors `gsplat/cuda` to
   `gsplat/hip` copying only source extensions it recognises — which silently
   drops glm's `.inl` files and breaks it.

6. **glm's compiler detection.** hipcc defines `__CUDACC__`, so
   `glm/simd/platform.h` takes its CUDA branch (which precedes the `__HIP__`
   branch) and errors with "GLM requires CUDA 7.0 or higher" because
   `CUDA_VERSION` is undefined. Passing `-DCUDA_VERSION=8000` selects
   `GLM_COMPILER_CUDA80`, whose only effect is the `__device__ __host__`
   decoration hipcc wants anyway.

7. **Host `.cpp` files compiled with MSVC.** On Windows,
   `torch.utils.cpp_extension` sends `.cpp` to `cl.exe`, but gsplat's host files
   include HIP headers (`amd_hip_vector_types.h`) whose `__attribute__` syntax
   `cl` cannot parse. `setup.py` now generates a one-line `.cu` shim per `.cpp`
   under `gsplat/cuda/csrc/_hip_host/` so the HIP toolchain compiles them.

### Solved (continued)

8. **CUDA cooperative groups.** HIP's `cooperative_groups` provides
   `thread_block_tile`, `tiled_partition`, `binary_partition` and
   `coalesced_threads`, but neither `cg::reduce` nor `cg::labeled_partition`,
   and there is no `<cooperative_groups/reduce.h>`. All of it is now shimmed in
   `gsplat/cuda/include/Utils.cuh`, behind `#ifdef USE_ROCM` so the CUDA path
   is untouched:

   * `cg::reduce` — every use is a warp-wide sum or max (float, plus one int
     max), which a `shfl_xor` butterfly computes exactly. Exposed as
     `GSPLAT_WARP_SUM` / `GSPLAT_WARP_MAX` / `GSPLAT_WARP_MAX_INT`.
   * `cg::labeled_partition` — used only in the backward projection kernels, to
     group threads sharing a gaussian/camera id so their gradients are summed
     once and written with a single atomic. The shim returns a one-thread
     group, which makes `warpSum` a no-op and gives every thread
     `thread_rank() == 0`, i.e. each thread does its own `atomicAdd`. Same
     result, more atomic traffic, and **no edits to the kernels themselves**.
   * The CUDA-only `<cooperative_groups/reduce.h>` include is now guarded.

9. **`hipFuncSetAttribute` signature.** hipify rewrites the name but not the
   argument; HIP's overload needs an explicit `(const void *)` cast. Applied to
   all 8 call sites.

### Also solved

10. **hipify does not translate `cub::` symbols.** It rewrites the include to
    `<hipcub/hipcub.hpp>` but leaves the call sites as `cub::DeviceRadixSort`.
    Fixed with `namespace cub = hipcub;` under `USE_ROCM`.
11. **`namespace cg` alias conflict.** Self-inflicted: the shim namespace was
    first placed *inside* `namespace gsplat`, so `cg` bound to
    `gsplat::cooperative_groups` in one translation unit and to the real
    `::cooperative_groups` in another. The shim now sits at global scope,
    before `namespace gsplat {`.
12. **MSVC `<array>` in device code** — `Cameras.cuh` called
    `std::array::at()` inside a `__host__ __device__` function; `.at()` is
    bounds-checked and calls MSVC's host-only `_Xran()`. The 10 indices are
    compile-time constants in range, so unchecked `operator[]` is equivalent.
13. **`import gsplat` pulled in `torch.distributed.nn`**, absent on builds
    without a distributed backend. Softened, like section 4 — that module is
    multi-GPU only.

### Verified

```
Successfully installed gsplat-1.5.3
forward : (1, 128, 128, 3) finite=True range=[0.0000, 0.8878] alpha_mean=0.0576
backward: means=ok, quats=ok, scales=ok, opacities=ok, colors=ok
```

The backward pass is the meaningful check: `cg::labeled_partition` is used
*only* there, so finite gradients on all five tensors exercise both shims.
Reproduce with `tools/t_gsplat.py`.

Confirmed end-to-end through the pipeline as well:

```
$ ./scripts/run_worldrecon.sh examples/worldrecon/realistic/Desk --save_rendered
[Inference] Done in 155.84s
Video saved to .../desk_video2/rendered (mode: split)
    Case total  161.100s
```

producing `rendered/rendered_rgb.mp4` alongside the usual point cloud and
splats. Two dependency snags on that last mile, both recorded in
`requirements-rocm.txt`:

* `moviepy` is not in upstream's requirements at all, so `--save_rendered`
  ended with "video rendering failed: No module named 'moviepy'".
* It must be pinned `moviepy<2`: `render_utils.py` imports `moviepy.editor`,
  which moviepy 2.0 removed (its contents moved to the top-level package).

Novel-view rendering, video export and 3DGS training are therefore unblocked on
ROCm.


## Panorama generation (HY-Pano 2.0 / Qwen-Image-Edit backend)

The generation path needed two changes beyond the shared compat layer.

### 10. The panogen entry point never loaded the compat layer

`hyworld2/panogen/pipeline_with_qwen_image.py` is designed to be run as a
script *from inside* `hyworld2/panogen`, so `sys.path[0]` is that directory and
the `hyworld2` package is not importable — meaning
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` was never set and every attention in
the 20B DiT and the Qwen2.5-VL text encoder silently fell back to the MATH
backend. Confirmed from a run's own warnings:

```
UserWarning: Flash Efficient attention on Current AMD GPU is still experimental.
Enable it with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1.
  attn_output = torch.nn.functional.scaled_dot_product_attention(
```

The file now puts the repo root on `sys.path` and imports `hyworld2.compat`
before torch is used. `scripts/run_panogen.sh` additionally sources
`scripts/rocm_env.sh`, so both the library and the shell route are covered.

### 11. The Qwen VAE is a 3D causal-conv video VAE, and it hits a size cliff

`QwenImageCausalConv3d` wraps `nn.Conv3d`. MIOpen handles 3D convolutions on
gfx1151 reasonably up to a point and then collapses. Measured with 128
channels, 3x3x3, batch 1:

| input | Conv3d bf16 | Conv2d bf16 (same area) |
|---|---|---|
| 240x488 | 15.5 ms (6.69 TFLOP/s) | 3.4 ms |
| 480x976 | 57.5 ms (7.21 TFLOP/s) | 12.7 ms |
| **960x1952** | **6971.9 ms (0.24 TFLOP/s)** | 52.2 ms |

A 4x larger input costs **450x** more time. fp32 behaves identically
(6868.1 ms), so the fallback is selected by size, not dtype.

For a 1952x960 panorama the two directions of the VAE land on opposite sides of
that cliff, which makes a single tiling switch wrong either way. Measured
standalone on the real VAE:

| | no tiling | tiling (256) |
|---|---|---|
| encode 768x1360 | **3.41 s** | 135.78 s |
| decode -> 960x1952 | minutes (cliff) | **96.83 s** |

So tiling is a **37x pessimisation for the encode** and a large win for the
decode — dozens of small convolutions whose fixed overhead dominates, versus
one convolution that falls off the cliff.

`hyworld2/compat/vae_tiling.py::enable_decode_only_tiling()` enables diffusers'
tiling and wraps `vae.encode` to clear `use_tiling` for the duration of the
call, so each direction gets the treatment it needs. It is applied by default
on ROCm from `HunyuanPanoPipeline.from_pretrained`.

Two things that sound plausible here but are **not** the problem, both measured:

* *Unified-memory pressure.* With the 38.3 GiB transformer resident, the same
  VAE encode ran in 1.38 s versus 3.41 s alone — no slowdown at all.
* *CPU offload as a workaround.* `enable_model_cpu_offload()` segfaults on this
  ROCm/Windows build (access violation 0xC0000005 inside `c10.dll`) at the
  first denoising step. It is available as `cpu_offload=True` but off by
  default.

Folding the single-frame `Conv3d` into an equivalent `Conv2d` (the temporal
padding is zeros, so for `T=1` only the last temporal weight slice
contributes) is mathematically exact and looks attractive given the Conv2d
numbers above, but a first attempt measured *slower* (60.6 s vs 3.4 s),
probably because the non-contiguous weight slice defeats MIOpen. Left as
future work rather than shipped unproven.

## Memory split: why panorama generation did not fit, and the fix

This turned out to be a hardware budget problem, not a porting defect —
and, it later emerged, a *configuration* problem rather than a lack of RAM
(see the resolution at the end of this section).

`torch.cuda.get_device_properties(0).total_memory` reports **107.87 GiB**, which
is the GTT/GART aperture the driver advertises. The machine has **31.6 GiB of
physical RAM** plus a 95.9 GiB page file. The Qwen-Image-Edit-2509 base is
~54 GB in bf16, so most of it is backed by the page file, and every GPU access
to those pages is a disk fault.

Everything downstream of that follows:

* The run appears to *hang* inside `QwenImageCausalConv3d` at 100% of one CPU
  core. That is the HIP busy-wait spinning while pages fault in from disk, not
  a slow kernel.
* `HIP out of memory. Tried to allocate 5.10 GiB ... 46.53 GiB is free` — the
  driver's GTT accounting says there is room; the OS cannot commit it.
* `OSError: [WinError 1455] The paging file is too small for this operation` —
  the commit limit itself is reached.
* The isolated VAE (250 MB of weights) encodes the same 1376x768 tensor in
  3.4 s, because nothing else is paged out.

Measurements that were made along the way and are still valid: tiling is a 37x
pessimisation for the VAE encode and a win for the decode (hence
`hyworld2/compat/vae_tiling.py`), Conv3d does fall off a size cliff, and
`enable_model_cpu_offload()` segfaults on this build. None of them is the
reason the panorama does not finish.

**To run panorama generation on this machine**, pick one:

* Give the host more RAM — ~64 GiB makes the bf16 base comfortable. **(Done, see below.)**
* Use a quantized base (fp8 / int8 Qwen-Image-Edit) to bring the resident set
  under ~30 GB.
* `enable_sequential_cpu_offload()` — bounded memory, much slower per step, and
  note that `enable_model_cpu_offload()` (the faster variant) crashes here.

### Resolved: the 31.6 GB was a BIOS setting, not the installed RAM

The box has **128 GB** installed (8 × 16 GB). Strix Halo lets the BIOS carve
part of it out as dedicated VRAM, and it was set to **96 GB VRAM / 32 GB RAM**
— exactly backwards for this workload, because every model is staged through
host RAM on its way to the GPU while reconstruction only ever needs ~2.4 GB
of GPU memory. (WMI's `AdapterRAM` truncates to 4 GB; the real carve-out is
in the registry key `HardwareInformation.qwMemorySize`.)

ROCm on Windows can also reach system memory through GTT (roughly half of
it), so shrinking the carve-out costs the GPU much less than it gives the host:

| BIOS split (VRAM / RAM) | GPU-addressable (`total_memory`) | Windows sees |
|---|---|---|
| 96 GB / 32 GB (before) | 107.9 GiB | 31.6 GB |
| **64 GB / 64 GB (now)** | **99.7 GiB** | **63.6 GB** |

With 64/64 the panorama pipeline runs unchanged, with `cpu_offload=False`
and the full bf16 base resident on the GPU:

| Stage (1952×960, 40 steps, `examples/worldrecon/realistic/Landmark/frame_0000.png`) | Measured |
|---|---|
| Denoising, 40 steps | **21 s** (1.6–2.0 it/s) |
| Whole first run, cold (model load + text encoder + VAE) | ~56 min wall |

The denoiser is fast; the cold start is not. `safetensors` shards "load" in a
second because they are memory-mapped, so the real cost is paid when the
54 GB is faulted in page by page during `.to(device)`. Loading straight to the
device (`device_map="cuda"` in diffusers 0.36) is the obvious next
experiment; the web UI keeps the pipeline resident between generations so the
cost is paid once per session either way.

A note on `expandable_segments:True`: it is set by default in
`hyworld2/compat/backend.py` and `scripts/rocm_env.sh`, but the ROCm runtime on
**Windows does not implement it** — it warns "expandable_segments not supported
on this platform" and is a no-op there. It is kept for Linux ROCm. It was
therefore *not* the fix for the fragmentation-shaped OOM above; that OOM is a
symptom of the commit limit, not of allocator layout.

Reconstruction (WorldMirror 2.0, ~2.4 GB resident) is unaffected and fully
working — it never approaches these limits.

## The MIOpen Conv3d shape cliff, and folding it away

The panorama pipeline appeared to hang on an ordinary 1800x1127 photograph:
15+ minutes inside one convolution, one CPU core pinned, the GPU idle. It was
not hung. The Qwen-Image VAE is a *video* VAE built from
`QwenImageCausalConv3d`, and MIOpen's solver choice for those on gfx1151 is
decided by the exact spatial shape. Measured on the real layer (128 channels,
3x3x3, bf16, one frame):

| VAE input | source image | Conv3d | Conv2d, folded |
|---|---|---|---|
| 1376x768 | 16:9 photo | **0.03 TFLOP/s** | 10.76 TFLOP/s |
| 1184x896 | 4:3 photo | **0.03 TFLOP/s** | 10.82 TFLOP/s |
| 896x1184 | portrait | **0.03 TFLOP/s** | 10.71 TFLOP/s |
| 1280x800 | the run that hung | 1.83 TFLOP/s | 10.83 TFLOP/s |
| 1024x1024 | square | 1.87 TFLOP/s | 10.30 TFLOP/s |

Note what this rules out: *area* does not select the bad path -- every row is
about a megapixel. Two ordinary aspect ratios land on a fallback 60x slower
than the good case, and even the "good" case is 6x off what the hardware does.

`hyworld2/compat/conv3d_fold.py` removes the lottery. With a single frame the
causal layer left-pads time by `kernel_d - 1` zeros and convolves with no
padding of its own, so the window is `[0, 0, x]` and only the last temporal
weight slice multiplies real data: `conv2d` with `weight[:, :, -1]` is exactly
equivalent. An earlier attempt at this was abandoned for being slower -- the
difference is `.contiguous()`, without which a strided weight view sends MIOpen
down a generic path and hands the win straight back.

It is the same arithmetic, not an approximation, and that was checked rather
than asserted -- both bf16 paths were compared against an fp32 reference of the
same VAE on a real photograph:

| path | vs fp32 reference |
|---|---|
| bf16, stock Conv3d | mean 0.0107, **PSNR 38.9 dB** |
| bf16, folded Conv2d | mean 0.0107, **PSNR 38.8 dB** |

The two bf16 paths differ from *each other* by a mean of 0.011 on a [-1, 1]
image, which looks alarming until it is measured against ground truth: folding
is exactly as close to fp32 as the stock path. The disagreement is bf16
rounding reshuffled, not error introduced.

### Folding and tiling are complementary, not alternatives

Disabling the decode tiling once folding applied looked like a tidy
simplification and was wrong. Tiling was never only a speed workaround -- it
also bounds the decode's peak memory. With the 54 GB model resident, **61.4 GB
of the 64 GB carve-out is in use**, and an untiled decode to 1952x960 has
nowhere to put its activations: the run stops dead with the CPU idle *and* the
GPU idle, blocked in the first synchronising copy after the denoising loop.
Folding makes each tile cheap; tiling keeps the peak bounded. Ship both.

End to end on the reference machine, on the photograph that used to hang:

| stage | before | after |
|---|---|---|
| model load (cold) | ~55 min (32 GB RAM) | **77.8 s** |
| VAE encode | 15+ min, never finished | **23.1 s** (first call, incl. MIOpen compile) |
| denoising, 40 steps | -- | **20 s** (1.24 it/s) |

### The second wall was not a wall: tqdm was lying

After the folding fix the panorama still looked stuck in the UI -- 40/40
denoising steps in 32 s, then many minutes of silence with one CPU core pinned,
the stack parked on the first synchronising copy after the loop. Two runs were
killed at 22 and 7 minutes on the assumption that they would never finish.

They would have. **The denoising loop never synchronises**, so tqdm counts
Python iterations while the GPU work merely queues; the entire cost then
surfaces at the first synchronising call afterwards, which is precisely the
line the stack always pointed at. Line up every measurement and the phantom
disappears:

| run | steps | total | per step |
|---|---|---|---|
| CLI, fresh process | 8 | 315 s | 39 s |
| CLI, in a worker thread | 10 | 382 s | 38 s |
| UI, via a subprocess worker | 40 | ~1500 s | 37.5 s |

One speed, three configurations. A 1952x960 panorama at 40 steps takes about
**25 minutes** on this hardware, and the run that "hung" wrote its
`panorama.png` exactly when that arithmetic says it should have.

Everything eliminated along the way was eliminated correctly -- output size,
step count, allocator cache, worker thread, the UI's stage wrappers and its
stdout capture were each replicated in `tools/t_pano_memory.py` and each
completed. The error was in the premise: there was nothing left to find,
because nothing was broken.

Two things came out of it that are worth keeping:

* `pipeline_with_qwen_image.forward(sync_each_step=...)`, on by default for any
  GPU backend, synchronises after each denoising step. It costs nothing and
  makes the progress bar and its ETA describe work actually done. A bar that
  races to 100% and then stops is worse than no bar at all.
* A caution about diagnosis on this platform. "CPU busy, GPU idle" was read as
  a stalled kernel; the GPU half of that reading is meaningless (see the
  counter warning above) and the CPU half was a HIP busy-wait doing exactly
  what it should. `py-spy dump` gives the true Python frame, but a frame parked
  on a synchronising call says "waiting for the GPU", not "wedged" -- the way
  to tell them apart is arithmetic against a known-good run, not intuition.

### Allocator cache hygiene

Separately, and still worth doing: the panorama peaks at 58.4 GiB inside the
64 GB carve-out regardless of output size, so only ~5.5 GB of headroom. A
long-lived process can lose that to cached-but-unused blocks.
`gradio_app._free_gpu_cache()` releases them under the GPU lock before each job
and prints what it recovered. This was originally introduced as a fix for the
stall above; it was not one, and it is kept only as hygiene.

Worth knowing if headroom ever does run out: the 20B transformer is dead weight
by the time the VAE decodes, so parking it on CPU would free ~40 GB rather than
a few. `tools/t_pano_memory.py --free-transformer` implements and measures that.

## Web UI

Upstream ships a Gradio demo for WorldMirror only (`hyworld2/worldrecon/gradio_app.py`,
multi-GPU oriented, `gr.Model3D` for display). The port adds its own UI at the
**wrapper root** (one level above this repo), modelled on the sibling
Hunyuan3D 2.1 port: a trilingual (English / 中文 / Русский) Gradio app with a
three.js viewer, so a generated scene is shown in the browser straight away.

```
hy-world-2.0-mac-rocm/
├── gradio_app.py        ← the UI: panorama, 3D scene, panorama → 3D, system tab
├── pano3d.py            ← equirectangular → ring of pinhole views (+ camera priors)
├── viewer/index.html    ← three.js + GaussianSplats3D viewer, served next to the app
├── launch.ps1 / launch.sh
└── HY-World-2.0/        ← this repo (branch rocm-port)
```

* **Panorama** — HY-Pano 2.0 (Qwen-Image-Edit backend). The pipeline is loaded
  on first use and kept resident, so the 54 GB cold start is paid once per
  session. The result is shown inside a sphere (drag to look around).
* **3D scene** — WorldMirror 2.0 on uploaded photos / a video, or a bundled
  example. Output: `gaussians.ply` (rendered with
  [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) on top of
  three.js), `points.ply`, camera frusta from `camera_params.json`, depth and
  normal previews, optional fly-through video (needs gsplat).
* **Panorama → 3D** — `pano3d.split_panorama()` re-projects the panorama
  into N pinhole views (default 8 × 90° around the horizon, optionally three
  pitch rows) and writes a `prior_cameras.json` in the layout
  `load_prior_camera` expects, so WorldMirror gets the exact synthetic cameras
  instead of having to estimate them. All views share one centre, so the
  scene is a depth relief rather than something with parallax.
* **System** — memory stats and per-model unload.

A scrolling log cannot distinguish "busy and quiet" from "wedged", which is the
question a user actually has, so above each log there is a status line: the
current stage, how long it has been in it, a spinner that proves the generator
is still yielding, the process's live CPU share, and a warning once output has
been silent for 90 s. The stages are read back out of the pipeline's own
stdout; the panorama backend prints nothing between "start" and the first
denoising step, so `gradio_app._announce_stages` wraps its text encoder and VAE
to mark that gap -- which is exactly where the Conv3d cliff used to sit. The
CPU figure is the diagnostic that mattered here: CPU busy with the GPU idle is
the signature of a bad convolution shape, and CPU at zero with the job still
"running" is a real stall.

The app is mounted on FastAPI so `/viewer/` and `/outputs/` are plain static
routes; the viewer takes its files by URL (`?splat=…&points=…&cams=…` or
`?pano=…`) and can be opened on its own. Splat sorting runs in a worker
without `SharedArrayBuffer`, so no cross-origin-isolation headers are needed.
Console output of a job is captured (with `
`-aware line handling for tqdm)
and streamed into the page.

Verified in the browser on the reference machine (2026-09-02):

| Flow | Result |
|---|---|
| 3D scene, bundled `realistic/Desk` (2 views, 714×952, bf16) | 161 s pipeline, 178 s incl. model load; 3DGS + cameras rendered in the page |
| Panorama → 3D, 8 views × 90° at 756×756 with camera priors | 1295 s inference (first run at this resolution, MIOpen compiling), 133 MB `gaussians.ply` |
| 360° viewer on a generated 1952×960 panorama | OK |
| Language switch EN → RU → all tabs, labels, buttons | OK |
| Status line streaming (`tools/t_ui_stream.py`, via the Gradio API) | 387 frames, stages advanced to `✅ Finished in 3:19` |

Notes for anyone extending it: Gradio's dropdown does not take programmatic
`form_input` from browser automation — click it; stdout must be unbuffered
(`python -u`, which the launchers do) or the server's own prints never reach
the terminal; and only one HIP process may use this GPU at a time, so stop the
UI server before running a CLI pipeline.
