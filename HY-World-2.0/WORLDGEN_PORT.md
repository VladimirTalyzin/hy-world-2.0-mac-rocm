# World Generation (worldgen) — ROCm port plan

Companion to `ROCM_PORT.md`. That document covers what is **done**:
reconstruction (WorldMirror 2.0), panorama generation (HY-Pano 2.0 /
Qwen-Image-Edit) and gsplat-on-HIP. This one covers the part that is **not**
ported and is the reason locally generated scenes look like a panorama rather
than the navigable 3D worlds in the upstream README.

## Why the current output is not 3D

The web UI's *panorama → 3D* tab re-projects one equirectangular image into a
ring of pinhole views taken **from a single point** and hands them to
WorldMirror. Zero baseline between cameras means zero parallax, so the best
possible result is a textured shell around the capture point. Everything
occluded behind an object is missing, because nothing ever generated it.

Upstream fills that in with four further stages (`hyworld2/worldgen`), none of
which were ported:

| Stage | Script | What it contributes |
|---|---|---|
| 1. Trajectory planning | `traj_generate.py` | VLM picks targets, NavMesh plans camera paths |
| 2. Trajectory rendering | `traj_render.py` | renders the point cloud along those paths + captions |
| 3. World expansion | `video_gen.py` | **WorldStereo 2.0 hallucinates the occluded geometry** |
| 4. GS data prep | `gen_gs_data.py` | frames, aligned depth, normals, cameras |
| 5. 3DGS training | `world_gs_trainer.py` | optimises the final splat world |

Stage 3 is the one that turns a shell into a world. Stage 5 is what makes it
renderable in real time. Everything else is plumbing between them.

## Reference machine

Same box as `ROCM_PORT.md`: gfx1151, 128 GB installed, BIOS split 64 GB VRAM
carve-out / 64 GB system RAM. Upstream recommends ≥4 GPUs and tested on 8× H20,
so every stage that shards across ranks has to be collapsed to one process.

## Weight inventory

Upstream would download ~125 GB for stage 3 alone. Two findings cut that to
~59 GB:

* `hanshanxue/WorldStereo/worldstereo-memory-dmd/model.safetensors` is
  **34.9 GB / 17.43 B params, all bf16**, and its header (read over an HTTP
  range request, no download) shows it carries the *complete* transformer:
  `blocks` 16.15 B + `controlnet` 0.998 B + `condition_embedder` 0.240 B +
  `camera_embedding` 0.039 B + `patch_embedding` + `proj_out`.
* Compared against the Wan index, it covers **1303 of 1343** backbone keys.
  The only gap is `blocks.{0..39}.attn2.norm_added_q.weight` — and those turn
  out to be dead weight in the literal sense. Fetched by byte range (800 KiB
  total, `tools/fetch_wan_extras.py`) they are **all exactly zero**, while the
  neighbouring `norm_added_k` carries real trained values around 0.995. The
  reason is in diffusers itself: `WanTransformer3DModel` declares
  `_keys_to_ignore_on_load_unexpected = ["norm_added_q"]`, its attention builds
  only `norm_added_k`, and only `norm_added_k` is applied. The parameter exists
  in Wan's checkpoint and in nothing else. **No Wan transformer weights are
  needed at all.**

So the 65.9 GB `transformer/*.safetensors` shards of
`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` are **not needed**: only its
`transformer/config.json` (466 bytes) for the architecture, plus the 40 missing
vectors fetched by byte range. What is genuinely needed from the Wan repo is
the auxiliary stack:

| Component | Size | Needed for |
|---|---|---|
| `text_encoder/` (UMT5-XXL) | 22.72 GB | prompt conditioning |
| `image_encoder/` (CLIP) | 1.18 GB | I2V image conditioning |
| `vae/` (Wan 3D VAE) | 0.47 GB | latent encode/decode |
| tokenizer, scheduler, image_processor, configs | < 0.02 GB | plumbing |
| `worldstereo-memory-dmd` | 34.86 GB | the model itself |
| **Total** | **≈ 59.3 GB** | (vs 125 GB upstream) |

Later stages add MoGe-2, SAM 3 (gated on HF) and a VLM; budgeted separately.

Weights land in the **standard HF cache**, not `weights/`, because the repo ids
are hardcoded (`"hanshanxue/WorldStereo"` in `video_gen.py`, `base_model` in
the checkpoint config). That keeps `local_files_only=True` working with no
path patching.

## Memory budget for stage 3, and why it does not fit as written

Loading as upstream wrote it:

1. `WorldStereoRefSModel.from_pretrained(base_model, subfolder="transformer")`
   builds the 14 B backbone from Wan shards → ~33 GB host RAM in bf16.
2. `build_controlnet()` adds ~1 B more.
3. `load_safetensors(weights_path, device="cpu")` materialises the **entire
   34.9 GB checkpoint as a second copy**.
4. `load_state_dict` then copies it in.

Peak is ~70 GB of host RAM against 64 GB installed — over the limit before the
GPU is even touched. The text encoder is loaded `torch_dtype=torch.float32`
(22.7 GB) on top of that.

Fix direction: instantiate from config on `meta`, stream the checkpoint tensor
by tensor with `safetensors.safe_open` + `assign=True`, and load/offload the
text encoder around the transformer rather than alongside it.

## TODO

- [x] **0. Weight reconnaissance.** Read the checkpoint header by HTTP range;
      prove the Wan transformer shards are redundant. *Done — see above.*
- [x] **1. Download.** `tools/dl_worldgen.py`: WorldStereo
      `worldstereo-memory-dmd` (34.9 GB) plus the Wan auxiliary stack (23 GB),
      skipping `transformer/*.safetensors`. 58 GB on disk against upstream's
      125 GB. The `norm_added_q` question it was also meant to settle is
      settled — `tools/fetch_wan_extras.py`, see above.
- [x] **2. Single-process execution.** Done — see *Single-process execution*
      below.
- [x] **3. Memory-lean loading.** Done and measured: the 17.4 B transformer
      loads in **38.8 s** using **0.60 GiB of host RAM**, against ~70 GB for
      upstream's sequence. See *Memory-lean loading* below.
- [x] **4. Smoke test.** **Passed** — stage 3 runs end to end and produces a
      coherent forward camera move with real parallax. It is, however, **42.6
      minutes per clip** at 480×832 with 21 keyframes, which is the open
      problem; see *The clip, and what it cost* below.
- [ ] **5. Stages 1–2 (WorldNav).** MoGe-2 + SAM 3 + an OpenAI-compatible VLM.
      `src/vlm_utils.py` talks plain OpenAI chat completions with base64
      images, so a local `llama-server` with Qwen3-VL can stand in for vLLM.
      `third_party/navmesh` needs recastnavigation built with MSVC.
- [ ] **6. Stage 4 (GS data).** Depth alignment, sky split, normals.
- [ ] **7. Stage 5 (3DGS training).** Build `gsplat_maskgaussian` with the
      HIP recipe already proven for stock gsplat (`ROCM_PORT.md` §gsplat).
      Single GPU means `--max_steps 8000` per upstream's scaling note.
- [ ] **8. UI.** Replace the *panorama → 3D* tab's single-point re-projection
      with the real pipeline, keeping the current path as a fast preview mode.

Ordering is deliberate: item 4 is the largest single risk, and it is reachable
without any of items 5–8. If stage 3 cannot fit or is impossibly slow on this
part, that is known after one evening rather than after a week.


## Single-process execution — what was changed

`torch.distributed` on this build imports as a hollow module: `is_available()`
returns False and `get_rank`, `barrier`, `init_process_group`,
`all_gather_object` and the rest are simply **absent**, so the first collective
raises `AttributeError` rather than degrading to one rank. Worldgen calls them
unconditionally — 27 `barrier()` calls alone.

* **`hyworld2/compat/distributed.py`** (new) installs single-rank stand-ins for
  the missing entry points: a barrier over one rank is a no-op, an all-gather is
  a copy, a broadcast from rank 0 to itself is a copy. Collectives that cannot
  mean anything on one rank (the sequence-parallel all-to-alls) raise a
  descriptive error instead of returning plausible nonsense. It deliberately
  leaves `is_available()` reporting **False**, because every *guarded* call site
  upstream spells the check as `dist.is_available() and dist.is_initialized()`
  and its else-branch is already the correct single-process path. Nothing is
  installed on a build with a real backend, and existing attributes are never
  overwritten.
* **`video_gen.py`, `traj_render.py`, `gen_gs_data.py`** gained the same
  `sys.path` bootstrap `hyworld2/panogen/pipeline_with_qwen_image.py` already
  used (§10 of `ROCM_PORT.md`): they are run as scripts from inside
  `hyworld2/worldgen`, so without it neither the attention flag nor the
  distributed stand-ins would ever be applied. Their `init_process_group` calls
  then become no-ops and needed no edit.
* **Device meshes** are the one thing a stand-in cannot fake:
  `init_device_mesh` is a stub in this build and rejects `mesh_dim_names`
  outright. `video_gen.py` and `src/sp_utils/parallel_states.py` now skip mesh
  construction when there is no backend or the world is one rank. Nothing
  indexes the mesh unless `--fsdp` is passed, which now fails with a sentence.
* **`models/worldstereo_wrapper.py`** imported `torch.distributed.fsdp` at
  module scope, which pulls in `torch._C._distributed_c10d` and made the whole
  module unimportable. Softened exactly like §4 of `ROCM_PORT.md`, with a
  `_require_fsdp()` guard at the two call sites.
* **`src/general_utils.py`** imported `decord` at module scope for one
  function, `get_last_video_frame`. decord has no Python 3.12 wheels for
  Windows — `requirements-rocm.txt` already said so — and everything else in
  that file already reads video through OpenCV. The import is now soft and the
  function falls back to an OpenCV seek, with a sequential scan behind it for
  codecs where seeking to the final frame returns nothing.

With those five changes `from models.worldstereo_wrapper import WorldStereo`
imports and reports `Flash Attention NOT available, using SDPA` — which on this
port is the AOTriton flash path, not the slow one. WorldMirror still imports and
runs, so nothing regressed.

## Memory-lean loading — measured

`tools/t_worldstereo_meta.py` builds the model under
`accelerate.init_empty_weights()` from the Wan architecture config plus the
checkpoint's `controlnet_cfg`, with no weights anywhere:

| | |
|---|---|
| parameters | **1799 tensors, 17.432 B** |
| checkpoint | **1799 tensors, 17.432 B** — a 1:1 match, no missing or unexpected keys |
| bf16 footprint | 32.5 GiB |
| buffers | 4, all RoPE tables, **created on CPU with real values** |

The last row is the one that could have sunk the approach. `freqs_cos` /
`freqs_sin` for `rope` and `controlnet_rope` are non-persistent buffers: they
appear in no checkpoint, so streaming weights in cannot materialise them, and
left on `meta` they would fail at the first forward pass. `init_empty_weights()`
defaults to `include_buffers=False`, so they are computed normally at
construction and only the 17.4 B parameters go to `meta`. Verified finite.

`hyworld2/compat/streaming_load.py` (new) does the load: build on meta, then
`safe_open` the checkpoint and assign one tensor at a time, so the transient
cost is a single tensor rather than a second full model. Peak host RAM should be
the model plus the largest tensor, against ~70 GB for the upstream sequence.
Round-trip tested on a small module: nothing left on meta, dtype and values
match the reference, forward pass runs.


## Memory-lean loading — measured on the real checkpoint

`tools/t_worldstereo_load.py`, gfx1151, `worldstereo-memory-dmd`:

```
transformer loaded    38.8s | device 32.55 GiB (peak 32.55, reserved 32.58) | host RSS 0.60 GiB
parameters : 17.432 B, dtypes ['torch.bfloat16']
```

The host figure is the point. Upstream's sequence peaks near **70 GB** of
system RAM — 33 GB for the backbone built from the base checkpoint, plus the
34.9 GB checkpoint materialised whole as a second copy — on a machine with 64 GB.
Building on `meta` and streaming tensor by tensor straight to the GPU never
stages anything: **0.60 GiB**, which is the interpreter and CUDA context rather
than weights. Device memory lands exactly on the predicted bf16 footprint with
no transient overshoot (peak == final), so the allocator never saw a second copy
either.

That leaves ~31 GiB of the 64 GiB carve-out for the rest of the pipeline, which
is why the text-encoder question below matters and now fits.

Two further changes were needed before this could run:

* **`torch.compile` cannot work here.** `_load_aux` compiles the text encoder
  and the VAE unconditionally. Inductor lowers GPU kernels through Triton, and
  there is no Triton for ROCm on Windows — confirmed by `import triton` failing
  and `torch.compile` raising `InductorError: No module named 'triton'`.
  Because compilation is lazy the call itself succeeds and the failure lands at
  the first forward pass, far from its cause. `hyworld2/compat/backend.py` now
  exposes `can_compile()` / `maybe_compile()`, and the two call sites use it, so
  both run eager. Neither is a significant share of this pipeline's cost next to
  the transformer, which upstream never compiles.
* **UMT5-XXL in fp32 does not fit.** Upstream loads the 11 B text encoder at
  `torch_dtype=torch.float32`: 21.2 GiB, affordable across eight cards, not
  next to a 32.5 GiB transformer on one. It now defaults to the half dtype on
  ROCm and MPS, which halves it and keeps fp32's exponent range — the overflow
  that makes fp16 unusable for T5-family encoders does not apply to bf16, and
  diffusers' own Wan pipelines run this encoder in bf16 as standard.
  `HYWORLD_TEXT_ENCODER_DTYPE=fp32` restores upstream's behaviour. CUDA boxes
  are unaffected.

### A note on the loader's own safety

`_assign` replaces a meta placeholder with a real tensor, and the placeholder is
the only record of the shape the architecture expects. Installing a
differently-shaped tensor there would produce a model that fails much later,
somewhere unrelated, so the loader compares shapes at assignment time and raises
naming the tensor. Tested both ways.

## Download note

`snapshot_download` did **not** resume the Wan shards after a reboot: five
partial blobs of 1.9–2.6 GB were discarded and restarted from zero. Budget for
that when a large fetch is interrupted — the checkpoint that had already
completed (WorldStereo, 34.9 GB) was kept, so the loss is bounded by whatever
was in flight.


## The MIOpen cache was being filled twice

Chasing the first clip's apparent hang turned up a defect in the port itself.

MIOpen benchmarks candidate convolution solvers the first time it meets a
problem shape and writes the winner to its *user find database*. That search is
the expensive part — `ROCM_PORT.md` already notes it costs "minutes of
mostly-CPU time" for WorldMirror's DPT heads, and the Wan VAE's shapes are worse.
Where those answers are kept therefore matters.

`scripts/rocm_env.sh` points MIOpen at `~/.cache/miopen`. Nothing did so for code
entered directly through Python — `hyworld2/compat/backend.py` set the AOTriton
flag and the allocator config but never the MIOpen paths — so those runs used
MIOpen's own default, `~/.miopen/db`. Both files then filled up independently:

```
~/.miopen/db/gfx1151_20...ufdb.txt        801 records
~/.cache/miopen/gfx1151_20...ufdb.txt     184 records
keys present in both                      132
```

132 problem shapes had been searched **twice**, once per entry route, and each
cache was blind to the other's work.

`configure_rocm_env()` now sets `MIOPEN_USER_DB_PATH` and
`MIOPEN_CUSTOM_CACHE_DIR` to the same location the shell script uses, with
`setdefault` so an explicit setting still wins. `tools/merge_miopen_findb.py`
folds the orphaned database into the canonical one rather than discarding it:
records are keyed by everything left of the `=`, the destination wins a
collision unless told otherwise, and the file is written through a temporary so
an interrupted merge cannot leave MIOpen with half a database. The compiled
kernel cache (`.ukdb`, SQLite) is deliberately left alone — kernels rebuild
cheaply once the search result is known.

### How to tell this apart from a hang

The diagnostic sequence, for next time, since all three look identical from
outside:

* `py-spy dump --pid N` repeatedly. Parked in the same `_conv_forward` across
  samples means a convolution, not a deadlock.
* CPU delta. Exactly **1.0 core** is HIP's busy-wait while the GPU works; well
  under one core suggests MIOpen is searching on the CPU.
* **Check whether the find database is growing.** `wc -l` on the `ufdb.txt`
  every 20 s settles it: entries appearing means MIOpen is still searching and
  the run will finish and be fast next time; a static file during a long stall
  means the kernel it already chose is simply slow, which is the size-cliff case
  from `ROCM_PORT.md`.

That last check is what distinguished the two here — and it is worth doing
*before* concluding anything, because the first run of a new model on this part
looks exactly like a hang for as long as the search takes.


## The Wan VAE lands on the Conv3d cliff, and it is the whole cost

The first clip loaded fine and then sat inside `WanCausalConv3d`. Following the
diagnostic above, the find database *was* growing, so the first minutes were
MIOpen searching. But the search finishing did not help. Measured with
`tools/bench_wan_vae.py`, which loads the VAE alone (126.9 M params, 243 MiB)
so the question can be asked without the 46 GiB the full pipeline needs:

```
21 frames at 480x832, bf16, 6 encoder passes
  first call (includes MIOpen's solver search)   476.9 s
  steady state                                   178.5 s   (29.8 s per pass)
```

178 seconds is the *best kernel MIOpen could find*, not a warm-up. For scale,
this part does 34.5 TFLOP/s of bf16 matmul; the VAE is 127 M parameters. This is
the same cliff `ROCM_PORT.md` documents for the Qwen VAE, roughly two orders of
magnitude off what the arithmetic deserves.

It also dominates everything else. A clip encodes the conditioning image, encodes
the point-cloud render, and decodes the result — three passes over 21 frames —
against four denoising steps of the transformer. Left alone, the VAE would be
most of stage 3's runtime.

### Why the existing fold does not apply, and what does

`hyworld2/compat/conv3d_fold.py` handles the Qwen VAE by observing that causal
padding leaves only the last temporal weight slice multiplying real data — true
for a **single frame**. `AutoencoderKLWan._encode` walks a clip one frame first
and then **four at a time**, so most of its convolutions see four frames and the
shortcut does not hold.

The general identity does. For temporal kernel `kt`, stride `st` and dilation
`dt`, over an input already padded the way the causal layer wants it:

```
out[:, :, o] = sum over k of  conv2d( x[:, :, o*st + k*dt],  weight[:, :, k] )
```

— `kt` plain 2D convolutions, each over every output frame at once with the
temporal axis folded into the batch, summed. Identical arithmetic; Conv2d has no
cliff. That is `hyworld2/compat/conv3d_unroll.py`, and it subsumes the
single-frame case as `kt = 1` output frame.

`tools/t_conv3d_unroll.py` checks it against the unmodified layer across the
shapes the Wan encoder actually uses — one frame and four, temporal stride 1 and
2, spatial stride 1 and 2, with and without the streaming cache, for both
routes:

```
worst fp32 relative error: 1.12e-06
```

which is fp32 rounding, i.e. the same convolution. bf16 sits at ~6e-3 from the
fp32 reference either way, which is bf16's own precision (2^-8) and not
something the rewrite introduced — the comparison is deliberately against fp32
rather than against the bf16 original, because two bf16 paths can agree with
each other while both drift from the truth.

As in the Qwen fold, `.contiguous()` on each weight slice is load-bearing: a
strided view sends MIOpen down a generic path and hands the win straight back.

### The result, and where the time actually was

```
encode, 21 frames at 480x832, bf16, 6 passes
  as shipped     177.6 s     29.60 s per pass     0.36 TFLOP/s
  unrolled         5.5 s      0.92 s per pass    11.6  TFLOP/s      32x
```

The throughput column is what makes this legible, and it comes from
`tools/t_wan_vae_flops.py`, which walks the encoder on the `meta` device and
adds up the multiply-accumulates: **10.63 TFLOP per pass**. 0.36 TFLOP/s is the
cliff figure from `ROCM_PORT.md` almost exactly; 11.6 is the rate that document
measured for folded Conv2d. So the unrolled path is not merely faster, it is at
the part's normal speed for this kind of work, and there is little left to win.

A warning about measuring this: the *first* unrolled run timed at 91 s, an
unremarkable 2x, and it would have been easy to conclude the rewrite was a
disappointment. That number was MIOpen searching for the newly-introduced Conv2d
shapes. Warm, it is 5.5 s. Any comparison here has to be made with both sides
warm, or it measures the search rather than the kernel.

Per-shape, the cliff turns out to be almost entirely **one layer**
(`tools/bench_wan_conv_shapes.py`, 3x3x3, bf16, 4 frames):

| channels | spatial | conv3d | conv3d1 | conv2d |
|---|---|---|---|---|
| 96 | 480x832 | **8635.8 ms — 0.09 T/s** | 204.7 ms — 3.88 | 109.1 ms — 7.29 |
| 192 | 240x416 | 99.7 ms — 7.98 | 120.5 ms — 6.60 | 71.3 ms — 11.16 |
| 384 | 120x208 | 55.0 ms — 14.45 | 57.6 ms — 13.81 | 33.0 ms — 24.09 |
| 384 | 60x104 | 13.3 ms — 14.91 | 14.4 ms — 13.81 | 9.2 ms — 21.70 |

A single convolution at 96 channels and 480x832 takes **8.6 seconds**, and the
encoder runs several at that resolution. Two things are worth keeping from that
table beyond the headline. Conv2d wins at *every* shape, not only the cliff one,
permute and copy included — so the rewrite is not a workaround that costs
something elsewhere. And `conv3d1`, which keeps the 5D layout and avoids the
permute entirely, rescues the cliff shape (0.09 → 3.88) but loses to Conv2d
everywhere, which is why `conv2d` is the default and `conv3d1` stays available
as an option rather than a guess.

The VAE loader applies this on ROCm automatically;
`HYWORLD_VAE_CONV_UNROLL=0` turns it off.


## What item 5 will run into (reconnaissance, not yet attempted)

Noted while working on stage 3, so the next session does not rediscover it:

* **The memory bank is not optional.** `video_gen.py` constructs
  `PanoramaMemoryBank` unconditionally, and `load_mutli_traj_dataset` then reads
  `memory_inputs/<model_type>.mp4` and `..._ref_w2cs.json` from it. So a real
  scene run needs MoGe-2 and SAM 3 even though the smoke test does not. There is
  no `--no-memory` path to lean on.
* **SAM 3 needs `transformers>=5`.** Upstream pins `transformers==5.2.0`; this
  environment has 4.57.1, where `Sam3VideoModel` does not exist.
  `requirements-rocm.txt` pins no version at all, so nothing yet forces the
  issue. Upgrading is the direction upstream expects — `worldstereo_wrapper.py`
  already carries `if _tr.__version__ >= "5.0.0"` patches for the CLIP and UMT5
  API changes — but it shares an interpreter with the working HY-Pano
  (Qwen-Image-Edit) path, so the upgrade needs testing against that before it is
  made, not after.
* **`open3d` 0.18 as pinned has no Python 3.12 wheel; 0.19.0 does.** `utils3d`
  is on PyPI. `moge` installs from git per `requirements_git.txt`.
* **`decord` is already handled** — see the single-process section; the OpenCV
  fallback covers the one function that used it.
* **`third_party/navmesh`** wraps recastnavigation and has to be built with
  MSVC. Not investigated.


## `nframe` counts keyframes, not frames

Worth writing down, because getting it wrong produces an error a long way from
its cause. The first corrected clip attempt died with:

```
RuntimeError: The size of tensor a (9360) must match the size of tensor b (32760)
  at .../worldstereo.py:272 in _prepare_controlnet_inputs
```

9360 is 6 x 30 x 52 and 32760 is 21 x 30 x 52 — the same spatial grid with a
different number of temporal slots. The checkpoint's `nframe: 21` reads like a
clip length, and passing `num_frames=21` to the pipeline is the obvious thing to
do. It is wrong.

The conditioning stack — `render_video`, `render_mask`, `camera_embedding` — is
one entry per **keyframe**, and `keyframe_vae_encode` encodes each keyframe
*independently* into a single latent frame, so 21 keyframes give 21 latent
frames. The clip those latents stand for is four times longer: `num_frames`
means output video frames, and for `nframe = 21` it is `4 * (21 - 1) + 1 = 81`.
Pass 21 and the ordinary video path in the VAE produces `(21-1)/4+1 = 6` latent
frames for the hidden states while the controlnet's mask branch — which has
`mask_downsample: 1`, so it does not touch the temporal axis at all — still
produces 21. Hence the mismatch, two files away from the argument that caused it.

The pipeline's own default (`num_frames: int = 81`) is the tell, as is
`if render_video.shape[2] == num_frames // 4 + 1` choosing the keyframe encoder.

Practical consequence: a clip is 32760 tokens, not 9360, so a denoising step
costs roughly 3.5x what the shorter reading would suggest.


## The carve-out is the real constraint, and the symptom is a slow step

With the VAE fixed, the first clip reached the denoising loop and reported:

```
 25%|##5    | 1/4 [00:43<02:10,  43.42s/it]
 50%|#####  | 2/4 [08:32<09:47, 293.72s/it]
```

A step going from 43 s to 294 s is not a step getting harder — the four DMD
timesteps do identical work. Checking the machine while it ran gives the reason:

```
GPU Process Memory (local)   63.5 GiB      of a 64 GiB carve-out
Memory\Committed Bytes      252.2 GiB      of a 253.3 GiB commit limit
Memory\Available MBytes       8.7 GiB
```

The model is 46.13 GiB resident and the activations for 32760 tokens add ~17
GiB, which lands exactly on the carve-out. Past that, allocations are served
from GTT — system memory the GPU reaches over the fabric rather than its own
pool — and everything slows by roughly the ratio seen above. The 43 s first step
is the honest one: the loop does not synchronise, so the CPU races through the
first step's enqueue and only blocks once the queue is full, which is why
`py-spy` parks on an innocent-looking `einops.rearrange` two hundred lines
before the work.

**Do not read the tqdm bar as per-step cost here.** It measures when the CPU got
back, not when the GPU finished. Take the total wall time across a synchronised
boundary instead.

The run was stopped rather than left to finish: 1.1 GiB of commit headroom is
not a state to leave a machine in for another ten minutes, and the answer it
would eventually print would be a measurement of thrashing.

### The fix: send UMT5 home before denoising

The text encoder is 10.6 GiB in bf16 and is used exactly once, before any
denoising. `encode_prompt` skips the text encoder entirely when handed
`prompt_embeds`, so the smoke test now encodes the prompt itself, moves the
encoder back to the host, and passes the embeddings in. That is ~10.6 GiB
returned to the activation budget, taking the peak from 63.5 to roughly 53 GiB.

The image encoder cannot be treated the same way and is not worth it anyway:
`check_inputs` makes `image` and `image_embeds` mutually exclusive while `image`
itself is still needed for the conditioning latents, and CLIP is only 1.2 GiB.

For a production path, `RefKFDMDGeneratorPipeline` inherits `DiffusionPipeline`
and so has `enable_model_cpu_offload()`, which would do this generically: within
one call it moves each component to the GPU as its turn comes and the previous
one back, so the transformer arrives once and stays for all four steps. Worth
using in the UI; the explicit version above is preferred in the smoke test
because it keeps what is resident, and when, obvious in the measurement.


## The clip, and what it cost

```
model loaded          75.9 s | device 46.13 GiB
text encoder offloaded        device 35.55 GiB
clip generated      2559.0 s | device 35.90 GiB (peak 47.19, reserved 41.01)
  denoising          1870 s (31:10) — 42.5, 359.6, 479.7, 536.8 s per step
  VAE encode+decode   ~690 s
output: 21 keyframes at 832x480
```

The output is right: a stylised interior with the camera dollying forward, the
fireplace growing, the shelves parting at the edges, geometry holding together
between frames. That is novel-view synthesis with parallax — the thing the
single-point re-projection in the UI cannot do — and it was produced from the
*weakest* inputs the pipeline accepts: no point-cloud guidance at all, and one
reference frame. Stage 3 is ported.

Note the output is **21 frames, not 81**: this is the keyframe pipeline, and
keyframes are what it is for. `num_frames=81` sets the latent temporal geometry;
the frames that come back are the keyframes themselves.

### The open problem: the steps get slower

```
step 1    42.5 s        step 3   479.7 s
step 2   359.6 s        step 4   536.8 s
```

The four DMD timesteps do identical work, so a monotonic climb from 42 s to 537 s
is not the model. (Step 1 is not really 42 s — the loop does not synchronise, so
that figure is how long the *enqueue* took.) Something degrades as the run
proceeds, and the leading suspect is memory: torch's own peak allocation is
47.19 GiB, but Windows reported the process holding **62.5 GiB** of the 64 GiB
carve-out throughout — the ~15 GiB gap being MIOpen workspaces, the HIP context
and fragmentation. Offloading the text encoder freed 10.6 GiB and did **not**
help; the activations simply grew into it, which is itself informative.

### Where the time is *not*

`tools/bench_worldstereo_block.py` times one transformer block against sequence
length, which settles the memory question immediately:

| tokens | TFLOP | time | TFLOP/s | x40 layers | peak GiB |
|---|---|---|---|---|---|
| 4680 | 3.25 | 190.3 ms | 17.05 | 7.6 s | 1.41 |
| 9360 | 7.39 | 509.5 ms | 14.50 | 20.4 s | 1.97 |
| 16380 | 15.29 | 1626.6 ms | 9.40 | 65.1 s | 2.81 |
| 23400 | 25.20 | 4247.2 ms | 5.93 | 169.9 s | 3.65 |
| 32760 | 41.56 | 8285.7 ms | 5.02 | **331.4 s** | 4.77 |

Peak memory at the full sequence is **4.77 GiB** — nowhere near the carve-out.
So hypothesis (1) is dead: this is not spilling. And x40 layers gives 331 s,
which is the observed step cost. The block itself is simply slow at length.

The obvious next suspect was attention, since it is the O(N²) term and over
half the arithmetic at this length. Measured directly on the model's own shape,
`(1, 40, N, 128)` bf16:

| tokens | TFLOP | time | TFLOP/s | backends offered |
|---|---|---|---|---|
| 4096 | 0.34 | 20.0 ms | 17.18 | flash, mem_eff, math |
| 9360 | 1.79 | 106.6 ms | 16.82 | flash, mem_eff, math |
| 16380 | 5.49 | 317.3 ms | 17.32 | flash, mem_eff |
| 32760 | 21.98 | 1240.4 ms | **17.72** | flash, mem_eff |

Attention is *fine*: a flat 17.7 TFLOP/s all the way up, and the AOTriton flash
kernel is being used (MATH drops out entirely past 16k, as it must — its score
matrix at 32760 would be 85 GB).

So neither hypothesis survives. At 32760 tokens the block spends 8.29 s, of
which attention is 1.24 s and the dense projections and FFN account for roughly
1.15 s at the 17 TFLOP/s the same hardware manages elsewhere. **About 70% of the
block's time is in neither.** That is the thing to profile next, and it is a much
better-posed question than "why is it slow".

Cheap levers regardless, if a usable scene is wanted before that is understood:
fewer keyframes, or a smaller trained aspect from `scale_map`, both cut the token
count directly — and the table above is steeply superlinear, so halving the
tokens is worth much more than half the time.

For scale: upstream runs this on 8 GPUs with sequence parallelism, so each card
sees 32760/8 ≈ 4100 tokens — the regime at the *top* of that table, where this
part manages 17 TFLOP/s too.
