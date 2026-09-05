#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate one WorldStereo 2.0 clip, to time stage 3 and size its memory.

Stage 3 normally consumes stage 1-2 output: a planned trajectory, a point-cloud
render along it, a VLM caption and a memory bank. None of that is ported yet, so
this drives ``RefKFDMDGeneratorPipeline`` directly with inputs built here:

* **frame 0** is a perspective crop out of a panorama -- a real image, so the
  image encoder and the VAE see something meaningful;
* **the trajectory** is a straight dolly forward, which is the simplest camera
  move that produces parallax and therefore the simplest thing that can look
  right or wrong at a glance;
* **the point-cloud render** is empty except for frame 0, with the render mask
  saying so. That is the honest degenerate case: the model is asked to invent
  the whole move with no geometric guidance, which is *harder* than what stage 2
  hands it, so the timings here are an upper bound rather than a flattering one;
* **the reference frames** the memory branch requires are the starting frame
  again. In a real run these are views of the same scene generated on earlier
  trajectories, so this too is the weakest possible input rather than a
  flattering one.

Note the frame convention, which is easy to get wrong: the checkpoint's
``nframe`` (21) counts **keyframes**, one per entry of the conditioning stack,
and each is encoded on its own into a single latent frame. The clip those
latents stand for is ``4 * (nframe - 1) + 1 = 81`` frames, and that is what
``num_frames`` means to the pipeline.

What this measures is peak memory and seconds per denoising step. What it does
**not** measure is the quality stage 2's guidance would buy, so judge the output
as "plausible camera move" and not as a sample of the finished pipeline.

Only one HIP process at a time on this box -- stop the UI server first.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "HY-World-2.0"
sys.path.insert(0, str(ROOT))            # pano3d.py lives at the wrapper root
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "hyworld2" / "worldgen"))

import hyworld2  # noqa: E402,F401  (compat: ROCm flags + distributed stand-ins)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from diffusers.utils import export_to_video  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

import pano3d  # noqa: E402
from models.camera import get_camera_embedding  # noqa: E402
from models.worldstereo_wrapper import WorldStereo  # noqa: E402
from src.data_utils import assign_scale  # noqa: E402

WS_REPO = "hanshanxue/WorldStereo"


def host_rss_gib() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / 2**30


def report(label: str, t0: float) -> None:
    rss = host_rss_gib()
    line = (f"{label:<26} {time.time() - t0:7.1f}s | device "
            f"{torch.cuda.memory_allocated() / 2**30:6.2f} GiB "
            f"(peak {torch.cuda.max_memory_allocated() / 2**30:6.2f}, "
            f"reserved {torch.cuda.memory_reserved() / 2**30:6.2f})")
    if rss is not None:
        line += f" | host RSS {rss:6.2f} GiB"
    print(line, flush=True)


def first_frame(panorama: Path, height: int, width: int, *,
                yaw: float, pitch: float, fov: float) -> tuple[Image.Image, float]:
    """A perspective crop of the panorama, plus the focal length it implies.

    ``pano3d.equirect_to_perspective`` renders a square view; cropping it
    vertically to the target aspect keeps the horizontal field of view and the
    principal point, so the intrinsics stay exact.
    """
    pano = np.asarray(Image.open(panorama).convert("RGB"))
    square = pano3d.equirect_to_perspective(pano, yaw, pitch, fov, width)
    top = (width - height) // 2
    if top < 0:
        raise ValueError(f"height {height} must not exceed width {width}")
    view = square[top:top + height]
    focal = (width / 2.0) / math.tan(math.radians(fov) / 2.0)
    return Image.fromarray(view), focal


def dolly_trajectory(n: int, distance: float, device) -> torch.Tensor:
    """World-to-camera matrices for a camera translating straight ahead.

    Camera axes follow OpenCV (x right, y down, z forward), matching the rest of
    the port. With no rotation the world-to-camera rotation is the identity and
    the translation is simply minus the camera centre.
    """
    w2cs = torch.eye(4, device=device).repeat(n, 1, 1)
    steps = torch.linspace(0.0, distance, n, device=device)
    w2cs[:, 2, 3] = -steps
    return w2cs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-type", default="worldstereo-memory-dmd")
    ap.add_argument("--panorama", type=Path,
                    default=REPO / "examples" / "worldgen" / "case000" / "panorama.png")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "worldstereo_smoke")
    ap.add_argument("--frames", type=int, default=None,
                    help="Defaults to the checkpoint's own nframe (21).")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--distance", type=float, default=0.4,
                    help="How far the camera travels over the clip. The value "
                         "is almost arbitrary: get_camera_embedding normalises "
                         "the trajectory so the first camera sits at a fixed "
                         "distance from the centroid, so only the *shape* of "
                         "the motion survives, not its scale.")
    ap.add_argument("--render", choices=["black", "repeat"], default="black",
                    help="What to put in the point-cloud render channel beyond "
                         "frame 0. 'black' matches what stage 2 emits where "
                         "nothing projects; 'repeat' holds frame 0, which is "
                         "easier for the model but contradicts the camera move.")
    ap.add_argument("--prompt", default="A steady camera moving forward through the scene.")
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--references", type=int, default=1,
                    help="How many reference keyframes to hand the memory "
                         "branch. Must be at least 1: the model indexes "
                         "ref_index unconditionally.")
    ap.add_argument("--keep-text-encoder", action="store_true",
                    help="Leave UMT5 on the GPU during denoising. It is 10.6 GiB "
                         "and is finished with after the prompt is encoded, so "
                         "by default it is moved back to the host first.")
    ap.add_argument("--vae-tiling", action="store_true",
                    help="Encode and decode the VAE in tiles.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build and print the inputs, load nothing.")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg_path = hf_hub_download(WS_REPO, "config.json", subfolder=args.model_type,
                               local_files_only=True)
    raw_cfg = json.load(open(cfg_path, encoding="utf-8"))
    n = args.frames or int(raw_cfg.get("nframe", 21))

    height, width = args.height, args.width
    if [height, width] not in raw_cfg.get("scale_map", [[480, 832]]):
        height, width = assign_scale(height, width, raw_cfg.get("scale_map"))
        print(f"note: snapped to the nearest trained aspect: {height}x{width}")

    # ---- inputs -----------------------------------------------------------
    image, focal = first_frame(args.panorama, height, width,
                               yaw=args.yaw, pitch=args.pitch, fov=args.fov)

    frame0 = torch.from_numpy(np.asarray(image)).float().permute(2, 0, 1) / 255.0
    frame0 = frame0 * 2 - 1                                  # [3,h,w] in [-1,1]
    if args.render == "black":
        # What stage 2 actually produces where the point cloud projects to
        # nothing: black pixels, and a mask saying so.
        render_video = torch.full((1, 3, n, height, width), -1.0, device=device)
        render_video[:, :, 0] = frame0.to(device)
    else:
        # Frame 0 held for the whole clip. Cheap, but it tells the model the
        # scene is static while the cameras move, which is a contradiction.
        render_video = frame0[None, :, None].repeat(1, 1, n, 1, 1).to(device)
    render_mask = torch.zeros(1, 1, n, height, width, device=device)
    render_mask[:, :, 0] = 1.0                               # only frame 0 is known

    w2cs = dolly_trajectory(n, args.distance, device)
    intrinsics = torch.eye(3, device=device).repeat(n, 1, 1)
    intrinsics[:, 0, 0] = focal
    intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = width / 2.0
    intrinsics[:, 1, 2] = height / 2.0
    camera_embedding = get_camera_embedding(intrinsics, w2cs, n, height, width,
                                            normalize=True, is_w2c=True).to(device)

    # `nframe` counts *keyframes*, not output frames. The conditioning stack --
    # render video, render mask, camera embedding -- carries one entry per
    # keyframe, and `keyframe_vae_encode` encodes each one on its own into a
    # single latent frame, so the transformer sees `nframe` latent frames. The
    # clip the VAE finally decodes is the 4x-longer video those latents stand
    # for, which is what `num_frames` means to the pipeline.
    num_frames = 4 * (n - 1) + 1

    print(f"keyframes   : {n} at {height}x{width}, focal {focal:.1f}px")
    print(f"output clip : {num_frames} frames")
    print(f"render_video: {tuple(render_video.shape)} in "
          f"[{render_video.min():.2f}, {render_video.max():.2f}]")
    print(f"render_mask : {tuple(render_mask.shape)}, known frames "
          f"{int(render_mask[:, :, :, 0, 0].sum())}/{n}")
    print(f"camera_emb  : {tuple(camera_embedding.shape)}, finite="
          f"{bool(torch.isfinite(camera_embedding).all())}")

    # The "memory" variants always take reference frames: `forward` builds
    # ref_rotary_emb from `ref_index` unconditionally, before it ever checks
    # whether a reference latent exists, so leaving them out fails on
    # `ref_index + 1` rather than degrading. In a real run these come from the
    # memory bank -- views of the same scene generated on earlier trajectories.
    # Here the starting frame stands in for one, which is the weakest possible
    # memory and therefore not a flattering test.
    reference_video = frame0[None, :, None].repeat(1, 1, args.references, 1, 1).to(device)
    ref_index = np.arange(args.references)
    print(f"reference   : {tuple(reference_video.shape)}, ref_index {ref_index.tolist()}")
    if args.dry_run:
        return 0

    # ---- model ------------------------------------------------------------
    t0 = time.time()
    worldstereo = WorldStereo.from_pretrained(
        WS_REPO, subfolder=args.model_type, local_files_only=True,
        sp_world_size=1, fsdp=False, device_mesh=None, device=device,
    )
    report("model loaded", t0)

    if args.vae_tiling:
        # Tiling bounds the spatial extent each 3D convolution sees. That caps
        # peak memory, and on this part it also changes which MIOpen solver is
        # selected, since the choice keys off the exact spatial shape.
        vae = getattr(worldstereo.pipeline, "vae", None)
        if vae is not None and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
            print(f"vae tiling  : on (min tile "
                  f"{vae.tile_sample_min_height}x{vae.tile_sample_min_width})")

    # UMT5 is 10.6 GiB and is used exactly once, before any denoising. Leaving
    # it resident cost the first attempt dearly: 46.13 GiB of model plus ~17 GiB
    # of activations reached 63.5 GiB of a 64 GiB carve-out, at which point
    # allocations start being served from GTT and a denoising step went from
    # 43 s to 294 s. Encoding the prompt here and handing the pipeline the
    # embeddings lets the encoder go home first -- `encode_prompt` skips the
    # text encoder entirely when it is given `prompt_embeds`.
    #
    # The image encoder cannot be treated the same way: `image_embeds` and
    # `image` are mutually exclusive in check_inputs, and `image` itself is
    # needed later for the conditioning latents. At 1.2 GiB it is not worth
    # fighting over.
    prompt_embeds = None
    prompt_arg = args.prompt
    if not args.keep_text_encoder:
        with torch.no_grad():
            prompt_embeds, _ = worldstereo.pipeline.encode_prompt(
                prompt=args.prompt,
                do_classifier_free_guidance=False,
                device=device,
            )
        worldstereo.pipeline.text_encoder.to("cpu")
        prompt_arg = None
        gc.collect()
        torch.cuda.empty_cache()
        print(f"text encoder offloaded; device now "
              f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    kwargs = dict(
        image=image,
        render_video=render_video,
        render_mask=render_mask,
        camera_embedding=camera_embedding,
        extrinsics=w2cs,
        intrinsics=intrinsics,
        reference_video=reference_video,
        ref_index=ref_index,
        prompt=prompt_arg,
        prompt_embeds=prompt_embeds,
        negative_prompt=raw_cfg.get("negative_prompt", ""),
        height=height,
        width=width,
        num_frames=num_frames,
        generator=generator,
        output_type="pt",
        latent_cond_mode=raw_cfg.get("latent_cond_mode", "first_frame_only"),
        mode="test",
    )

    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        result = worldstereo.pipeline(**kwargs)
    torch.cuda.synchronize()
    elapsed = time.time() - t1
    report("clip generated", t1)

    frames = result.frames[0].float()                        # [f,c,h,w]
    steps = int(getattr(worldstereo.pipeline, "_num_timesteps", 0) or 0)
    if steps:
        # Deliberately not `elapsed / steps`: the elapsed time also covers the
        # VAE encode of the conditioning stack and the decode of the result,
        # which together are a large minority of it. The denoising bar's own
        # total is the honest per-step figure -- and note that its *first* step
        # is always short, because the loop does not synchronise and that step
        # only measures how long the enqueue took.
        print(f"denoising   : {steps} steps; see the progress bar's total for "
              f"the per-step cost, not {elapsed:.0f}s / {steps}")

    args.out.mkdir(parents=True, exist_ok=True)
    video = frames.permute(0, 2, 3, 1).cpu().numpy()
    path = args.out / f"{args.model_type}_smoke.mp4"
    export_to_video(video, str(path), fps=16)
    image.save(args.out / "first_frame.png")
    print(f"wrote       : {path}")

    del worldstereo
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
