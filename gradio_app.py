#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HY-World 2.0 — local web interface (ROCm / MPS / CUDA / CPU).

Run:
    ./launch.sh      (or)   powershell -File launch.ps1      (or)   python gradio_app.py

Three things, each on its own tab, all rendered with three.js in the browser
(viewer/index.html, served next to the app):

  * Panorama        HY-Pano 2.0 (Qwen-Image-Edit backend): one photo -> a 360°
                    equirectangular panorama, shown from inside a sphere.
  * 3D scene        WorldMirror 2.0: photos or a video -> 3D Gaussian Splatting
                    + point cloud + camera frusta, in a splat viewer.
  * Panorama -> 3D  the panorama is re-projected into a ring of pinhole views
                    (pano3d.py) and reconstructed by WorldMirror, with the exact
                    cameras of those views handed over as priors.
  * Results         every earlier run under outputs/ (from this UI or the CLI
                    scripts), reopened in the viewer and exportable.

Each result tab has an export panel (export3d.py): 3DGS .ply / .splat, point
cloud .ply / .glb / .xyz, a textured relief mesh (.glb / OBJ), three.js JSON,
and the viewer itself as a ready-to-open HTML5 + three.js scene.

The heavy lifting is the upstream code under HY-World-2.0/ running on the
hyworld2.compat layer; this file only wires it to a UI. Console logs stay in
English; the UI is localised (English / 中文 / Русский) like the sibling
Hunyuan3D port.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlencode

# ---- Paths -----------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR / "HY-World-2.0"
PANOGEN_DIR = REPO_DIR / "hyworld2" / "panogen"
VIEWER_DIR = PROJECT_DIR / "viewer"
WEIGHTS_DIR = Path(os.environ.get("HYWORLD_WEIGHTS", PROJECT_DIR / "weights")).resolve()
OUTPUTS_DIR = Path(os.environ.get("HYWORLD_OUTPUT", PROJECT_DIR / "outputs")).resolve()
EXAMPLES_DIR = REPO_DIR / "examples" / "worldrecon"

# The upstream package lives in the subdirectory; the panorama backend is a
# script-style module that imports its siblings (`qwen_image`) by bare name, so
# its directory goes on the path too -- at the END, so nothing there can shadow
# a real package.
sys.path.insert(0, str(REPO_DIR))
sys.path.append(str(PANOGEN_DIR))
# WorldMirror resolves `skyseg.onnx` relative to the working directory.
os.chdir(REPO_DIR)

# hyworld2.compat sets the ROCm/MPS environment (AOTriton SDPA etc.) and must
# be imported before anything touches the GPU.
from hyworld2 import compat  # noqa: E402

import gradio as gr  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(PROJECT_DIR))
import export3d  # noqa: E402
import pano3d  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".webm", ".gif"}
VIEWER_HEIGHT = 620

# Panorama sizes HY-Pano was trained around (2:1, multiples of 16).
PANO_SIZES = {"1952x960": (1952, 960), "1536x768": (1536, 768), "1024x512": (1024, 512)}
# Rows of views for panorama -> 3D: pitch angles in degrees.
PITCH_ROWS = {"1": (0.0,), "3": (20.0, 0.0, -20.0)}

# ==== i18n ===================================================================
# Console output stays in English regardless of the UI language.
LANG_CODES = ("en", "zh", "ru")
LANG_DISPLAY = {"en": "English", "zh": "中文", "ru": "Русский"}
LANG_BY_DISPLAY = {v: k for k, v in LANG_DISPLAY.items()}
DEFAULT_LANG = "en"

I18N: dict[str, dict[str, str]] = {
    "en": {
        "lang_label": "Interface language",
        "description": (
            "# HY-World 2.0 — local interface\n\n"
            "Panorama generation (HY-Pano 2.0), feed-forward 3D reconstruction (WorldMirror 2.0) "
            "and a panorama → 3D path, rendered in the browser with three.js. Every result exports to "
            "PLY / SPLAT / GLB / OBJ / three.js JSON and as a ready-to-open HTML5 + three.js scene.\n\n"
            "**Device:** {device} • **attention:** `{attn}` • **dtype:** `{dtype}`  \n"
            "**Weights:** `{weights}` • **Outputs:** `{outputs}`"
        ),
        "tab_pano": "🌅 Panorama", "tab_recon": "🧊 3D scene", "tab_p3d": "🌐 Panorama → 3D", "tab_system": "⚙️ System",
        "log_label": "Execution log",
        # --- panorama ---
        "pano_input": "Input photo",
        "pano_prompt": "Prompt (what the surroundings look like)",
        "pano_prompt_ph": "e.g. Venice, Grand Canal, sunny afternoon, gondolas",
        "pano_negative": "Extra negative prompt",
        "pano_params": "Generation parameters",
        "pano_seed": "Seed (−1 = random)",
        "pano_steps": "Diffusion steps",
        "pano_guidance": "Guidance scale",
        "pano_true_cfg": "True CFG scale (negative-prompt strength)",
        "pano_size": "Panorama size",
        "size_1952x960": "1952 × 960 (default)", "size_1536x768": "1536 × 768", "size_1024x512": "1024 × 512 (quick test)",
        "pano_blend": "Seam blend width (px)",
        "pano_crop": "Crop input border (fraction)",
        "pano_cpu_offload": "CPU offload (bounded memory; crashes on some ROCm/Windows builds)",
        "pano_btn": "🌅 Generate panorama",
        "pano_result": "Panorama (equirectangular)",
        "pano_viewer_md": "### 360° view — drag to look around, wheel to zoom",
        "pano_download": "Download",
        "pano_help": (
            "Each generation runs in a separate process (see `pano_worker.py`), so the ~54 GB base model is "
            "loaded per run -- about 80 s -- and the memory is fully released afterwards. Budget roughly "
            "**25 minutes** for 1952x960 at 40 steps on this hardware: denoising costs ~38 s per step and "
            "the progress bar now reports that honestly. Halve the steps or drop to 1024x512 to iterate faster."
        ),
        # --- reconstruction ---
        "recon_files": "Photos (several views of one scene) or a video",
        "recon_examples": "…or pick a bundled example",
        "recon_params": "Reconstruction parameters",
        "recon_target": "Working resolution (long side, px)",
        "recon_bf16": "bf16 (faster; keeps critical layers in fp32)",
        "recon_sky": "Remove sky",
        "recon_edge": "Remove edge artefacts",
        "recon_conf": "Confidence mask (drop the 10% least confident points)",
        "recon_max_gs": "Max Gaussians",
        "recon_render": "Render a fly-through video (needs gsplat)",
        "recon_interp": "Interpolated frames between cameras",
        "recon_btn": "🧊 Reconstruct 3D scene",
        "recon_viewer_md": "### 3D scene — Gaussians, points and cameras",
        "recon_files_out": "Result files",
        "recon_video": "Fly-through video",
        "recon_gallery": "Depth and normals",
        "recon_help": (
            "WorldMirror 2.0 is feed-forward: no per-scene optimisation, one pass over all views. "
            "Order the photos as they were taken; 2–32 views work well. The first run of a new "
            "resolution compiles convolution kernels (MIOpen) and is a few minutes slower."
        ),
        # --- panorama -> 3D ---
        "p3d_input": "Panorama (equirectangular, 2:1)",
        "p3d_use_last": "⬅ Use the last generated panorama",
        "p3d_params": "View sampling",
        "p3d_views": "Views around the horizon",
        "p3d_fov": "Field of view per view (°)",
        "p3d_size": "View size (px)",
        "p3d_rows": "Rows of views",
        "rows_1": "1 row (horizon)", "rows_3": "3 rows (±20° pitch)",
        "p3d_prior": "Pass the exact view cameras to WorldMirror as priors",
        "p3d_btn": "🌐 Build 3D scene from the panorama",
        "p3d_gallery": "Generated views",
        "p3d_help": (
            "The panorama is re-projected into pinhole views from its centre and reconstructed like a "
            "photo set. Because every view comes from the same point there is no parallax — depth comes "
            "from what the model has learned, so expect a relief-like scene rather than a walkable one."
        ),
        # --- system ---
        "sys_md": "### Models and memory",
        "sys_mem_btn": "🔄 Refresh memory stats",
        "sys_unload_pano": "Unload HY-Pano 2.0",
        "sys_unload_recon": "Unload WorldMirror 2.0",
        "sys_out": "Status",
        # --- dynamic ---
        # --- progress / status line ---
        # The status line exists to answer one question a scrolling log cannot:
        # is this working or wedged? Hence the CPU figure and the quiet warning.
        "st_idle":   "Idle.",
        "st_cpu":    "CPU {cpu:.0f}%",
        "st_quiet":  ("⚠️ No new output for {quiet}. A non-zero CPU figure above means the process "
                      "is still doing something; zero means it is blocked. Long silences are normal "
                      "while MIOpen compiles kernels or the VAE runs."),
        "st_done":   "✅ Finished in {elapsed}.",
        "st_failed": "❌ Failed after {elapsed}.",
        "stage_starting":   "Starting…",
        "stage_load_base":  "Loading the base model (~54 GB, first run only)",
        "stage_lora":       "Loading the HY-Pano LoRA",
        "stage_load_recon": "Loading WorldMirror 2.0",
        "stage_ready":      "Model ready",
        "stage_text":       "Encoding the prompt",
        "stage_vae_encode": "Encoding the input image (VAE)",
        "stage_vae_decode": "Decoding the result (VAE)",
        "stage_denoise":    "Denoising",
        "stage_infer":      "Reconstructing geometry",
        "stage_postproc":   "Masks and filtering",
        "stage_save":       "Writing results",
        "stage_views":      "Slicing the panorama into views",
        "err_no_image": "⚠️ Upload an input image first.",
        "err_no_files": "⚠️ Upload photos / a video, or pick an example.",
        "err_failed": "❌ Failed: {e}",
        "done": "✅ Done in {elapsed:.1f} s",
        "log_views_ready": "Wrote {n} perspective views; reconstructing…",
        "unloaded": "Unloaded {name}.",
        "mem_no_gpu": "No GPU device — nothing to report.",
        "mem_report_mps": (
            "Metal allocated: {alloc:.2f} GiB • held by the driver: {reserved:.2f} GiB • "
            "unified memory available to the GPU: {total:.2f} GiB (of {ram:.0f} GB installed)\n"
            "HY-Pano 2.0 loaded: {pano} • WorldMirror 2.0 loaded: {recon}"
        ),
        "mem_report": (
            "GPU allocated: {alloc:.2f} GiB • reserved: {reserved:.2f} GiB • free: {free:.2f} / {total:.2f} GiB\n"
            "HY-Pano 2.0 loaded: {pano} • WorldMirror 2.0 loaded: {recon}"
        ),
        # --- export ---
        "export_acc": "⬇️ Export the result",
        "export_fmt": "Format",
        "export_detail": "Mesh detail",
        "detail_1": "Full (a vertex per pixel, large files)", "detail_2": "Half (recommended)", "detail_4": "Quarter (small files)",
        "export_btn": "⬇️ Export",
        "export_file": "Exported file",
        "export_help_scene": (
            "**What the formats are for.** `gaussians.ply` — 3D Gaussian Splatting exactly as produced • `scene.splat` — "
            "the same Gaussians in the compact layout SuperSplat, the Unity/Unreal plugins and most web viewers read • "
            "`points.ply` / `points.glb` / `points.xyz` — the point cloud for three.js, Blender, CloudCompare, MeshLab • "
            "`mesh.glb` / OBJ — a textured relief mesh built from the predicted depth maps, so the scene opens anywhere a "
            "mesh does • `scene.three.json` — three.js JSON Object format for `THREE.ObjectLoader` • **HTML5 scene** — the "
            "viewer above: as a folder with a one-click local server (every layer, Gaussians included) or as one HTML file "
            "that opens straight from disk (mesh, points, cameras; Gaussians appear once it is served over http).\n\n"
            "glTF, OBJ and three.js files are y-up; PLY and SPLAT keep WorldMirror's camera frame (y down, z forward)."
        ),
        "export_help_pano": (
            "**What the formats are for.** PNG / JPEG — the equirectangular image • **cube map** — six faces in three.js "
            "`CubeTextureLoader` order, for skyboxes • `panorama.glb` — an inward-facing textured sphere for any glTF viewer • "
            "**HTML5 viewer** — the 360° view above, as one file or as a folder with a local server."
        ),
        "err_no_result": "⚠️ Nothing to export yet — generate something first, or open an earlier run on the Results tab.",
        "export_done": "✅ Exported {name} ({size}) in {elapsed}.",
        "stage_export": "Exporting",
        "fmt_gaussians_ply": "gaussians.ply — 3D Gaussian Splatting, as produced",
        "fmt_splat": "scene.splat — 3D Gaussian Splatting, compact (SuperSplat, Unity/Unreal, web viewers)",
        "fmt_points_ply": "points.ply — point cloud, as produced",
        "fmt_points_glb": "points.glb — point cloud, glTF 2.0 (three.js, Blender, Godot)",
        "fmt_points_xyz": "points.xyz — point cloud, ASCII x y z r g b (CloudCompare, MeshLab)",
        "fmt_mesh_glb": "mesh.glb — textured mesh from the depth maps, glTF 2.0",
        "fmt_mesh_obj": "mesh_obj.zip — textured mesh, OBJ + MTL + textures",
        "fmt_three_json": "scene.three.json — three.js JSON Object format (points + cameras)",
        "fmt_cameras": "camera_params.json — intrinsics + camera-to-world matrices",
        "fmt_web_zip": "scene_web.zip — HTML5 + three.js scene, folder with a one-click local server",
        "fmt_web_html": "scene.html — HTML5 + three.js scene, single self-contained file",
        "fmt_everything": "scene_all.zip — everything above in one archive",
        "pfmt_png": "panorama.png — equirectangular, as produced",
        "pfmt_jpg": "panorama.jpg — equirectangular JPEG (smaller)",
        "pfmt_cubemap": "cubemap.zip — cube map, 6 faces (three.js CubeTextureLoader order)",
        "pfmt_sphere_glb": "panorama.glb — textured sphere, glTF 2.0 (any glTF viewer)",
        "pfmt_web_html": "panorama.html — 360° HTML5 + three.js viewer, single file",
        "pfmt_web_zip": "panorama_web.zip — 360° HTML5 + three.js viewer, folder with a local server",
        # --- results tab ---
        "tab_results": "📁 Results",
        "results_md": (
            "### Earlier runs\n"
            "Everything under the outputs directory, whether it came from this interface or from the command-line "
            "scripts. Open a run to view it again and export it in any format."
        ),
        "results_scenes": "3D scenes (newest first)",
        "results_panos": "Panoramas (newest first)",
        "results_refresh": "🔄 Refresh",
        "results_open": "Open",
        "results_empty": "No runs found yet.",
    },
    "zh": {
        "lang_label": "界面语言",
        "description": (
            "# HY-World 2.0 — 本地界面\n\n"
            "全景生成（HY-Pano 2.0）、前馈式三维重建（WorldMirror 2.0）以及全景 → 三维流程，"
            "结果在浏览器中用 three.js 渲染。所有结果都可导出为 PLY / SPLAT / GLB / OBJ / three.js JSON，"
            "以及可直接打开的 HTML5 + three.js 场景。\n\n"
            "**设备：** {device} • **注意力：** `{attn}` • **精度：** `{dtype}`  \n"
            "**权重：** `{weights}` • **输出：** `{outputs}`"
        ),
        "tab_pano": "🌅 全景图", "tab_recon": "🧊 三维场景", "tab_p3d": "🌐 全景 → 三维", "tab_system": "⚙️ 系统",
        "log_label": "运行日志",
        "pano_input": "输入照片",
        "pano_prompt": "提示词（描述周围环境）",
        "pano_prompt_ph": "例如：威尼斯，大运河，晴朗的午后，贡多拉",
        "pano_negative": "附加负面提示词",
        "pano_params": "生成参数",
        "pano_seed": "随机种子（−1 = 随机）",
        "pano_steps": "扩散步数",
        "pano_guidance": "引导系数",
        "pano_true_cfg": "True CFG 系数（负面提示强度）",
        "pano_size": "全景图尺寸",
        "size_1952x960": "1952 × 960（默认）", "size_1536x768": "1536 × 768", "size_1024x512": "1024 × 512（快速测试）",
        "pano_blend": "接缝融合宽度（像素）",
        "pano_crop": "裁掉输入边缘（比例）",
        "pano_cpu_offload": "CPU 卸载（内存受限；部分 ROCm/Windows 构建会崩溃）",
        "pano_btn": "🌅 生成全景图",
        "pano_result": "全景图（等距柱状投影）",
        "pano_viewer_md": "### 360° 视图 — 拖动环视，滚轮缩放",
        "pano_download": "下载",
        "pano_help": (
            "每次生成都在独立进程中运行（见 `pano_worker.py`），因此约 54 GB 的基础模型每次都要加载（约 80 秒），"
            "结束后内存会完全释放。在本机上，1952×960、40 步大约需要 **25 分钟**：每步去噪约 38 秒，"
            "进度条现在如实反映这一点。想快速迭代，可减半步数或改用 1024×512。"
        ),
        "recon_files": "照片（同一场景的多个视角）或视频",
        "recon_examples": "…或选择内置示例",
        "recon_params": "重建参数",
        "recon_target": "工作分辨率（长边，像素）",
        "recon_bf16": "bf16（更快；关键层保持 fp32）",
        "recon_sky": "去除天空",
        "recon_edge": "去除边缘伪影",
        "recon_conf": "置信度掩码（丢弃置信度最低的 10% 点）",
        "recon_max_gs": "最大高斯数量",
        "recon_render": "渲染漫游视频（需要 gsplat）",
        "recon_interp": "相机之间的插值帧数",
        "recon_btn": "🧊 重建三维场景",
        "recon_viewer_md": "### 三维场景 — 高斯、点云与相机",
        "recon_files_out": "结果文件",
        "recon_video": "漫游视频",
        "recon_gallery": "深度与法线",
        "recon_help": (
            "WorldMirror 2.0 是前馈模型：无需逐场景优化，一次前向处理所有视角。"
            "请按拍摄顺序排列照片；2–32 个视角效果最好。新分辨率的首次运行需要编译卷积核（MIOpen），会慢几分钟。"
        ),
        "p3d_input": "全景图（等距柱状投影，2:1）",
        "p3d_use_last": "⬅ 使用上次生成的全景图",
        "p3d_params": "视角采样",
        "p3d_views": "水平方向视角数量",
        "p3d_fov": "每个视角的视场角（°）",
        "p3d_size": "视角图像尺寸（像素）",
        "p3d_rows": "视角行数",
        "rows_1": "1 行（地平线）", "rows_3": "3 行（俯仰 ±20°）",
        "p3d_prior": "将各视角的精确相机参数作为先验传给 WorldMirror",
        "p3d_btn": "🌐 由全景图构建三维场景",
        "p3d_gallery": "生成的视角",
        "p3d_help": (
            "全景图从其中心重投影为针孔视角，然后像照片集一样重建。由于所有视角来自同一点，没有视差——"
            "深度完全来自模型的先验知识，因此结果更像浮雕而非可行走的场景。"
        ),
        "sys_md": "### 模型与内存",
        "sys_mem_btn": "🔄 刷新内存统计",
        "sys_unload_pano": "卸载 HY-Pano 2.0",
        "sys_unload_recon": "卸载 WorldMirror 2.0",
        "sys_out": "状态",
        # --- progress / status line ---
        "st_idle":   "空闲。",
        "st_cpu":    "CPU {cpu:.0f}%",
        "st_quiet":  ("⚠️ 已有 {quiet} 没有新输出。上面的 CPU 占用不为零说明进程仍在做事，为零则说明它被阻塞。"
                      "MIOpen 编译内核或 VAE 运行时，长时间没有输出是正常的。"),
        "st_done":   "✅ 完成，用时 {elapsed}。",
        "st_failed": "❌ 失败，耗时 {elapsed}。",
        "stage_starting":   "正在启动…",
        "stage_load_base":  "加载基础模型（约 54 GB，仅首次）",
        "stage_lora":       "加载 HY-Pano LoRA",
        "stage_load_recon": "加载 WorldMirror 2.0",
        "stage_ready":      "模型就绪",
        "stage_text":       "编码提示词",
        "stage_vae_encode": "编码输入图像（VAE）",
        "stage_vae_decode": "解码结果（VAE）",
        "stage_denoise":    "去噪中",
        "stage_infer":      "重建几何",
        "stage_postproc":   "掩码与过滤",
        "stage_save":       "写入结果",
        "stage_views":      "将全景图切分为视角",
        "err_no_image": "⚠️ 请先上传输入图像。",
        "err_no_files": "⚠️ 请上传照片/视频，或选择一个示例。",
        "err_failed": "❌ 失败：{e}",
        "done": "✅ 完成，用时 {elapsed:.1f} 秒",
        "log_views_ready": "已生成 {n} 个透视视角；正在重建…",
        "unloaded": "已卸载 {name}。",
        "mem_no_gpu": "没有 GPU 设备——无可报告内容。",
        "mem_report_mps": (
            "Metal 已分配：{alloc:.2f} GiB • 驱动占用：{reserved:.2f} GiB • "
            "GPU 可用的统一内存：{total:.2f} GiB（共安装 {ram:.0f} GB）\n"
            "HY-Pano 2.0 已加载：{pano} • WorldMirror 2.0 已加载：{recon}"
        ),
        "mem_report": (
            "GPU 已分配：{alloc:.2f} GiB • 已保留：{reserved:.2f} GiB • 可用：{free:.2f} / {total:.2f} GiB\n"
            "HY-Pano 2.0 已加载：{pano} • WorldMirror 2.0 已加载：{recon}"
        ),
        # --- export ---
        "export_acc": "⬇️ 导出结果",
        "export_fmt": "格式",
        "export_detail": "网格精度",
        "detail_1": "完整（每像素一个顶点，文件很大）", "detail_2": "一半（推荐）", "detail_4": "四分之一（文件小）",
        "export_btn": "⬇️ 导出",
        "export_file": "导出的文件",
        "export_help_scene": (
            "**各格式用途。** `gaussians.ply` — 原始输出的 3D 高斯泼溅 • `scene.splat` — 同样的高斯，紧凑格式，"
            "SuperSplat、Unity/Unreal 插件和多数网页查看器可读 • `points.ply` / `points.glb` / `points.xyz` — 点云，"
            "用于 three.js、Blender、CloudCompare、MeshLab • `mesh.glb` / OBJ — 由预测深度图构建的带纹理浮雕网格，"
            "任何支持网格的软件都能打开 • `scene.three.json` — three.js JSON Object 格式（`THREE.ObjectLoader`） • "
            "**HTML5 场景** — 上方的查看器：一个带一键本地服务器的文件夹（所有图层，含高斯），或一个可直接从磁盘打开的"
            "单一 HTML 文件（网格、点云、相机；通过 http 打开时也显示高斯）。\n\n"
            "glTF、OBJ 与 three.js 文件为 y 轴向上；PLY 与 SPLAT 保持 WorldMirror 的相机坐标系（y 向下，z 向前）。"
        ),
        "export_help_pano": (
            "**各格式用途。** PNG / JPEG — 等距柱状图 • **立方体贴图** — 六个面，按 three.js `CubeTextureLoader` 顺序，"
            "用于天空盒 • `panorama.glb` — 朝内的带纹理球体，任何 glTF 查看器可打开 • **HTML5 查看器** — 上方的 360° 视图，"
            "单文件或带本地服务器的文件夹。"
        ),
        "err_no_result": "⚠️ 还没有可导出的内容——请先生成，或在“结果”标签页打开之前的运行。",
        "export_done": "✅ 已导出 {name}（{size}），用时 {elapsed}。",
        "stage_export": "导出中",
        "fmt_gaussians_ply": "gaussians.ply — 3D 高斯泼溅，原始输出",
        "fmt_splat": "scene.splat — 3D 高斯泼溅，紧凑格式（SuperSplat、Unity/Unreal、网页查看器）",
        "fmt_points_ply": "points.ply — 点云，原始输出",
        "fmt_points_glb": "points.glb — 点云，glTF 2.0（three.js、Blender、Godot）",
        "fmt_points_xyz": "points.xyz — 点云，ASCII x y z r g b（CloudCompare、MeshLab）",
        "fmt_mesh_glb": "mesh.glb — 由深度图构建的带纹理网格，glTF 2.0",
        "fmt_mesh_obj": "mesh_obj.zip — 带纹理网格，OBJ + MTL + 贴图",
        "fmt_three_json": "scene.three.json — three.js JSON Object 格式（点云 + 相机）",
        "fmt_cameras": "camera_params.json — 内参 + 相机到世界矩阵",
        "fmt_web_zip": "scene_web.zip — HTML5 + three.js 场景，带一键本地服务器的文件夹",
        "fmt_web_html": "scene.html — HTML5 + three.js 场景，单一自包含文件",
        "fmt_everything": "scene_all.zip — 以上全部打包",
        "pfmt_png": "panorama.png — 等距柱状图，原始输出",
        "pfmt_jpg": "panorama.jpg — 等距柱状图 JPEG（更小）",
        "pfmt_cubemap": "cubemap.zip — 立方体贴图，6 个面（three.js CubeTextureLoader 顺序）",
        "pfmt_sphere_glb": "panorama.glb — 带纹理球体，glTF 2.0（任何 glTF 查看器）",
        "pfmt_web_html": "panorama.html — 360° HTML5 + three.js 查看器，单文件",
        "pfmt_web_zip": "panorama_web.zip — 360° HTML5 + three.js 查看器，带本地服务器的文件夹",
        # --- results tab ---
        "tab_results": "📁 结果",
        "results_md": (
            "### 之前的运行\n"
            "输出目录下的全部内容，无论来自本界面还是命令行脚本。打开一次运行即可重新查看并导出为任意格式。"
        ),
        "results_scenes": "三维场景（最新在前）",
        "results_panos": "全景图（最新在前）",
        "results_refresh": "🔄 刷新",
        "results_open": "打开",
        "results_empty": "尚未找到任何运行。",
    },
    "ru": {
        "lang_label": "Язык интерфейса",
        "description": (
            "# HY-World 2.0 — локальный интерфейс\n\n"
            "Генерация панорам (HY-Pano 2.0), прямая 3D-реконструкция (WorldMirror 2.0) "
            "и путь панорама → 3D; результат показывается в браузере через three.js. Любой результат "
            "экспортируется в PLY / SPLAT / GLB / OBJ / three.js JSON и в готовую к открытию HTML5 + three.js сцену.\n\n"
            "**Устройство:** {device} • **attention:** `{attn}` • **dtype:** `{dtype}`  \n"
            "**Веса:** `{weights}` • **Результаты:** `{outputs}`"
        ),
        "tab_pano": "🌅 Панорама", "tab_recon": "🧊 3D-сцена", "tab_p3d": "🌐 Панорама → 3D", "tab_system": "⚙️ Система",
        "log_label": "Журнал выполнения",
        "pano_input": "Исходное фото",
        "pano_prompt": "Промпт (что вокруг)",
        "pano_prompt_ph": "например: Венеция, Гранд-канал, солнечный день, гондолы",
        "pano_negative": "Дополнительный негативный промпт",
        "pano_params": "Параметры генерации",
        "pano_seed": "Seed (−1 = случайный)",
        "pano_steps": "Шагов диффузии",
        "pano_guidance": "Guidance scale",
        "pano_true_cfg": "True CFG (сила негативного промпта)",
        "pano_size": "Размер панорамы",
        "size_1952x960": "1952 × 960 (по умолчанию)", "size_1536x768": "1536 × 768", "size_1024x512": "1024 × 512 (быстрый тест)",
        "pano_blend": "Ширина сшивки шва (px)",
        "pano_crop": "Обрезать край входа (доля)",
        "pano_cpu_offload": "CPU offload (ограниченная память; на некоторых сборках ROCm/Windows падает)",
        "pano_btn": "🌅 Сгенерировать панораму",
        "pano_result": "Панорама (равнопромежуточная проекция)",
        "pano_viewer_md": "### Обзор 360° — тяните мышью, колесо — зум",
        "pano_download": "Скачать",
        "pano_help": (
            "Каждая генерация идёт в отдельном процессе (см. `pano_worker.py`), поэтому базовая модель (~54 ГБ) "
            "загружается заново — около 80 с, — зато память полностью освобождается после. Закладывайте примерно "
            "**25 минут** на 1952×960 при 40 шагах: один шаг стоит ~38 с, и полоса прогресса теперь показывает это "
            "честно. Чтобы быстрее пробовать варианты, уменьшите число шагов вдвое или возьмите 1024×512."
        ),
        "recon_files": "Фотографии (несколько ракурсов одной сцены) или видео",
        "recon_examples": "…или выберите встроенный пример",
        "recon_params": "Параметры реконструкции",
        "recon_target": "Рабочее разрешение (длинная сторона, px)",
        "recon_bf16": "bf16 (быстрее; критичные слои остаются в fp32)",
        "recon_sky": "Убрать небо",
        "recon_edge": "Убрать артефакты на границах",
        "recon_conf": "Маска уверенности (отбросить 10 % наименее уверенных точек)",
        "recon_max_gs": "Макс. число гауссиан",
        "recon_render": "Отрендерить видео-пролёт (нужен gsplat)",
        "recon_interp": "Кадров интерполяции между камерами",
        "recon_btn": "🧊 Реконструировать 3D-сцену",
        "recon_viewer_md": "### 3D-сцена — гауссианы, точки и камеры",
        "recon_files_out": "Файлы результата",
        "recon_video": "Видео-пролёт",
        "recon_gallery": "Глубина и нормали",
        "recon_help": (
            "WorldMirror 2.0 работает без оптимизации по сцене: один проход по всем ракурсам. "
            "Располагайте фото в порядке съёмки; хорошо работает 2–32 ракурса. Первый запуск на новом "
            "разрешении компилирует свёрточные ядра (MIOpen) и идёт на несколько минут дольше."
        ),
        "p3d_input": "Панорама (равнопромежуточная, 2:1)",
        "p3d_use_last": "⬅ Взять последнюю сгенерированную панораму",
        "p3d_params": "Нарезка ракурсов",
        "p3d_views": "Ракурсов по горизонту",
        "p3d_fov": "Угол обзора ракурса (°)",
        "p3d_size": "Размер ракурса (px)",
        "p3d_rows": "Рядов ракурсов",
        "rows_1": "1 ряд (горизонт)", "rows_3": "3 ряда (наклон ±20°)",
        "p3d_prior": "Передать WorldMirror точные камеры ракурсов как приор",
        "p3d_btn": "🌐 Собрать 3D-сцену из панорамы",
        "p3d_gallery": "Полученные ракурсы",
        "p3d_help": (
            "Панорама перепроецируется из своего центра в набор пинхол-ракурсов и реконструируется как "
            "набор фотографий. Все ракурсы сняты из одной точки, параллакса нет — глубина берётся из "
            "того, что модель выучила, поэтому ждите рельефную сцену, а не такую, по которой можно ходить."
        ),
        "sys_md": "### Модели и память",
        "sys_mem_btn": "🔄 Обновить статистику памяти",
        "sys_unload_pano": "Выгрузить HY-Pano 2.0",
        "sys_unload_recon": "Выгрузить WorldMirror 2.0",
        "sys_out": "Состояние",
        # --- progress / status line ---
        "st_idle":   "Простой.",
        "st_cpu":    "CPU {cpu:.0f}%",
        "st_quiet":  ("⚠️ Нового вывода нет уже {quiet}. Ненулевой CPU выше означает, что процесс "
                      "всё ещё что-то делает; ноль — что он заблокирован. Долгие паузы нормальны, "
                      "пока MIOpen компилирует ядра или работает VAE."),
        "st_done":   "✅ Готово за {elapsed}.",
        "st_failed": "❌ Ошибка через {elapsed}.",
        "stage_starting":   "Запуск…",
        "stage_load_base":  "Загрузка базовой модели (~54 ГБ, только первый раз)",
        "stage_lora":       "Загрузка LoRA HY-Pano",
        "stage_load_recon": "Загрузка WorldMirror 2.0",
        "stage_ready":      "Модель готова",
        "stage_text":       "Кодирование промпта",
        "stage_vae_encode": "Кодирование входного изображения (VAE)",
        "stage_vae_decode": "Декодирование результата (VAE)",
        "stage_denoise":    "Шумоподавление",
        "stage_infer":      "Реконструкция геометрии",
        "stage_postproc":   "Маски и фильтрация",
        "stage_save":       "Запись результатов",
        "stage_views":      "Нарезка панорамы на ракурсы",
        "err_no_image": "⚠️ Сначала загрузите исходное изображение.",
        "err_no_files": "⚠️ Загрузите фото / видео или выберите пример.",
        "err_failed": "❌ Ошибка: {e}",
        "done": "✅ Готово за {elapsed:.1f} с",
        "log_views_ready": "Записано ракурсов: {n}; реконструкция…",
        "unloaded": "Выгружено: {name}.",
        "mem_no_gpu": "GPU не обнаружен — отчитываться не о чем.",
        "mem_report_mps": (
            "Metal занято: {alloc:.2f} GiB • удерживает драйвер: {reserved:.2f} GiB • "
            "единой памяти доступно GPU: {total:.2f} GiB (из {ram:.0f} GB установленных)\n"
            "HY-Pano 2.0 загружена: {pano} • WorldMirror 2.0 загружена: {recon}"
        ),
        "mem_report": (
            "GPU занято: {alloc:.2f} GiB • зарезервировано: {reserved:.2f} GiB • свободно: {free:.2f} / {total:.2f} GiB\n"
            "HY-Pano 2.0 загружена: {pano} • WorldMirror 2.0 загружена: {recon}"
        ),
        # --- export ---
        "export_acc": "⬇️ Экспорт результата",
        "export_fmt": "Формат",
        "export_detail": "Детализация меша",
        "detail_1": "Полная (вершина на каждый пиксель, большие файлы)", "detail_2": "Половина (рекомендуется)",
        "detail_4": "Четверть (маленькие файлы)",
        "export_btn": "⬇️ Экспортировать",
        "export_file": "Файл экспорта",
        "export_help_scene": (
            "**Для чего какой формат.** `gaussians.ply` — 3D Gaussian Splatting как есть • `scene.splat` — те же гауссианы "
            "в компактной раскладке, которую читают SuperSplat, плагины Unity/Unreal и большинство веб-просмотрщиков • "
            "`points.ply` / `points.glb` / `points.xyz` — облако точек для three.js, Blender, CloudCompare, MeshLab • "
            "`mesh.glb` / OBJ — текстурированный рельефный меш, собранный из предсказанных карт глубины: открывается везде, "
            "где открываются меши • `scene.three.json` — формат three.js JSON Object для `THREE.ObjectLoader` • "
            "**HTML5-сцена** — просмотрщик, что выше: папка с локальным сервером в один клик (все слои, включая гауссианы) "
            "или один HTML-файл, который открывается прямо с диска (меш, точки, камеры; гауссианы появятся, если отдать "
            "его по http).\n\n"
            "glTF, OBJ и three.js — с осью Y вверх; PLY и SPLAT сохраняют систему камер WorldMirror (Y вниз, Z вперёд)."
        ),
        "export_help_pano": (
            "**Для чего какой формат.** PNG / JPEG — равнопромежуточное изображение • **кубическая карта** — шесть граней "
            "в порядке three.js `CubeTextureLoader`, для скайбоксов • `panorama.glb` — текстурированная сфера, вывернутая "
            "внутрь, для любого glTF-просмотрщика • **HTML5-просмотрщик** — обзор 360° как выше, одним файлом или папкой "
            "с локальным сервером."
        ),
        "err_no_result": "⚠️ Экспортировать пока нечего — сначала сгенерируйте что-нибудь или откройте прошлый запуск на вкладке «Результаты».",
        "export_done": "✅ Экспортировано: {name} ({size}) за {elapsed}.",
        "stage_export": "Экспорт",
        "fmt_gaussians_ply": "gaussians.ply — 3D Gaussian Splatting как есть",
        "fmt_splat": "scene.splat — 3D Gaussian Splatting, компактный (SuperSplat, Unity/Unreal, веб-просмотрщики)",
        "fmt_points_ply": "points.ply — облако точек как есть",
        "fmt_points_glb": "points.glb — облако точек, glTF 2.0 (three.js, Blender, Godot)",
        "fmt_points_xyz": "points.xyz — облако точек, текст x y z r g b (CloudCompare, MeshLab)",
        "fmt_mesh_glb": "mesh.glb — текстурированный меш из карт глубины, glTF 2.0",
        "fmt_mesh_obj": "mesh_obj.zip — текстурированный меш, OBJ + MTL + текстуры",
        "fmt_three_json": "scene.three.json — формат three.js JSON Object (точки + камеры)",
        "fmt_cameras": "camera_params.json — интринсики + матрицы камера→мир",
        "fmt_web_zip": "scene_web.zip — HTML5 + three.js сцена, папка с локальным сервером в один клик",
        "fmt_web_html": "scene.html — HTML5 + three.js сцена, один самодостаточный файл",
        "fmt_everything": "scene_all.zip — всё перечисленное одним архивом",
        "pfmt_png": "panorama.png — равнопромежуточная, как есть",
        "pfmt_jpg": "panorama.jpg — равнопромежуточная JPEG (меньше)",
        "pfmt_cubemap": "cubemap.zip — кубическая карта, 6 граней (порядок three.js CubeTextureLoader)",
        "pfmt_sphere_glb": "panorama.glb — текстурированная сфера, glTF 2.0 (любой glTF-просмотрщик)",
        "pfmt_web_html": "panorama.html — обзор 360° на HTML5 + three.js, один файл",
        "pfmt_web_zip": "panorama_web.zip — обзор 360° на HTML5 + three.js, папка с локальным сервером",
        # --- results tab ---
        "tab_results": "📁 Результаты",
        "results_md": (
            "### Прошлые запуски\n"
            "Всё, что лежит в каталоге результатов — из этого интерфейса или из скриптов командной строки. "
            "Откройте запуск, чтобы снова посмотреть его и экспортировать в любой формат."
        ),
        "results_scenes": "3D-сцены (новые сверху)",
        "results_panos": "Панорамы (новые сверху)",
        "results_refresh": "🔄 Обновить",
        "results_open": "Открыть",
        "results_empty": "Запусков пока нет.",
    },
}


def t(lang: str, key: str, **kw) -> str:
    """Translate a key into the target language, falling back to English."""
    text = I18N.get(lang, {}).get(key)
    if text is None:
        text = I18N["en"].get(key, key)
    try:
        return text.format(**kw) if kw else text
    except Exception:
        return text


class _L10n:
    """Registry of (component, attribute -> i18n key) pairs, so the language
    switch is one loop rather than a hand-maintained tuple."""

    def __init__(self) -> None:
        self.items: list[tuple[object, dict[str, str], Optional[Callable[[str], dict]]]] = []

    def bind(self, comp, extra: Optional[Callable[[str], dict]] = None, **keys: str):
        self.items.append((comp, keys, extra))
        return comp

    def components(self) -> list:
        return [c for c, _, _ in self.items]

    def updates(self, lang: str) -> list:
        out = []
        for _, keys, extra in self.items:
            kw = {attr: t(lang, key) for attr, key in keys.items()}
            if extra is not None:
                kw.update(extra(lang))
            out.append(gr.update(**kw))
        return out


# ==== Live log capture =======================================================
class _LogBuffer:
    """Line buffer that understands carriage returns, so tqdm progress bars
    overwrite their line instead of piling up.

    It also timestamps the last write. A scrolling log cannot distinguish "busy
    and quiet" from "wedged", and that distinction is the whole point of the
    status line above it, so the time since the last byte of output is tracked
    here rather than guessed at.
    """

    def __init__(self) -> None:
        self._lines = [""]
        self._lock = threading.Lock()
        self.last_write = time.monotonic()

    def write(self, s: str) -> None:
        with self._lock:
            for part in re.split(r"(\r\n|\r|\n)", s):
                if part in ("\n", "\r\n"):
                    self._lines.append("")
                elif part == "\r":
                    self._lines[-1] = ""
                elif part:
                    self._lines[-1] += part
            if len(self._lines) > 2000:
                del self._lines[:-2000]
            self.last_write = time.monotonic()

    def text(self, max_lines: int = 300) -> str:
        with self._lock:
            return "\n".join(self._lines[-max_lines:])

    def tail(self, n: int = 12) -> list[str]:
        """The most recent lines, including the one still being written."""
        with self._lock:
            return [ln for ln in self._lines[-n:] if ln]

    def silent_for(self) -> float:
        return time.monotonic() - self.last_write


class _Tee(io.TextIOBase):
    def __init__(self, orig, buf: _LogBuffer) -> None:
        self._orig, self._buf = orig, buf

    def write(self, s: str) -> int:
        try:
            self._orig.write(s)
        except Exception:
            pass
        self._buf.write(s)
        return len(s)

    def flush(self) -> None:
        try:
            self._orig.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self._orig, "encoding", None) or "utf-8"


@contextlib.contextmanager
def _capture(buf: _LogBuffer):
    # Process-global by nature (sys.stdout is), which is fine for a single-user
    # local app; jobs are serialised by the queue anyway.
    out, err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(out, buf), _Tee(err, buf)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = out, err


# ==== Progress reporting =====================================================
# Only the running job knows where it is, and it says so on stdout, so the
# stage is read back out of the captured console. Each entry maps a line the
# pipeline prints to an i18n key; the last match wins.
_STAGE_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\[Init\] Loading base model|Loading checkpoint shards"), "stage_load_base"),
    (re.compile(r"\[Init\] Loading LoRA"), "stage_lora"),
    (re.compile(r"\[Pipeline\] Single-GPU|\[Init\] Found local model"), "stage_load_recon"),
    (re.compile(r"\[Init\] Model ready|\[Init\] LoRA weights loaded"), "stage_ready"),
    (re.compile(r"\[Stage\] text"), "stage_text"),
    (re.compile(r"\[Stage\] vae-encode"), "stage_vae_encode"),
    (re.compile(r"\[Stage\] vae-decode"), "stage_vae_decode"),
    (re.compile(r"^\s*\d+%\|.*(it/s|s/it)"), "stage_denoise"),
    (re.compile(r"\[Input\] |\[Inference\] \d+ images"), "stage_infer"),
    (re.compile(r"\[Inference\] Done"), "stage_postproc"),
    (re.compile(r"\[Save\]|Results saved"), "stage_save"),
    (re.compile(r"\[Export\] "), "stage_export"),
)

# A stage quiet for longer than this is called out. Not an error: MIOpen kernel
# compilation and the VAE legitimately go minutes without printing.
_QUIET_WARN_S = 90.0

# ASCII on purpose: this string can end up on a Windows console running a
# legacy code page (cp1251 for a Russian locale), where a Braille spinner
# raises UnicodeEncodeError and takes the status line with it.
_SPINNER = ("|", "/", "-", "\\")


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}" if m else f"{s} s"


class _StageTracker:
    """Follows the job's stage and how long it has been there."""

    def __init__(self) -> None:
        self.key = "stage_starting"
        self.since = time.monotonic()
        self.detail = ""

    def update(self, buf: _LogBuffer) -> None:
        for line in buf.tail():
            for pattern, key in _STAGE_PATTERNS:
                if pattern.search(line):
                    if key != self.key:
                        self.key, self.since = key, time.monotonic()
                    # tqdm's own "16/40 [00:08<00:12, 1.9it/s]" beats anything
                    # this module could invent, so it is shown verbatim.
                    self.detail = line.strip() if key == "stage_denoise" else ""

    def elapsed(self) -> float:
        return time.monotonic() - self.since


def _cpu_line(proc, lang: str) -> str:
    """Whether the process is computing, which is what "stuck?" really asks.

    A wedged run and a slow one look identical in a log; they do not look
    identical here -- a process that is computing accrues CPU time, one that
    is blocked does not.

    Deliberately *not* paired with a GPU utilisation figure: Windows' WDDM
    counters do not see ROCm compute on this iGPU at all. Measured against a
    continuous 8192-cube bf16 matmul, the GPU Engine utilisation counter
    reads 2.5%, so quoting it would invite exactly the wrong conclusion.
    """
    try:
        cpu = proc.cpu_percent()          # since the previous call, i.e. this poll
    except Exception:
        return ""
    return t(lang, "st_cpu", cpu=cpu)


def _status_md(tracker: _StageTracker, buf: _LogBuffer, tick: int, proc, lang: str) -> str:
    spin = _SPINNER[tick % len(_SPINNER)]
    parts = [f"{spin} **{t(lang, tracker.key)}** · {_fmt_duration(tracker.elapsed())}"]
    if tracker.detail:
        parts.append(f"`{tracker.detail}`")
    cpu = _cpu_line(proc, lang)
    if cpu:
        parts.append(cpu)
    line = " · ".join(parts)

    quiet = buf.silent_for()
    if quiet > _QUIET_WARN_S:
        line += "\n\n" + t(lang, "st_quiet", quiet=_fmt_duration(quiet))
    return line


def _run_job(fn: Callable[[], object], lang: str = DEFAULT_LANG):
    """Run ``fn`` in a thread and yield ``(status, log, result, error)`` while it
    works, so a Gradio generator can stream both to the page."""
    buf = _LogBuffer()
    tracker = _StageTracker()
    outcome: dict = {}
    proc = psutil.Process()
    proc.cpu_percent()   # prime the counter; the first reading is meaningless

    def worker():
        with _capture(buf):
            try:
                outcome["value"] = fn()
            except BaseException as e:  # noqa: BLE001 - reported to the UI
                outcome["error"] = e
                traceback.print_exc()

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    tick = 0
    started = time.monotonic()
    while th.is_alive():
        th.join(timeout=0.5)
        tracker.update(buf)
        tick += 1
        yield _status_md(tracker, buf, tick, proc, lang), buf.text(), None, None

    total = _fmt_duration(time.monotonic() - started)
    if "error" in outcome:
        yield t(lang, "st_failed", elapsed=total), buf.text(), None, outcome["error"]
    else:
        yield t(lang, "st_done", elapsed=total), buf.text(), outcome.get("value"), None


# ==== Models =================================================================
_pano_pipe = None
_recon_pipe = None
_recon_bf16: Optional[bool] = None
_gpu_lock = threading.Lock()


def _announce_stages(pipe) -> None:
    """Make the panorama pipeline say which stage it is in.

    Between "Start generating panorama" and the first denoising step the
    pipeline prints nothing at all, yet that gap contains the text encoder and
    the VAE encode -- and the VAE encode is exactly where a bad convolution
    shape used to sit for a quarter of an hour. Wrapping the three components
    turns that silence into stage markers `_STAGE_PATTERNS` can pick up.
    """
    if getattr(pipe, "_hyworld_announced", False):
        return

    def wrap(owner, name, tag):
        original = getattr(owner, name)

        def announced(*args, **kwargs):
            print(f"[Stage] {tag}", flush=True)
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                print(f"[Stage] {tag} done in {time.perf_counter() - t0:.1f}s", flush=True)

        setattr(owner, name, announced)

    if hasattr(pipe, "vae"):
        wrap(pipe.vae, "encode", "vae-encode")
        wrap(pipe.vae, "decode", "vae-decode")
    if hasattr(pipe, "encode_prompt"):
        wrap(pipe, "encode_prompt", "text")
    pipe._hyworld_announced = True


def _load_pano(cpu_offload: bool):
    """Load HY-Pano in *this* process. No longer used by the UI -- see
    `pano_worker.py` for why the panorama runs in a subprocess -- but kept
    because it is the shortest way to reproduce the in-process behaviour when
    someone wants to attack that stall again."""
    global _pano_pipe
    if _pano_pipe is None:
        from pipeline_with_qwen_image import HunyuanPanoPipeline  # HY-World-2.0/hyworld2/panogen

        base = WEIGHTS_DIR / "Qwen-Image-Edit-2509"
        base = str(base) if base.is_dir() else HunyuanPanoPipeline.DEFAULT_MODEL_ID
        lora = str(WEIGHTS_DIR) if (WEIGHTS_DIR / "HY-Pano-2.0").is_dir() else HunyuanPanoPipeline.DEFAULT_LORA_PATH
        t0 = time.perf_counter()
        _pano_pipe = HunyuanPanoPipeline.from_pretrained(
            base, lora_path=lora, lora_subfolder="HY-Pano-2.0", cpu_offload=bool(cpu_offload))
        print(f"[UI] HY-Pano 2.0 ready in {time.perf_counter() - t0:.1f} s")
    _announce_stages(_pano_pipe.pipe)
    return _pano_pipe


def _load_recon(enable_bf16: bool):
    global _recon_pipe, _recon_bf16
    if _recon_pipe is not None and _recon_bf16 != bool(enable_bf16):
        _unload("recon")
    if _recon_pipe is None:
        from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

        base = str(WEIGHTS_DIR) if (WEIGHTS_DIR / "HY-WorldMirror-2.0").is_dir() else "tencent/HY-World-2.0"
        t0 = time.perf_counter()
        _recon_pipe = WorldMirrorPipeline.from_pretrained(base, enable_bf16=bool(enable_bf16))
        _recon_bf16 = bool(enable_bf16)
        print(f"[UI] WorldMirror 2.0 ready in {time.perf_counter() - t0:.1f} s (bf16={_recon_bf16})")
    return _recon_pipe


def _unload(which: str) -> None:
    global _pano_pipe, _recon_pipe
    if which == "pano":
        _pano_pipe = None
    else:
        _recon_pipe = None
    gc.collect()
    compat.empty_cache()


def _memory_report(lang: str) -> str:
    loaded = dict(pano="✔" if _pano_pipe is not None else "—", recon="✔" if _recon_pipe is not None else "—")
    if compat.device_type() == "mps":
        # Unified memory: there is no "free on device" figure. recommended_max_memory
        # is the share of RAM Metal lets a process use; driver_allocated is what the
        # allocator currently holds (allocated + cached), which is what a second
        # process would compete with.
        budget = getattr(torch.mps, "recommended_max_memory", lambda: 0)()   # torch >= 2.3
        return t(lang, "mem_report_mps",
                 alloc=torch.mps.current_allocated_memory() / 2**30,
                 reserved=torch.mps.driver_allocated_memory() / 2**30,
                 total=budget / 2**30,
                 ram=psutil.virtual_memory().total / 2**30, **loaded)
    if compat.device_type() != "cuda":
        return t(lang, "mem_no_gpu")
    free, total = torch.cuda.mem_get_info()
    return t(lang, "mem_report",
             alloc=torch.cuda.memory_allocated() / 2**30, reserved=torch.cuda.memory_reserved() / 2**30,
             free=free / 2**30, total=total / 2**30, **loaded)


# ==== Files, runs and the viewer ============================================
def _new_run_dir(kind: str) -> Path:
    d = OUTPUTS_DIR / "ui" / f"{kind}_{datetime.now():%Y%m%d_%H%M%S}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _outputs_url(path) -> str:
    return "/outputs/" + Path(path).resolve().relative_to(OUTPUTS_DIR).as_posix()


def _viewer_iframe(**params) -> str:
    """<iframe> onto viewer/index.html with the given files (splat / points / cams / pano)."""
    qs = {k: _outputs_url(v) for k, v in params.items() if v}
    qs["v"] = str(int(time.time()))  # defeat iframe caching between runs
    return (f'<iframe src="/viewer/?{urlencode(qs)}" '
            f'style="width:100%;height:{VIEWER_HEIGHT}px;border:0;border-radius:8px;background:#111" '
            f'allow="fullscreen"></iframe>')


def _ensure_splat(result_dir: Path) -> Optional[Path]:
    """`scene.splat` next to `gaussians.ply`, written on first use.

    The viewer shows this file rather than the .ply: WorldMirror stores its
    opacities after the sigmoid, while every INRIA-layout viewer (the one in
    the page included) applies the sigmoid again and renders the .ply too
    translucent. The .splat carries the true alpha and is half the size.
    """
    ply = result_dir / "gaussians.ply"
    if not ply.is_file():
        return None
    splat = result_dir / "scene.splat"
    if not splat.is_file() or splat.stat().st_mtime < ply.stat().st_mtime:
        export3d.write_splat(ply, splat)
    return splat


def _collect_recon_outputs(result_dir: Path):
    """(viewer html, downloadable files, video path or None, depth/normal previews)."""
    result_dir = Path(result_dir)
    splat = _ensure_splat(result_dir)
    files = [result_dir / n for n in ("gaussians.ply", "scene.splat", "points.ply", "camera_params.json", "pipeline_timing.json")]
    files = [f for f in files if f.is_file()]
    video = result_dir / "rendered" / "rendered_rgb.mp4"
    video = video if video.is_file() else None
    if video:
        files.append(video)
    gallery = sorted((result_dir / "depth").glob("*.png"))[:8] + sorted((result_dir / "normal").glob("*.png"))[:8]
    have = {n: (result_dir / n) for n in ("points.ply", "camera_params.json") if (result_dir / n).is_file()}
    html = _viewer_iframe(splat=splat, points=have.get("points.ply"), cams=have.get("camera_params.json"))
    return html, [str(f) for f in files], (str(video) if video else None), [str(g) for g in gallery]


# ==== Earlier runs and export ===============================================
def _list_runs() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(scenes, panoramas) under the outputs directory as (label, path) pairs,
    newest first. A scene is any directory holding gaussians.ply; a panorama
    is any panorama.png that is not merely the input of a panorama -> 3D run."""
    scenes, panos = [], []
    if not OUTPUTS_DIR.is_dir():
        return scenes, panos
    for gs in OUTPUTS_DIR.rglob("gaussians.ply"):
        d = gs.parent
        rel = d.relative_to(OUTPUTS_DIR).as_posix()
        label = rel[:-len("/result")] if rel.endswith("/result") else rel
        scenes.append((gs.stat().st_mtime, label, str(d)))
    for png in OUTPUTS_DIR.rglob("panorama.png"):
        if png.parent.name.startswith("pano3d_"):
            continue
        rel = png.parent.relative_to(OUTPUTS_DIR).as_posix() or png.parent.name
        panos.append((png.stat().st_mtime, rel, str(png)))
    scenes.sort(reverse=True)
    panos.sort(reverse=True)
    return [(lb, p) for _, lb, p in scenes], [(lb, p) for _, lb, p in panos]


def _fmt_size(n: int) -> str:
    return f"{n / 2**20:.1f} MiB" if n >= 2**20 else f"{n / 2**10:.0f} KiB"


def run_export_scene(result_dir, fmt, detail, lang):
    """Export a reconstruction result (streams status + log like a generation)."""
    u = gr.update()
    if not result_dir or not Path(result_dir).is_dir():
        yield t(lang, "st_idle"), t(lang, "err_no_result"), u
        return
    result_dir = Path(result_dir)
    step = int(detail or 2)

    def job():
        return export3d.export_scene(result_dir, fmt, result_dir / "export", mesh_step=step,
                                     title=f"HY-World 2.0 — {result_dir.parent.name}")

    t0 = time.monotonic()
    for status, log, result, err in _run_job(job, lang):
        if err is not None:
            yield status, log + "\n" + t(lang, "err_failed", e=err), u
            return
        if result is None:
            yield status, log, u
            continue
        out = Path(result)
        yield (t(lang, "export_done", name=out.name, size=_fmt_size(out.stat().st_size),
                 elapsed=_fmt_duration(time.monotonic() - t0)), log, str(out))


def run_export_pano(pano_path, fmt, lang):
    u = gr.update()
    if not pano_path or not Path(pano_path).is_file():
        yield t(lang, "st_idle"), t(lang, "err_no_result"), u
        return
    pano_path = Path(pano_path)

    def job():
        return export3d.export_panorama(pano_path, fmt, pano_path.parent / "export",
                                        title=f"HY-World 2.0 — {pano_path.parent.name}")

    t0 = time.monotonic()
    for status, log, result, err in _run_job(job, lang):
        if err is not None:
            yield status, log + "\n" + t(lang, "err_failed", e=err), u
            return
        if result is None:
            yield status, log, u
            continue
        out = Path(result)
        yield (t(lang, "export_done", name=out.name, size=_fmt_size(out.stat().st_size),
                 elapsed=_fmt_duration(time.monotonic() - t0)), log, str(out))


def _stage_inputs(paths: list[str], run_dir: Path) -> Path:
    """Copy the uploads into the run directory: a video as-is, images renamed
    with an index prefix so the upload order survives the pipeline's sort."""
    videos = [p for p in paths if Path(p).suffix.lower() in VIDEO_EXTS]
    if videos:
        dst = run_dir / Path(videos[0]).name
        shutil.copy(videos[0], dst)
        return dst
    images = [p for p in paths if Path(p).suffix.lower() in IMAGE_EXTS]
    if not images:
        raise ValueError("no supported images or video among the uploads")
    img_dir = run_dir / "images"
    img_dir.mkdir(exist_ok=True)
    for i, p in enumerate(images):
        shutil.copy(p, img_dir / f"{i:03d}_{Path(p).name}")
    return img_dir


def _list_examples() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for cat in ("realistic", "stylistic"):
        d = EXAMPLES_DIR / cat
        if d.is_dir():
            for s in sorted(p for p in d.iterdir() if p.is_dir()):
                out[f"{cat}/{s.name}"] = s
    return out


EXAMPLES = _list_examples()


def _example_files(name: Optional[str]) -> Optional[list[str]]:
    d = EXAMPLES.get(name or "")
    if d is None:
        return None
    return sorted(str(p) for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)


# ==== Jobs ==================================================================
def _run_worker(cmd: list[str]) -> None:
    """Run a generation subprocess, echoing its output so the UI can follow it.

    Everything printed here goes through the capture in `_run_job`, so the log
    streams and the stage tracker sees the worker's `[Stage]` markers exactly as
    if the work were happening in-process. stderr is merged in because tqdm
    writes its progress bars there.
    """
    proc = subprocess.Popen(
        cmd, cwd=str(REPO_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            # end="" keeps the worker's own newlines and, importantly, its
            # carriage returns, so tqdm bars still overwrite themselves.
            print(line, end="", flush=True)
        code = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    if code != 0:
        raise RuntimeError(f"worker exited with code {code} (see the log above)")


def _free_gpu_cache(tag: str) -> None:
    """Return cached-but-unused blocks before a job starts.

    The caching allocator keeps freed blocks for reuse, which is normally a
    win. It is not a win here: the panorama peaks at 58.4 GiB inside a 64 GB
    carve-out, so a few GB held over from a previous run is the difference
    between finishing and stopping dead. Measured on this machine -- a fresh
    process reserves 55.4 GiB for the same work where a long-lived UI session
    had grown to 60.9 GiB, and that session wedged after the denoising loop
    with the CPU spinning and the GPU idle: the allocator could not find room
    and neither could it give up.
    """
    if compat.device_type() == "mps":
        before = torch.mps.driver_allocated_memory()
        compat.empty_cache()
        print(f"[Memory] {tag}: released {(before - torch.mps.driver_allocated_memory()) / 2**30:.2f} GiB of cache, "
              f"{torch.mps.driver_allocated_memory() / 2**30:.2f} GiB held by Metal", flush=True)
        return
    if compat.device_type() != "cuda":
        return
    before = torch.cuda.memory_reserved()
    compat.empty_cache()
    freed = (before - torch.cuda.memory_reserved()) / 2**30
    free, total = torch.cuda.mem_get_info()
    print(f"[Memory] {tag}: released {freed:.2f} GiB of cache, "
          f"{free / 2**30:.1f} / {total / 2**30:.1f} GiB free on device", flush=True)


def _reconstruct(input_path: Path, out_dir: Path, target_size, enable_bf16, sky, edge, conf,
                 max_gs, render, interp, prior_cam: Optional[str] = None) -> str:
    with _gpu_lock:
        _free_gpu_cache("before reconstruction")
        pipe = _load_recon(enable_bf16)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = pipe(
                str(input_path), strict_output_path=str(out_dir),
                target_size=int(target_size),
                apply_sky_mask=bool(sky), apply_edge_mask=bool(edge), apply_confidence_mask=bool(conf),
                save_sky_mask=bool(sky),        # the mesh export masks the sky with it
                compress_gs_max_points=int(max_gs),
                save_rendered=bool(render), render_interp_per_pair=int(interp),
                prior_cam_path=prior_cam,
            )
        if out is None:
            raise RuntimeError("the pipeline skipped this input (see the log above)")
        print(f"[UI] reconstruction finished in {time.perf_counter() - t0:.1f} s -> {out}")
        _keep_video_frames(input_path, Path(out))
        _ensure_splat(Path(out))
        return out


def _keep_video_frames(input_path: Path, result_dir: Path) -> None:
    """Copy the frames the pipeline extracted from a video next to the result.

    prepare_input() writes them to /tmp/frames_<stem> (C:\\tmp on Windows) and
    never cleans up; the mesh export needs them as textures, so they are kept
    with the run where the export code and the user can find them.
    """
    if input_path.is_dir():
        return
    src = Path("/tmp") / f"frames_{input_path.stem}"
    if not src.is_dir():
        return
    dst = result_dir.parent / "frames"
    dst.mkdir(exist_ok=True)
    n = 0
    for p in sorted(src.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            shutil.copy(p, dst / p.name)
            n += 1
    if n:
        print(f"[UI] kept {n} extracted video frames in {dst}")


def run_panorama(image, prompt, negative, seed, steps, guidance, true_cfg, size_key, blend, crop, cpu_offload, lang):
    u = gr.update()
    if image is None:
        yield t(lang, "st_idle"), u, u, u, t(lang, "err_no_image"), u
        return
    run_dir = _new_run_dir("pano")
    in_path = run_dir / "input.png"
    image.save(in_path)
    width, height = PANO_SIZES.get(size_key, PANO_SIZES["1952x960"])
    seed = int(seed)
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    out_path = run_dir / "panorama.png"

    def job():
        with _gpu_lock:
            _free_gpu_cache("before panorama")
            t0 = time.perf_counter()
            _run_worker([
                sys.executable, "-u", str(PROJECT_DIR / "pano_worker.py"),
                "--image", str(in_path), "--out", str(out_path),
                "--weights", str(WEIGHTS_DIR),
                "--prompt", prompt or "", "--negative-prompt", negative or "",
                "--seed", str(seed), "--width", str(width), "--height", str(height),
                "--steps", str(int(steps)), "--guidance-scale", str(float(guidance)),
                "--true-cfg-scale", str(float(true_cfg)),
                "--blend-width", str(int(blend)), "--crop-border", str(float(crop)),
            ] + (["--cpu-offload"] if cpu_offload else []))
            if not out_path.is_file():
                raise RuntimeError("the worker finished without writing a panorama")
            print(f"[UI] panorama done in {time.perf_counter() - t0:.1f} s (seed={seed})")
            return str(out_path)

    t0 = time.time()
    for status, log, result, err in _run_job(job, lang):
        if err is not None:
            yield status, u, u, u, log + "\n" + t(lang, "err_failed", e=err), u
            return
        if result is None:
            yield status, u, u, u, log, u
            continue
        yield (status, result, _viewer_iframe(pano=result), result,
               log + "\n" + t(lang, "done", elapsed=time.time() - t0), result)


def run_reconstruction(files, example, target_size, enable_bf16, sky, edge, conf, max_gs, render, interp, lang):
    u = gr.update()
    paths = list(files or [])
    if not paths and example:
        paths = _example_files(example) or []
    if not paths:
        yield t(lang, "st_idle"), u, u, u, u, t(lang, "err_no_files"), u
        return
    run_dir = _new_run_dir("recon")
    try:
        input_path = _stage_inputs(paths, run_dir)
    except ValueError as e:
        yield t(lang, "st_idle"), u, u, u, u, t(lang, "err_failed", e=e), u
        return

    def job():
        return _reconstruct(input_path, run_dir / "result", target_size, enable_bf16, sky, edge, conf,
                            max_gs, render, interp)

    t0 = time.time()
    for status, log, result, err in _run_job(job, lang):
        if err is not None:
            yield status, u, u, u, u, log + "\n" + t(lang, "err_failed", e=err), u
            return
        if result is None:
            yield status, u, u, u, u, log, u
            continue
        html, out_files, video, gallery = _collect_recon_outputs(Path(result))
        yield (status, html, out_files, video, gallery,
               log + "\n" + t(lang, "done", elapsed=time.time() - t0), str(result))


def run_pano_to_3d(pano_path, n_views, fov, size, rows_key, use_prior, target_size, enable_bf16,
                   sky, edge, conf, max_gs, render, interp, lang):
    u = gr.update()
    if not pano_path:
        yield t(lang, "st_idle"), u, u, u, u, t(lang, "err_no_image"), u
        return
    run_dir = _new_run_dir("pano3d")
    pano_copy = run_dir / "panorama.png"
    Image.open(pano_path).convert("RGB").save(pano_copy)
    views, cams = pano3d.split_panorama(
        pano_copy, run_dir, n_views=int(n_views), fov_deg=float(fov), size=int(size),
        pitch_rows=PITCH_ROWS.get(rows_key, PITCH_ROWS["1"]))
    yield t(lang, "stage_views"), views, u, u, u, t(lang, "log_views_ready", n=len(views)), u

    def job():
        return _reconstruct(run_dir / "images", run_dir / "result", target_size, enable_bf16, sky, edge, conf,
                            max_gs, render, interp, prior_cam=cams if use_prior else None)

    t0 = time.time()
    for status, log, result, err in _run_job(job, lang):
        if err is not None:
            yield status, u, u, u, u, log + "\n" + t(lang, "err_failed", e=err), u
            return
        if result is None:
            yield status, u, u, u, u, log, u
            continue
        html, out_files, video, gallery = _collect_recon_outputs(Path(result))
        yield (status, u, html, out_files, video,
               log + "\n" + t(lang, "done", elapsed=time.time() - t0), str(result))


def open_scene_run(result_dir, lang):
    """Show an earlier reconstruction in the viewer (Results tab)."""
    u = gr.update()
    if not result_dir or not Path(result_dir).is_dir():
        return t(lang, "results_empty"), u, u, u, u, None
    html, out_files, video, gallery = _collect_recon_outputs(Path(result_dir))
    return t(lang, "st_idle"), html, out_files, video, gallery, str(result_dir)


def open_pano_run(pano_path, lang):
    u = gr.update()
    if not pano_path or not Path(pano_path).is_file():
        return t(lang, "results_empty"), u, u, None
    return t(lang, "st_idle"), _viewer_iframe(pano=pano_path), pano_path, pano_path


# ==== UI ====================================================================
CSS = """
.hy-log textarea { font-family: ui-monospace, Consolas, monospace; font-size: 12px; }
.hy-status { font-size: 13px; padding: 6px 10px; border-radius: 6px;
             background: var(--background-fill-secondary); min-height: 2.2em; }
.hy-status p { margin: 0.2em 0; }
"""


def _desc_kwargs() -> dict:
    return dict(
        device=compat.describe(),
        attn=compat.attention_backend_name(),
        dtype=str(compat.preferred_dtype()).replace("torch.", ""),
        weights=WEIGHTS_DIR,
        outputs=OUTPUTS_DIR,
    )


def _recon_controls(L: _L10n, lang: str, key_prefix: str):
    """The WorldMirror parameter block, shared by the two reconstruction tabs."""
    with L.bind(gr.Accordion(t(lang, "recon_params"), open=False), label="recon_params"):
        target = L.bind(gr.Dropdown(choices=[518, 714, 952, 1204], value=952, label=t(lang, "recon_target")),
                        label="recon_target")
        bf16 = L.bind(gr.Checkbox(value=compat.supports_bf16(), label=t(lang, "recon_bf16")), label="recon_bf16")
        sky = L.bind(gr.Checkbox(value=True, label=t(lang, "recon_sky")), label="recon_sky")
        edge = L.bind(gr.Checkbox(value=True, label=t(lang, "recon_edge")), label="recon_edge")
        conf = L.bind(gr.Checkbox(value=False, label=t(lang, "recon_conf")), label="recon_conf")
        max_gs = L.bind(gr.Slider(minimum=200_000, maximum=5_000_000, step=100_000, value=2_000_000,
                                  label=t(lang, "recon_max_gs")), label="recon_max_gs")
        render = L.bind(gr.Checkbox(value=False, label=t(lang, "recon_render")), label="recon_render")
        interp = L.bind(gr.Slider(minimum=5, maximum=60, step=5, value=15, label=t(lang, "recon_interp")),
                        label="recon_interp")
    return [target, bf16, sky, edge, conf, max_gs, render, interp]


def _scene_fmt_choices(lg: str):
    return [(t(lg, f"fmt_{k}"), k) for k in export3d.SCENE_FORMATS]


def _pano_fmt_choices(lg: str):
    return [(t(lg, f"pfmt_{k}"), k) for k in export3d.PANO_FORMATS]


def _detail_choices(lg: str):
    return [(t(lg, f"detail_{k}"), k) for k in ("1", "2", "4")]


def _export_panel(L: _L10n, lang: str, kind: str):
    """Format picker + Export button + download slot, for a scene or a panorama.

    Returns (format, detail, button, file); ``detail`` is None for panoramas.
    """
    with L.bind(gr.Accordion(t(lang, "export_acc"), open=True), label="export_acc"):
        with gr.Row():
            if kind == "scene":
                fmt = L.bind(gr.Dropdown(choices=_scene_fmt_choices(lang), value="web_zip", label=t(lang, "export_fmt"),
                                         scale=3), extra=lambda lg: {"choices": _scene_fmt_choices(lg)}, label="export_fmt")
                detail = L.bind(gr.Dropdown(choices=_detail_choices(lang), value="2", label=t(lang, "export_detail"),
                                            scale=1), extra=lambda lg: {"choices": _detail_choices(lg)}, label="export_detail")
            else:
                fmt = L.bind(gr.Dropdown(choices=_pano_fmt_choices(lang), value="web_html", label=t(lang, "export_fmt"),
                                         scale=3), extra=lambda lg: {"choices": _pano_fmt_choices(lg)}, label="export_fmt")
                detail = None
        with gr.Row():
            btn = L.bind(gr.Button(t(lang, "export_btn"), variant="secondary", scale=1), value="export_btn")
            file = L.bind(gr.File(label=t(lang, "export_file"), scale=3), label="export_file")
        L.bind(gr.Markdown(t(lang, "export_help_scene" if kind == "scene" else "export_help_pano")),
               value="export_help_scene" if kind == "scene" else "export_help_pano")
    return fmt, detail, btn, file


def build_demo() -> gr.Blocks:
    L = _L10n()
    lang = DEFAULT_LANG

    def size_choices(lg):
        return [(t(lg, f"size_{k}"), k) for k in PANO_SIZES]

    def rows_choices(lg):
        return [(t(lg, f"rows_{k}"), k) for k in PITCH_ROWS]

    scenes0, panos0 = _list_runs()

    with gr.Blocks(title="HY-World 2.0", theme=gr.themes.Soft(), css=CSS) as demo:
        lang_state = gr.State(lang)
        pano_state = gr.State(None)  # path of the last generated panorama
        recon_result = gr.State(None)  # result dir of the last reconstruction, for export
        p3d_result = gr.State(None)
        res_scene_state = gr.State(None)  # the run opened on the Results tab
        res_pano_state = gr.State(None)

        lang_radio = L.bind(gr.Radio(choices=[LANG_DISPLAY[k] for k in LANG_CODES], value=LANG_DISPLAY[lang],
                                     label=t(lang, "lang_label")), label="lang_label")
        L.bind(gr.Markdown(t(lang, "description", **_desc_kwargs())),
               extra=lambda lg: {"value": t(lg, "description", **_desc_kwargs())})

        with gr.Tabs():
            # ---------------------------------------------------------------- panorama
            with L.bind(gr.Tab(t(lang, "tab_pano")), label="tab_pano"):
                with gr.Row():
                    with gr.Column(scale=1):
                        pano_in = L.bind(gr.Image(type="pil", label=t(lang, "pano_input"), height=320,
                                                  sources=["upload", "clipboard"]), label="pano_input")
                        pano_prompt = L.bind(gr.Textbox(label=t(lang, "pano_prompt"), lines=2,
                                                        placeholder=t(lang, "pano_prompt_ph")),
                                             label="pano_prompt", placeholder="pano_prompt_ph")
                        pano_neg = L.bind(gr.Textbox(label=t(lang, "pano_negative"), lines=1), label="pano_negative")
                        with L.bind(gr.Accordion(t(lang, "pano_params"), open=False), label="pano_params"):
                            pano_seed = L.bind(gr.Number(value=42, precision=0, label=t(lang, "pano_seed")), label="pano_seed")
                            pano_steps = L.bind(gr.Slider(minimum=10, maximum=60, step=1, value=40,
                                                          label=t(lang, "pano_steps")), label="pano_steps")
                            pano_guidance = L.bind(gr.Slider(minimum=1.0, maximum=5.0, step=0.1, value=1.0,
                                                             label=t(lang, "pano_guidance")), label="pano_guidance")
                            pano_cfg = L.bind(gr.Slider(minimum=1.0, maximum=12.0, step=0.5, value=7.5,
                                                        label=t(lang, "pano_true_cfg")), label="pano_true_cfg")
                            pano_size = L.bind(gr.Dropdown(choices=size_choices(lang), value="1952x960",
                                                           label=t(lang, "pano_size")),
                                               extra=lambda lg: {"choices": size_choices(lg)}, label="pano_size")
                            pano_blend = L.bind(gr.Slider(minimum=0, maximum=128, step=8, value=32,
                                                          label=t(lang, "pano_blend")), label="pano_blend")
                            pano_crop = L.bind(gr.Slider(minimum=0.0, maximum=0.2, step=0.01, value=0.0,
                                                         label=t(lang, "pano_crop")), label="pano_crop")
                            pano_offload = L.bind(gr.Checkbox(value=False, label=t(lang, "pano_cpu_offload")),
                                                  label="pano_cpu_offload")
                        pano_btn = L.bind(gr.Button(t(lang, "pano_btn"), variant="primary"), value="pano_btn")
                        pano_status = gr.Markdown(t(lang, "st_idle"), elem_classes=["hy-status"])
                        pano_log = L.bind(gr.Textbox(label=t(lang, "log_label"), lines=12, max_lines=30,
                                                     interactive=False, elem_classes=["hy-log"]), label="log_label")
                    with gr.Column(scale=2):
                        L.bind(gr.Markdown(t(lang, "pano_viewer_md")), value="pano_viewer_md")
                        pano_viewer = gr.HTML(_viewer_iframe())
                        pano_out = L.bind(gr.Image(label=t(lang, "pano_result"), interactive=False, height=260),
                                          label="pano_result")
                        pano_dl = L.bind(gr.File(label=t(lang, "pano_download")), label="pano_download")
                        pano_exp_fmt, _, pano_exp_btn, pano_exp_file = _export_panel(L, lang, "pano")
                L.bind(gr.Markdown(t(lang, "pano_help")), value="pano_help")

            # ---------------------------------------------------------------- reconstruction
            with L.bind(gr.Tab(t(lang, "tab_recon")), label="tab_recon"):
                with gr.Row():
                    with gr.Column(scale=1):
                        recon_files = L.bind(gr.File(file_count="multiple", type="filepath",
                                                     file_types=["image", "video"],
                                                     label=t(lang, "recon_files")), label="recon_files")
                        recon_example = L.bind(gr.Dropdown(choices=list(EXAMPLES), value=None,
                                                           label=t(lang, "recon_examples")), label="recon_examples")
                        recon_ctrls = _recon_controls(L, lang, "recon")
                        recon_btn = L.bind(gr.Button(t(lang, "recon_btn"), variant="primary"), value="recon_btn")
                        recon_status = gr.Markdown(t(lang, "st_idle"), elem_classes=["hy-status"])
                        recon_log = L.bind(gr.Textbox(label=t(lang, "log_label"), lines=12, max_lines=30,
                                                      interactive=False, elem_classes=["hy-log"]), label="log_label")
                    with gr.Column(scale=2):
                        L.bind(gr.Markdown(t(lang, "recon_viewer_md")), value="recon_viewer_md")
                        recon_viewer = gr.HTML(_viewer_iframe())
                        with gr.Row():
                            recon_out_files = L.bind(gr.File(label=t(lang, "recon_files_out"), file_count="multiple"),
                                                     label="recon_files_out")
                            recon_video = L.bind(gr.Video(label=t(lang, "recon_video"), height=240), label="recon_video")
                        recon_gallery = L.bind(gr.Gallery(label=t(lang, "recon_gallery"), columns=4, height=200),
                                               label="recon_gallery")
                        recon_exp_fmt, recon_exp_detail, recon_exp_btn, recon_exp_file = _export_panel(L, lang, "scene")
                L.bind(gr.Markdown(t(lang, "recon_help")), value="recon_help")

            # ---------------------------------------------------------------- panorama -> 3D
            with L.bind(gr.Tab(t(lang, "tab_p3d")), label="tab_p3d"):
                with gr.Row():
                    with gr.Column(scale=1):
                        p3d_in = L.bind(gr.Image(type="filepath", label=t(lang, "p3d_input"), height=220,
                                                 sources=["upload", "clipboard"]), label="p3d_input")
                        p3d_use_last = L.bind(gr.Button(t(lang, "p3d_use_last"), variant="secondary"), value="p3d_use_last")
                        with L.bind(gr.Accordion(t(lang, "p3d_params"), open=True), label="p3d_params"):
                            p3d_views = L.bind(gr.Slider(minimum=4, maximum=16, step=1, value=8,
                                                         label=t(lang, "p3d_views")), label="p3d_views")
                            p3d_fov = L.bind(gr.Slider(minimum=50, maximum=120, step=5, value=90,
                                                       label=t(lang, "p3d_fov")), label="p3d_fov")
                            p3d_size = L.bind(gr.Dropdown(choices=[512, 768, 1024], value=768,
                                                          label=t(lang, "p3d_size")), label="p3d_size")
                            p3d_rows = L.bind(gr.Dropdown(choices=rows_choices(lang), value="1",
                                                          label=t(lang, "p3d_rows")),
                                              extra=lambda lg: {"choices": rows_choices(lg)}, label="p3d_rows")
                            p3d_prior = L.bind(gr.Checkbox(value=True, label=t(lang, "p3d_prior")), label="p3d_prior")
                        p3d_ctrls = _recon_controls(L, lang, "p3d")
                        p3d_btn = L.bind(gr.Button(t(lang, "p3d_btn"), variant="primary"), value="p3d_btn")
                        p3d_status = gr.Markdown(t(lang, "st_idle"), elem_classes=["hy-status"])
                        p3d_log = L.bind(gr.Textbox(label=t(lang, "log_label"), lines=12, max_lines=30,
                                                    interactive=False, elem_classes=["hy-log"]), label="log_label")
                    with gr.Column(scale=2):
                        L.bind(gr.Markdown(t(lang, "recon_viewer_md")), value="recon_viewer_md")
                        p3d_viewer = gr.HTML(_viewer_iframe())
                        p3d_gallery = L.bind(gr.Gallery(label=t(lang, "p3d_gallery"), columns=8, height=140),
                                             label="p3d_gallery")
                        with gr.Row():
                            p3d_out_files = L.bind(gr.File(label=t(lang, "recon_files_out"), file_count="multiple"),
                                                   label="recon_files_out")
                            p3d_video = L.bind(gr.Video(label=t(lang, "recon_video"), height=240), label="recon_video")
                        p3d_exp_fmt, p3d_exp_detail, p3d_exp_btn, p3d_exp_file = _export_panel(L, lang, "scene")
                L.bind(gr.Markdown(t(lang, "p3d_help")), value="p3d_help")

            # ---------------------------------------------------------------- results
            with L.bind(gr.Tab(t(lang, "tab_results")), label="tab_results"):
                L.bind(gr.Markdown(t(lang, "results_md")), value="results_md")
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            res_scene_dd = L.bind(gr.Dropdown(choices=scenes0, value=None, label=t(lang, "results_scenes"),
                                                              scale=4), label="results_scenes")
                            res_scene_open = L.bind(gr.Button(t(lang, "results_open"), scale=1), value="results_open")
                        with gr.Row():
                            res_pano_dd = L.bind(gr.Dropdown(choices=panos0, value=None, label=t(lang, "results_panos"),
                                                             scale=4), label="results_panos")
                            res_pano_open = L.bind(gr.Button(t(lang, "results_open"), scale=1), value="results_open")
                        res_refresh = L.bind(gr.Button(t(lang, "results_refresh")), value="results_refresh")
                        res_status = gr.Markdown(t(lang, "st_idle"), elem_classes=["hy-status"])
                        res_log = L.bind(gr.Textbox(label=t(lang, "log_label"), lines=8, max_lines=30,
                                                    interactive=False, elem_classes=["hy-log"]), label="log_label")
                    with gr.Column(scale=2):
                        L.bind(gr.Markdown(t(lang, "recon_viewer_md")), value="recon_viewer_md")
                        res_viewer = gr.HTML(_viewer_iframe())
                        with gr.Row():
                            res_files = L.bind(gr.File(label=t(lang, "recon_files_out"), file_count="multiple"),
                                               label="recon_files_out")
                            res_video = L.bind(gr.Video(label=t(lang, "recon_video"), height=240), label="recon_video")
                        res_gallery = L.bind(gr.Gallery(label=t(lang, "recon_gallery"), columns=4, height=160),
                                             label="recon_gallery")
                        res_exp_fmt, res_exp_detail, res_exp_btn, res_exp_file = _export_panel(L, lang, "scene")
                        L.bind(gr.Markdown(t(lang, "pano_viewer_md")), value="pano_viewer_md")
                        res_pano_viewer = gr.HTML(_viewer_iframe())
                        res_pano_img = L.bind(gr.Image(label=t(lang, "pano_result"), interactive=False, height=200),
                                              label="pano_result")
                        res_pano_fmt, _, res_pano_btn, res_pano_file = _export_panel(L, lang, "pano")

            # ---------------------------------------------------------------- system
            with L.bind(gr.Tab(t(lang, "tab_system")), label="tab_system"):
                L.bind(gr.Markdown(t(lang, "sys_md")), value="sys_md")
                with gr.Row():
                    sys_mem_btn = L.bind(gr.Button(t(lang, "sys_mem_btn")), value="sys_mem_btn")
                    sys_unload_pano = L.bind(gr.Button(t(lang, "sys_unload_pano")), value="sys_unload_pano")
                    sys_unload_recon = L.bind(gr.Button(t(lang, "sys_unload_recon")), value="sys_unload_recon")
                sys_out = L.bind(gr.Textbox(label=t(lang, "sys_out"), lines=3, interactive=False), label="sys_out")

        # ---- events ----------------------------------------------------------
        def apply_language(display):
            lg = LANG_BY_DISPLAY.get(display, "en")
            return [lg] + L.updates(lg)

        lang_radio.change(apply_language, inputs=[lang_radio], outputs=[lang_state] + L.components())

        pano_btn.click(
            run_panorama,
            inputs=[pano_in, pano_prompt, pano_neg, pano_seed, pano_steps, pano_guidance, pano_cfg, pano_size,
                    pano_blend, pano_crop, pano_offload, lang_state],
            outputs=[pano_status, pano_out, pano_viewer, pano_dl, pano_log, pano_state],
            concurrency_id="gpu",
        )

        recon_example.change(lambda name: _example_files(name), inputs=[recon_example], outputs=[recon_files])
        recon_btn.click(
            run_reconstruction,
            inputs=[recon_files, recon_example] + recon_ctrls + [lang_state],
            outputs=[recon_status, recon_viewer, recon_out_files, recon_video, recon_gallery, recon_log, recon_result],
            concurrency_id="gpu",
        )

        p3d_use_last.click(lambda p: p, inputs=[pano_state], outputs=[p3d_in])
        p3d_btn.click(
            run_pano_to_3d,
            inputs=[p3d_in, p3d_views, p3d_fov, p3d_size, p3d_rows, p3d_prior] + p3d_ctrls + [lang_state],
            outputs=[p3d_status, p3d_gallery, p3d_viewer, p3d_out_files, p3d_video, p3d_log, p3d_result],
            concurrency_id="gpu",
        )

        # ---- export (CPU only, but serialised with the GPU jobs so the log is not interleaved)
        pano_exp_btn.click(run_export_pano, inputs=[pano_state, pano_exp_fmt, lang_state],
                           outputs=[pano_status, pano_log, pano_exp_file], concurrency_id="gpu")
        recon_exp_btn.click(run_export_scene, inputs=[recon_result, recon_exp_fmt, recon_exp_detail, lang_state],
                            outputs=[recon_status, recon_log, recon_exp_file], concurrency_id="gpu")
        p3d_exp_btn.click(run_export_scene, inputs=[p3d_result, p3d_exp_fmt, p3d_exp_detail, lang_state],
                          outputs=[p3d_status, p3d_log, p3d_exp_file], concurrency_id="gpu")

        # ---- results tab
        def refresh_runs():
            scenes, panos = _list_runs()
            return gr.update(choices=scenes, value=None), gr.update(choices=panos, value=None)

        res_refresh.click(refresh_runs, outputs=[res_scene_dd, res_pano_dd])
        res_scene_open.click(open_scene_run, inputs=[res_scene_dd, lang_state],
                             outputs=[res_status, res_viewer, res_files, res_video, res_gallery, res_scene_state])
        res_pano_open.click(open_pano_run, inputs=[res_pano_dd, lang_state],
                            outputs=[res_status, res_pano_viewer, res_pano_img, res_pano_state])
        res_exp_btn.click(run_export_scene, inputs=[res_scene_state, res_exp_fmt, res_exp_detail, lang_state],
                          outputs=[res_status, res_log, res_exp_file], concurrency_id="gpu")
        res_pano_btn.click(run_export_pano, inputs=[res_pano_state, res_pano_fmt, lang_state],
                           outputs=[res_status, res_log, res_pano_file], concurrency_id="gpu")

        sys_mem_btn.click(_memory_report, inputs=[lang_state], outputs=[sys_out])

        def unload_pano(lg):
            _unload("pano")
            return t(lg, "unloaded", name="HY-Pano 2.0") + "\n" + _memory_report(lg)

        def unload_recon(lg):
            _unload("recon")
            return t(lg, "unloaded", name="WorldMirror 2.0") + "\n" + _memory_report(lg)

        sys_unload_pano.click(unload_pano, inputs=[lang_state], outputs=[sys_out], concurrency_id="gpu")
        sys_unload_recon.click(unload_recon, inputs=[lang_state], outputs=[sys_out], concurrency_id="gpu")

    return demo


# ==== Entry point ===========================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="HY-World 2.0 web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_PORT", "7860")))
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    print(f"[launch] {compat.describe()} | attention: {compat.attention_backend_name()}")
    print(f"[launch] weights: {WEIGHTS_DIR}")
    print(f"[launch] outputs: {OUTPUTS_DIR}")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    demo = build_demo()
    demo.queue(default_concurrency_limit=1)

    app = FastAPI(title="HY-World 2.0")
    # The viewer and the outputs are plain static files; mount them before the
    # Gradio app so its catch-all route at "/" does not swallow them.
    app.mount("/viewer", StaticFiles(directory=str(VIEWER_DIR), html=True), name="viewer")
    app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
    app = gr.mount_gradio_app(app, demo, path="/", allowed_paths=[str(OUTPUTS_DIR), str(EXAMPLES_DIR)])

    browse_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{browse_host}:{args.port}"
    print(f"[launch] {url}")
    if not args.no_browser:
        threading.Timer(2.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
