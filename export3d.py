#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export HY-World 2.0 results into popular formats.

A reconstruction (WorldMirror 2.0) leaves a result directory with
``gaussians.ply`` (3D Gaussian Splatting, INRIA layout), ``points.ply``,
``camera_params.json``, per-view depth maps (``depth/*.npy``) and, optionally,
sky masks and a fly-through video. A panorama run leaves ``panorama.png``.
Everything here derives other formats from those files, on the CPU, without
touching the model:

  3D scene
    scene.splat        compact 3DGS (antimatter15 layout: SuperSplat, Unity/Unreal
                       plugins, most web viewers)
    points.glb         glTF 2.0 point cloud (three.js, Blender, Godot, ...)
    points.xyz         ASCII "x y z r g b" (CloudCompare, MeshLab)
    mesh.glb           textured relief mesh built from the depth maps
    mesh_obj.zip       the same mesh as OBJ + MTL + JPEG textures
    scene.three.json   three.js JSON Object format (THREE.ObjectLoader)
    scene_web.zip      a ready-to-open HTML5 + three.js scene (with a tiny server)
    scene.html         the same scene as a single self-contained HTML file

  Panorama
    cubemap.zip        six cube faces (px, nx, py, ny, pz, nz) for skyboxes
    panorama.glb       an inward-facing textured sphere (glTF viewers, Blender)
    panorama.html / panorama_web.zip   360-degree HTML5 viewers

Coordinate frames. WorldMirror works in the OpenCV convention (x right, y down,
z forward), and ``gaussians.ply`` / ``points.ply`` / ``camera_params.json`` /
``scene.splat`` keep it, so they stay interchangeable with the rest of the 3DGS
tool chain. glTF, OBJ and three.js are y-up, so ``mesh.glb``, ``points.glb``,
``mesh_obj.zip`` and ``scene.three.json`` are rotated 180 degrees about x
(y -> -y, z -> -z), which maps "forward" onto glTF's -z as well. The web
viewer undoes that rotation for the mesh so all layers line up.

The module has no torch dependency; it is imported by the web UI and can be
run from the command line on any result directory:

    python export3d.py outputs/ui/recon_20260905_120000/result --all
    python export3d.py outputs/ui/pano_20260905_120000/panorama.png --all
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
VIEWER_HTML = PROJECT_DIR / "viewer" / "index.html"

SH_C0 = 0.28209479177387814
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# Rotation that turns the OpenCV frame (y down, z forward) into glTF's (y up,
# z backward). A proper rotation (det = +1), so triangle winding survives it.
TO_YUP = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

Log = Callable[[str], None]


def _log_default(msg: str) -> None:
    print(msg, flush=True)


# ============================================================================
# Result directories
# ============================================================================
def find_inputs(scene_dir: Path) -> list[Path]:
    """The photographs a scene was reconstructed from, in pipeline order.

    The web UI stages uploads into ``<run>/images`` (index-prefixed so the
    upload order survives sorting) and copies extracted video frames into
    ``<run>/frames``; the CLI batches keep them in ``<run>/input``. The
    pipeline itself sorts the file names, which is what ``sorted`` does here.
    """
    for base in (scene_dir, scene_dir.parent):
        for name in ("frames", "images", "input"):
            d = base / name
            if d.is_dir():
                files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXT)
                if files:
                    return files
    return []


@dataclass
class SceneResult:
    """What a WorldMirror result directory contains."""
    dir: Path
    gaussians: Optional[Path] = None
    points: Optional[Path] = None
    cameras: Optional[Path] = None
    video: Optional[Path] = None
    depths: list[Path] = field(default_factory=list)
    sky_masks: list[Path] = field(default_factory=list)
    inputs: list[Path] = field(default_factory=list)
    panorama: Optional[Path] = None

    @classmethod
    def open(cls, result_dir) -> "SceneResult":
        d = Path(result_dir).resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"result directory not found: {d}")

        def have(name: str) -> Optional[Path]:
            p = d / name
            return p if p.is_file() else None

        pano = d.parent / "panorama.png"
        return cls(
            dir=d,
            gaussians=have("gaussians.ply"),
            points=have("points.ply"),
            cameras=have("camera_params.json"),
            video=have("rendered/rendered_rgb.mp4"),
            depths=sorted((d / "depth").glob("depth_*.npy")) if (d / "depth").is_dir() else [],
            sky_masks=sorted((d / "sky_mask").glob("sky_mask_*.png")) if (d / "sky_mask").is_dir() else [],
            inputs=find_inputs(d),
            panorama=pano if pano.is_file() else None,
        )

    def camera_data(self) -> dict:
        if self.cameras is None:
            return {"num_cameras": 0, "extrinsics": [], "intrinsics": []}
        with open(self.cameras, encoding="utf-8") as f:
            return json.load(f)

    @property
    def is_empty(self) -> bool:
        return self.gaussians is None and self.points is None and not self.depths


# ============================================================================
# PLY readers
# ============================================================================
def _read_ply_vertices(path: Path):
    from plyfile import PlyData  # in requirements-rocm.txt; imported lazily so the CLI help works without it
    return PlyData.read(str(path))["vertex"]


def load_points(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """(xyz float32 [N, 3], rgb uint8 [N, 3] or None)."""
    v = _read_ply_vertices(path)
    names = v.data.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    rgb = None
    if "red" in names:
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    return xyz, rgb


def load_gaussians(path: Path) -> dict:
    """Gaussians in the INRIA layout: log scales, logit opacities, SH DC colour."""
    v = _read_ply_vertices(path)
    return {
        "xyz": np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32),
        "log_scale": np.stack([v[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float32),
        "rot": np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float32),
        "f_dc": np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float32),
        "opacity": np.asarray(v["opacity"], dtype=np.float32),
    }


# ============================================================================
# 3DGS -> .splat
# ============================================================================
def write_splat(gaussians_ply: Path, out: Path, log: Log = _log_default) -> Path:
    """Convert an INRIA-style ``.ply`` into the compact 32-byte-per-splat
    ``.splat`` layout: position (3 x f32), scale (3 x f32), RGBA (4 x u8),
    rotation quaternion (4 x u8, ``q * 128 + 128``, w first).

    Splats are ordered by opacity x volume, largest first, like the reference
    converter, so viewers that stream the file show the important ones first.
    """
    g = load_gaussians(gaussians_ply)
    scale = np.exp(g["log_scale"])
    # WorldMirror writes opacities *after* its sigmoid (act_gs.reg_dense_opacities),
    # not as logits like the INRIA reference, so viewers that apply the sigmoid a
    # second time render its .ply too translucent. The .splat carries the true
    # alpha; the sigmoid is only applied to a file that really holds logits.
    op = g["opacity"]
    alpha = op if (op.min() >= 0.0 and op.max() <= 1.0) else 1.0 / (1.0 + np.exp(-op))
    rgb = np.clip((0.5 + SH_C0 * g["f_dc"]) * 255.0, 0, 255)
    q = g["rot"] / np.maximum(np.linalg.norm(g["rot"], axis=1, keepdims=True), 1e-12)
    order = np.argsort(-(scale.prod(axis=1) * alpha), kind="stable")

    dt = np.dtype([("pos", "<f4", (3,)), ("scale", "<f4", (3,)), ("rgba", "u1", (4,)), ("rot", "u1", (4,))])
    arr = np.empty(len(order), dtype=dt)
    arr["pos"] = g["xyz"][order]
    arr["scale"] = scale[order]
    arr["rgba"][:, :3] = rgb[order].astype(np.uint8)
    arr["rgba"][:, 3] = np.clip(alpha[order] * 255.0, 0, 255).astype(np.uint8)
    arr["rot"] = np.clip(q[order] * 128.0 + 128.0, 0, 255).astype(np.uint8)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(str(out))
    log(f"[Export] {out.name}: {len(arr):,} splats, {out.stat().st_size / 2**20:.1f} MiB")
    return out


# ============================================================================
# Point cloud formats
# ============================================================================
def write_points_xyz(points_ply: Path, out: Path, log: Log = _log_default) -> Path:
    xyz, rgb = load_points(points_ply)
    out = Path(out)
    if rgb is None:
        np.savetxt(str(out), xyz, fmt="%.5f")
    else:
        rows = np.concatenate([xyz, rgb.astype(np.float32)], axis=1)
        np.savetxt(str(out), rows, fmt="%.5f %.5f %.5f %d %d %d")
    log(f"[Export] {out.name}: {len(xyz):,} points")
    return out


def write_points_glb(points_ply: Path, out: Path, log: Log = _log_default) -> Path:
    import trimesh
    xyz, rgb = load_points(points_ply)
    xyz = xyz @ TO_YUP.T
    colors = None
    if rgb is not None:
        colors = np.concatenate([rgb, np.full((len(rgb), 1), 255, np.uint8)], axis=1)
    cloud = trimesh.PointCloud(xyz, colors=colors)
    out = Path(out)
    out.write_bytes(cloud.export(file_type="glb"))
    log(f"[Export] {out.name}: {len(xyz):,} points (y-up), {out.stat().st_size / 2**20:.1f} MiB")
    return out


# ============================================================================
# Relief mesh from the depth maps
# ============================================================================
@dataclass
class ViewMesh:
    name: str
    vertices: np.ndarray          # [N, 3] float32, OpenCV world frame
    faces: np.ndarray             # [M, 3] int32
    uv: np.ndarray                # [N, 2] float32, OBJ convention (v = 0 at the bottom)
    texture: Optional[Image.Image]
    vertex_colors: Optional[np.ndarray] = None   # [N, 3] uint8, when there is no texture


def _resize_dims(orig_w: int, orig_h: int, max_dim: int, patch: int = 14) -> tuple[int, int]:
    # Mirrors the pipeline's _calculate_resize_dims so the texture lands on the
    # same pixels the depth map was predicted for.
    if orig_w >= orig_h:
        new_w = max_dim
        new_h = round(orig_h * (new_w / orig_w) / patch) * patch
    else:
        new_h = max_dim
        new_w = round(orig_w * (new_h / orig_h) / patch) * patch
    return new_w, new_h


def _aligned_texture(image_path: Path, width: int, height: int, max_side: int) -> Image.Image:
    """The input photo resized and centre-cropped exactly as the pipeline did,
    so pixel (u, v) of the texture is the pixel depth[v, u] was predicted for."""
    img = Image.open(image_path)
    if img.mode == "RGBA":
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    img = img.convert("RGB")
    new_w, new_h = _resize_dims(img.width, img.height, max(width, height))
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    if new_h > height:
        top = (new_h - height) // 2
        img = img.crop((0, top, new_w, top + height))
    if new_w > width:
        left = (new_w - width) // 2
        img = img.crop((left, 0, left + width, img.height))
    if img.size != (width, height):          # the "pad" strategy, or an odd size: fall back to a plain resize
        img = img.resize((width, height), Image.Resampling.BICUBIC)
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.Resampling.LANCZOS)
    # Re-encode as JPEG so the GLB embeds a JPEG rather than a PNG several times larger.
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return Image.open(io.BytesIO(buf.getvalue()))


def _colors_from_points(vertices: np.ndarray, points_ply: Optional[Path]) -> Optional[np.ndarray]:
    """Nearest-neighbour colours from the point cloud, for scenes whose input
    frames are not on disk (a video processed by the CLI, say)."""
    if points_ply is None:
        return None
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    xyz, rgb = load_points(points_ply)
    if rgb is None or len(xyz) == 0:
        return None
    _, idx = cKDTree(xyz).query(vertices, k=1, workers=-1)
    return rgb[idx]


def build_relief_meshes(res: SceneResult, *, step: int = 2, depth_jump: float = 0.06,
                        max_texture: int = 2048, log: Log = _log_default) -> list[ViewMesh]:
    """One textured height-field mesh per view, from its predicted depth map.

    Every ``step``-th pixel becomes a vertex, unprojected through the view's
    intrinsics and placed in the world with its camera-to-world matrix.
    Triangles are dropped across depth discontinuities (a relative jump above
    ``depth_jump`` along any edge), over the sky mask, and where depth is
    invalid, so foreground objects are not "curtained" to the background.
    Per view because WorldMirror predicts one depth map per view; the meshes
    overlap where the views do, which is what the Gaussians do as well.
    """
    if not res.depths:
        raise FileNotFoundError("no depth maps (depth/depth_*.npy) in the result directory")
    cams = res.camera_data()
    extr = {int(e["camera_id"]): np.asarray(e["matrix"], np.float64) for e in cams.get("extrinsics", [])}
    intr = {int(e["camera_id"]): np.asarray(e["matrix"], np.float64) for e in cams.get("intrinsics", [])}
    if len(extr) != len(res.depths):
        raise ValueError(f"{len(res.depths)} depth maps but {len(extr)} cameras in camera_params.json")

    inputs = res.inputs if len(res.inputs) == len(res.depths) else []
    if not inputs:
        log("[Export] input frames not found next to the result -- colouring the mesh from the point cloud")
    meshes: list[ViewMesh] = []
    step = max(1, int(step))
    for i, depth_path in enumerate(res.depths):
        depth = np.load(depth_path).astype(np.float32)
        H, W = depth.shape[:2]
        K, c2w = intr[i], extr[i]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        valid = np.isfinite(depth) & (depth > 1e-6)
        if i < len(res.sky_masks):
            sky = np.asarray(Image.open(res.sky_masks[i]).convert("L").resize((W, H), Image.Resampling.NEAREST))
            valid &= sky > 127          # the mask is white where it is NOT sky

        ys = np.arange(0, H, step)
        xs = np.arange(0, W, step)
        if ys[-1] != H - 1:
            ys = np.append(ys, H - 1)
        if xs[-1] != W - 1:
            xs = np.append(xs, W - 1)
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        d = depth[gy, gx]
        v_ok = valid[gy, gx]
        rows, cols = d.shape

        # Unproject (pixel centre) and move to the world.
        X = (gx + 0.5 - cx) / fx * d
        Y = (gy + 0.5 - cy) / fy * d
        cam = np.stack([X, Y, d], axis=-1).reshape(-1, 3)
        world = cam @ c2w[:3, :3].T + c2w[:3, 3]

        # Triangles. Split each cell along the diagonal with the smaller depth
        # difference, and keep a triangle only if all its corners are valid and
        # no edge crosses a depth discontinuity.
        idx = np.arange(rows * cols).reshape(rows, cols)
        tl, tr = idx[:-1, :-1], idx[:-1, 1:]
        bl, br = idx[1:, :-1], idx[1:, 1:]
        dtl, dtr, dbl, dbr = d[:-1, :-1], d[:-1, 1:], d[1:, :-1], d[1:, 1:]
        ok = v_ok[:-1, :-1] & v_ok[:-1, 1:] & v_ok[1:, :-1] & v_ok[1:, 1:]

        def jump(a, b):
            return np.abs(a - b) / np.maximum(np.minimum(a, b), 1e-6) > depth_jump

        edge_top, edge_bottom = jump(dtl, dtr), jump(dbl, dbr)
        edge_left, edge_right = jump(dtl, dbl), jump(dtr, dbr)
        diag_a, diag_b = jump(dtl, dbr), jump(dtr, dbl)
        use_a = np.abs(dtl - dbr) <= np.abs(dtr - dbl)       # split tl-br, else tr-bl

        # Winding chosen so the normals face the camera (see the module notes on frames).
        fa1 = np.stack([tl, bl, br], -1); fa2 = np.stack([tl, br, tr], -1)
        fb1 = np.stack([tl, bl, tr], -1); fb2 = np.stack([tr, bl, br], -1)
        keep_a1 = ok & use_a & ~edge_left & ~edge_bottom & ~diag_a
        keep_a2 = ok & use_a & ~edge_top & ~edge_right & ~diag_a
        keep_b1 = ok & ~use_a & ~edge_left & ~edge_top & ~diag_b
        keep_b2 = ok & ~use_a & ~edge_right & ~edge_bottom & ~diag_b
        faces = np.concatenate([fa1[keep_a1], fa2[keep_a2], fb1[keep_b1], fb2[keep_b2]], axis=0)
        if len(faces) == 0:
            log(f"[Export] view {i}: no valid surface, skipped")
            continue

        # Compact to the vertices that are actually referenced.
        used, remap = np.unique(faces, return_inverse=True)
        faces = remap.reshape(-1, 3).astype(np.int32)
        vertices = world[used].astype(np.float32)
        u = (gx.reshape(-1)[used] + 0.5) / W
        v = (gy.reshape(-1)[used] + 0.5) / H
        uv = np.stack([u, 1.0 - v], axis=-1).astype(np.float32)

        texture, vcol = None, None
        if inputs:
            texture = _aligned_texture(inputs[i], W, H, max_texture)
        else:
            vcol = _colors_from_points(vertices, res.points)
        meshes.append(ViewMesh(f"view_{i:02d}", vertices, faces, uv, texture, vcol))
        log(f"[Export] view {i}: {len(vertices):,} vertices, {len(faces):,} triangles")
    if not meshes:
        raise RuntimeError("no view produced a mesh (all depth invalid or masked)")
    return meshes


def write_mesh_glb(meshes: list[ViewMesh], out: Path, log: Log = _log_default) -> Path:
    import trimesh
    from trimesh.visual.material import PBRMaterial
    scene = trimesh.Scene()
    for m in meshes:
        verts = m.vertices @ TO_YUP.T
        if m.texture is not None:
            mat = PBRMaterial(baseColorTexture=m.texture, doubleSided=True, metallicFactor=0.0, roughnessFactor=1.0)
            visual = trimesh.visual.TextureVisuals(uv=m.uv, material=mat)
            tm = trimesh.Trimesh(verts, m.faces, visual=visual, process=False)
        else:
            tm = trimesh.Trimesh(verts, m.faces, process=False)
            if m.vertex_colors is not None:
                tm.visual.vertex_colors = np.concatenate(
                    [m.vertex_colors, np.full((len(m.vertex_colors), 1), 255, np.uint8)], axis=1)
        scene.add_geometry(tm, node_name=m.name, geom_name=m.name)
    out = Path(out)
    out.write_bytes(scene.export(file_type="glb"))
    n_v = sum(len(m.vertices) for m in meshes)
    n_f = sum(len(m.faces) for m in meshes)
    log(f"[Export] {out.name}: {len(meshes)} views, {n_v:,} vertices, {n_f:,} triangles, {out.stat().st_size / 2**20:.1f} MiB")
    return out


def write_mesh_obj_zip(meshes: list[ViewMesh], out: Path, log: Log = _log_default) -> Path:
    """OBJ + MTL + one JPEG per view, zipped. Written by hand: the format is
    trivial and this keeps one material per view without a library detour."""
    obj = ["# HY-World 2.0 relief mesh (y-up)", "mtllib scene.mtl", ""]
    mtl = []
    textures: dict[str, bytes] = {}
    offset = 1
    for m in meshes:
        verts = m.vertices @ TO_YUP.T
        obj.append(f"o {m.name}")
        obj.append(f"usemtl {m.name}")
        if m.texture is not None or m.vertex_colors is None:
            obj.extend(f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts)
        else:
            obj.extend(f"v {x:.5f} {y:.5f} {z:.5f} {r / 255:.4f} {g / 255:.4f} {b / 255:.4f}"
                       for (x, y, z), (r, g, b) in zip(verts, m.vertex_colors))
        if m.texture is not None:
            obj.extend(f"vt {u:.5f} {v:.5f}" for u, v in m.uv)
            obj.extend(f"f {a + offset}/{a + offset} {b + offset}/{b + offset} {c + offset}/{c + offset}"
                       for a, b, c in m.faces)
            buf = io.BytesIO()
            m.texture.convert("RGB").save(buf, format="JPEG", quality=88)
            textures[f"{m.name}.jpg"] = buf.getvalue()
            mtl += [f"newmtl {m.name}", "Ka 1.0 1.0 1.0", "Kd 1.0 1.0 1.0", "Ks 0.0 0.0 0.0", "d 1.0",
                    "illum 1", f"map_Kd {m.name}.jpg", ""]
        else:
            obj.extend(f"f {a + offset} {b + offset} {c + offset}" for a, b, c in m.faces)
            mtl += [f"newmtl {m.name}", "Kd 0.8 0.8 0.8", "illum 1", ""]
        obj.append("")
        offset += len(verts)
    out = Path(out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("scene.obj", "\n".join(obj))
        z.writestr("scene.mtl", "\n".join(mtl))
        for name, data in textures.items():
            z.writestr(name, data)
    log(f"[Export] {out.name}: {len(meshes)} objects, {out.stat().st_size / 2**20:.1f} MiB")
    return out


# ============================================================================
# three.js JSON Object format
# ============================================================================
def _frustum_segments(cams: dict, scale: float) -> np.ndarray:
    """Line segments (pairs of points, world frame) drawing every camera frustum."""
    intr = {str(e["camera_id"]): np.asarray(e["matrix"], np.float64) for e in cams.get("intrinsics", [])}
    segs = []
    for e in cams.get("extrinsics", []):
        K = intr.get(str(e["camera_id"]))
        if K is None:
            continue
        c2w = np.asarray(e["matrix"], np.float64)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        w, h, d = 2 * cx, 2 * cy, scale
        corners = np.array([[0, 0, 0],
                            [(0 - cx) / fx * d, (0 - cy) / fy * d, d], [(w - cx) / fx * d, (0 - cy) / fy * d, d],
                            [(w - cx) / fx * d, (h - cy) / fy * d, d], [(0 - cx) / fx * d, (h - cy) / fy * d, d]])
        corners = corners @ c2w[:3, :3].T + c2w[:3, 3]
        for a, b in ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)):
            segs.append(corners[a]); segs.append(corners[b])
    return np.asarray(segs, np.float64).reshape(-1, 3)


def write_three_json(res: SceneResult, out: Path, *, max_points: int = 0, log: Log = _log_default) -> Path:
    """The scene as a three.js JSON object (``THREE.ObjectLoader().parse``):
    the point cloud as ``Points`` with per-vertex colour and the camera
    frusta as ``LineSegments``, y-up."""
    geometries, materials, children = [], [], []

    def uid() -> str:
        return str(uuid.uuid4())

    radius = 1.0
    if res.points is not None:
        xyz, rgb = load_points(res.points)
        if max_points and len(xyz) > max_points:
            pick = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
            xyz, rgb = xyz[pick], (rgb[pick] if rgb is not None else None)
        xyz = (xyz @ TO_YUP.T).astype(np.float32)
        centre = xyz.mean(axis=0) if len(xyz) else np.zeros(3)
        radius = float(np.linalg.norm(xyz - centre, axis=1).max()) if len(xyz) else 1.0
        g_id, m_id = uid(), uid()
        attributes = {"position": {"itemSize": 3, "type": "Float32Array", "normalized": False,
                                   "array": np.round(xyz, 5).reshape(-1).tolist()}}
        if rgb is not None:
            attributes["color"] = {"itemSize": 3, "type": "Uint8Array", "normalized": True,
                                   "array": rgb.reshape(-1).tolist()}
        geometries.append({"uuid": g_id, "type": "BufferGeometry", "data": {"attributes": attributes}})
        materials.append({"uuid": m_id, "type": "PointsMaterial", "size": radius * 0.004,
                          "sizeAttenuation": True, "vertexColors": rgb is not None, "color": 16777215})
        children.append({"uuid": uid(), "type": "Points", "name": "points", "layers": 1,
                         "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                         "geometry": g_id, "material": m_id})

    cams = res.camera_data()
    if cams.get("extrinsics"):
        segs = _frustum_segments(cams, radius * 0.08) @ TO_YUP.T
        g_id, m_id = uid(), uid()
        geometries.append({"uuid": g_id, "type": "BufferGeometry", "data": {"attributes": {
            "position": {"itemSize": 3, "type": "Float32Array", "normalized": False,
                         "array": np.round(segs, 5).reshape(-1).tolist()}}}})
        materials.append({"uuid": m_id, "type": "LineBasicMaterial", "color": 0x50A0FF})
        children.append({"uuid": uid(), "type": "LineSegments", "name": "cameras", "layers": 1,
                         "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                         "geometry": g_id, "material": m_id})

    doc = {
        "metadata": {"version": 4.6, "type": "Object", "generator": "HY-World 2.0 ROCm/MPS port (export3d.py)"},
        "geometries": geometries, "materials": materials,
        "object": {"uuid": uid(), "type": "Scene", "name": res.dir.parent.name or "scene", "layers": 1,
                   "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], "up": [0, 1, 0],
                   "children": children},
    }
    out = Path(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    log(f"[Export] {out.name}: {len(children)} objects, {out.stat().st_size / 2**20:.1f} MiB")
    return out


# ============================================================================
# Panorama formats
# ============================================================================
_CUBE_FACES = ("px", "nx", "py", "ny", "pz", "nz")


def _cube_face_dirs(face: str, n: int) -> np.ndarray:
    """Unit directions for every pixel of a cube face (OpenGL / three.js layout,
    rows top to bottom)."""
    a, b = np.meshgrid((np.arange(n) + 0.5) / n * 2 - 1, (np.arange(n) + 0.5) / n * 2 - 1)
    one = np.ones_like(a)
    d = {"px": (one, -b, -a), "nx": (-one, -b, a), "py": (a, one, b),
         "ny": (a, -one, -b), "pz": (a, -b, one), "nz": (-a, -b, -one)}[face]
    v = np.stack(d, axis=-1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _equirect_uv(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """three.js ``equirectUv``: u from atan2(z, x), v = 0 at the top row."""
    u = np.arctan2(dirs[..., 2], dirs[..., 0]) / (2 * math.pi) + 0.5
    v = 0.5 - np.arcsin(np.clip(dirs[..., 1], -1, 1)) / math.pi
    return u, v


def _cube_lookup_dirs(face: str, n: int) -> np.ndarray:
    """Where to sample the panorama for every pixel of a cube face.

    Cube maps are defined for a viewer *outside* the cube, so an engine that
    draws one as a skybox from the inside mirrors it -- three.js samples a
    CubeTexture at (-x, y, z). Mirroring x here makes the six faces show
    exactly what ``EquirectangularReflectionMapping`` shows for the same
    panorama (checked side by side in the browser).
    """
    d = _cube_face_dirs(face, n).copy()
    d[..., 0] *= -1.0
    return d



def write_cubemap_zip(pano_path: Path, out: Path, *, face_size: int = 1024, log: Log = _log_default) -> Path:
    """Six cube faces named px/nx/py/ny/pz/nz (``THREE.CubeTextureLoader`` order)."""
    import cv2
    pano = np.asarray(Image.open(pano_path).convert("RGB"))
    H, W = pano.shape[:2]
    padded = np.concatenate([pano, pano[:, :1]], axis=1)      # wrap the seam for bilinear sampling
    out = Path(out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for face in _CUBE_FACES:
            u, v = _equirect_uv(_cube_lookup_dirs(face, face_size))
            map_x = (u * W - 0.5).astype(np.float32)
            map_y = (v * H - 0.5).astype(np.float32)
            img = cv2.remap(padded, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="PNG")
            z.writestr(f"{face}.png", buf.getvalue())
        z.writestr("README.txt",
                   "Cube map faces in OpenGL / three.js order: px nx py ny pz nz.\n"
                   "three.js:  new THREE.CubeTextureLoader().load(['px.png','nx.png','py.png','ny.png','pz.png','nz.png'])\n"
                   "Same orientation as THREE.EquirectangularReflectionMapping on the source panorama.\n")
    log(f"[Export] {out.name}: 6 faces of {face_size}x{face_size}")
    return out


def write_pano_sphere_glb(pano_path: Path, out: Path, *, max_width: int = 4096, segments: int = 96,
                          log: Log = _log_default) -> Path:
    """An inward-facing sphere with the panorama as its texture, so any glTF
    viewer (three.js, Blender, Windows 3D Viewer, ...) shows the 360 image
    from the centre. Orientation matches three.js' equirectangular mapping."""
    import trimesh
    from trimesh.visual.material import PBRMaterial
    img = Image.open(pano_path).convert("RGB")
    if img.width > max_width:
        img = img.resize((max_width, max(1, round(img.height * max_width / img.width))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    tex = Image.open(io.BytesIO(buf.getvalue()))

    n_u, n_v = segments, segments // 2
    u = np.linspace(0, 1, n_u + 1)
    v = np.linspace(0, 1, n_v + 1)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    theta, phi = 2 * math.pi * uu, math.pi * vv
    x = -np.cos(theta) * np.sin(phi)
    y = np.cos(phi)
    z = -np.sin(theta) * np.sin(phi)
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3) * 10.0
    uv = np.stack([uu, 1.0 - vv], axis=-1).reshape(-1, 2)     # OBJ convention; trimesh flips for glTF
    idx = np.arange((n_u + 1) * (n_v + 1)).reshape(n_u + 1, n_v + 1)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    mesh = trimesh.Trimesh(verts, faces, process=False)
    # Face the triangles inwards: flip if the first normal points away from the centre.
    if np.dot(mesh.face_normals[0], mesh.triangles_center[0]) > 0:
        faces = faces[:, ::-1]
    mat = PBRMaterial(baseColorTexture=tex, doubleSided=True, metallicFactor=0.0, roughnessFactor=1.0,
                      emissiveFactor=[0.0, 0.0, 0.0])
    mesh = trimesh.Trimesh(verts, faces, visual=trimesh.visual.TextureVisuals(uv=uv, material=mat), process=False)
    out = Path(out)
    out.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
    log(f"[Export] {out.name}: {len(verts):,} vertices, texture {tex.width}x{tex.height}, {out.stat().st_size / 2**20:.1f} MiB")
    return out


# ============================================================================
# HTML5 + three.js scenes
# ============================================================================
_SERVE_PY = r'''#!/usr/bin/env python3
"""Serve this folder on localhost and open the scene in the default browser.

Browsers do not let a page opened from a file:// URL read the files next to
it (the point cloud, the Gaussians, the cameras), so the scene is served over
HTTP instead. Nothing leaves your machine: the server listens on 127.0.0.1
only. Close this window (or press Ctrl+C) to stop it.
"""
import functools, http.server, os, socketserver, threading, webbrowser

os.chdir(os.path.dirname(os.path.abspath(__file__)))
Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=".")
Handler.extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                          ".splat": "application/octet-stream", ".ply": "application/octet-stream",
                          ".glb": "model/gltf-binary", ".json": "application/json"}
socketserver.TCPServer.allow_reuse_address = True
for port in range(8765, 8800):
    try:
        httpd = socketserver.TCPServer(("127.0.0.1", port), Handler)
        break
    except OSError:
        continue
else:
    raise SystemExit("no free port between 8765 and 8799")
url = f"http://127.0.0.1:{port}/index.html"
print(f"Serving the scene at {url}  (Ctrl+C to stop)")
threading.Timer(0.6, lambda: webbrowser.open(url)).start()
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
'''

_OPEN_BAT = r'''@echo off
cd /d "%~dp0"
where python >nul 2>nul && (python serve.py & goto :done)
where py >nul 2>nul && (py -3 serve.py & goto :done)
echo Python 3 was not found. Install it from https://www.python.org/ and run serve.py,
echo or open index.html through any local web server.
pause
:done
'''

_OPEN_COMMAND = r'''#!/bin/bash
cd "$(dirname "$0")"
exec python3 serve.py
'''

_BUNDLE_README = """HY-World 2.0 - exported scene
=============================

index.html      the viewer (three.js + GaussianSplats3D, loaded from a CDN)
scene/          the data: Gaussians (.splat), point cloud (.ply), cameras
                (.json), relief mesh (.glb), panorama (.png) -- whichever the
                scene has
serve.py        a tiny local web server that opens the viewer in your browser

How to open
-----------
Windows:  double-click  "Open scene (Windows).bat"
macOS:    double-click  "Open scene (macOS).command"   (first time: chmod +x it)
Linux:    python3 serve.py

Why a server: browsers refuse to read local files from a page opened as
file://, so index.html cannot load scene/ on its own. Any static web server
works (python -m http.server, nginx, GitHub Pages, ...); the page needs no
special headers.

Controls: drag to orbit, wheel to zoom, right-drag to pan. The checkboxes in
the corner switch the layers (Gaussians / points / mesh / cameras).
"""


def _viewer_template() -> str:
    return VIEWER_HTML.read_text(encoding="utf-8")


def _inject_config(html: str, cfg: dict, title: str) -> str:
    """Embed the scene description into the viewer page, replacing the query
    string it would otherwise read."""
    script = "<script>window.HYWORLD_SCENE = " + json.dumps(cfg, separators=(",", ":")) + ";</script>"
    html = html.replace("<!-- HYWORLD_SCENE -->", script, 1)
    return re.sub(r"<title>.*?</title>", f"<title>{title}</title>", html, count=1, flags=re.S)


def _data_url(path: Path, mime: str = "application/octet-stream") -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _scene_assets(res: SceneResult, tmp: Path, *, splat: bool, points: bool, mesh: bool, mesh_step: int,
                  log: Log) -> dict[str, Path]:
    """The files a web scene is made of, written under ``tmp``."""
    assets: dict[str, Path] = {}
    if splat and res.gaussians is not None:
        assets["splat"] = write_splat(res.gaussians, tmp / "scene.splat", log)
    if points and res.points is not None:
        assets["points"] = res.points
    if res.cameras is not None:
        assets["cams"] = res.cameras
    if mesh and res.depths:
        try:
            assets["glb"] = write_mesh_glb(build_relief_meshes(res, step=mesh_step, log=log), tmp / "mesh.glb", log)
        except Exception as e:  # noqa: BLE001 - the mesh is optional in a web scene
            log(f"[Export] mesh skipped: {e}")
    if res.panorama is not None:
        assets["pano"] = res.panorama
    return assets


def write_scene_bundle(res: SceneResult, out_zip: Path, *, title: str = "HY-World 2.0 scene",
                       splat: bool = True, points: bool = True, mesh: bool = True, mesh_step: int = 2,
                       log: Log = _log_default) -> Path:
    """A folder (zipped) with the viewer, the data and a one-click local server."""
    out_zip = Path(out_zip)
    with tempfile.TemporaryDirectory(prefix="hyworld_web_") as td:
        tmp = Path(td)
        assets = _scene_assets(res, tmp, splat=splat, points=points, mesh=mesh, mesh_step=mesh_step, log=log)
        names = {"splat": "scene.splat", "points": "points.ply", "cams": "cameras.json", "glb": "mesh.glb",
                 "pano": "panorama.png"}
        cfg = {k: f"scene/{names[k]}" for k in assets}
        cfg["title"] = title
        cfg["glb_yup"] = True
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("index.html", _inject_config(_viewer_template(), cfg, title))
            for k, p in assets.items():
                z.write(p, f"scene/{names[k]}")
            z.writestr("serve.py", _SERVE_PY)
            z.writestr("Open scene (Windows).bat", _OPEN_BAT.replace("\n", "\r\n"))
            info = zipfile.ZipInfo("Open scene (macOS).command")
            info.external_attr = 0o755 << 16
            z.writestr(info, _OPEN_COMMAND)
            z.writestr("README.txt", _BUNDLE_README)
    log(f"[Export] {out_zip.name}: {', '.join(assets)} -- {out_zip.stat().st_size / 2**20:.1f} MiB")
    return out_zip


def write_scene_single_html(res: SceneResult, out_html: Path, *, title: str = "HY-World 2.0 scene",
                            splat: bool = True, points: bool = True, mesh: bool = True, mesh_step: int = 2,
                            max_embed_mb: float = 200.0, log: Log = _log_default) -> Path:
    """One HTML file with the data embedded as base64, so it opens from disk
    with a double-click. Gaussians are included too, but browsers only run
    the splat sorter's worker on http(s) pages, so from ``file://`` the page
    shows the mesh, the points and the cameras and says why.

    Assets are embedded largest-last until ``max_embed_mb`` would be exceeded;
    what is left out is reported through ``log``.
    """
    out_html = Path(out_html)
    with tempfile.TemporaryDirectory(prefix="hyworld_html_") as td:
        tmp = Path(td)
        assets = _scene_assets(res, tmp, splat=splat, points=points, mesh=mesh, mesh_step=mesh_step, log=log)
        cfg: dict = {"title": title, "glb_yup": True, "embedded": True}
        budget = max_embed_mb * 2**20
        used = 0
        # Small things first so the important ones survive the budget.
        for k, p in sorted(assets.items(), key=lambda kv: kv[1].stat().st_size):
            size = p.stat().st_size * 4 / 3
            if used + size > budget:
                log(f"[Export] {k} ({p.stat().st_size / 2**20:.0f} MiB) left out of {out_html.name}: "
                    f"over the {max_embed_mb:.0f} MiB embed budget -- use the zip bundle for it")
                continue
            used += size
            if k == "cams":
                cfg[k] = json.loads(p.read_text(encoding="utf-8"))
            elif k == "pano":
                cfg[k] = _data_url(p, "image/png")
            else:
                cfg[k] = _data_url(p)
        out_html.write_text(_inject_config(_viewer_template(), cfg, title), encoding="utf-8")
    log(f"[Export] {out_html.name}: {out_html.stat().st_size / 2**20:.1f} MiB")
    return out_html


def write_pano_bundle(pano_path: Path, out_zip: Path, *, title: str = "HY-World 2.0 panorama",
                      log: Log = _log_default) -> Path:
    out_zip = Path(out_zip)
    cfg = {"pano": "scene/panorama.png", "title": title}
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("index.html", _inject_config(_viewer_template(), cfg, title))
        z.write(pano_path, "scene/panorama.png")
        z.writestr("serve.py", _SERVE_PY)
        z.writestr("Open scene (Windows).bat", _OPEN_BAT.replace("\n", "\r\n"))
        info = zipfile.ZipInfo("Open scene (macOS).command")
        info.external_attr = 0o755 << 16
        z.writestr(info, _OPEN_COMMAND)
        z.writestr("README.txt", _BUNDLE_README)
    log(f"[Export] {out_zip.name}: {out_zip.stat().st_size / 2**20:.1f} MiB")
    return out_zip


def write_pano_single_html(pano_path: Path, out_html: Path, *, title: str = "HY-World 2.0 panorama",
                           log: Log = _log_default) -> Path:
    """A single HTML file with the panorama embedded: works from file://."""
    out_html = Path(out_html)
    mime = "image/jpeg" if Path(pano_path).suffix.lower() in (".jpg", ".jpeg") else "image/png"
    cfg = {"pano": _data_url(Path(pano_path), mime), "title": title, "embedded": True}
    out_html.write_text(_inject_config(_viewer_template(), cfg, title), encoding="utf-8")
    log(f"[Export] {out_html.name}: {out_html.stat().st_size / 2**20:.1f} MiB")
    return out_html


# ============================================================================
# Catalogue used by the UI and the CLI
# ============================================================================
# key -> (file name, description). The description doubles as the UI help.
SCENE_FORMATS: dict[str, tuple[str, str]] = {
    "gaussians_ply": ("gaussians.ply", "3D Gaussian Splatting, INRIA .ply (as produced)"),
    "splat": ("scene.splat", "3D Gaussian Splatting, compact .splat (SuperSplat, Unity/Unreal, web viewers)"),
    "points_ply": ("points.ply", "Point cloud .ply (as produced)"),
    "points_glb": ("points.glb", "Point cloud, glTF 2.0 (three.js, Blender, Godot)"),
    "points_xyz": ("points.xyz", "Point cloud, ASCII x y z r g b (CloudCompare, MeshLab)"),
    "mesh_glb": ("mesh.glb", "Textured mesh from the depth maps, glTF 2.0"),
    "mesh_obj": ("mesh_obj.zip", "Textured mesh, OBJ + MTL + textures"),
    "three_json": ("scene.three.json", "three.js JSON Object format (points + cameras)"),
    "cameras": ("camera_params.json", "Cameras: intrinsics + camera-to-world matrices"),
    "web_zip": ("scene_web.zip", "HTML5 + three.js scene, folder with a one-click local server"),
    "web_html": ("scene.html", "HTML5 + three.js scene, single self-contained file"),
    "everything": ("scene_all.zip", "Everything above in one archive"),
}

PANO_FORMATS: dict[str, tuple[str, str]] = {
    "png": ("panorama.png", "Equirectangular PNG (as produced)"),
    "jpg": ("panorama.jpg", "Equirectangular JPEG (smaller)"),
    "cubemap": ("cubemap.zip", "Cube map, 6 faces (three.js CubeTextureLoader order, skyboxes)"),
    "sphere_glb": ("panorama.glb", "Textured sphere, glTF 2.0 (any glTF viewer)"),
    "web_html": ("panorama.html", "360-degree HTML5 + three.js viewer, single file"),
    "web_zip": ("panorama_web.zip", "360-degree HTML5 + three.js viewer, folder with a local server"),
}


def export_scene(result_dir, fmt: str, out_dir, *, mesh_step: int = 2, title: Optional[str] = None,
                 log: Log = _log_default) -> Path:
    """Write one export of a reconstruction result and return its path."""
    res = SceneResult.open(result_dir)
    if res.is_empty:
        raise FileNotFoundError(f"nothing to export in {res.dir}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name, _ = SCENE_FORMATS[fmt]
    out = out_dir / name
    title = title or f"HY-World 2.0 - {res.dir.parent.name}"
    t0 = time.perf_counter()

    if fmt == "gaussians_ply":
        _require(res.gaussians, "gaussians.ply"); out = res.gaussians
    elif fmt == "points_ply":
        _require(res.points, "points.ply"); out = res.points
    elif fmt == "cameras":
        _require(res.cameras, "camera_params.json"); out = res.cameras
    elif fmt == "splat":
        _require(res.gaussians, "gaussians.ply"); write_splat(res.gaussians, out, log)
    elif fmt == "points_glb":
        _require(res.points, "points.ply"); write_points_glb(res.points, out, log)
    elif fmt == "points_xyz":
        _require(res.points, "points.ply"); write_points_xyz(res.points, out, log)
    elif fmt == "mesh_glb":
        write_mesh_glb(build_relief_meshes(res, step=mesh_step, log=log), out, log)
    elif fmt == "mesh_obj":
        write_mesh_obj_zip(build_relief_meshes(res, step=mesh_step, log=log), out, log)
    elif fmt == "three_json":
        write_three_json(res, out, log=log)
    elif fmt == "web_zip":
        write_scene_bundle(res, out, title=title, mesh_step=mesh_step, log=log)
    elif fmt == "web_html":
        write_scene_single_html(res, out, title=title, mesh_step=mesh_step, log=log)
    elif fmt == "everything":
        _export_everything(res, out, title, mesh_step, log)
    else:
        raise ValueError(f"unknown scene format: {fmt}")
    log(f"[Export] done in {time.perf_counter() - t0:.1f} s -> {out}")
    return out


def _export_everything(res: SceneResult, out_zip: Path, title: str, mesh_step: int, log: Log) -> None:
    with tempfile.TemporaryDirectory(prefix="hyworld_all_") as td:
        tmp = Path(td)
        files: list[tuple[Path, str]] = []
        for k in ("gaussians_ply", "points_ply", "cameras", "splat", "points_glb", "points_xyz",
                  "mesh_glb", "mesh_obj", "three_json", "web_zip", "web_html"):
            try:
                p = export_scene(res.dir, k, tmp, mesh_step=mesh_step, title=title, log=log)
                files.append((p, SCENE_FORMATS[k][0]))
            except Exception as e:  # noqa: BLE001 - one missing format must not sink the archive
                log(f"[Export] {k} skipped: {e}")
        extras = [p for p in (res.video,) if p]
        extras += sorted((res.dir / "depth").glob("*.png")) + sorted((res.dir / "normal").glob("*.png"))
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for p, name in files:
                z.write(p, name)
            for p in extras:
                z.write(p, str(p.relative_to(res.dir)).replace("\\", "/"))
    log(f"[Export] {out_zip.name}: {out_zip.stat().st_size / 2**20:.1f} MiB")


def export_panorama(pano_path, fmt: str, out_dir, *, title: Optional[str] = None, face_size: int = 1024,
                    log: Log = _log_default) -> Path:
    pano_path = Path(pano_path)
    if not pano_path.is_file():
        raise FileNotFoundError(f"panorama not found: {pano_path}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name, _ = PANO_FORMATS[fmt]
    out = out_dir / name
    title = title or f"HY-World 2.0 - {pano_path.parent.name}"
    t0 = time.perf_counter()
    if fmt == "png":
        if pano_path.suffix.lower() != ".png":
            Image.open(pano_path).convert("RGB").save(out)
        else:
            out = pano_path
    elif fmt == "jpg":
        Image.open(pano_path).convert("RGB").save(out, quality=92)
    elif fmt == "cubemap":
        write_cubemap_zip(pano_path, out, face_size=face_size, log=log)
    elif fmt == "sphere_glb":
        write_pano_sphere_glb(pano_path, out, log=log)
    elif fmt == "web_html":
        write_pano_single_html(pano_path, out, title=title, log=log)
    elif fmt == "web_zip":
        write_pano_bundle(pano_path, out, title=title, log=log)
    else:
        raise ValueError(f"unknown panorama format: {fmt}")
    log(f"[Export] done in {time.perf_counter() - t0:.1f} s -> {out}")
    return out


def _require(path: Optional[Path], what: str) -> None:
    if path is None:
        raise FileNotFoundError(f"{what} is missing from the result directory")


# ============================================================================
# Command line
# ============================================================================
def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Export a HY-World 2.0 result into other formats.")
    ap.add_argument("source", help="a reconstruction result directory (with gaussians.ply / points.ply) "
                                   "or a panorama image")
    ap.add_argument("--format", "-f", action="append", default=[],
                    help="format key (repeatable). Scene: " + ", ".join(SCENE_FORMATS)
                         + ". Panorama: " + ", ".join(PANO_FORMATS))
    ap.add_argument("--all", action="store_true", help="every format that applies")
    ap.add_argument("--out", "-o", default=None, help="output directory (default: <source>/export)")
    ap.add_argument("--mesh-step", type=int, default=2, help="mesh vertex spacing in pixels (1 = full detail)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    src = Path(args.source)
    is_pano = src.is_file() and src.suffix.lower() in IMAGE_EXT
    catalogue = PANO_FORMATS if is_pano else SCENE_FORMATS
    formats = list(catalogue) if args.all else args.format
    if is_pano and args.all:
        formats = [k for k in formats if k != "png"]
    if not is_pano and args.all:
        formats = [k for k in formats if k not in ("everything", "gaussians_ply", "points_ply", "cameras")]
    if not formats:
        ap.error("pick at least one --format, or --all")
    out_dir = Path(args.out) if args.out else (src.parent if is_pano else src) / "export"
    for k in formats:
        if k not in catalogue:
            print(f"unknown format for this source: {k}")
            return 2
        if is_pano:
            export_panorama(src, k, out_dir)
        else:
            export_scene(src, k, out_dir, mesh_step=args.mesh_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
