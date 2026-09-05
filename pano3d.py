#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Panorama helpers for the HY-World 2.0 web UI.

The "panorama -> 3D" path re-projects an equirectangular image into a ring of
pinhole views and hands them to WorldMirror 2.0 together with the cameras they
were rendered from. WorldMirror is a multi-view model: it wants perspective
images, and it accepts camera priors (``prior_cam_path``) in the same JSON
layout as its own ``camera_params.json`` -- so this module writes that file
too, with the exact intrinsics and extrinsics of the synthetic views. The model
then only has to predict geometry instead of recovering cameras it could have
been told.

Conventions match WorldMirror / OpenCV: camera x right, y down, z forward;
extrinsics are camera-to-world. Every view is rendered from the panorama's
centre, so the extrinsics are pure rotations.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


def rotation_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Camera-to-world rotation for a view turned ``yaw`` right and ``pitch`` up.

    Yaw is a rotation about the world y axis (which points down, so a positive
    yaw turns the view to the right), pitch about the camera x axis. Pitch is
    applied first, in the camera frame, then yaw in the world frame.
    """
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    cy, sy, cp, sp = math.cos(y), math.sin(y), math.cos(p), math.sin(p)
    r_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return r_yaw @ r_pitch


def intrinsics(fov_deg: float, size: int) -> np.ndarray:
    """Pinhole intrinsics of a square ``size`` x ``size`` view with horizontal FOV ``fov_deg``."""
    f = (size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[f, 0.0, size / 2.0], [0.0, f, size / 2.0], [0.0, 0.0, 1.0]])


def equirect_to_perspective(pano: np.ndarray, yaw_deg: float, pitch_deg: float,
                            fov_deg: float, size: int) -> np.ndarray:
    """Sample one pinhole view out of an equirectangular image.

    For every output pixel the camera ray is rotated into the world frame and
    converted to longitude/latitude, which index the panorama. Sampling is
    bilinear; the panorama is padded by one wrapped column so the 360-degree
    seam interpolates correctly instead of clamping.
    """
    h, w = pano.shape[:2]
    k = intrinsics(fov_deg, size)
    rot = rotation_yaw_pitch(yaw_deg, pitch_deg)

    u, v = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    rays = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], axis=-1)
    rays = rays @ rot.T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    lon = np.arctan2(rays[..., 0], rays[..., 2])            # 0 straight ahead, positive to the right
    lat = np.arcsin(np.clip(-rays[..., 1], -1.0, 1.0))      # positive up (image y points down)

    map_x = ((lon / (2.0 * math.pi)) + 0.5) * w - 0.5
    map_y = (0.5 - lat / math.pi) * h - 0.5

    padded = np.concatenate([pano, pano[:, :1]], axis=1)
    return cv2.remap(padded, map_x.astype(np.float32), map_y.astype(np.float32),
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def split_panorama(pano_path: str | Path, out_dir: str | Path, *,
                   n_views: int = 8, fov_deg: float = 90.0, size: int = 768,
                   pitch_rows: Iterable[float] = (0.0,), yaw_offset: float = 0.0
                   ) -> tuple[list[str], str]:
    """Write a ring of perspective views plus a WorldMirror camera-prior file.

    Returns ``(image_paths, prior_cameras_json)``. Views are named
    ``view_NN.png`` and the JSON uses those stems as ``camera_id``, which is
    how ``load_prior_camera`` matches cameras to images.
    """
    pano = np.asarray(Image.open(pano_path).convert("RGB"))
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    k = intrinsics(fov_deg, size)
    paths, extrinsics, intrinsics_ = [], [], []
    idx = 0
    for pitch in pitch_rows:
        for i in range(n_views):
            yaw = yaw_offset + i * 360.0 / n_views
            view = equirect_to_perspective(pano, yaw, pitch, fov_deg, size)
            name = f"view_{idx:02d}"
            path = img_dir / f"{name}.png"
            Image.fromarray(view).save(path)

            c2w = np.eye(4)
            c2w[:3, :3] = rotation_yaw_pitch(yaw, pitch)
            extrinsics.append({"camera_id": name, "matrix": c2w.tolist()})
            intrinsics_.append({"camera_id": name, "matrix": k.tolist()})
            paths.append(str(path))
            idx += 1

    cam_path = out_dir / "prior_cameras.json"
    with open(cam_path, "w", encoding="utf-8") as f:
        json.dump({"num_cameras": idx, "extrinsics": extrinsics, "intrinsics": intrinsics_}, f, indent=2)
    return paths, str(cam_path)
