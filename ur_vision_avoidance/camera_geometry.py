"""Pinhole back-projection helpers (pure numpy, no ROS)."""

from __future__ import annotations

import math

import numpy as np


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def backproject(u, v, depth, fx, fy, cx, cy, depth_is_range: bool = False):
    """Pixel + depth -> camera-frame point and unit ray (optical frame, z forward).

    ``depth`` is the optical-axis distance (z-depth) unless ``depth_is_range``,
    in which case it is the Euclidean range along the ray.
    """
    xn = (u - cx) / fx
    yn = (v - cy) / fy
    ray = np.array([xn, yn, 1.0])
    ray = ray / np.linalg.norm(ray)
    if depth_is_range:
        z = depth / math.sqrt(1.0 + xn * xn + yn * yn)
    else:
        z = depth
    return np.array([xn * z, yn * z, z]), ray, z


def estimate_radius(bbox_xyxy, z, fx, fy, r_min: float, r_max: float) -> float:
    """Half of the mean metric extent of the box, clamped to [r_min, r_max]."""
    x1, y1, x2, y2 = bbox_xyxy
    width_m = (x2 - x1) * z / fx
    height_m = (y2 - y1) * z / fy
    return float(np.clip(0.25 * (width_m + height_m), r_min, r_max))


def detection_to_camera_points(
    bbox_xyxy,
    distance_m,
    fx,
    fy,
    cx,
    cy,
    depth_is_range: bool = False,
    push_to_center: bool = True,
    r_min: float = 0.03,
    r_max: float = 0.40,
    anchor_uv=None,
):
    """Return (surface_cam, center_cam, radius_m) for one detection.

    The sampled depth belongs to the visible surface, so the obstacle centre is
    approximated by pushing the surface point along the ray by the estimated
    radius (``push_to_center``). ``anchor_uv`` replaces the bbox centre as the
    pixel of the surface point (depth-fallback regions provide one).
    """
    x1, y1, x2, y2 = bbox_xyxy
    if anchor_uv is not None:
        u, v = float(anchor_uv[0]), float(anchor_uv[1])
    else:
        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)
    surface, ray, z = backproject(u, v, distance_m, fx, fy, cx, cy, depth_is_range)
    radius = estimate_radius(bbox_xyxy, z, fx, fy, r_min, r_max)
    center = surface + ray * radius if push_to_center else surface.copy()
    return surface, center, radius
