"""Automatic 2D labels from 3D ground-truth boxes (pure numpy, no ROS).

Each ground-truth box (config/scene_ground_truth.yaml) is projected into the
camera with the pinhole intrinsics and the base_link->optical transform at the
image time. The label is the bounding rectangle of the visible corners, clipped
to the image. Occlusion is checked against the depth image: a label is kept
only if enough depth samples inside the rectangle fall within the box's depth
range ("depth support"). This is approximate (box-shaped objects, coarse
occlusion test) but needs no manual annotation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ProjectedBox:
    bbox_xyxy: tuple[float, float, float, float]
    near_z: float  # optical-axis depth of the closest corner
    far_z: float
    inside_ratio: float  # clipped area / unclipped area (truncation)


def box_corners(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    half = 0.5 * np.asarray(size, dtype=float)
    return np.array(
        [np.asarray(center, dtype=float) + half * np.array(s) for s in itertools.product((-1, 1), repeat=3)]
    )


def project_box(
    center, size, rotation_cam_from_base, translation_cam_from_base,
    fx, fy, cx, cy, width, height, near_clip=0.05,
) -> Optional[ProjectedBox]:
    """Project a base_link box into the image; None if not visible."""
    corners = box_corners(center, size) @ np.asarray(rotation_cam_from_base).T + translation_cam_from_base
    z = corners[:, 2]
    if np.all(z <= near_clip):
        return None
    front = corners[z > near_clip]
    u = fx * front[:, 0] / front[:, 2] + cx
    v = fy * front[:, 1] / front[:, 2] + cy
    x1, x2, y1, y2 = float(u.min()), float(u.max()), float(v.min()), float(v.max())
    full_area = max((x2 - x1) * (y2 - y1), 1e-9)
    cx1, cx2 = float(np.clip(x1, 0, width)), float(np.clip(x2, 0, width))
    cy1, cy2 = float(np.clip(y1, 0, height)), float(np.clip(y2, 0, height))
    if cx2 - cx1 < 1.0 or cy2 - cy1 < 1.0:
        return None
    return ProjectedBox(
        (cx1, cy1, cx2, cy2),
        float(max(z.min(), near_clip)),
        float(z.max()),
        float((cx2 - cx1) * (cy2 - cy1) / full_area),
    )


def depth_support(depth_m: np.ndarray, box: ProjectedBox, tolerance_m: float = 0.05, stride: int = 4) -> float:
    """Fraction of depth samples in the rectangle that lie within the box depth range."""
    x1, y1, x2, y2 = (int(round(v)) for v in box.bbox_xyxy)
    patch = np.asarray(depth_m[y1:y2:stride, x1:x2:stride], dtype=np.float32)
    if patch.size == 0:
        return 0.0
    ok = np.isfinite(patch) & (patch >= box.near_z - tolerance_m) & (patch <= box.far_z + tolerance_m)
    return float(ok.mean())


def yolo_line(class_id: int, bbox_xyxy, width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    return (
        f"{class_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
        f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
    )
