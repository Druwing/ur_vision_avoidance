"""Class-agnostic near-obstacle detection from the depth image (pure numpy).

YOLO only reports objects of the classes it was trained on, and even those are
missed at short range or in partial views (in Gazebo the camera ended up under
the chair seat with no detection at all). This module is the redundant path:
it finds connected regions of the depth image that are closer than
``max_range_m`` and returns them as bounding boxes, whatever they are.

The image is sampled on a coarse grid (``stride`` pixels) and 4-connected cells
are grouped when their depth differs by less than ``max_depth_step_m``, so an
object in front of the background is not merged with it. Regions already
covered by a YOLO box are skipped by the caller through ``exclude_boxes``.

Limitations: the floor (or any surface) closer than ``max_range_m`` also forms
a region; the localizer drops measurements below ``min_height_m`` in base_link.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DepthBlob:
    bbox_xyxy: tuple[float, float, float, float]  # depth-image pixels
    distance_m: float  # low percentile: conservative range for stop decisions
    area_ratio: float  # fraction of the sampled image covered by the region
    # Representative surface point for 3D localization: the region centroid
    # and the depth measured there. Using the bbox centre with the low
    # percentile instead lifts extended surfaces (a floor strip ends up ~7 cm
    # above the floor) and defeats the localizer's floor filter.
    anchor_uv: tuple[float, float] = (0.0, 0.0)
    anchor_depth_m: float = 0.0


def _inside_any(x: float, y: float, boxes: Sequence[Sequence[float]]) -> bool:
    return any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes)


def find_near_blobs(
    depth_m: np.ndarray,
    max_range_m: float,
    min_range_m: float = 0.05,
    stride: int = 8,
    min_area_ratio: float = 0.02,
    max_depth_step_m: float = 0.10,
    percentile: float = 25.0,
    exclude_boxes: Sequence[Sequence[float]] = (),
    max_excluded_fraction: float = 0.5,
) -> list[DepthBlob]:
    """Return near regions sorted by distance (closest first)."""
    if depth_m is None or depth_m.ndim < 2:
        return []
    stride = max(int(stride), 1)
    height, width = depth_m.shape[:2]
    offset = stride // 2
    grid = np.asarray(depth_m[offset::stride, offset::stride], dtype=np.float32)
    rows, cols = grid.shape[:2]
    if rows == 0 or cols == 0:
        return []
    near = np.isfinite(grid) & (grid >= min_range_m) & (grid <= max_range_m)
    labels = np.zeros((rows, cols), dtype=np.int32)
    total = float(rows * cols)
    blobs: list[DepthBlob] = []
    next_label = 0

    for r0, c0 in zip(*np.nonzero(near)):
        if labels[r0, c0]:
            continue
        next_label += 1
        labels[r0, c0] = next_label
        stack = [(r0, c0)]
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            here = grid[r, c]
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (
                    0 <= rr < rows
                    and 0 <= cc < cols
                    and near[rr, cc]
                    and not labels[rr, cc]
                    and abs(float(grid[rr, cc]) - float(here)) <= max_depth_step_m
                ):
                    labels[rr, cc] = next_label
                    stack.append((rr, cc))

        area_ratio = len(cells) / total
        if area_ratio < min_area_ratio:
            continue
        cell_rows = np.array([cell[0] for cell in cells])
        cell_cols = np.array([cell[1] for cell in cells])
        centres_x = cell_cols * stride + offset
        centres_y = cell_rows * stride + offset
        if exclude_boxes:
            excluded = sum(
                _inside_any(float(x), float(y), exclude_boxes)
                for x, y in zip(centres_x, centres_y)
            )
            if excluded / len(cells) >= max_excluded_fraction:
                continue
        x1 = float(max(cell_cols.min() * stride, 0))
        y1 = float(max(cell_rows.min() * stride, 0))
        x2 = float(min((cell_cols.max() + 1) * stride, width))
        y2 = float(min((cell_rows.max() + 1) * stride, height))
        values = grid[cell_rows, cell_cols]
        distance = float(np.percentile(values, percentile))
        centroid_r, centroid_c = cell_rows.mean(), cell_cols.mean()
        nearest = int(np.argmin((cell_rows - centroid_r) ** 2 + (cell_cols - centroid_c) ** 2))
        around = (np.abs(cell_rows - cell_rows[nearest]) <= 1) & (np.abs(cell_cols - cell_cols[nearest]) <= 1)
        anchor_uv = (float(cell_cols[nearest] * stride + offset), float(cell_rows[nearest] * stride + offset))
        blobs.append(
            DepthBlob(
                (x1, y1, x2, y2), distance, float(area_ratio),
                anchor_uv, float(np.median(values[around])),
            )
        )

    blobs.sort(key=lambda blob: blob.distance_m)
    return blobs
