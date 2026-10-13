"""Episode metrics against the scene ground truth (pure numpy, no ROS).

The ground truth is the list of obstacle boxes in ``config/scene_ground_truth.yaml``
(the poses of ``worlds/obstacle_world.sdf``, world == base_link). The robot is
represented by the same sample points the safety filter uses (TCP, wrist and
arm points from ``UR5eKinematics``) plus the camera body.

Metrics per episode (see ``EpisodeRecorder.summary``):
  safety      min true clearance robot<->boxes, collision flag
  control     outcome, time to goal, path length, max deviation from the line,
              time spent holding (each non-"ok" status)
  perception  detection age (image stamp -> publication), YOLO inference time,
              frames with a stop request, detections per source (yolo/depth)
  localization  distance of each track's surface point to the nearest GT box
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

CAMERA_OFFSET_TOOL0 = np.array([0.0, 0.0, 0.08])  # camera_mount_joint in the URDF


@dataclass(frozen=True)
class GroundTruthBox:
    name: str
    label: str
    center: np.ndarray
    size: np.ndarray
    class_name: str = ""


def load_ground_truth(path: Path | str) -> list[GroundTruthBox]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    return [
        GroundTruthBox(
            name=str(item["name"]),
            label=str(item.get("label", "")),
            center=np.asarray(item["center_m"], dtype=float),
            size=np.asarray(item["size_m"], dtype=float),
            class_name=str(item.get("class_name", item.get("label", ""))),
        )
        for item in document.get("obstacles", [])
    ]


def box_signed_distance(points: np.ndarray, center: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Signed distance of each point to an axis-aligned box (negative inside)."""
    q = np.abs(np.atleast_2d(points) - center) - 0.5 * size
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def robot_points(kin, q: np.ndarray) -> np.ndarray:
    transform = kin.fk(q)
    camera = transform[:3, 3] + transform[:3, :3] @ CAMERA_OFFSET_TOOL0
    return np.vstack((transform[:3, 3], kin.link_points(q), camera))


def true_clearance(kin, q: np.ndarray, boxes: Sequence[GroundTruthBox]) -> Optional[float]:
    if not boxes:
        return None
    points = robot_points(kin, q)
    return float(min(np.min(box_signed_distance(points, b.center, b.size)) for b in boxes))


def nearest_box(point: Sequence[float], boxes: Sequence[GroundTruthBox]):
    """(box, signed distance) of the box closest to ``point``."""
    best, best_d = None, math.inf
    for box in boxes:
        d = float(box_signed_distance(np.asarray(point, dtype=float), box.center, box.size)[0])
        if abs(d) < abs(best_d):
            best, best_d = box, d
    return best, best_d


def _stats(values: Sequence[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(array.mean()), 4),
        "p95": round(float(np.percentile(array, 95)), 4),
        "max": round(float(array.max()), 4),
    }


class EpisodeRecorder:
    """Accumulates one episode; feed it messages, then call ``summary``."""

    def __init__(self, kin, boxes: Sequence[GroundTruthBox], start_s: float, task_state: dict) -> None:
        self.kin = kin
        self.boxes = list(boxes)
        self.start_s = start_s
        self.start = np.asarray(task_state.get("start", [0, 0, 0]), dtype=float)
        self.goal = np.asarray(task_state.get("goal", [0, 0, 0]), dtype=float)
        self.mode = task_state.get("mode")
        self.duration_s = float(task_state.get("episode_duration_s", 0.0))
        self.min_clearance = math.inf
        self.min_estimated_clearance = math.inf
        self.path_length = 0.0
        self.max_deviation = 0.0
        self.last_tcp: Optional[np.ndarray] = None
        self.status_time: Counter = Counter()
        self.last_status_s: Optional[float] = None
        self.last_status: Optional[str] = None
        self.reached_s: Optional[float] = None
        self.detection_ages: list[float] = []
        self.inference_ms: list[float] = []
        self.frames = 0
        self.stop_frames = 0
        self.sources: Counter = Counter()
        self.surface_errors: list[float] = []
        self.surface_errors_by_label: dict[str, list[float]] = {}
        self.track_labels: Counter = Counter()

    def on_joints(self, q: np.ndarray) -> None:
        tcp = self.kin.position(q)
        if self.last_tcp is not None:
            self.path_length += float(np.linalg.norm(tcp - self.last_tcp))
        self.last_tcp = tcp
        line = self.goal - self.start
        norm = float(np.linalg.norm(line))
        if norm > 1e-9:
            axis = line / norm
            rel = tcp - self.start
            self.max_deviation = max(self.max_deviation, float(np.linalg.norm(rel - (rel @ axis) * axis)))
        clearance = true_clearance(self.kin, q, self.boxes)
        if clearance is not None:
            self.min_clearance = min(self.min_clearance, clearance)

    def on_task_state(self, state: dict, now_s: float) -> None:
        status = str(state.get("status"))
        if self.last_status is not None and self.last_status_s is not None:
            self.status_time[self.last_status] += max(now_s - self.last_status_s, 0.0)
        self.last_status, self.last_status_s = status, now_s
        estimated = state.get("min_clearance_m")
        if estimated is not None:
            self.min_estimated_clearance = min(self.min_estimated_clearance, float(estimated))
        if state.get("reached") and self.reached_s is None:
            self.reached_s = now_s

    def on_detections(self, payload: dict) -> None:
        self.frames += 1
        if payload.get("obstacle"):
            self.stop_frames += 1
        if payload.get("age_s") is not None:
            self.detection_ages.append(float(payload["age_s"]))
        if payload.get("inference_ms") is not None:
            self.inference_ms.append(float(payload["inference_ms"]))
        for detection in payload.get("detections", []):
            self.sources[str(detection.get("source", "yolo"))] += 1

    def on_tracks(self, payload: dict) -> None:
        for track in payload.get("tracks", []):
            if not track.get("visible", False):
                continue
            label = str(track.get("label", ""))
            self.track_labels[label] += 1
            if self.boxes and track.get("surface") is not None:
                _, distance = nearest_box(track["surface"], self.boxes)
                self.surface_errors.append(abs(distance))
                self.surface_errors_by_label.setdefault(label, []).append(abs(distance))

    def summary(self, end_s: float, outcome: str) -> dict[str, Any]:
        def finite(value):
            return None if value is None or not math.isfinite(value) else round(float(value), 4)

        return {
            "mode": self.mode,
            "outcome": outcome,
            "time_to_goal_s": None if self.reached_s is None else round(self.reached_s - self.start_s, 3),
            "episode_time_s": round(end_s - self.start_s, 3),
            "collision": bool(math.isfinite(self.min_clearance) and self.min_clearance <= 0.0),
            "min_true_clearance_m": finite(self.min_clearance),
            "min_estimated_clearance_m": finite(self.min_estimated_clearance),
            "path_length_m": round(self.path_length, 4),
            "max_deviation_m": round(self.max_deviation, 4),
            "status_time_s": {k: round(v, 2) for k, v in self.status_time.items()},
            "frames": self.frames,
            "stop_frame_ratio": None if not self.frames else round(self.stop_frames / self.frames, 4),
            "detection_age_s": _stats(self.detection_ages),
            "inference_ms": _stats(self.inference_ms),
            "detections_by_source": dict(self.sources),
            "track_labels": dict(self.track_labels),
            "track_surface_error_m": _stats(self.surface_errors),
            "track_surface_error_by_label_m": {
                label: _stats(values) for label, values in self.surface_errors_by_label.items()
            },
            "start": [round(float(x), 4) for x in self.start],
            "goal": [round(float(x), 4) for x in self.goal],
        }
