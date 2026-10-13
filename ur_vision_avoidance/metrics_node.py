#!/usr/bin/env python3
"""Record one row of metrics per action_adapter episode (methodology step 7).

Inputs : /task/state, /joint_states, /vision/detections, /obstacle/tracks
Ground truth: config/scene_ground_truth.yaml (boxes in base_link).
Output : /metrics/episode (JSON) and, in ``output_dir``, episodes.jsonl plus
         episodes.csv (flattened) - one line per finished episode.

An episode starts when /task/state reports ``episode_active`` with a new start
pose (or elapsed time going back), and ends when the goal is reached, the
adapter stops the episode, or ``timeout_factor`` x episode duration elapses.
See ur_vision_avoidance/metrics.py for the definition of each metric.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .metrics import EpisodeRecorder, load_ground_truth
from .ur_kinematics import UR5eKinematics

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _default_ground_truth() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        path = Path(get_package_share_directory("ur_vision_avoidance")) / "config" / "scene_ground_truth.yaml"
        if path.exists():
            return str(path)
    except Exception:  # noqa: BLE001 - ament is optional outside ROS
        pass
    return str(Path(__file__).resolve().parents[1] / "config" / "scene_ground_truth.yaml")


# Summary fields whose keys vary between episodes are stored as JSON strings so
# that every CSV row has the same columns.
VARIABLE_KEY_FIELDS = (
    "status_time_s", "detections_by_source", "track_labels", "track_surface_error_by_label_m",
)


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in summary.items():
        if isinstance(value, dict) and key not in VARIABLE_KEY_FIELDS:
            for sub_key, sub_value in value.items():
                flat[f"{key}.{sub_key}"] = sub_value
        elif isinstance(value, (dict, list)):
            flat[key] = json.dumps(value)
        else:
            flat[key] = value
    return flat


class MetricsNode(Node):
    def __init__(self) -> None:
        super().__init__("metrics_node")
        self.declare_parameter("ground_truth_file", "")
        self.declare_parameter("output_dir", "/tmp/ur_vision_avoidance_metrics")
        self.declare_parameter("run_label", "")
        self.declare_parameter("timeout_factor", 2.0)
        self.declare_parameter("base_yaw", float(np.pi))

        ground_truth = str(self.get_parameter("ground_truth_file").value) or _default_ground_truth()
        self.boxes = load_ground_truth(ground_truth)
        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_label = str(self.get_parameter("run_label").value)
        self.timeout_factor = float(self.get_parameter("timeout_factor").value)
        self.kin = UR5eKinematics(base_yaw=float(self.get_parameter("base_yaw").value))

        self.recorder: Optional[EpisodeRecorder] = None
        self.episode_start: Optional[list[float]] = None
        self.last_elapsed = 0.0
        self.episode_index = 0

        self.publisher = self.create_publisher(String, "/metrics/episode", 10)
        self.create_subscription(String, "/task/state", self.on_task_state, 10)
        self.create_subscription(JointState, "/joint_states", self.on_joints, 20)
        self.create_subscription(String, "/vision/detections", self.on_detections, 10)
        self.create_subscription(String, "/obstacle/tracks", self.on_tracks, 10)
        self.get_logger().info(
            f"metrics_node: {len(self.boxes)} ground-truth obstacles from {ground_truth}; "
            f"writing to {self.output_dir}"
        )

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    @staticmethod
    def _parse(message: String) -> Optional[dict]:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def on_task_state(self, message: String) -> None:
        state = self._parse(message)
        if state is None:
            return
        now = self.now_s()
        active = bool(state.get("episode_active", False))
        elapsed = float(state.get("elapsed_s", 0.0))
        start = state.get("start")

        if self.recorder is not None:
            if not active:
                self.finish(now, "stopped")
            elif start != self.episode_start or elapsed + 1e-6 < self.last_elapsed:
                self.finish(now, "restarted")

        if self.recorder is None and active and not state.get("reached", False):
            self.recorder = EpisodeRecorder(self.kin, self.boxes, now - elapsed, state)
            self.episode_start = start
            self.get_logger().info(f"Episode {self.episode_index + 1} started (mode {state.get('mode')})")

        if self.recorder is not None:
            self.recorder.on_task_state(state, now)
            if state.get("reached", False):
                self.finish(now, "reached")
            elif elapsed > self.timeout_factor * float(state.get("episode_duration_s", 1e9)):
                self.finish(now, "timeout")
        self.last_elapsed = elapsed

    def on_joints(self, message: JointState) -> None:
        if self.recorder is None:
            return
        positions = dict(zip(message.name, message.position))
        if all(name in positions for name in JOINT_NAMES):
            self.recorder.on_joints(np.array([positions[name] for name in JOINT_NAMES], dtype=float))

    def on_detections(self, message: String) -> None:
        payload = self._parse(message)
        if self.recorder is not None and payload is not None:
            self.recorder.on_detections(payload)

    def on_tracks(self, message: String) -> None:
        payload = self._parse(message)
        if self.recorder is not None and payload is not None:
            self.recorder.on_tracks(payload)

    def finish(self, now: float, outcome: str) -> None:
        if self.recorder is None:
            return
        self.episode_index += 1
        summary = {"episode": self.episode_index, "run_label": self.run_label,
                   **self.recorder.summary(now, outcome)}
        self.recorder = None  # episode_start is kept: a lingering "reached" state opens nothing
        self.publisher.publish(String(data=json.dumps(summary)))
        with open(self.output_dir / "episodes.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary) + "\n")
        flat = flatten_summary(summary)
        csv_path = self.output_dir / "episodes.csv"
        new_file = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(flat)
        self.get_logger().info(
            f"Episode {self.episode_index} {outcome}: collision={summary['collision']} "
            f"min_true_clearance={summary['min_true_clearance_m']} "
            f"time_to_goal={summary['time_to_goal_s']} path={summary['path_length_m']}"
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MetricsNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.recorder is not None:
            node.finish(node.now_s(), "shutdown")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
