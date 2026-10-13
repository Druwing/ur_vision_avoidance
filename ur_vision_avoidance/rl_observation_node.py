"""Publish the RL observation (schema v2) from the task state and obstacle tracks.

Inputs
  /task/state       published by action_adapter (TCP, goal, task frame, timing)
  /obstacle/tracks  published by obstacle_localizer (3D tracks in base_link)
  /rl/action        previous policy action (to fill the "previous action" slot)

Output
  /rl/observation   JSON with the 16-element ``vector`` that the policy expects,
                    built by ``rl_contract.build_observation`` - the same
                    function the training environment uses. All vectors are
                    relative to the TCP and expressed in the task frame
                    (x along the start->goal path), exactly as in training.

When no obstacle track exists the observation is marked ``obstacle.valid =
false``: the fixed-scene policy never saw an empty scene, so the policy node
publishes a zero offset in that case instead of extrapolating.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from .rl_contract import OBSERVATION_SCHEMA, action_scale, build_observation
from .scene_config import load_scene


class RLObservationNode(Node):
    """Join task state, obstacle tracks and previous action into one observation."""

    def __init__(self) -> None:
        super().__init__("rl_observation_adapter")
        self.declare_parameter("task_state_topic", "/task/state")
        self.declare_parameter("tracks_topic", "/obstacle/tracks")
        self.declare_parameter("action_topic", "/rl/action")
        self.declare_parameter("observation_topic", "/rl/observation")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("sensor_timeout_s", 0.50)
        self.declare_parameter("perception_timeout_s", 1.0)
        self.declare_parameter("scene_config", "")
        self.declare_parameter("scene_section", "ur5e_scene")
        self.declare_parameter("absent_obstacle_rel", [0.0, 3.0, 0.0])

        self.sensor_timeout_s = float(self.get_parameter("sensor_timeout_s").value)
        self.perception_timeout_s = float(self.get_parameter("perception_timeout_s").value)
        self.absent_rel = np.asarray(self.get_parameter("absent_obstacle_rel").value, dtype=float)
        scene = load_scene(
            str(self.get_parameter("scene_config").value) or None,
            str(self.get_parameter("scene_section").value),
        )
        self.scale = action_scale(scene.max_offset_speed, scene.max_angular)

        self.task_state: Optional[dict[str, Any]] = None
        self.task_time: Optional[float] = None
        self.tracks: Optional[dict[str, Any]] = None
        self.tracks_time: Optional[float] = None
        self.prev_action = np.zeros(6, dtype=np.float32)

        self.publisher = self.create_publisher(
            String, str(self.get_parameter("observation_topic").value), 10
        )
        self.create_subscription(
            String, str(self.get_parameter("task_state_topic").value), self.on_task_state, 10
        )
        self.create_subscription(
            String, str(self.get_parameter("tracks_topic").value), self.on_tracks, 10
        )
        self.create_subscription(
            Float32MultiArray, str(self.get_parameter("action_topic").value), self.on_action, 10
        )
        period = 1.0 / max(float(self.get_parameter("publish_rate_hz").value), 1.0)
        self.create_timer(period, self.publish_observation)

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def on_task_state(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            self.task_state = payload
            self.task_time = self.now_s()

    def on_tracks(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            self.tracks = payload
            self.tracks_time = self.now_s()

    def on_action(self, message: Float32MultiArray) -> None:
        if len(message.data) == 6:
            normalized = np.asarray(message.data, dtype=np.float32) / self.scale
            self.prev_action = np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def publish_observation(self) -> None:
        now = self.now_s()
        state = self.task_state
        task_age = float("inf") if self.task_time is None else now - self.task_time
        tracks_age = float("inf") if self.tracks_time is None else now - self.tracks_time
        perception_age = None if self.tracks is None else self.tracks.get("perception_age_s")

        state_ok = (
            state is not None
            and bool(state.get("episode_active", False))
            and task_age <= self.sensor_timeout_s
        )
        perception_ok = (
            self.tracks is not None
            and tracks_age <= self.sensor_timeout_s
            and perception_age is not None
            and float(perception_age) <= self.perception_timeout_s
        )
        sensor_valid = bool(state_ok and perception_ok)

        vector = np.zeros(16, dtype=np.float32)
        obstacle: dict[str, Any] = {"valid": False, "visible": False}
        goal_rel_task = [0.0, 0.0, 0.0]
        if state_ok:
            tcp = np.asarray(state["tcp"], dtype=float)
            rotation = np.asarray(state["task_rotation"], dtype=float).reshape(3, 3)
            to_task = rotation.T  # base_link -> task frame
            goal_rel = to_task @ (np.asarray(state["goal"], dtype=float) - tcp)
            velocity = to_task @ np.asarray(state["tcp_velocity"], dtype=float)
            goal_rel_task = [float(x) for x in goal_rel]

            nearest = None
            if self.tracks is not None and tracks_age <= self.sensor_timeout_s:
                candidates = self.tracks.get("tracks", [])
                nearest = min(
                    candidates,
                    key=lambda t: float(np.linalg.norm(np.asarray(t["position"]) - tcp)),
                    default=None,
                )
            if nearest is not None:
                center_rel = np.asarray(nearest["position"], dtype=float) - tcp
                obstacle_rel = to_task @ center_rel
                obstacle = {
                    "valid": True,
                    "visible": bool(nearest.get("visible", False)),
                    "track_id": int(nearest.get("id", -1)),
                    "distance_m": float(np.linalg.norm(center_rel)),
                    "radius_m": float(nearest.get("radius_m", 0.0)),
                    "confidence": float(nearest.get("confidence", 0.0)),
                    "rel_task": [float(x) for x in obstacle_rel],
                }
            else:
                obstacle_rel = self.absent_rel

            duration = max(float(state.get("episode_duration_s", 1.0)), 1e-6)
            remaining = float(np.clip(1.0 - float(state.get("elapsed_s", 0.0)) / duration, 0.0, 1.0))
            vector = build_observation(goal_rel, obstacle_rel, velocity, self.prev_action, remaining)

        payload = {
            "schema": OBSERVATION_SCHEMA,
            "stamp_ns": int(now * 1e9),
            "frame": "task",
            "vector": [float(x) for x in vector],
            "goal_rel_task": goal_rel_task,
            "obstacle": obstacle,
            "task_status": None if state is None else state.get("status"),
            "task_state_age_s": None if task_age == float("inf") else round(task_age, 3),
            "tracks_age_s": None if tracks_age == float("inf") else round(tracks_age, 3),
            "perception_age_s": perception_age,
            "sensor_valid": sensor_valid,
            "ard_enabled": False,
        }
        self.publisher.publish(String(data=json.dumps(payload, separators=(",", ":"))))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RLObservationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
