"""Publish a structured RL observation from existing ROS 2 topics.

This starter adapter keeps the policy interface explicit without coupling the
training scaffold to a custom ROS message. The JSON payload is versioned and
contains the latest detection, joint state, confidence, and sensor age.
"""

from __future__ import annotations

import json
import math
from typing import Any

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class RLObservationNode(Node):
    """Combine perception and joint state into a timestamped JSON observation."""

    def __init__(self) -> None:
        super().__init__("rl_observation_adapter")
        self.declare_parameter("detections_topic", "/vision/detections")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("observation_topic", "/rl/observation")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("sensor_timeout_s", 0.50)

        self.sensor_timeout_s = float(self.get_parameter("sensor_timeout_s").value)
        self.latest_detections: dict[str, Any] = {"detections": [], "depth_fresh": False}
        self.latest_joint_state: dict[str, Any] = {"positions": [], "velocities": [], "names": []}
        self.latest_detection_time = self.get_clock().now()
        self.latest_joint_time = self.get_clock().now()

        self.publisher = self.create_publisher(
            String, str(self.get_parameter("observation_topic").value), 10
        )
        self.create_subscription(
            String,
            str(self.get_parameter("detections_topic").value),
            self.detection_callback,
            10,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self.joint_callback,
            20,
        )
        period = 1.0 / max(float(self.get_parameter("publish_rate_hz").value), 1.0)
        self.create_timer(period, self.publish_observation)

    def detection_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if isinstance(payload, dict):
                self.latest_detections = payload
                self.latest_detection_time = self.get_clock().now()
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Ignoring malformed detection JSON: {exc}")

    def joint_callback(self, message: JointState) -> None:
        self.latest_joint_state = {
            "names": list(message.name),
            "positions": [float(value) for value in message.position],
            "velocities": [float(value) for value in message.velocity],
        }
        self.latest_joint_time = self.get_clock().now()

    def publish_observation(self) -> None:
        now = self.get_clock().now()
        detection_age = (now - self.latest_detection_time).nanoseconds / 1e9
        joint_age = (now - self.latest_joint_time).nanoseconds / 1e9
        detections = self.latest_detections.get("detections", [])
        nearest = min(
            (item for item in detections if item.get("distance_m") is not None),
            key=lambda item: float(item["distance_m"]),
            default={},
        )
        obstacle = {
            "position_camera": [
                float(nearest.get("lateral_offset", 0.0)),
                0.0,
                float(nearest.get("distance_m", -1.0)),
            ],
            "distance_m": float(nearest.get("distance_m", -1.0)),
            "lateral_offset": float(nearest.get("lateral_offset", 0.0)),
            "confidence": float(nearest.get("confidence", 0.0)),
            "valid": bool(nearest) and bool(self.latest_detections.get("depth_fresh", False)),
        }
        payload = {
            "schema": "ur_vision_avoidance.rl_observation.v1",
            "stamp_ns": now.nanoseconds,
            "obstacle": obstacle,
            "detections": detections,
            "joint_state": self.latest_joint_state,
            "detection_age_s": detection_age,
            "joint_state_age_s": joint_age,
            "sensor_valid": detection_age <= self.sensor_timeout_s and joint_age <= self.sensor_timeout_s,
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
