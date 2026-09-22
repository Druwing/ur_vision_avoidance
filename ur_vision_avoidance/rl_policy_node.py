"""Run a trained fixed-scene PPO policy on the structured RL observation.

The node publishes a bounded action vector only. It deliberately does not send
trajectory goals to the UR controller; a separate safety-filtered adapter must
be added after the fixed-scene policy is evaluated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String


class RLPolicyNode(Node):
    """Load PPO and convert JSON observations to bounded actions."""

    def __init__(self) -> None:
        super().__init__("rl_policy_node")
        self.declare_parameter("model_path", "")
        self.declare_parameter("observation_topic", "/rl/observation")
        self.declare_parameter("action_topic", "/rl/action")
        self.declare_parameter("max_linear_mps", 0.12)
        self.declare_parameter("max_angular_rps", 0.35)
        self.declare_parameter("sensor_timeout_s", 0.50)
        self.model = None
        model_path = str(self.get_parameter("model_path").value)
        if model_path:
            try:
                from stable_baselines3 import PPO

                if not Path(model_path).exists() and not Path(f"{model_path}.zip").exists():
                    raise FileNotFoundError(model_path)
                self.model = PPO.load(model_path)
                self.get_logger().info(f"Loaded fixed-scene PPO policy: {model_path}")
            except Exception as exc:
                self.get_logger().error(f"Policy unavailable; publishing zero actions: {exc}")
        else:
            self.get_logger().warning("model_path is empty; publishing zero actions")

        self.max_linear = float(self.get_parameter("max_linear_mps").value)
        self.max_angular = float(self.get_parameter("max_angular_rps").value)
        self.sensor_timeout_s = float(self.get_parameter("sensor_timeout_s").value)
        self.publisher = self.create_publisher(
            Float32MultiArray, str(self.get_parameter("action_topic").value), 10
        )
        self.create_subscription(
            String,
            str(self.get_parameter("observation_topic").value),
            self.observation_callback,
            10,
        )

    @staticmethod
    def observation_vector(payload: dict) -> np.ndarray:
        obstacle = payload.get("obstacle", {})
        joints = payload.get("joint_state", {})
        position = joints.get("positions", [])
        velocity = joints.get("velocities", [])
        # Fixed-scene training vector: 3 goal-relative values, 3 obstacle-relative
        # placeholders, 3 velocity placeholders, 6 previous-action placeholders,
        # and remaining-time placeholder. Deployment should replace this adapter
        # with the exact state used by the trained environment.
        vector = np.zeros(16, dtype=np.float32)
        vector[3:6] = np.asarray(obstacle.get("position_camera", [0, 0, -1]), dtype=np.float32)[:3]
        vector[6:9] = np.asarray(velocity[:3], dtype=np.float32)
        if position:
            vector[0] = float(position[0])
        if payload.get("sensor_valid") is False:
            vector[:] = 0.0
        return vector

    def observation_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                return
        except json.JSONDecodeError:
            return
        action = np.zeros(6, dtype=np.float32)
        age = max(
            float(payload.get("detection_age_s", float("inf"))),
            float(payload.get("joint_state_age_s", float("inf"))),
        )
        if self.model is not None and payload.get("sensor_valid", False) and age <= self.sensor_timeout_s:
            try:
                raw_action, _ = self.model.predict(self.observation_vector(payload), deterministic=True)
                action = np.asarray(raw_action, dtype=np.float32).reshape(6)
                action = np.clip(action, -1.0, 1.0)
            except Exception as exc:
                self.get_logger().error(f"Policy inference failed; using zero action: {exc}")
        scale = np.array(
            [self.max_linear, self.max_linear, self.max_linear, self.max_angular, self.max_angular, self.max_angular],
            dtype=np.float32,
        )
        output = Float32MultiArray()
        output.data = (action * scale).tolist()
        self.publisher.publish(output)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RLPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
