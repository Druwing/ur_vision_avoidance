"""Run a trained fixed-scene PPO policy on the RL observation (schema v2).

The node publishes a bounded action vector (task frame, m/s and rad/s) on
/rl/action. It does not command the robot: ``action_adapter`` (mode ``rl``)
rotates the action into base_link, adds it to the nominal motion and runs it
through the safety layer.

Zero actions are published when no model is loaded, the observation is not
schema v2, sensors are stale, there is no obstacle track, or the obstacle is
farther than ``engage_distance_m`` (outside what the fixed-scene policy was
trained on). In ``rl`` mode the adapter then falls back to the deterministic
potential field only when the action is *missing*; a zero action means
"nominal motion, protected by the filter".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from .rl_contract import ACT_DIM, OBS_DIM, OBSERVATION_SCHEMA, action_scale
from .scene_config import load_scene


def load_policy(model_path: str):
    """Load a PPO checkpoint independently of the NumPy version used to save it.

    SB3 pickles the spaces, schedules and rollout buffers with NumPy. A policy
    trained with NumPy 2.x then fails to load next to the ROS cv_bridge, which
    needs NumPy 1.x ("No module named 'numpy._core.numeric'"), and the node
    silently published zero actions. Inference only needs the network weights
    (a torch state dict), so the pickled objects are replaced here.
    """
    from gymnasium import spaces
    from stable_baselines3 import PPO

    custom_objects = {
        "observation_space": spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32),
        "action_space": spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32),
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.2,
        "_last_obs": None,
        "_last_episode_starts": None,
        "_last_original_obs": None,
        "ep_info_buffer": None,
        "ep_success_buffer": None,
    }
    return PPO.load(model_path, custom_objects=custom_objects, device="cpu")


class RLPolicyNode(Node):
    """Load PPO and convert the observation vector to a bounded action."""

    def __init__(self) -> None:
        super().__init__("rl_policy_node")
        self.declare_parameter("model_path", "")
        self.declare_parameter("observation_topic", "/rl/observation")
        self.declare_parameter("action_topic", "/rl/action")
        self.declare_parameter("scene_config", "")
        self.declare_parameter("scene_section", "ur5e_scene")
        # 0.0 -> use the scene's values (must match the training scene)
        self.declare_parameter("max_linear_mps", 0.0)
        self.declare_parameter("max_angular_rps", 0.0)
        self.declare_parameter("engage_distance_m", 0.0)  # 0.0 -> 1.25 x training distance
        self.declare_parameter("sensor_timeout_s", 0.50)

        scene = load_scene(
            str(self.get_parameter("scene_config").value) or None,
            str(self.get_parameter("scene_section").value),
        )
        max_linear = float(self.get_parameter("max_linear_mps").value) or scene.max_offset_speed
        max_angular = float(self.get_parameter("max_angular_rps").value) or scene.max_angular
        self.scale = action_scale(max_linear, max_angular)
        engage = float(self.get_parameter("engage_distance_m").value)
        self.engage_distance_m = engage or 1.25 * float(np.linalg.norm(scene.obstacle_offset))
        self.sensor_timeout_s = float(self.get_parameter("sensor_timeout_s").value)
        self.warned_schema = False

        self.model = None
        model_path = str(self.get_parameter("model_path").value)
        if model_path:
            try:
                if not Path(model_path).exists() and not Path(f"{model_path}.zip").exists():
                    raise FileNotFoundError(model_path)
                self.model = load_policy(model_path)
                self.get_logger().info(
                    f"Loaded PPO policy {model_path} (engages within {self.engage_distance_m:.2f} m)"
                )
            except Exception as exc:
                self.get_logger().error(f"Policy unavailable; publishing zero actions: {exc}")
        else:
            self.get_logger().warning("model_path is empty; publishing zero actions")

        self.publisher = self.create_publisher(
            Float32MultiArray, str(self.get_parameter("action_topic").value), 10
        )
        self.create_subscription(
            String,
            str(self.get_parameter("observation_topic").value),
            self.observation_callback,
            10,
        )

    def compute_action(self, payload: dict) -> np.ndarray:
        """Normalized action in [-1, 1]^6 for one observation payload."""
        zero = np.zeros(ACT_DIM, dtype=np.float32)
        if self.model is None:
            return zero
        if payload.get("schema") != OBSERVATION_SCHEMA:
            if not self.warned_schema:
                self.get_logger().error(
                    f"Unexpected observation schema {payload.get('schema')!r}; "
                    f"expected {OBSERVATION_SCHEMA}. Publishing zero actions."
                )
                self.warned_schema = True
            return zero
        obstacle = payload.get("obstacle", {})
        vector = payload.get("vector", [])
        if (
            not payload.get("sensor_valid", False)
            or not obstacle.get("valid", False)
            or len(vector) != OBS_DIM
            or float(obstacle.get("distance_m", np.inf)) > self.engage_distance_m
        ):
            return zero
        try:
            raw, _ = self.model.predict(np.asarray(vector, dtype=np.float32), deterministic=True)
            return np.clip(np.asarray(raw, dtype=np.float32).reshape(ACT_DIM), -1.0, 1.0)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Policy inference failed; using zero action: {exc}")
            return zero

    def observation_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                return
        except json.JSONDecodeError:
            return
        action = self.compute_action(payload)
        output = Float32MultiArray()
        output.data = (action * self.scale).tolist()
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
