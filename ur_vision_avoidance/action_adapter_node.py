#!/usr/bin/env python3
"""Convert task-space motion + obstacle tracks into safe joint commands.

This node closes the loop that the original package left open:

    /obstacle/tracks  ->  clearance-aware motion  ->  joint trajectory point

Modes (parameter ``mode``):
  potential_field  deterministic avoidance (classical baseline, no learning)
  rl               nominal motion + the offset published on /rl/action; falls
                   back to potential_field when the action is stale or absent
  nominal          nominal motion toward the goal, protected only by the filter

Whatever the mode, the same safety layer runs: approach-speed limiting for the
TCP, wrist and arm points, speed and workspace limits, orientation hold,
damped least squares, joint-speed and joint-limit handling. Stale joint states
or (by default) stale perception make the node hold position.

The node is the owner of the episode (start pose, goal, task frame) and always
publishes /task/state, which the RL observation node consumes. Commands to the
real controller topic are sent only when ``enable_command`` is true; otherwise
the computed command is only published on /avoidance/adapter_command.

Episodes start only on /task/start (std_msgs/Empty) unless ``auto_start`` is
true; /task/stop ends the episode. While no episode runs the node publishes
nothing to the controller, so other tools can move the arm.

Simulation only. Not a safety-rated function. Do not run together with
motion_loop or avoidance_supervisor: they would command the same controller.
"""

from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float32MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .avoidance_logic import (
    AvoidanceParams,
    ObstacleView,
    StepResult,
    check_goal_reachable,
    compute_step,
    pick_escape_direction,
)
from .rl_contract import task_rotation
from .scene_config import load_scene
from .ur_kinematics import UR5eKinematics

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
TASK_STATE_SCHEMA = "ur_vision_avoidance.task_state.v1"


class ActionAdapter(Node):
    def __init__(self) -> None:
        super().__init__("action_adapter")
        self.declare_parameter("mode", "potential_field")
        self.declare_parameter("enable_command", False)
        self.declare_parameter("auto_start", False)
        self.declare_parameter("scene_config", "")
        self.declare_parameter("scene_section", "ur5e_scene")
        # Path direction and "up", both in base_link. They define the task frame
        # (x along the path) in which the RL observation and action live.
        self.declare_parameter("path_direction", [0.0, -1.0, 0.0])
        self.declare_parameter("up_direction", [0.0, 0.0, 1.0])
        self.declare_parameter("auto_escape", True)
        self.declare_parameter("escape_direction", [0.0, 0.0, 1.0])
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("base_yaw", math.pi)
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("tracks_topic", "/obstacle/tracks")
        self.declare_parameter("action_topic", "/rl/action")
        self.declare_parameter("task_state_topic", "/task/state")
        self.declare_parameter("reset_topic", "/task/reset")
        self.declare_parameter("start_topic", "/task/start")
        self.declare_parameter("stop_topic", "/task/stop")
        self.declare_parameter(
            "command_topic", "/scaled_joint_trajectory_controller/joint_trajectory"
        )
        self.declare_parameter("debug_command_topic", "/avoidance/adapter_command")
        self.declare_parameter("sensor_timeout_s", 0.50)
        self.declare_parameter("perception_timeout_s", 1.0)
        self.declare_parameter("action_timeout_s", 0.50)
        self.declare_parameter("require_fresh_perception", True)
        # safety layer
        self.declare_parameter("stop_clearance_m", 0.10)
        self.declare_parameter("slow_clearance_m", 0.22)
        self.declare_parameter("influence_m", 0.35)
        self.declare_parameter("link_stop_clearance_m", 0.04)
        self.declare_parameter("max_speed_mps", 0.12)
        self.declare_parameter("max_joint_vel_rps", 0.6)
        self.declare_parameter("z_min_m", 0.05)
        self.declare_parameter("goal_fade_m", 0.15)
        self.declare_parameter("horizon_s", 0.25)

        self.mode = str(self.get_parameter("mode").value)
        if self.mode not in ("potential_field", "rl", "nominal"):
            raise ValueError(f"mode must be potential_field, rl or nominal, got '{self.mode}'")
        self.enable_command = bool(self.get_parameter("enable_command").value)
        self.started = bool(self.get_parameter("auto_start").value)
        self.require_fresh_perception = bool(self.get_parameter("require_fresh_perception").value)
        self.sensor_timeout_s = float(self.get_parameter("sensor_timeout_s").value)
        self.perception_timeout_s = float(self.get_parameter("perception_timeout_s").value)
        self.action_timeout_s = float(self.get_parameter("action_timeout_s").value)
        self.auto_escape = bool(self.get_parameter("auto_escape").value)

        scene_path = str(self.get_parameter("scene_config").value) or None
        self.scene = load_scene(scene_path, str(self.get_parameter("scene_section").value))
        self.params = AvoidanceParams(
            nominal_speed=self.scene.nominal_speed,
            offset_max=self.scene.max_offset_speed,
            goal_tolerance=self.scene.goal_tolerance,
            max_speed=float(self.get_parameter("max_speed_mps").value),
            stop_clearance_m=float(self.get_parameter("stop_clearance_m").value),
            slow_clearance_m=float(self.get_parameter("slow_clearance_m").value),
            influence_m=float(self.get_parameter("influence_m").value),
            link_stop_clearance_m=float(self.get_parameter("link_stop_clearance_m").value),
            max_joint_vel=float(self.get_parameter("max_joint_vel_rps").value),
            z_min_m=float(self.get_parameter("z_min_m").value),
            goal_fade_m=float(self.get_parameter("goal_fade_m").value),
            horizon_s=float(self.get_parameter("horizon_s").value),
            escape_direction=tuple(
                float(x) for x in self.get_parameter("escape_direction").value
            ),
        )
        self.kin = UR5eKinematics(base_yaw=float(self.get_parameter("base_yaw").value))
        self.path_direction = np.asarray(self.get_parameter("path_direction").value, dtype=float)
        self.up_direction = np.asarray(self.get_parameter("up_direction").value, dtype=float)

        # sensor state
        self.q: Optional[np.ndarray] = None
        self.q_dot: Optional[np.ndarray] = None
        self.q_time = 0.0
        self.obstacles: list[ObstacleView] = []
        self.tracks_time: Optional[float] = None
        self.perception_age: Optional[float] = None
        self.rl_offset_task: Optional[np.ndarray] = None
        self.rl_time: Optional[float] = None
        # episode state
        self.episode_active = False
        self.reset_requested = False
        self.start = np.zeros(3)
        self.goal = np.zeros(3)
        self.rotation = np.eye(3)
        self.r_ref = np.eye(3)
        self.t0 = 0.0
        self.goal_ok = True
        self.last_result: Optional[StepResult] = None

        self.create_subscription(
            JointState, str(self.get_parameter("joint_state_topic").value), self.on_joints, 20
        )
        self.create_subscription(
            String, str(self.get_parameter("tracks_topic").value), self.on_tracks, 10
        )
        self.create_subscription(
            Float32MultiArray, str(self.get_parameter("action_topic").value), self.on_action, 10
        )
        self.create_subscription(
            Empty, str(self.get_parameter("reset_topic").value), self.on_reset, 1
        )
        self.create_subscription(
            Empty, str(self.get_parameter("start_topic").value), self.on_start, 1
        )
        self.create_subscription(
            Empty, str(self.get_parameter("stop_topic").value), self.on_stop, 1
        )
        self.state_pub = self.create_publisher(
            String, str(self.get_parameter("task_state_topic").value), 10
        )
        self.debug_pub = self.create_publisher(
            JointTrajectory, str(self.get_parameter("debug_command_topic").value), 10
        )
        self.command_pub = self.create_publisher(
            JointTrajectory, str(self.get_parameter("command_topic").value), 10
        )
        rate = max(float(self.get_parameter("rate_hz").value), 1.0)
        self.create_timer(1.0 / rate, self.tick)

        self.get_logger().warn(
            f"action_adapter mode={self.mode}, controller command "
            f"{'ENABLED' if self.enable_command else 'DISABLED (debug topic only)'}. "
            "Simulation only; not a safety function."
        )

    # ---- callbacks -------------------------------------------------------
    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def on_joints(self, message: JointState) -> None:
        positions = dict(zip(message.name, message.position))
        if not all(name in positions for name in JOINT_NAMES):
            return
        self.q = np.array([positions[name] for name in JOINT_NAMES], dtype=float)
        velocities = dict(zip(message.name, message.velocity))
        if all(name in velocities for name in JOINT_NAMES):
            self.q_dot = np.array([velocities[name] for name in JOINT_NAMES], dtype=float)
        else:
            self.q_dot = None
        self.q_time = self.now_s()

    def on_tracks(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        self.obstacles = [
            ObstacleView(
                center=np.asarray(track["position"], dtype=float),
                radius=float(track.get("radius_m", 0.0)),
            )
            for track in payload.get("tracks", [])
        ]
        self.perception_age = payload.get("perception_age_s")
        self.tracks_time = self.now_s()

    def on_action(self, message: Float32MultiArray) -> None:
        if len(message.data) >= 3:
            self.rl_offset_task = np.asarray(message.data[:3], dtype=float)
            self.rl_time = self.now_s()

    def on_reset(self, _message: Empty) -> None:
        self.reset_requested = True

    def on_start(self, _message: Empty) -> None:
        self.started = True
        self.reset_requested = True
        self.get_logger().info("Start requested: a new episode begins at the current pose.")

    def on_stop(self, _message: Empty) -> None:
        self.started = False
        self.episode_active = False
        self.get_logger().info("Stop requested: no more controller commands until /task/start.")

    # ---- episode -----------------------------------------------------------
    def start_episode(self, q: np.ndarray) -> None:
        transform = self.kin.fk(q)
        self.start = transform[:3, 3].copy()
        self.r_ref = transform[:3, :3].copy()
        self.rotation = task_rotation(self.path_direction, self.up_direction)
        self.goal = self.start + self.rotation @ np.asarray(self.scene.goal_offset, dtype=float)
        self.t0 = self.now_s()
        self.goal_ok, error, worst = check_goal_reachable(self.kin, q, self.goal)
        if not self.goal_ok:
            self.get_logger().error(
                f"Goal {np.round(self.goal, 3).tolist()} is not reachable from this pose "
                f"(final error {error:.3f} m, worst manipulability {worst:.4f}). "
                "Move the arm to a well-conditioned pose (for example the first motion_loop "
                "waypoint), or change path_direction, then publish std_msgs/Empty on /task/reset. "
                "Holding position."
            )
        if self.auto_escape:
            direction, scores = pick_escape_direction(
                self.kin, q, self.rotation[:, 0], z_min_m=self.params.z_min_m, up=self.up_direction
            )
            self.params.escape_direction = tuple(float(x) for x in direction)
            self.get_logger().info(
                f"Escape direction {np.round(direction, 2).tolist()} (scores {scores})"
            )
        self.episode_active = True
        self.reset_requested = False
        self.get_logger().info(
            f"Episode started: start={np.round(self.start, 3).tolist()} "
            f"goal={np.round(self.goal, 3).tolist()}"
        )

    # ---- control loop ------------------------------------------------------
    def tick(self) -> None:
        if self.q is None:
            return
        now = self.now_s()
        joint_age = now - self.q_time
        if joint_age > self.sensor_timeout_s:
            self.publish_task_state(now, "stale_joint_state", None)
            return
        if not self.started:
            self.publish_task_state(now, "waiting_start", None)
            return
        if self.reset_requested or not self.episode_active:
            self.start_episode(self.q)

        tracks_fresh = (
            self.tracks_time is not None and (now - self.tracks_time) <= self.sensor_timeout_s
        )
        perception_ok = (
            tracks_fresh
            and self.perception_age is not None
            and float(self.perception_age) <= self.perception_timeout_s
        )
        obstacles = self.obstacles if tracks_fresh else []

        if not self.goal_ok:
            result = StepResult(np.zeros(6), np.zeros(3), "goal_unreachable", self.mode, None)
        elif self.require_fresh_perception and not perception_ok:
            result = StepResult(np.zeros(6), np.zeros(3), "perception_stale", self.mode, None)
        else:
            rl_offset = None
            if (
                self.mode == "rl"
                and self.rl_time is not None
                and (now - self.rl_time) <= self.action_timeout_s
                and self.rl_offset_task is not None
            ):
                rl_offset = self.rotation @ self.rl_offset_task  # task -> base_link
            result = compute_step(
                self.kin, self.q, self.goal, obstacles, self.params, self.mode, rl_offset, self.r_ref
            )
        self.last_result = result

        trajectory = self.build_trajectory(self.q, result.qdot)
        self.debug_pub.publish(trajectory)
        if self.enable_command:
            self.command_pub.publish(trajectory)
        self.publish_task_state(now, result.status, result)

    def build_trajectory(self, q: np.ndarray, qdot: np.ndarray) -> JointTrajectory:
        horizon = self.params.horizon_s
        trajectory = JointTrajectory()
        trajectory.joint_names = list(JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in (q + qdot * horizon)]
        # No velocities: the joint trajectory controller rejects a final point
        # with non-zero velocity ("Velocity of last trajectory point ... is not
        # zero"), which silently dropped every command in Gazebo.
        seconds = int(math.floor(horizon))
        point.time_from_start = DurationMsg(sec=seconds, nanosec=int((horizon - seconds) * 1e9))
        trajectory.points = [point]
        return trajectory

    def publish_task_state(self, now: float, status: str, result: Optional[StepResult]) -> None:
        tcp = self.kin.position(self.q)
        tcp_velocity = np.zeros(3)
        if self.q_dot is not None:
            tcp_velocity = (self.kin.jacobian(self.q) @ self.q_dot)[:3]
        payload = {
            "schema": TASK_STATE_SCHEMA,
            "stamp_ns": int(now * 1e9),
            "frame": "base_link",
            "tcp": [float(x) for x in tcp],
            "tcp_velocity": [float(x) for x in tcp_velocity],
            "start": [float(x) for x in self.start],
            "goal": [float(x) for x in self.goal],
            "task_rotation": [float(x) for x in self.rotation.reshape(-1)],
            "elapsed_s": float(now - self.t0) if self.episode_active else 0.0,
            "episode_duration_s": float(self.scene.episode_duration_s),
            "status": status,
            "mode": result.mode_used if result is not None else self.mode,
            "reached": bool(result.reached) if result is not None else False,
            "min_clearance_m": None
            if result is None or result.min_clearance_m is None
            else float(result.min_clearance_m),
            "command_enabled": self.enable_command,
            "episode_active": self.episode_active,
        }
        self.state_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ActionAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
