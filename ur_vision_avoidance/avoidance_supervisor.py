#!/usr/bin/env python3
"""Simulation-only obstacle supervisor for the UR arm.

This example is intentionally conservative: the detector requests a stop, and,
when explicitly enabled, the supervisor sends one small shoulder-pan retreat to
the simulated scaled joint trajectory controller. It is not a safety-rated stop
or collision-avoidance implementation.
"""

from __future__ import annotations

import math
from typing import Optional

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class AvoidanceSupervisor(Node):
    """Convert a high-level obstacle request into a small simulated retreat."""

    def __init__(self) -> None:
        super().__init__("avoidance_supervisor")

        self.declare_parameter("enable_avoidance", False)
        self.declare_parameter(
            "action_name", "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        )
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("obstacle_topic", "/obstacle/stop")
        self.declare_parameter("lateral_topic", "/obstacle/lateral_offset")
        self.declare_parameter("retreat_radians", 0.35)
        self.declare_parameter("retreat_duration_s", 2.0)
        self.declare_parameter("sensor_timeout_s", 0.75)
        self.declare_parameter("min_command_interval_s", 3.0)
        self.declare_parameter("shoulder_pan_min", -3.0)
        self.declare_parameter("shoulder_pan_max", 3.0)

        self.enable_avoidance = bool(self.get_parameter("enable_avoidance").value)
        self.retreat_radians = float(self.get_parameter("retreat_radians").value)
        self.retreat_duration_s = float(self.get_parameter("retreat_duration_s").value)
        self.sensor_timeout_s = float(self.get_parameter("sensor_timeout_s").value)
        self.min_command_interval_s = float(
            self.get_parameter("min_command_interval_s").value
        )
        self.shoulder_pan_min = float(self.get_parameter("shoulder_pan_min").value)
        self.shoulder_pan_max = float(self.get_parameter("shoulder_pan_max").value)

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.latest_positions: Optional[list[float]] = None
        self.latest_joint_state_time = self.get_clock().now()
        self.latest_lateral = 0.0
        self.obstacle_active = False
        self.last_command_time = self.get_clock().now()
        self.goal_in_progress = False

        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("action_name").value),
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self.joint_state_callback,
            20,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("obstacle_topic").value),
            self.obstacle_callback,
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("lateral_topic").value),
            self.lateral_callback,
            10,
        )
        self.command_pub = self.create_publisher(JointTrajectory, "/avoidance/command", 10)
        self.timer = self.create_timer(0.10, self.watchdog)

        mode = "ENABLED" if self.enable_avoidance else "DISABLED (stop request only)"
        self.get_logger().warn(
            f"Avoidance mode: {mode}. This node is simulation-only and not a safety function."
        )

    def joint_state_callback(self, message: JointState) -> None:
        """Cache the current six UR joint positions in controller order."""
        positions = dict(zip(message.name, message.position))
        if not all(name in positions for name in self.joint_names):
            return
        self.latest_positions = [float(positions[name]) for name in self.joint_names]
        self.latest_joint_state_time = self.get_clock().now()

    def lateral_callback(self, message: Float32) -> None:
        self.latest_lateral = float(message.data)

    def obstacle_callback(self, message: Bool) -> None:
        """Handle the rising edge of a close obstacle request."""
        requested = bool(message.data)
        rising_edge = requested and not self.obstacle_active
        self.obstacle_active = requested
        if rising_edge:
            self.get_logger().warn(
                "Obstacle request received: simulated retreat requested. "
                "The robot will not be commanded unless enable_avoidance:=true."
            )
            if self.enable_avoidance:
                self.send_retreat_if_safe()

    def watchdog(self) -> None:
        """Fail closed at the high-level decision layer when sensor data expire."""
        age = (self.get_clock().now() - self.latest_joint_state_time).nanoseconds / 1e9
        if age > self.sensor_timeout_s and self.goal_in_progress:
            self.get_logger().error(
                "Joint-state data are stale while a retreat is active. "
                "Do not use this example as a real safety stop."
            )

    def send_retreat_if_safe(self) -> None:
        """Send one small joint-space retreat using the current joint state."""
        now = self.get_clock().now()
        since_last = (now - self.last_command_time).nanoseconds / 1e9
        state_age = (now - self.latest_joint_state_time).nanoseconds / 1e9
        if self.goal_in_progress:
            self.get_logger().warn("A retreat goal is already in progress; ignoring trigger.")
            return
        if since_last < self.min_command_interval_s:
            self.get_logger().warn("Retreat rate limit active; ignoring trigger.")
            return
        if self.latest_positions is None or state_age > self.sensor_timeout_s:
            self.get_logger().error(
                "No fresh joint state is available; refusing to send a trajectory."
            )
            return

        if not self.action_client.wait_for_server(timeout_sec=0.25):
            self.get_logger().error(
                "Trajectory action server is unavailable. Start the UR GZ simulation "
                "and confirm scaled_joint_trajectory_controller is active."
            )
            return

        # Positive image offset means the obstacle is on the right. The sign can
        # be inverted after checking the camera's optical-frame convention.
        direction = -1.0 if self.latest_lateral >= 0.0 else 1.0
        target = list(self.latest_positions)
        target[0] = max(
            self.shoulder_pan_min,
            min(self.shoulder_pan_max, target[0] + direction * self.retreat_radians),
        )

        trajectory = JointTrajectory()
        trajectory.joint_names = list(self.joint_names)
        point = JointTrajectoryPoint()
        point.positions = target
        point.time_from_start.sec = int(math.floor(self.retreat_duration_s))
        point.time_from_start.nanosec = int(
            (self.retreat_duration_s - math.floor(self.retreat_duration_s)) * 1e9
        )
        trajectory.points = [point]
        self.command_pub.publish(trajectory)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        self.goal_in_progress = True
        self.last_command_time = now
        future = self.action_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)
        self.get_logger().warn(
            f"Sending a {self.retreat_radians:.2f} rad shoulder-pan retreat over "
            f"{self.retreat_duration_s:.1f} s."
        )

    def goal_response_callback(self, future: object) -> None:
        """Monitor the action response without blocking the ROS executor."""
        try:
            goal_handle = future.result()  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - transport failure path
            self.goal_in_progress = False
            self.get_logger().error(f"Trajectory goal transport failed: {exc}")
            return
        if not goal_handle.accepted:
            self.goal_in_progress = False
            self.get_logger().error("Trajectory goal was rejected by the controller.")
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future: object) -> None:
        """Clear the in-progress flag when the controller finishes."""
        self.goal_in_progress = False
        try:
            status = future.result().status  # type: ignore[attr-defined]
            self.get_logger().info(f"Retreat action completed with status {status}.")
        except Exception as exc:  # pragma: no cover - transport failure path
            self.get_logger().error(f"Could not read trajectory result: {exc}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = AvoidanceSupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
