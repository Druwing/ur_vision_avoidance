"""Continuous fixed-scene motion loop for simulation testing.

The loop cycles through conservative joint waypoints. It pauses and cancels its
active trajectory when the perception layer requests a stop. This is a starter
trajectory generator, not a planner, RL controller, or safety function.
"""

from __future__ import annotations

from typing import Optional

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class MotionLoop(Node):
    """Cycle through fixed joint waypoints while respecting obstacle stop state."""

    def __init__(self) -> None:
        super().__init__("motion_loop")
        self.declare_parameter(
            "action_name", "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        )
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("obstacle_topic", "/obstacle/stop")
        self.declare_parameter("sensor_timeout_s", 0.75)
        self.declare_parameter("waypoint_duration_s", 4.0)
        self.declare_parameter("start_immediately", True)

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        # Conservative fixed-scene loop. Keep this in simulation only.
        self.waypoints = [
            [0.00, -1.20, 1.40, -1.70, -1.57, 0.00],
            [0.55, -1.20, 1.40, -1.70, -1.57, 0.00],
            [0.55, -1.05, 1.20, -1.55, -1.57, 0.00],
            [0.00, -1.05, 1.20, -1.55, -1.57, 0.00],
        ]
        self.waypoint_duration_s = float(self.get_parameter("waypoint_duration_s").value)
        self.sensor_timeout_s = float(self.get_parameter("sensor_timeout_s").value)
        self.latest_positions: Optional[list[float]] = None
        self.latest_joint_time = self.get_clock().now()
        self.obstacle_active = False
        self.goal_handle = None
        self.waypoint_index = 0
        self.start_immediately = bool(self.get_parameter("start_immediately").value)

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
        self.create_timer(0.10, self.loop_timer)
        self.get_logger().warn(
            "Motion loop enabled: fixed waypoints, simulation-only, no RL or safety guarantees."
        )

    def joint_state_callback(self, message: JointState) -> None:
        positions = dict(zip(message.name, message.position))
        if all(name in positions for name in self.joint_names):
            self.latest_positions = [float(positions[name]) for name in self.joint_names]
            self.latest_joint_time = self.get_clock().now()

    def obstacle_callback(self, message: Bool) -> None:
        requested = bool(message.data)
        if requested and not self.obstacle_active:
            self.get_logger().warn("Obstacle active: pausing motion loop and cancelling current goal.")
            if self.goal_handle is not None:
                future = self.goal_handle.cancel_goal_async()
                future.add_done_callback(self.cancel_done_callback)
            self.goal_handle = None
        elif not requested and self.obstacle_active:
            self.get_logger().info("Obstacle cleared: resuming motion loop.")
        self.obstacle_active = requested

    def cancel_done_callback(self, future: object) -> None:
        try:
            future.result()  # type: ignore[attr-defined]
        except Exception as exc:
            self.get_logger().error(f"Could not cancel motion-loop goal: {exc}")

    def loop_timer(self) -> None:
        if self.obstacle_active or self.goal_handle is not None:
            return
        if not self.start_immediately and self.latest_positions is None:
            return
        if self.latest_positions is None:
            return
        age = (self.get_clock().now() - self.latest_joint_time).nanoseconds / 1e9
        if age > self.sensor_timeout_s:
            self.get_logger().warning("Motion loop waiting for fresh joint state.")
            return
        if not self.action_client.wait_for_server(timeout_sec=0.05):
            return
        self.send_next_waypoint()

    def send_next_waypoint(self) -> None:
        target = self.waypoints[self.waypoint_index]
        trajectory = JointTrajectory()
        trajectory.joint_names = list(self.joint_names)
        point = JointTrajectoryPoint()
        point.positions = list(target)
        duration = max(self.waypoint_duration_s, 0.1)
        point.time_from_start = Duration(seconds=duration).to_msg()
        trajectory.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        future = self.action_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)
        self.get_logger().info(
            f"Motion loop sending waypoint {self.waypoint_index + 1}/{len(self.waypoints)}."
        )
        self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)

    def goal_response_callback(self, future: object) -> None:
        try:
            goal_handle = future.result()  # type: ignore[attr-defined]
        except Exception as exc:
            self.get_logger().error(f"Motion-loop goal failed: {exc}")
            self.goal_handle = None
            return
        if not goal_handle.accepted:
            self.get_logger().error("Motion-loop goal was rejected.")
            self.goal_handle = None
            return
        self.goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future: object) -> None:
        self.goal_handle = None
        try:
            status = future.result().status  # type: ignore[attr-defined]
            self.get_logger().info(f"Motion-loop waypoint completed with status {status}.")
        except Exception as exc:
            self.get_logger().error(f"Could not read motion-loop result: {exc}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MotionLoop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
