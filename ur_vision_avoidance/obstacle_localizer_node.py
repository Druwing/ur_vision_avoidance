#!/usr/bin/env python3
"""Turn 2D detections + depth into persistent 3D obstacle tracks in base_link.

Input : /vision/detections (JSON from vision_node), /camera/camera_info, TF.
Output: /obstacle/tracks (JSON), /obstacle/markers (RViz spheres).

Each detection is back-projected with the camera intrinsics, transformed to the
target frame with the TF available at the image timestamp (the camera moves
with the end effector), and fed to an ObstacleTracker that keeps obstacles in
memory after they leave the field of view.

Fail closed: if the transform is not available at the image time, the detection
is dropped (no guessing with a newer pose).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .camera_geometry import detection_to_camera_points, quat_to_matrix
from .obstacle_tracker import Measurement, ObstacleTracker

TRACKS_SCHEMA = "ur_vision_avoidance.obstacle_tracks.v1"


class ObstacleLocalizerNode(Node):
    def __init__(self) -> None:
        super().__init__("obstacle_localizer")
        self.declare_parameter("detections_topic", "/vision/detections")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("tracks_topic", "/obstacle/tracks")
        self.declare_parameter("markers_topic", "/obstacle/markers")
        self.declare_parameter("target_frame", "base_link")
        # Frame of the camera image. The bridged CameraInfo frame_id may carry a
        # Gazebo scoped name, so it is configured explicitly.
        self.declare_parameter("camera_frame", "camera_optical_frame")
        self.declare_parameter("depth_is_range", False)
        self.declare_parameter("push_to_center", True)
        self.declare_parameter("radius_min_m", 0.03)
        self.declare_parameter("radius_max_m", 0.40)
        self.declare_parameter("min_confidence", 0.0)
        # Drop measurements whose visible surface is below this height in the
        # target frame: the floor seen by the depth fallback is not an obstacle.
        self.declare_parameter("min_height_m", 0.03)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("tf_timeout_s", 0.05)
        # tracker
        self.declare_parameter("gate_m", 0.30)
        self.declare_parameter("alpha", 0.5)
        self.declare_parameter("memory_s", 10.0)
        self.declare_parameter("min_hits", 2)
        self.declare_parameter("max_speed_mps", 2.0)

        self.target_frame = str(self.get_parameter("target_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.depth_is_range = bool(self.get_parameter("depth_is_range").value)
        self.push_to_center = bool(self.get_parameter("push_to_center").value)
        self.radius_min = float(self.get_parameter("radius_min_m").value)
        self.radius_max = float(self.get_parameter("radius_max_m").value)
        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.min_height = float(self.get_parameter("min_height_m").value)
        self.dropped_below_floor = 0
        self.tf_timeout = float(self.get_parameter("tf_timeout_s").value)

        self.tracker = ObstacleTracker(
            gate_m=float(self.get_parameter("gate_m").value),
            alpha=float(self.get_parameter("alpha").value),
            memory_s=float(self.get_parameter("memory_s").value),
            min_hits=int(self.get_parameter("min_hits").value),
            max_speed_mps=float(self.get_parameter("max_speed_mps").value),
        )

        self.intrinsics: tuple[float, float, float, float] | None = None
        self.last_perception_s: float | None = None
        self.dropped_for_tf = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.tracks_pub = self.create_publisher(
            String, str(self.get_parameter("tracks_topic").value), 10
        )
        self.markers_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("markers_topic").value), 10
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("detections_topic").value),
            self.detections_callback,
            10,
        )
        period = 1.0 / max(float(self.get_parameter("publish_rate_hz").value), 1.0)
        self.create_timer(period, self.publish_tracks)
        self.get_logger().info(
            f"Localizer: camera_frame={self.camera_frame} -> {self.target_frame}"
        )

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def camera_info_callback(self, message: CameraInfo) -> None:
        k = message.k
        if k[0] > 0.0 and k[4] > 0.0:
            self.intrinsics = (float(k[0]), float(k[4]), float(k[2]), float(k[5]))

    def _lookup(self, stamp_ns: int):
        try:
            return self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                Time(nanoseconds=stamp_ns),
                timeout=Duration(seconds=self.tf_timeout),
            )
        except TransformException as exc:
            self.dropped_for_tf += 1
            self.get_logger().warning(
                f"No TF {self.target_frame}<-{self.camera_frame} at image time; "
                f"dropping detections ({exc})",
                throttle_duration_sec=2.0,
            )
            return None

    def detections_callback(self, message: String) -> None:
        try:
            payload: dict[str, Any] = json.loads(message.data)
        except json.JSONDecodeError:
            return
        now = self.now_s()
        self.last_perception_s = now
        if self.intrinsics is None:
            self.get_logger().warning(
                "Waiting for CameraInfo intrinsics.", throttle_duration_sec=2.0
            )
            return
        fx, fy, cx, cy = self.intrinsics

        measurements: list[Measurement] = []
        detections = payload.get("detections", [])
        if detections and bool(payload.get("depth_fresh", False)):
            transform = self._lookup(int(payload.get("stamp_ns", 0)))
            if transform is None:
                return
            q = transform.transform.rotation
            t = transform.transform.translation
            rotation = quat_to_matrix(q.x, q.y, q.z, q.w)
            translation = np.array([t.x, t.y, t.z])
            for det in detections:
                distance = det.get("distance_m")
                if distance is None or float(det.get("confidence", 1.0)) < self.min_confidence:
                    continue
                anchor = det.get("anchor_uv")
                if anchor is not None and det.get("anchor_depth_m") is not None:
                    distance = det["anchor_depth_m"]
                surface_cam, center_cam, radius = detection_to_camera_points(
                    det["bbox_xyxy"],
                    float(distance),
                    fx,
                    fy,
                    cx,
                    cy,
                    depth_is_range=self.depth_is_range,
                    push_to_center=self.push_to_center,
                    r_min=self.radius_min,
                    r_max=self.radius_max,
                    anchor_uv=anchor,
                )
                surface = rotation @ surface_cam + translation
                if surface[2] < self.min_height:
                    self.dropped_below_floor += 1
                    continue
                measurements.append(
                    Measurement(
                        position=rotation @ center_cam + translation,
                        surface=surface,
                        radius_m=radius,
                        label=str(det.get("class", "")),
                        confidence=float(det.get("confidence", 1.0)),
                    )
                )
        self.tracker.update(measurements, now)

    def publish_tracks(self) -> None:
        now = self.now_s()
        self.tracker.prune(now)
        tracks = self.tracker.tracks(now)
        perception_age = (
            float("inf") if self.last_perception_s is None else now - self.last_perception_s
        )
        payload = {
            "schema": TRACKS_SCHEMA,
            "stamp_ns": int(now * 1e9),
            "frame": self.target_frame,
            "perception_age_s": None if perception_age == float("inf") else round(perception_age, 3),
            "dropped_for_tf": self.dropped_for_tf,
            "dropped_below_floor": self.dropped_below_floor,
            "tracks": tracks,
        }
        self.tracks_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))
        self._publish_markers(tracks)

    def _publish_markers(self, tracks: list[dict[str, Any]]) -> None:
        stamp = self.get_clock().now().to_msg()
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        for track in tracks:
            marker = Marker()
            marker.header.frame_id = self.target_frame
            marker.header.stamp = stamp
            marker.ns = "obstacle_tracks"
            marker.id = int(track["id"])
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = track["position"]
            marker.pose.orientation.w = 1.0
            diameter = 2.0 * float(track["radius_m"])
            marker.scale.x = marker.scale.y = marker.scale.z = diameter
            marker.color.r, marker.color.g, marker.color.b = (
                (1.0, 0.2, 0.2) if track["visible"] else (1.0, 0.7, 0.1)
            )
            marker.color.a = 0.6
            marker.lifetime = DurationMsg(sec=1)
            array.markers.append(marker)
        self.markers_pub.publish(array)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ObstacleLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
