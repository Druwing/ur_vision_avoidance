#!/usr/bin/env python3
"""YOLO + RGB-D obstacle detector for the UR Gazebo starter project.

This node is deliberately a perception component, not a safety-rated controller.
It publishes a high-level stop request and an optional lateral image offset for a
simulation supervisor.
"""

from __future__ import annotations

import json
import math
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String
from ultralytics import YOLO


class VisionNode(Node):
    """Run object detection on a ROS 2 RGB-D camera stream."""

    def __init__(self) -> None:
        super().__init__("vision_obstacle_detector")

        self.declare_parameter("model", "yolo26n.pt")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth_image_raw")
        self.declare_parameter("annotated_topic", "/vision/annotated")
        self.declare_parameter("detections_topic", "/vision/detections")
        self.declare_parameter("stop_topic", "/obstacle/stop")
        self.declare_parameter("distance_topic", "/obstacle/nearest_distance")
        self.declare_parameter("lateral_topic", "/obstacle/lateral_offset")
        self.declare_parameter("confidence", 0.45)
        self.declare_parameter("stop_distance_m", 0.80)
        self.declare_parameter("depth_timeout_s", 0.50)
        self.declare_parameter("require_depth", True)
        self.declare_parameter(
            "target_classes",
            ["person", "chair", "car", "truck", "bus", "bicycle", "motorcycle"],
        )

        model_name = str(self.get_parameter("model").value)
        self.confidence = float(self.get_parameter("confidence").value)
        self.stop_distance_m = float(self.get_parameter("stop_distance_m").value)
        self.depth_timeout_s = float(self.get_parameter("depth_timeout_s").value)
        self.require_depth = bool(self.get_parameter("require_depth").value)
        self.target_classes = {
            str(item).lower() for item in self.get_parameter("target_classes").value
        }

        self.bridge = CvBridge()
        self.model = YOLO(model_name)
        self.latest_depth: np.ndarray | None = None
        self.latest_depth_encoding = ""
        self.latest_depth_stamp_ns = 0
        self.warned_no_depth = False

        image_topic = str(self.get_parameter("image_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        self.image_pub = self.create_publisher(
            Image, str(self.get_parameter("annotated_topic").value), 5
        )
        self.detections_pub = self.create_publisher(
            String, str(self.get_parameter("detections_topic").value), 10
        )
        self.stop_pub = self.create_publisher(
            Bool, str(self.get_parameter("stop_topic").value), 10
        )
        self.distance_pub = self.create_publisher(
            Float32, str(self.get_parameter("distance_topic").value), 10
        )
        self.lateral_pub = self.create_publisher(
            Float32, str(self.get_parameter("lateral_topic").value), 10
        )

        self.create_subscription(
            Image, depth_topic, self.depth_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"Loaded {model_name}; RGB={image_topic}, depth={depth_topic}; "
            f"target classes={sorted(self.target_classes)}"
        )

    def depth_callback(self, message: Image) -> None:
        """Cache the most recent depth image for the next RGB frame."""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                message, desired_encoding="passthrough"
            )
            self.latest_depth_encoding = str(message.encoding)
            self.latest_depth_stamp_ns = (
                int(message.header.stamp.sec) * 1_000_000_000
                + int(message.header.stamp.nanosec)
            )
        except CvBridgeError as exc:
            self.get_logger().error(f"Could not convert depth image: {exc}")

    def image_callback(self, message: Image) -> None:
        """Infer on one RGB image and publish perception outputs."""
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"Could not convert RGB image: {exc}")
            return

        # stream=False is intentional: this callback processes one frame at a time.
        result = self.model.predict(
            source=image,
            conf=self.confidence,
            imgsz=640,
            device="cpu",
            verbose=False,
        )[0]
        annotated = result.plot(show=False)

        image_stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        depth_is_fresh = (
            self.latest_depth is not None
            and self.latest_depth_stamp_ns > 0
            and abs(image_stamp_ns - self.latest_depth_stamp_ns)
            <= int(self.depth_timeout_s * 1_000_000_000)
        )

        detections: list[dict[str, Any]] = []
        nearest_distance = math.inf
        nearest_lateral = 0.0
        obstacle_now = False

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                label = str(result.names[int(class_id)]).lower()
                if self.target_classes and label not in self.target_classes:
                    continue

                x1, y1, x2, y2 = [float(value) for value in box]
                center_x = 0.5 * (x1 + x2)
                center_y = 0.5 * (y1 + y2)
                lateral_offset = (center_x / max(float(image.shape[1]), 1.0)) - 0.5
                distance_m = self._estimate_depth(
                    center_x,
                    center_y,
                    image.shape[1],
                    image.shape[0],
                    self.latest_depth if depth_is_fresh else None,
                    self.latest_depth_encoding,
                )
                in_path = 0.20 <= (center_x / image.shape[1]) <= 0.80
                is_close = math.isfinite(distance_m) and distance_m <= self.stop_distance_m
                if self.require_depth and not depth_is_fresh:
                    is_close = False

                detection = {
                    "class": label,
                    "confidence": round(float(confidence), 4),
                    "bbox_xyxy": [round(v, 1) for v in (x1, y1, x2, y2)],
                    "distance_m": None if not math.isfinite(distance_m) else round(distance_m, 3),
                    "in_path": bool(in_path),
                    "close": bool(is_close),
                }
                detections.append(detection)

                if math.isfinite(distance_m) and distance_m < nearest_distance:
                    nearest_distance = distance_m
                    nearest_lateral = lateral_offset
                if in_path and is_close:
                    obstacle_now = True

        # If depth is unavailable, publish an explicit non-stop result while the
        # default require_depth policy prevents an unverified distance decision.
        # A real safety layer should instead fail safe on sensor loss.
        self.stop_pub.publish(Bool(data=obstacle_now))
        self.distance_pub.publish(
            Float32(data=float(nearest_distance) if math.isfinite(nearest_distance) else -1.0)
        )
        self.lateral_pub.publish(Float32(data=float(nearest_lateral)))
        self.detections_pub.publish(
            String(
                data=json.dumps(
                    {
                        "stamp_ns": image_stamp_ns,
                        "depth_fresh": depth_is_fresh,
                        "obstacle": obstacle_now,
                        "detections": detections,
                    }
                )
            )
        )

        try:
            output_message = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            output_message.header = message.header
            self.image_pub.publish(output_message)
        except CvBridgeError as exc:
            self.get_logger().error(f"Could not publish annotated image: {exc}")

        if self.latest_depth is None and not self.warned_no_depth:
            self.get_logger().warn(
                "No depth image received yet. Set require_depth:=false only for "
                "2-D debugging; RGB alone does not provide reliable range."
            )
            self.warned_no_depth = True

    @staticmethod
    def _estimate_depth(
        center_x: float,
        center_y: float,
        rgb_width: int,
        rgb_height: int,
        depth: np.ndarray | None,
        depth_encoding: str,
    ) -> float:
        """Return the median depth around a detection center in metres."""
        if depth is None or depth.ndim < 2:
            return math.inf
        depth_height, depth_width = depth.shape[:2]
        u = int(np.clip(center_x * depth_width / max(rgb_width, 1), 0, depth_width - 1))
        v = int(np.clip(center_y * depth_height / max(rgb_height, 1), 0, depth_height - 1))
        radius = 4
        patch = depth[
            max(v - radius, 0) : min(v + radius + 1, depth_height),
            max(u - radius, 0) : min(u + radius + 1, depth_width),
        ].astype(np.float32)
        if depth_encoding.upper() in {"16UC1", "MONO16"} or depth.dtype == np.uint16:
            patch *= 0.001  # Standard ROS 16UC1 depth convention: millimetres.
        valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 100.0)]
        return float(np.median(valid)) if valid.size else math.inf


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
