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

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32, String
from ultralytics import YOLO


class VisionNode(Node):
    """Run object detection on a ROS 2 RGB-D camera stream."""

    def __init__(self) -> None:
        super().__init__("vision_obstacle_detector")

        self.declare_parameter("model", "yolo26n.pt")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth_image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("annotated_topic", "/vision/annotated")
        self.declare_parameter("detections_topic", "/vision/detections")
        self.declare_parameter("stop_topic", "/obstacle/stop")
        self.declare_parameter("distance_topic", "/obstacle/nearest_distance")
        self.declare_parameter("lateral_topic", "/obstacle/lateral_offset")
        self.declare_parameter("confidence", 0.45)
        self.declare_parameter("stop_distance_m", 0.80)
        self.declare_parameter("require_depth", True)
        self.declare_parameter(
            "target_classes",
            ["person", "chair", "car", "truck", "bus", "bicycle", "motorcycle"],
        )
        self.declare_parameter("device", "")
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop_s", 0.08)
        self.declare_parameter("depth_roi_margin_ratio", 0.15)
        self.declare_parameter("depth_percentile", 25.0)
        self.declare_parameter("min_valid_depth_samples", 20)
        self.declare_parameter("depth_min_m", 0.05)
        self.declare_parameter("depth_max_m", 10.0)
        self.declare_parameter("path_left_ratio", 0.20)
        self.declare_parameter("path_right_ratio", 0.80)
        self.declare_parameter("min_path_overlap_ratio", 0.15)

        model_name = str(self.get_parameter("model").value)
        self.confidence = float(self.get_parameter("confidence").value)
        self.stop_distance_m = float(self.get_parameter("stop_distance_m").value)
        self.require_depth = bool(self.get_parameter("require_depth").value)
        self.target_classes = {
            str(item).lower() for item in self.get_parameter("target_classes").value
        }

        self.device = str(self.get_parameter("device").value).strip() or None
        self.sync_queue_size = int(self.get_parameter("sync_queue_size").value)
        self.sync_slop_s = float(self.get_parameter("sync_slop_s").value)
        self.depth_roi_margin_ratio = float(
            self.get_parameter("depth_roi_margin_ratio").value
        )
        self.depth_percentile = float(self.get_parameter("depth_percentile").value)
        self.min_valid_depth_samples = int(
            self.get_parameter("min_valid_depth_samples").value
        )
        self.depth_min_m = float(self.get_parameter("depth_min_m").value)
        self.depth_max_m = float(self.get_parameter("depth_max_m").value)
        self.path_left_ratio = float(self.get_parameter("path_left_ratio").value)
        self.path_right_ratio = float(self.get_parameter("path_right_ratio").value)
        self.min_path_overlap_ratio = float(
            self.get_parameter("min_path_overlap_ratio").value
        )

        if not 0.0 <= self.depth_roi_margin_ratio < 0.5:
            raise ValueError("depth_roi_margin_ratio must be in [0.0, 0.5)")
        if not 0.0 <= self.depth_percentile <= 100.0:
            raise ValueError("depth_percentile must be in [0, 100]")
        if self.min_valid_depth_samples < 1:
            raise ValueError("min_valid_depth_samples must be >= 1")
        if self.depth_min_m <= 0.0 or self.depth_max_m <= self.depth_min_m:
            raise ValueError("depth range is invalid")
        if not 0.0 <= self.path_left_ratio < self.path_right_ratio <= 1.0:
            raise ValueError("path ratios must satisfy 0 <= left < right <= 1")
        if not 0.0 <= self.min_path_overlap_ratio <= 1.0:
            raise ValueError("min_path_overlap_ratio must be in [0, 1]")

        self.bridge = CvBridge()
        self.model = YOLO(model_name)
        self.camera_info: CameraInfo | None = None
        self.warned_camera_mismatch = False
        self.warned_depth_conversion = False

        image_topic = str(self.get_parameter("image_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)

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

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

        self.rgb_sub = message_filters.Subscriber(
            self,
            Image,
            image_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            depth_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop_s,
        )
        self.sync.registerCallback(self.rgb_depth_callback)

        self.get_logger().info(
            f"Loaded {model_name}; RGB={image_topic}, depth={depth_topic}; "
            f"target classes={sorted(self.target_classes)}; "
            f"sync_slop={self.sync_slop_s:.3f}s"
        )

    def camera_info_callback(self, message: CameraInfo) -> None:
        self.camera_info = message

    def rgb_depth_callback(self, rgb_message: Image, depth_message: Image) -> None:
        """Process one approximately time-synchronized RGB/depth pair."""
        try:
            image = self.bridge.imgmsg_to_cv2(rgb_message, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(
                depth_message, desired_encoding="passthrough"
            )
        except CvBridgeError as exc:
            if not self.warned_depth_conversion:
                self.get_logger().error(f"Could not convert RGB/depth image: {exc}")
                self.warned_depth_conversion = True
            return

        if self.camera_info is not None:
            if (
                self.camera_info.width != image.shape[1]
                or self.camera_info.height != image.shape[0]
            ) and not self.warned_camera_mismatch:
                self.get_logger().warn(
                    "CameraInfo resolution does not match RGB image: "
                    f"CameraInfo={self.camera_info.width}x{self.camera_info.height}, "
                    f"RGB={image.shape[1]}x{image.shape[0]}"
                )
                self.warned_camera_mismatch = True

        result = self.model.predict(
            source=image,
            conf=self.confidence,
            imgsz=640,
            device=self.device,
            verbose=False,
        )[0]
        annotated = result.plot(show=False)

        image_stamp_ns = (
            int(rgb_message.header.stamp.sec) * 1_000_000_000
            + int(rgb_message.header.stamp.nanosec)
        )
        depth_stamp_ns = (
            int(depth_message.header.stamp.sec) * 1_000_000_000
            + int(depth_message.header.stamp.nanosec)
        )
        sync_delta_s = abs(image_stamp_ns - depth_stamp_ns) / 1_000_000_000.0
        depth_is_fresh = sync_delta_s <= self.sync_slop_s

        detections: list[dict[str, Any]] = []
        nearest_distance = math.inf
        nearest_lateral = 0.0
        obstacle_now = False
        image_height, image_width = image.shape[:2]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                label = str(result.names[int(class_id)]).lower()
                if self.target_classes and label not in self.target_classes:
                    continue

                x1, y1, x2, y2 = [float(value) for value in box]
                x1 = float(np.clip(x1, 0.0, image_width))
                x2 = float(np.clip(x2, 0.0, image_width))
                y1 = float(np.clip(y1, 0.0, image_height))
                y2 = float(np.clip(y2, 0.0, image_height))
                if x2 <= x1 or y2 <= y1:
                    continue

                bbox_width = max(x2 - x1, 1.0)
                center_x = 0.5 * (x1 + x2)
                lateral_offset = (center_x / max(float(image_width), 1.0)) - 0.5
                distance_m = self._estimate_depth(
                    x1,
                    y1,
                    x2,
                    y2,
                    image_width,
                    image_height,
                    depth,
                    str(depth_message.encoding),
                )

                path_left_px = self.path_left_ratio * image_width
                path_right_px = self.path_right_ratio * image_width
                overlap_px = max(
                    0.0,
                    min(x2, path_right_px) - max(x1, path_left_px),
                )
                overlap_ratio = min(max(overlap_px / bbox_width, 0.0), 1.0)
                in_path = overlap_ratio >= self.min_path_overlap_ratio

                is_close = (
                    math.isfinite(distance_m)
                    and distance_m <= self.stop_distance_m
                    and depth_is_fresh
                )
                if not self.require_depth and not math.isfinite(distance_m):
                    is_close = False

                detection = {
                    "class": label,
                    "confidence": round(float(confidence), 4),
                    "bbox_xyxy": [round(v, 1) for v in (x1, y1, x2, y2)],
                    "lateral_offset": round(float(lateral_offset), 4),
                    "distance_m": None
                    if not math.isfinite(distance_m)
                    else round(distance_m, 3),
                    "path_overlap_ratio": round(float(overlap_ratio), 4),
                    "in_path": bool(in_path),
                    "close": bool(is_close),
                }
                detections.append(detection)

                if math.isfinite(distance_m) and distance_m < nearest_distance:
                    nearest_distance = distance_m
                    nearest_lateral = lateral_offset
                if in_path and is_close:
                    obstacle_now = True

        if self.require_depth and not depth_is_fresh:
            obstacle_now = False

        self.stop_pub.publish(Bool(data=obstacle_now))
        self.distance_pub.publish(
            Float32(
                data=float(nearest_distance)
                if math.isfinite(nearest_distance)
                else -1.0
            )
        )
        self.lateral_pub.publish(Float32(data=float(nearest_lateral)))
        self.detections_pub.publish(
            String(
                data=json.dumps(
                    {
                        "stamp_ns": image_stamp_ns,
                        "depth_fresh": depth_is_fresh,
                        "sync_delta_s": round(sync_delta_s, 6),
                        "obstacle": obstacle_now,
                        "detections": detections,
                    }
                )
            )
        )
        self._publish_annotated_image(annotated, rgb_message)

    def _publish_annotated_image(self, annotated: np.ndarray, source: Image) -> None:
        """Publish an annotated image without relying on cv_bridge's type map."""
        try:
            array = np.asarray(annotated)
            if array.ndim == 2:
                encoding = "mono8"
            elif array.ndim == 3 and array.shape[2] == 3:
                encoding = "bgr8"
            else:
                raise ValueError(f"unsupported annotated image shape: {array.shape}")
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            array = np.ascontiguousarray(array)

            output_message = Image()
            output_message.header = source.header
            output_message.height = int(array.shape[0])
            output_message.width = int(array.shape[1])
            output_message.encoding = encoding
            output_message.is_bigendian = 0
            output_message.step = int(array.strides[0])
            output_message.data = array.tobytes()
            self.image_pub.publish(output_message)
        except Exception as exc:
            self.get_logger().error(f"Could not publish annotated image: {exc}")

    def _estimate_depth(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        rgb_width: int,
        rgb_height: int,
        depth: np.ndarray | None,
        depth_encoding: str,
    ) -> float:
        """Estimate obstacle distance from a robust ROI inside the detection box."""
        if depth is None or depth.ndim < 2:
            return math.inf

        depth_height, depth_width = depth.shape[:2]
        scale_x = depth_width / max(float(rgb_width), 1.0)
        scale_y = depth_height / max(float(rgb_height), 1.0)

        dx1 = int(np.floor(np.clip(x1 * scale_x, 0, depth_width - 1)))
        dy1 = int(np.floor(np.clip(y1 * scale_y, 0, depth_height - 1)))
        dx2 = int(np.ceil(np.clip(x2 * scale_x, 0, depth_width)))
        dy2 = int(np.ceil(np.clip(y2 * scale_y, 0, depth_height)))
        if dx2 <= dx1 or dy2 <= dy1:
            return math.inf

        margin_x = int(round((dx2 - dx1) * self.depth_roi_margin_ratio))
        margin_y = int(round((dy2 - dy1) * self.depth_roi_margin_ratio))
        roi_x1 = min(dx1 + margin_x, dx2 - 1)
        roi_y1 = min(dy1 + margin_y, dy2 - 1)
        roi_x2 = max(dx2 - margin_x, roi_x1 + 1)
        roi_y2 = max(dy2 - margin_y, roi_y1 + 1)

        patch = depth[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float32)
        if depth_encoding.upper() in {"16UC1", "MONO16"} or depth.dtype == np.uint16:
            patch *= 0.001

        valid = patch[np.isfinite(patch)]
        valid = valid[(valid >= self.depth_min_m) & (valid <= self.depth_max_m)]
        if valid.size < self.min_valid_depth_samples:
            return math.inf

        return float(np.percentile(valid, self.depth_percentile))


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
