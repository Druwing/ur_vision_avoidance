#!/usr/bin/env python3
"""Capture an auto-labelled RGB dataset from the simulated camera.

For every ``period_s`` (ROS time) the node saves the RGB frame and, for each
ground-truth obstacle visible in it, a 2D box obtained by projecting the 3D box
(config/scene_ground_truth.yaml) with CameraInfo and the TF at the image time
(see dataset_labels.py). Labels hidden behind something else (low depth
support) or truncated too much are skipped.

Output layout in ``output_dir`` (Ultralytics-compatible):
    images/frame_000001.png
    labels/frame_000001.txt     one "class cx cy w h" line per object
    annotations.jsonl           pixel boxes, depth range, support, pose
    data.yaml                   class names (for fine-tuning a custom model)

Move the arm while capturing (e.g. motion_loop with its stop topic remapped)
to get several viewpoints. Use scripts/benchmark_yolo.py on the result.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from cv_bridge import CvBridge, CvBridgeError
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from .camera_geometry import quat_to_matrix
from .dataset_labels import depth_support, project_box, yolo_line
from .metrics import load_ground_truth
from .metrics_node import _default_ground_truth


class DatasetCaptureNode(Node):
    def __init__(self) -> None:
        super().__init__("dataset_capture")
        self.declare_parameter("output_dir", "/tmp/ur_vision_avoidance_dataset")
        self.declare_parameter("ground_truth_file", "")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth_image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_optical_frame")
        self.declare_parameter("period_s", 0.5)
        self.declare_parameter("max_frames", 500)
        self.declare_parameter("min_depth_support", 0.15)
        self.declare_parameter("min_inside_ratio", 0.4)
        self.declare_parameter("min_box_px", 12.0)
        self.declare_parameter("save_empty_frames", True)

        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)
        gt_path = str(self.get_parameter("ground_truth_file").value) or _default_ground_truth()
        self.boxes = load_ground_truth(gt_path)
        self.class_names = sorted({box.class_name for box in self.boxes})
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.period_s = float(self.get_parameter("period_s").value)
        self.max_frames = int(self.get_parameter("max_frames").value)
        self.min_support = float(self.get_parameter("min_depth_support").value)
        self.min_inside = float(self.get_parameter("min_inside_ratio").value)
        self.min_box_px = float(self.get_parameter("min_box_px").value)
        self.save_empty = bool(self.get_parameter("save_empty_frames").value)

        self.frame_index = len(list((self.output_dir / "images").glob("frame_*.png")))
        self.last_saved_s = -1e9
        self.intrinsics = None
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        (self.output_dir / "data.yaml").write_text(
            f"path: {self.output_dir}\ntrain: images\nval: images\n"
            + "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(self.class_names)),
            encoding="utf-8",
        )

        self.create_subscription(
            CameraInfo, str(self.get_parameter("camera_info_topic").value),
            self.on_camera_info, qos_profile_sensor_data,
        )
        rgb = message_filters.Subscriber(
            self, Image, str(self.get_parameter("image_topic").value), qos_profile=qos_profile_sensor_data
        )
        depth = message_filters.Subscriber(
            self, Image, str(self.get_parameter("depth_topic").value), qos_profile=qos_profile_sensor_data
        )
        self.sync = message_filters.ApproximateTimeSynchronizer([rgb, depth], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.on_frame)
        self.get_logger().info(
            f"Capturing to {self.output_dir} every {self.period_s}s, classes={self.class_names}"
        )

    def on_camera_info(self, message: CameraInfo) -> None:
        k = message.k
        if k[0] > 0.0 and k[4] > 0.0:
            self.intrinsics = (float(k[0]), float(k[4]), float(k[2]), float(k[5]))

    def on_frame(self, rgb_message: Image, depth_message: Image) -> None:
        if self.intrinsics is None or self.frame_index >= self.max_frames:
            return
        stamp_ns = int(rgb_message.header.stamp.sec) * 1_000_000_000 + int(rgb_message.header.stamp.nanosec)
        if stamp_ns / 1e9 - self.last_saved_s < self.period_s:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, self.camera_frame, Time(nanoseconds=stamp_ns), timeout=Duration(seconds=0.05)
            )
            image = self.bridge.imgmsg_to_cv2(rgb_message, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_message, desired_encoding="passthrough").astype(np.float32)
        except (TransformException, CvBridgeError) as exc:
            self.get_logger().warning(f"Frame skipped: {exc}", throttle_duration_sec=2.0)
            return
        if depth_message.encoding.upper() in {"16UC1", "MONO16"}:
            depth *= 0.001
        q, t = transform.transform.rotation, transform.transform.translation
        rotation_base_from_cam = quat_to_matrix(q.x, q.y, q.z, q.w)
        translation_base_from_cam = np.array([t.x, t.y, t.z])
        rotation_cam_from_base = rotation_base_from_cam.T
        translation_cam_from_base = -rotation_cam_from_base @ translation_base_from_cam

        fx, fy, cx, cy = self.intrinsics
        height, width = image.shape[:2]
        objects = []
        for box in self.boxes:
            projected = project_box(
                box.center, box.size, rotation_cam_from_base, translation_cam_from_base,
                fx, fy, cx, cy, width, height,
            )
            if projected is None:
                continue
            x1, y1, x2, y2 = projected.bbox_xyxy
            support = depth_support(depth, projected)
            if (
                projected.inside_ratio < self.min_inside
                or support < self.min_support
                or min(x2 - x1, y2 - y1) < self.min_box_px
            ):
                continue
            objects.append({
                "name": box.name,
                "class_name": box.class_name,
                "class_id": self.class_names.index(box.class_name),
                "bbox_xyxy": [round(v, 1) for v in projected.bbox_xyxy],
                "near_z_m": round(projected.near_z, 3),
                "depth_support": round(support, 3),
                "inside_ratio": round(projected.inside_ratio, 3),
            })
        if not objects and not self.save_empty:
            return

        self.frame_index += 1
        self.last_saved_s = stamp_ns / 1e9
        stem = f"frame_{self.frame_index:06d}"
        cv2.imwrite(str(self.output_dir / "images" / f"{stem}.png"), image)
        (self.output_dir / "labels" / f"{stem}.txt").write_text(
            "".join(yolo_line(o["class_id"], o["bbox_xyxy"], width, height) + "\n" for o in objects),
            encoding="utf-8",
        )
        with open(self.output_dir / "annotations.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "image": f"images/{stem}.png", "stamp_ns": stamp_ns, "width": width, "height": height,
                "camera_position": [round(float(v), 4) for v in translation_base_from_cam],
                "objects": objects,
            }) + "\n")
        if self.frame_index % 20 == 0:
            self.get_logger().info(f"{self.frame_index} frames captured")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DatasetCaptureNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
