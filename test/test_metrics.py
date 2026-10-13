import unittest

import numpy as np

from _common import Q_HOME, ROOT
from ur_vision_avoidance.metrics import (
    EpisodeRecorder,
    GroundTruthBox,
    box_signed_distance,
    load_ground_truth,
    nearest_box,
    true_clearance,
)
from ur_vision_avoidance.ur_kinematics import UR5eKinematics


def box(center, size=(0.1, 0.1, 0.1), name="b"):
    return GroundTruthBox(name, "unknown", np.asarray(center, float), np.asarray(size, float))


class MetricsTest(unittest.TestCase):
    def test_signed_distance_outside_on_face_and_inside(self):
        center, size = np.zeros(3), np.array([0.2, 0.2, 0.2])
        points = np.array([[0.3, 0, 0], [0.1, 0, 0], [0.0, 0, 0], [0.2, 0.2, 0]])
        d = box_signed_distance(points, center, size)
        np.testing.assert_allclose(d, [0.2, 0.0, -0.1, np.sqrt(0.02)], atol=1e-12)

    def test_ground_truth_file_matches_the_world(self):
        boxes = {b.name: b for b in load_ground_truth(ROOT / "config" / "scene_ground_truth.yaml")}
        self.assertIn("box_obstacle", boxes)
        self.assertIn("chair_obstacle", boxes)
        world = (ROOT / "worlds" / "obstacle_world.sdf").read_text()
        self.assertIn("<pose>0.63 -0.09 0.37 0 0 0</pose>", world)
        np.testing.assert_allclose(boxes["box_obstacle"].center, [0.63, -0.09, 0.37])

    def test_initial_pose_is_clear_of_every_obstacle(self):
        # Regression: the old chair pose enclosed the TCP at the initial pose.
        kin = UR5eKinematics()
        boxes = load_ground_truth(ROOT / "config" / "scene_ground_truth.yaml")
        self.assertGreater(true_clearance(kin, Q_HOME, boxes), 0.05)
        old_chair = box((0.80, -0.0415, 0.4085), (0.578, 0.513, 0.817))
        self.assertLess(true_clearance(kin, Q_HOME, [old_chair]), 0.0)

    def test_nearest_box(self):
        boxes = [box((0, 0, 0), name="a"), box((1, 0, 0), name="b")]
        found, distance = nearest_box([0.9, 0, 0], boxes)
        self.assertEqual(found.name, "b")
        self.assertAlmostEqual(distance, 0.05)  # 0.05 m outside the face at x = 0.95

    def test_episode_summary(self):
        kin = UR5eKinematics()
        tcp = kin.position(Q_HOME)
        state = {"start": tcp.tolist(), "goal": (tcp + [0, -0.45, 0]).tolist(),
                 "mode": "potential_field", "episode_duration_s": 20.0}
        recorder = EpisodeRecorder(kin, [box(tcp + [0, -0.3, 0])], 10.0, state)
        recorder.on_joints(Q_HOME)
        recorder.on_joints(Q_HOME + [0.05, 0, 0, 0, 0, 0])
        recorder.on_task_state({"status": "ok", "min_clearance_m": 0.2}, 10.0)
        recorder.on_task_state({"status": "hold_obstacle", "min_clearance_m": 0.12}, 11.0)
        recorder.on_task_state({"status": "goal_reached", "reached": True}, 12.5)
        recorder.on_detections({"obstacle": True, "age_s": 0.1, "inference_ms": 30.0,
                                "detections": [{"source": "depth"}, {"source": "yolo"}]})
        recorder.on_detections({"obstacle": False, "age_s": 0.2, "inference_ms": 40.0, "detections": []})
        recorder.on_tracks({"tracks": [{"visible": True, "label": "unknown",
                                        "surface": (tcp + [0, -0.25, 0]).tolist()}]})
        summary = recorder.summary(13.0, "reached")
        self.assertEqual(summary["time_to_goal_s"], 2.5)
        self.assertFalse(summary["collision"])
        self.assertGreater(summary["min_true_clearance_m"], 0.0)
        self.assertEqual(summary["min_estimated_clearance_m"], 0.12)
        self.assertGreater(summary["path_length_m"], 0.0)
        self.assertGreater(summary["max_deviation_m"], 0.0)
        self.assertEqual(summary["status_time_s"], {"ok": 1.0, "hold_obstacle": 1.5})
        self.assertEqual(summary["stop_frame_ratio"], 0.5)
        self.assertEqual(summary["detections_by_source"], {"depth": 1, "yolo": 1})
        self.assertAlmostEqual(summary["track_surface_error_m"]["mean"], 0.0, places=6)
        self.assertAlmostEqual(summary["inference_ms"]["mean"], 35.0)


if __name__ == "__main__":
    unittest.main()
