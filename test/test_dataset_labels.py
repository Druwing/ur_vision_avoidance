import unittest

import numpy as np

import _common  # noqa: F401
from ur_vision_avoidance.dataset_labels import depth_support, project_box, yolo_line

FX = FY = 554.25
CX, CY, W, H = 320.0, 240.0, 640, 480
# camera at the origin of base_link looking along +x (optical z = base x)
R_CAM_FROM_BASE = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
T = np.zeros(3)


class DatasetLabelsTest(unittest.TestCase):
    def test_box_straight_ahead_projects_to_a_centred_rectangle(self):
        box = project_box([1.0, 0, 0], [0.2, 0.2, 0.2], R_CAM_FROM_BASE, T, FX, FY, CX, CY, W, H)
        x1, y1, x2, y2 = box.bbox_xyxy
        self.assertAlmostEqual((x1 + x2) / 2, CX, places=6)
        self.assertAlmostEqual((y1 + y2) / 2, CY, places=6)
        # nearest face at 0.9 m, 0.2 m wide -> 0.2 * fx / 0.9 pixels
        self.assertAlmostEqual(x2 - x1, 0.2 * FX / 0.9, places=4)
        self.assertAlmostEqual(box.near_z, 0.9)
        self.assertAlmostEqual(box.far_z, 1.1)
        self.assertAlmostEqual(box.inside_ratio, 1.0)

    def test_box_behind_or_outside_the_view_is_dropped(self):
        self.assertIsNone(project_box([-1.0, 0, 0], [0.2] * 3, R_CAM_FROM_BASE, T, FX, FY, CX, CY, W, H))
        self.assertIsNone(project_box([1.0, 5.0, 0], [0.2] * 3, R_CAM_FROM_BASE, T, FX, FY, CX, CY, W, H))

    def test_truncated_box_reports_inside_ratio(self):
        box = project_box([1.0, 0.6, 0], [0.2] * 3, R_CAM_FROM_BASE, T, FX, FY, CX, CY, W, H)
        self.assertEqual(box.bbox_xyxy[0], 0.0)
        self.assertLess(box.inside_ratio, 1.0)

    def test_depth_support_detects_occlusion(self):
        box = project_box([1.0, 0, 0], [0.2] * 3, R_CAM_FROM_BASE, T, FX, FY, CX, CY, W, H)
        visible = np.full((H, W), 0.9, np.float32)
        occluded = np.full((H, W), 0.4, np.float32)  # something in front
        self.assertGreater(depth_support(visible, box), 0.9)
        self.assertLess(depth_support(occluded, box), 0.1)

    def test_yolo_line_is_normalized(self):
        self.assertEqual(yolo_line(1, (160, 120, 480, 360), W, H), "1 0.500000 0.500000 0.500000 0.500000")


if __name__ == "__main__":
    unittest.main()
