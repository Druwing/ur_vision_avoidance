import math
import unittest

import numpy as np

import _common  # noqa: F401
from ur_vision_avoidance.camera_geometry import (
    backproject,
    detection_to_camera_points,
    estimate_radius,
    quat_to_matrix,
)

FX = FY = 554.25  # 640 px wide, 60 degree horizontal FOV
CX, CY = 320.0, 240.0


class CameraGeometryTest(unittest.TestCase):
    def test_principal_point_maps_to_optical_axis(self):
        point, ray, z = backproject(CX, CY, 1.5, FX, FY, CX, CY)
        np.testing.assert_allclose(point, [0.0, 0.0, 1.5], atol=1e-9)
        np.testing.assert_allclose(ray, [0.0, 0.0, 1.0], atol=1e-9)

    def test_pixel_offset_maps_to_lateral_distance(self):
        point, _, _ = backproject(CX + FX * 0.1, CY, 2.0, FX, FY, CX, CY)
        self.assertAlmostEqual(point[0], 0.2, places=6)  # 0.1 * z

    def test_range_depth_is_converted_to_z_depth(self):
        point, _, z = backproject(CX + FX, CY, math.sqrt(2.0), FX, FY, CX, CY, depth_is_range=True)
        self.assertAlmostEqual(z, 1.0, places=6)  # xn = 1 -> range = z*sqrt(2)
        self.assertAlmostEqual(point[0], 1.0, places=6)

    def test_quaternion_rotation(self):
        s = math.sqrt(0.5)
        rotation = quat_to_matrix(0.0, 0.0, s, s)  # +90 deg about z
        np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-9)
        with self.assertRaises(ValueError):
            quat_to_matrix(0, 0, 0, 0)

    def test_radius_estimate_is_clamped(self):
        self.assertEqual(estimate_radius((0, 0, 1, 1), 1.0, FX, FY, 0.03, 0.4), 0.03)
        self.assertEqual(estimate_radius((0, 0, 640, 480), 3.0, FX, FY, 0.03, 0.4), 0.4)

    def test_center_is_pushed_behind_surface_along_ray(self):
        bbox = (CX - 50, CY - 50, CX + 50, CY + 50)
        surface, center, radius = detection_to_camera_points(bbox, 1.0, FX, FY, CX, CY)
        self.assertAlmostEqual(center[2] - surface[2], radius, places=6)
        flat = detection_to_camera_points(bbox, 1.0, FX, FY, CX, CY, push_to_center=False)[1]
        np.testing.assert_allclose(flat, surface)

    def test_round_trip_through_a_moving_camera_pose(self):
        """Project a known base_link point into a camera, back-project, transform back."""
        rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # optical axis = +x
        translation = np.array([0.4, 0.1, 0.5])
        target_base = np.array([1.1, 0.25, 0.42])

        target_cam = rotation.T @ (target_base - translation)  # base -> camera
        u = FX * target_cam[0] / target_cam[2] + CX
        v = FY * target_cam[1] / target_cam[2] + CY
        surface_cam, _, _ = backproject(u, v, target_cam[2], FX, FY, CX, CY)
        recovered = rotation @ surface_cam + translation
        np.testing.assert_allclose(recovered, target_base, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
