import unittest

import numpy as np

import _common  # noqa: F401  (adds the package to sys.path)
from ur_vision_avoidance.depth_fallback import find_near_blobs


def scene(background=3.0):
    depth = np.full((480, 640), background, dtype=np.float32)
    return depth


class DepthFallbackTest(unittest.TestCase):
    def test_near_box_is_found_with_its_extent_and_distance(self):
        depth = scene()
        depth[100:300, 200:400] = 0.30
        blobs = find_near_blobs(depth, max_range_m=0.6)
        self.assertEqual(len(blobs), 1)
        x1, y1, x2, y2 = blobs[0].bbox_xyxy
        self.assertLessEqual(abs(x1 - 200), 8)
        self.assertLessEqual(abs(x2 - 400), 8)
        self.assertLessEqual(abs(y1 - 100), 8)
        self.assertLessEqual(abs(y2 - 300), 8)
        self.assertAlmostEqual(blobs[0].distance_m, 0.30, places=5)

    def test_anchor_is_the_measured_surface_at_the_region_centre(self):
        # A floor seen from above: depth grows with the image row. The low
        # percentile is nearer than the floor at the region centre; the anchor
        # must report the depth actually measured at its pixel.
        depth = scene()
        rows = np.arange(480, dtype=np.float32)[:, None]
        depth[360:480, :] = np.broadcast_to(0.80 - (rows[360:480] - 360) * 0.002, (120, 640))
        blob = find_near_blobs(depth, max_range_m=0.85, max_depth_step_m=0.05)[0]
        u, v = blob.anchor_uv
        self.assertAlmostEqual(blob.anchor_depth_m, float(depth[int(v), int(u)]), delta=0.02)
        self.assertLess(blob.distance_m, blob.anchor_depth_m)

    def test_far_background_and_invalid_depth_produce_nothing(self):
        depth = scene()
        depth[0:50, 0:50] = np.nan
        depth[60:90, 60:90] = 0.0
        self.assertEqual(find_near_blobs(depth, max_range_m=0.6), [])

    def test_small_specks_are_ignored(self):
        depth = scene()
        depth[10:20, 10:20] = 0.2  # ~0.03 % of the image
        self.assertEqual(find_near_blobs(depth, max_range_m=0.6), [])

    def test_two_objects_at_different_depths_are_not_merged(self):
        depth = scene()
        depth[100:300, 100:300] = 0.25
        depth[100:300, 300:500] = 0.55  # touching, but 0.30 m behind
        blobs = find_near_blobs(depth, max_range_m=0.6)
        self.assertEqual(len(blobs), 2)
        self.assertLess(blobs[0].distance_m, blobs[1].distance_m)  # closest first

    def test_regions_already_covered_by_yolo_are_skipped(self):
        depth = scene()
        depth[100:300, 200:400] = 0.30
        blobs = find_near_blobs(depth, max_range_m=0.6, exclude_boxes=[(190, 90, 410, 310)])
        self.assertEqual(blobs, [])

    def test_millimetre_scale_mistake_is_not_silently_accepted(self):
        # 300 (mm interpreted as m) is out of range: the caller must convert units.
        depth = scene()
        depth[100:300, 200:400] = 300.0
        self.assertEqual(find_near_blobs(depth, max_range_m=0.6), [])


if __name__ == "__main__":
    unittest.main()
