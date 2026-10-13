import unittest

import numpy as np

import _common  # noqa: F401
from ur_vision_avoidance.obstacle_tracker import Measurement, ObstacleTracker


def meas(x, y, z, radius=0.05, label="chair"):
    p = np.array([x, y, z], dtype=float)
    return Measurement(position=p, surface=p - np.array([0.0, 0.0, radius]), radius_m=radius, label=label)


class TrackerTest(unittest.TestCase):
    def test_track_confirmed_after_min_hits(self):
        tracker = ObstacleTracker(min_hits=2)
        tracker.update([meas(0.5, 0.0, 0.3)], now_s=0.0)
        self.assertEqual(tracker.tracks(0.0), [])
        tracker.update([meas(0.5, 0.0, 0.3)], now_s=0.1)
        self.assertEqual(len(tracker.tracks(0.1)), 1)

    def test_unconfirmed_track_is_dropped_quickly(self):
        tracker = ObstacleTracker(min_hits=2, unconfirmed_ttl_s=1.0)
        tracker.update([meas(0.5, 0.0, 0.3)], now_s=0.0)
        tracker.update([], now_s=2.0)
        tracker.update([meas(0.5, 0.0, 0.3)], now_s=2.1)  # starts a NEW track
        self.assertEqual(tracker.tracks(2.1, confirmed_only=False)[0]["hits"], 1)

    def test_nearby_measurements_are_merged_and_smoothed(self):
        tracker = ObstacleTracker(alpha=0.5, min_hits=1)
        tracker.update([meas(0.50, 0.0, 0.3)], 0.0)
        tracker.update([meas(0.52, 0.0, 0.3)], 0.1)
        tracks = tracker.tracks(0.1)
        self.assertEqual(len(tracks), 1)
        self.assertAlmostEqual(tracks[0]["position"][0], 0.51, places=3)

    def test_distant_measurements_make_separate_tracks(self):
        tracker = ObstacleTracker(min_hits=1, gate_m=0.3)
        tracker.update([meas(0.5, 0.0, 0.3), meas(0.5, 0.6, 0.3)], 0.0)
        self.assertEqual(len(tracker.tracks(0.0)), 2)

    def test_memory_keeps_obstacle_after_it_leaves_view_then_expires(self):
        tracker = ObstacleTracker(min_hits=2, memory_s=5.0)
        for t in (0.0, 0.1, 0.2):
            tracker.update([meas(0.5, 0.0, 0.3)], t)
        tracker.update([], 2.0)  # camera no longer sees it
        remembered = tracker.tracks(2.0)
        self.assertEqual(len(remembered), 1)
        self.assertFalse(remembered[0]["visible"])
        self.assertGreater(remembered[0]["age_since_seen_s"], 1.5)
        tracker.update([], 6.0)  # beyond memory
        self.assertEqual(tracker.tracks(6.0), [])

    def test_velocity_estimated_for_moving_obstacle_and_zero_for_static(self):
        moving = ObstacleTracker(min_hits=1, alpha=1.0, velocity_alpha=1.0)
        for k in range(6):
            moving.update([meas(0.5 + 0.2 * 0.1 * k, 0.0, 0.3)], 0.1 * k)  # 0.2 m/s along x
        vx = moving.tracks(0.5)[0]["velocity"][0]
        self.assertAlmostEqual(vx, 0.2, delta=0.03)

        static = ObstacleTracker(min_hits=1)
        rng = np.random.default_rng(0)
        for k in range(30):
            noise = rng.normal(0.0, 0.003, 3)
            static.update([meas(0.5 + noise[0], noise[1], 0.3 + noise[2])], 0.1 * k)
        self.assertEqual(static.tracks(3.0)[0]["velocity"], [0.0, 0.0, 0.0])

    def test_reset_clears_everything(self):
        tracker = ObstacleTracker(min_hits=1)
        tracker.update([meas(0.5, 0.0, 0.3)], 0.0)
        tracker.reset()
        self.assertEqual(tracker.tracks(0.0), [])


if __name__ == "__main__":
    unittest.main()
