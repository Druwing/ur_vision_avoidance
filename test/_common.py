"""Shared helpers for the pure-Python tests (no ROS, no Gazebo required)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "config" / "rl_starter.yaml"
Q_HOME = np.array([0.0, -1.2, 1.4, -1.7, -1.57, 0.0])  # first motion_loop waypoint


def ensure_gymnasium() -> bool:
    """Install a minimal stand-in for gymnasium when it is not installed.

    Returns True if the stub is used. The stub only provides what
    FixedSceneAvoidanceEnv needs (Env, spaces.Box).
    """
    try:
        import gymnasium  # noqa: F401

        return False
    except ImportError:
        pass

    class Env:
        def __class_getitem__(cls, item):
            return cls

        def __init__(self):
            pass

        def reset(self, *, seed=None, options=None):
            self._seed = seed

    class Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    gym = types.ModuleType("gymnasium")
    gym.Env = Env
    spaces = types.ModuleType("gymnasium.spaces")
    spaces.Box = Box
    gym.spaces = spaces
    sys.modules["gymnasium"] = gym
    sys.modules["gymnasium.spaces"] = spaces
    return True


def true_clearance(kin, q, spheres) -> float:
    """Independent clearance check over TCP + arm sample points (not the controller's)."""
    points = np.vstack((kin.position(q), kin.link_points(q)))
    best = np.inf
    for center, radius in spheres:
        for point in points:
            best = min(best, float(np.linalg.norm(point - center)) - radius)
    return best
