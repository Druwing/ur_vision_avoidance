"""Persistent obstacle tracks in a fixed frame (pure numpy, no ROS).

The camera is mounted on the end effector, so an obstacle leaves the field of
view as soon as the arm moves away. Tracks are therefore kept for ``memory_s``
after the last detection, which prevents a stop/resume oscillation.

Limitation: memory only expires by time. There is no free-space clearing (a
track is not removed when the camera looks at its location and sees nothing),
so a moved obstacle can leave a ghost until ``memory_s`` elapses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Measurement:
    position: np.ndarray  # estimated obstacle centre in the tracking frame
    surface: np.ndarray  # visible surface point in the tracking frame
    radius_m: float
    label: str = ""
    confidence: float = 1.0


@dataclass
class Track:
    track_id: int
    position: np.ndarray
    surface: np.ndarray
    radius_m: float
    label: str
    confidence: float
    first_seen_s: float
    last_seen_s: float
    hits: int = 1
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))


class ObstacleTracker:
    def __init__(
        self,
        gate_m: float = 0.30,
        alpha: float = 0.5,
        velocity_alpha: float = 0.3,
        memory_s: float = 10.0,
        min_hits: int = 2,
        unconfirmed_ttl_s: float = 1.0,
        max_speed_mps: float = 2.0,
        static_speed_mps: float = 0.08,
        min_dt_for_velocity_s: float = 0.05,
    ) -> None:
        self.gate_m = gate_m
        self.alpha = alpha
        self.velocity_alpha = velocity_alpha
        self.memory_s = memory_s
        self.min_hits = min_hits
        self.unconfirmed_ttl_s = unconfirmed_ttl_s
        self.max_speed_mps = max_speed_mps
        self.static_speed_mps = static_speed_mps
        self.min_dt_for_velocity_s = min_dt_for_velocity_s
        self._tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, measurements: list[Measurement], now_s: float) -> None:
        pairs = []
        for ti, track in enumerate(self._tracks):
            dt = max(now_s - track.last_seen_s, 0.0)
            gate = self.gate_m + min(float(np.linalg.norm(track.velocity)) * dt, 1.0)
            for mi, meas in enumerate(measurements):
                dist = float(np.linalg.norm(meas.position - track.position))
                if dist <= gate:
                    pairs.append((dist, ti, mi))
        pairs.sort()

        used_tracks: set[int] = set()
        used_meas: set[int] = set()
        for _, ti, mi in pairs:
            if ti in used_tracks or mi in used_meas:
                continue
            used_tracks.add(ti)
            used_meas.add(mi)
            self._apply(self._tracks[ti], measurements[mi], now_s)

        for mi, meas in enumerate(measurements):
            if mi in used_meas:
                continue
            self._tracks.append(
                Track(
                    track_id=self._next_id,
                    position=np.array(meas.position, dtype=float),
                    surface=np.array(meas.surface, dtype=float),
                    radius_m=float(meas.radius_m),
                    label=meas.label,
                    confidence=float(meas.confidence),
                    first_seen_s=now_s,
                    last_seen_s=now_s,
                )
            )
            self._next_id += 1

        self._prune(now_s)

    def _apply(self, track: Track, meas: Measurement, now_s: float) -> None:
        dt = now_s - track.last_seen_s
        if dt >= self.min_dt_for_velocity_s:
            raw = (meas.position - track.position) / dt
            norm = float(np.linalg.norm(raw))
            if norm > self.max_speed_mps:
                raw = raw * (self.max_speed_mps / norm)
            track.velocity = (1 - self.velocity_alpha) * track.velocity + self.velocity_alpha * raw
            if float(np.linalg.norm(track.velocity)) < self.static_speed_mps:
                track.velocity = np.zeros(3)
        a = self.alpha
        track.position = (1 - a) * track.position + a * meas.position
        track.surface = (1 - a) * track.surface + a * meas.surface
        track.radius_m = (1 - a) * track.radius_m + a * float(meas.radius_m)
        track.confidence = (1 - a) * track.confidence + a * float(meas.confidence)
        track.label = meas.label or track.label
        track.hits += 1
        track.last_seen_s = now_s

    def _prune(self, now_s: float) -> None:
        kept = []
        for track in self._tracks:
            age = now_s - track.last_seen_s
            limit = self.memory_s if track.hits >= self.min_hits else self.unconfirmed_ttl_s
            if age <= limit:
                kept.append(track)
        self._tracks = kept

    def prune(self, now_s: float) -> None:
        self._prune(now_s)

    def tracks(
        self, now_s: float, confirmed_only: bool = True, visible_within_s: float = 0.5
    ) -> list[dict[str, Any]]:
        out = []
        for track in self._tracks:
            if confirmed_only and track.hits < self.min_hits:
                continue
            age = max(now_s - track.last_seen_s, 0.0)
            out.append(
                {
                    "id": track.track_id,
                    "label": track.label,
                    "confidence": round(float(track.confidence), 4),
                    "position": [round(float(x), 4) for x in track.position],
                    "surface": [round(float(x), 4) for x in track.surface],
                    "radius_m": round(float(track.radius_m), 4),
                    "velocity": [round(float(x), 4) for x in track.velocity],
                    "hits": track.hits,
                    "visible": age <= visible_within_s,
                    "age_since_seen_s": round(age, 3),
                }
            )
        return out
