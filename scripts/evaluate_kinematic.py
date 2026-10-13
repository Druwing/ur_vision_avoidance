#!/usr/bin/env python3
"""Compare avoidance strategies on the UR5e kinematic closed loop (no ROS).

The plant is an ideal joint integrator, as in test/test_closed_loop.py, but the
data path is the deployed one: task frame, rl_contract.build_observation, PPO,
and avoidance_logic.compute_step (safety layer always on). Obstacles are
spheres placed on a grid of positions relative to the nominal path, inside and
outside the training jitter of the scene.

Usage:
    python3 scripts/evaluate_kinematic.py --model models/ppo_ur5e_scene.zip \
        --csv /tmp/kinematic_eval.csv

Modes: rl (needs --model and stable-baselines3), potential_field, nominal.
This evaluates the decision layer only: no camera, latency or controller
dynamics. Use metrics_node in Gazebo for the integrated system.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ur_vision_avoidance.avoidance_logic import (  # noqa: E402
    AvoidanceParams,
    ObstacleView,
    compute_step,
    pick_escape_direction,
)
from ur_vision_avoidance.rl_contract import build_observation, task_rotation  # noqa: E402
from ur_vision_avoidance.scene_config import load_scene  # noqa: E402
from ur_vision_avoidance.ur_kinematics import UR5eKinematics  # noqa: E402

Q_HOME = np.array([0.0, -1.2, 1.4, -1.7, -1.57, 0.0])  # initial pose in the URDF


def true_clearance(kin, q, center, radius) -> float:
    points = np.vstack((kin.position(q), kin.link_points(q)))
    return float(np.min(np.linalg.norm(points - center, axis=1)) - radius)


def run_episode(kin, scene, mode, offset_task, radius, model, direction, time_factor=1.5):
    params = AvoidanceParams(
        nominal_speed=scene.nominal_speed,
        offset_max=scene.max_offset_speed,
        goal_tolerance=scene.goal_tolerance,
    )
    q = Q_HOME.copy()
    transform = kin.fk(q)
    start, r_ref = transform[:3, 3].copy(), transform[:3, :3].copy()
    rotation = task_rotation(direction)
    goal = start + rotation @ np.asarray(scene.goal_offset)
    center = start + rotation @ np.asarray(offset_task, dtype=float)
    obstacles = [ObstacleView(center, radius)]
    params.escape_direction = tuple(pick_escape_direction(kin, q, rotation[:, 0])[0])

    previous = np.zeros(6, np.float32)
    qdot = np.zeros(6)
    min_clearance, path_length, max_deviation = np.inf, 0.0, 0.0
    axis = rotation[:, 0]
    max_steps = int(scene.max_episode_steps * time_factor)
    reached, step = False, 0
    for step in range(max_steps):
        tcp = kin.position(q)
        rl_offset = None
        if mode == "rl":
            velocity = (kin.jacobian(q) @ qdot)[:3]
            observation = build_observation(
                rotation.T @ (goal - tcp),
                rotation.T @ (center - tcp),
                rotation.T @ velocity,
                previous,
                max(0.0, 1.0 - step / scene.max_episode_steps),
            )
            action, _ = model.predict(observation, deterministic=True)
            previous = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
            rl_offset = rotation @ (previous[:3] * scene.max_offset_speed)
        result = compute_step(kin, q, goal, obstacles, params, mode, rl_offset, r_ref)
        qdot = result.qdot
        q = q + qdot * scene.dt
        new_tcp = kin.position(q)
        path_length += float(np.linalg.norm(new_tcp - tcp))
        rel = new_tcp - start
        max_deviation = max(max_deviation, float(np.linalg.norm(rel - (rel @ axis) * axis)))
        min_clearance = min(min_clearance, true_clearance(kin, q, center, radius))
        if result.reached:
            reached = True
            break
    return {
        "reached": reached,
        "collision": min_clearance <= 0.0,
        "min_clearance_m": min_clearance,
        "steps": step + 1,
        "time_s": (step + 1) * scene.dt,
        "path_length_m": path_length,
        "max_deviation_m": max_deviation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="", help="PPO .zip; omit to skip the rl mode")
    parser.add_argument("--scene", default="ur5e_scene")
    parser.add_argument("--config", default=str(ROOT / "config" / "rl_starter.yaml"))
    parser.add_argument("--radius", type=float, default=0.06)
    parser.add_argument("--direction", type=float, nargs=3, default=[0.0, -1.0, 0.0])
    parser.add_argument("--csv", default="", help="write one row per (scenario, mode)")
    args = parser.parse_args()

    scene = load_scene(args.config, args.scene)
    kin = UR5eKinematics()
    model = None
    modes = ["potential_field", "nominal"]
    if args.model:
        from stable_baselines3 import PPO

        model = PPO.load(args.model)
        modes.insert(0, "rl")

    nominal = np.asarray(scene.obstacle_offset)
    jmin, jmax = np.asarray(scene.obstacle_jitter_min), np.asarray(scene.obstacle_jitter_max)
    grid = list(itertools.product((0.20, 0.22, 0.25, 0.30), (-0.10, -0.05, 0.0, 0.05, 0.10), (-0.06, 0.0, 0.06)))
    rows = []
    for along, lateral, vertical in grid:
        offset = np.array([along, lateral, vertical])
        in_dist = bool(np.all(offset - nominal >= jmin - 1e-9) and np.all(offset - nominal <= jmax + 1e-9))
        for mode in modes:
            result = run_episode(kin, scene, mode, offset, args.radius, model, np.asarray(args.direction))
            rows.append({"along": along, "lateral": lateral, "vertical": vertical,
                         "in_training_range": in_dist, "mode": mode, **result})

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.csv}")

    print(f"\n{len(grid)} obstacle placements, sphere radius {args.radius} m, scene {args.scene}")
    header = "| mode | subset | n | success | collisions | min clearance (m) | mean time (s) | mean path (m) |"
    print(header)
    print("|" + "---|" * 8)
    for mode in modes:
        for subset, keep in (("all", lambda r: True), ("in training range", lambda r: r["in_training_range"]),
                             ("outside", lambda r: not r["in_training_range"])):
            sel = [r for r in rows if r["mode"] == mode and keep(r)]
            if not sel:
                continue
            ok = [r for r in sel if r["reached"]]
            print(
                f"| {mode} | {subset} | {len(sel)} | {len(ok) / len(sel):.0%} | "
                f"{sum(r['collision'] for r in sel)} | {min(r['min_clearance_m'] for r in sel):.3f} | "
                f"{np.mean([r['time_s'] for r in ok]) if ok else float('nan'):.1f} | "
                f"{np.mean([r['path_length_m'] for r in ok]) if ok else float('nan'):.3f} |"
            )


if __name__ == "__main__":
    main()
