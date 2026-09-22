"""Train the fixed-scene PPO starter policy.

Usage:
    python3 -m ur_vision_avoidance.train_fixed_scene --timesteps 100000

Augmented randomized domain is intentionally rejected here. This script is only
for validating the observation/action contract in the fixed starter scene.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .fixed_scene_env import FixedSceneAvoidanceEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("/tmp/ur_vision_avoidance_ppo"))
    parser.add_argument("--ard", action="store_true", help="Rejected in the starter stage.")
    args = parser.parse_args()
    if args.ard:
        raise SystemExit("ARD is intentionally disabled in this fixed-scene starter stage.")
    if args.timesteps <= 0:
        raise SystemExit("--timesteps must be positive")

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "Training requires stable-baselines3 and gymnasium. Install requirements-rl.txt."
        ) from exc

    env = FixedSceneAvoidanceEnv(seed=args.seed)
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    env.close()
    print(f"Saved fixed-scene PPO policy to {args.output}.zip")


if __name__ == "__main__":
    main()
