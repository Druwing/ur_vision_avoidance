#!/usr/bin/env python3
"""Aggregate metrics_node episodes (episodes.jsonl) into a table per mode.

Usage:
    python3 scripts/summarize_metrics.py /tmp/ur_vision_avoidance_metrics/episodes.jsonl \
        [--labels potential_field_v5 rl_v3 ...] [--prefix rl_ potential_field_ nominal_]

Mean +- standard deviation over episodes; time and path only over successes.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict


def fmt(values, digits=3):
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl")
    parser.add_argument("--labels", nargs="*", default=None, help="keep only these run labels")
    parser.add_argument("--prefix", nargs="*", default=None, help="keep labels starting with any prefix")
    args = parser.parse_args()

    episodes = [json.loads(line) for line in open(args.jsonl, encoding="utf-8") if line.strip()]
    if args.labels:
        episodes = [e for e in episodes if e.get("run_label") in args.labels]
    if args.prefix:
        episodes = [e for e in episodes if any(str(e.get("run_label", "")).startswith(p) for p in args.prefix)]
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode.get("mode")].append(episode)

    print("| mode | n | success | collisions | min true clearance (m) | time to goal (s) | path (m) | max deviation (m) "
          "| detection age (s) | YOLO inference (ms) | depth-track error (m) |")
    print("|" + "---|" * 11)
    for mode, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ok = [e for e in items if e["outcome"] == "reached"]
        depth_err = [e.get("track_surface_error_by_label_m", {}).get("unknown", {}).get("mean") for e in items]
        print(
            f"| {mode} | {len(items)} | {len(ok)}/{len(items)} | {sum(e['collision'] for e in items)} | "
            f"{fmt([e['min_true_clearance_m'] for e in items])} | {fmt([e['time_to_goal_s'] for e in ok], 1)} | "
            f"{fmt([e['path_length_m'] for e in ok])} | {fmt([e['max_deviation_m'] for e in items])} | "
            f"{fmt([e['detection_age_s']['mean'] for e in items])} | {fmt([e['inference_ms']['mean'] for e in items], 0)} | "
            f"{fmt(depth_err)} |"
        )


if __name__ == "__main__":
    main()
