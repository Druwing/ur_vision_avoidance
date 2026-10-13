# Results of 2026-10-12

Raw data behind the tables in the main README ("Experiments and results").

- `gazebo_episodes.jsonl`: one metrics_node row per Gazebo episode (12 episodes, all fixes applied). Aggregate with `python3 scripts/summarize_metrics.py results/2026-10-12/gazebo_episodes.jsonl`.
- `kinematic_eval.csv`: `scripts/evaluate_kinematic.py --model models/ppo_ur5e_scene.zip` (60 placements x 3 modes).
- `yolo_benchmark.csv`: `scripts/benchmark_yolo.py` on 93 auto-labelled Gazebo frames (`--alias bench=chair`).
- `annotated_depth_fallback.png`: `/vision/annotated` at the start of an episode (box found by the depth fallback; the floor strip is removed later by the localizer).
- `dataset_label_check.png`: projected ground-truth labels drawn on captured frames.

Setup: 12-core CPU, no CUDA, Docker image from this repository (ROS 2 Jazzy, Gazebo Harmonic).
