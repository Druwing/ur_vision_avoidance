#!/usr/bin/env bash
# Capture an auto-labelled dataset in Gazebo for scripts/benchmark_yolo.py.
#
# Usage (inside the container, workspace sourced):
#   scripts/capture_dataset.sh [duration_s] [output_dir]
# The arm cycles through viewpoints (motion_loop with its stop topic remapped,
# so detections do not pause it) while dataset_capture saves one frame every
# 0.5 s with labels projected from config/scene_ground_truth.yaml.
# The default viewpoints keep >= 8 cm from the ground-truth obstacles
# (checked with ur_kinematics + metrics.true_clearance on the joint paths).
set -u
DURATION=${1:-180}
OUT=${2:-/tmp/ur_vision_avoidance_dataset}
WAYPOINTS=${WAYPOINTS:-"[0.0,-1.2,1.4,-1.7,-1.57,0.0,0.0,-1.2,1.4,-1.7,-1.57,0.5,0.3,-1.3,1.5,-1.75,-1.57,0.3,0.3,-1.0,1.1,-1.6,-1.57,-0.3,0.0,-1.4,1.8,-1.9,-1.57,0.0,0.55,-1.2,1.4,-1.7,-1.57,-0.6,0.15,-0.9,0.9,-1.5,-1.57,0.0]"}
mkdir -p "$OUT"

setsid ros2 launch ur_vision_avoidance vision_bringup.launch.py \
  gazebo_gui:=false launch_rviz:=false device:=cpu > "$OUT/launch.log" 2>&1 &
LAUNCH=$!
cleanup() {
  kill -INT -- "-$LAUNCH" "-${LOOP:-0}" "-${CAPTURE:-0}" 2>/dev/null
  sleep 6
  kill -KILL -- "-$LAUNCH" "-${LOOP:-0}" "-${CAPTURE:-0}" 2>/dev/null
  pkill -f "gz sim" 2>/dev/null
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  timeout 5 ros2 topic echo /camera/image_raw --once --field header.stamp >/dev/null 2>&1 && break
  sleep 2
done

setsid ros2 run ur_vision_avoidance dataset_capture --ros-args -p use_sim_time:=true \
  -p output_dir:="$OUT" -p period_s:=0.5 > "$OUT/capture.log" 2>&1 &
CAPTURE=$!
setsid ros2 run ur_vision_avoidance motion_loop --ros-args -p use_sim_time:=true \
  -p obstacle_topic:=/dataset/unused_stop -p waypoint_duration_s:=4.0 \
  -p waypoints:="$WAYPOINTS" > "$OUT/motion_loop.log" 2>&1 &
LOOP=$!

sleep "$DURATION"
echo "frames: $(ls "$OUT/images" | wc -l), labelled: $(grep -c '"objects": \[{' "$OUT/annotations.jsonl" 2>/dev/null)"
