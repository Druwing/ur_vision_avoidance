#!/usr/bin/env bash
# Run one headless Gazebo episode and record its metrics row.
#
# Usage (inside the container, workspace sourced):
#   scripts/run_gazebo_episode.sh <mode> [run_label] [timeout_s]
#     mode: potential_field | rl | nominal
# Environment:
#   OUT=/tmp/ur_vision_avoidance_metrics   metrics + logs directory
#   MODEL=/path/to/detector.pt             YOLO checkpoint (default: bundled)
#   EXTRA="key:=value ..."                 extra launch arguments
#   SNAPSHOT=1                             save /vision/annotated at the start
#
# Example (from the host):
#   docker compose run --rm ros2jazzy-gazebo bash -lc \
#     "source install/setup.bash && src/ur_vision_avoidance/scripts/run_gazebo_episode.sh rl"
set -u
MODE=${1:-potential_field}
LABEL=${2:-$MODE}
TIMEOUT=${3:-120}
OUT=${OUT:-/tmp/ur_vision_avoidance_metrics}
mkdir -p "$OUT"
JSONL="$OUT/episodes.jsonl"
BEFORE=$( [ -f "$JSONL" ] && wc -l < "$JSONL" || echo 0 )

RL_ARGS=""
[ "$MODE" = "rl" ] && RL_ARGS="enable_rl_starter:=true"

start_simulation() {
  # own process group, so cleanup stops exactly this simulation
  setsid ros2 launch ur_vision_avoidance vision_bringup.launch.py \
    gazebo_gui:=false launch_rviz:=false device:=cpu \
    enable_action_adapter:=true adapter_enable_command:=true adapter_mode:="$MODE" $RL_ARGS \
    enable_metrics:=true metrics_output_dir:="$OUT" run_label:="$LABEL" \
    ${MODEL:+model:=$MODEL} ${EXTRA:-} > "$OUT/launch_${LABEL}.log" 2>&1 &
  LAUNCH=$!
}

cleanup() {
  kill -INT -- "-$LAUNCH" 2>/dev/null
  sleep 6
  kill -KILL -- "-$LAUNCH" 2>/dev/null
  pkill -f "gz sim" 2>/dev/null
  sleep 2
}
trap cleanup EXIT

# Under CPU load the controller spawners can time out ("Switch controller
# timed out"): without joint_state_broadcaster there is no /joint_states and
# the episode never starts. Check readiness and restart the simulation if needed.
READY=0
for attempt in 1 2 3; do
  start_simulation
  echo "[$LABEL] waiting for joint states and perception (attempt $attempt) ..."
  JOINTS=0
  for _ in $(seq 1 30); do
    timeout 5 ros2 topic echo /joint_states --once >/dev/null 2>&1 && { JOINTS=1; break; }
    sleep 2
  done
  PERCEPTION=0
  if [ "$JOINTS" = 1 ]; then
    for _ in $(seq 1 40); do
      timeout 5 ros2 topic echo /vision/detections --once >/dev/null 2>&1 && { PERCEPTION=1; break; }
      sleep 2
    done
  fi
  if [ "$PERCEPTION" = 1 ]; then READY=1; break; fi
  echo "[$LABEL] simulation not ready (joint_states=$JOINTS, detections=$PERCEPTION); restarting"
  cp "$OUT/launch_${LABEL}.log" "$OUT/launch_${LABEL}_failed${attempt}.log" 2>/dev/null
  cleanup
done
if [ "$READY" != 1 ]; then
  echo "[$LABEL] giving up: simulation never became ready"
  exit 1
fi
sleep 5  # let the tracker confirm tracks

if [ -n "${SNAPSHOT:-}" ]; then
  timeout 15 python3 - "$OUT/annotated_${LABEL}.png" <<'PY'
import sys, time, numpy as np, cv2, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
rclpy.init(); node = Node("snapshot"); got = []
node.create_subscription(Image, "/vision/annotated",
    lambda m: got.append(np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)), 5)
t = time.time()
while not got and time.time() - t < 10: rclpy.spin_once(node, timeout_sec=0.2)
if got: cv2.imwrite(sys.argv[1], np.ascontiguousarray(got[-1])); print("snapshot", sys.argv[1])
PY
fi

echo "[$LABEL] starting episode"
timeout 10 ros2 topic pub --once /task/start std_msgs/msg/Empty "{}" >/dev/null
for _ in $(seq 1 "$TIMEOUT"); do
  AFTER=$( [ -f "$JSONL" ] && wc -l < "$JSONL" || echo 0 )
  [ "$AFTER" -gt "$BEFORE" ] && break
  sleep 1
done
if [ "${AFTER:-0}" -gt "$BEFORE" ]; then
  tail -n 1 "$JSONL"
else
  echo "[$LABEL] no metrics row after ${TIMEOUT}s; stopping episode"
  timeout 10 ros2 topic pub --once /task/stop std_msgs/msg/Empty "{}" >/dev/null
  sleep 3
  tail -n 1 "$JSONL" 2>/dev/null
fi
