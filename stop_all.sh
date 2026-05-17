#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

cd "$PROJECT_DIR"

log() {
  printf '[stop] %s\n' "$*"
}

stop_pattern() {
  local pattern="$1"

  if pgrep -f "$pattern" >/dev/null 2>&1; then
    log "stopping: $pattern"
    pkill -INT -f "$pattern" || true
    sleep 1
  fi

  if pgrep -f "$pattern" >/dev/null 2>&1; then
    pkill -TERM -f "$pattern" || true
    sleep 1
  fi

  if pgrep -f "$pattern" >/dev/null 2>&1; then
    pkill -KILL -f "$pattern" || true
  fi
}

stop_port() {
  local port="$1"
  local pids

  pids="$(ss -ltnp "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u)"

  if [[ -z "$pids" ]]; then
    return
  fi

  log "stopping process(es) on port ${port}: ${pids}"
  kill $pids || true
  sleep 1

  pids="$(ss -ltnp "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u)"

  if [[ -n "$pids" ]]; then
    kill -9 $pids || true
  fi
}

log "project: $PROJECT_DIR"
log "stopping project processes"

stop_pattern "${PROJECT_DIR}/ros_keyframe_recorder.py"
stop_pattern "${PROJECT_DIR}/semantic_nav_to_pose.py"
stop_pattern "${PROJECT_DIR}/launch/slam_nav2_launch.py"
stop_pattern "/slam_toolbox/async_slam_toolbox_node"
stop_pattern "/nav2_controller/controller_server"
stop_pattern "/nav2_planner/planner_server"
stop_pattern "/nav2_behaviors/behavior_server"
stop_pattern "/nav2_bt_navigator/bt_navigator"
stop_pattern "/nav2_lifecycle_manager/lifecycle_manager"
stop_pattern "/pointcloud_to_laserscan/pointcloud_to_laserscan_node"
stop_pattern "/tf2_ros/static_transform_publisher"
stop_pattern "web_control_panel.py --host ${HOST}"
stop_pattern "web_control_panel.py"

stop_port "$PORT"

if [[ -f "$PROJECT_DIR/compose.qdrant.yml" ]]; then
  log "stopping Qdrant"
  docker compose -f "$PROJECT_DIR/compose.qdrant.yml" down || true
fi

log "done"
