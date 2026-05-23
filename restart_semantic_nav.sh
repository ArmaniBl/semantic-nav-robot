#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
WEB_LOG="${WEB_LOG:-$PROJECT_DIR/logs/web/process.log}"
WEB_URL="http://${HOST}:${PORT}"

cd "$PROJECT_DIR"

log() {
  printf '[restart] %s\n' "$*"
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

wait_http() {
  local url="$1"
  local timeout_sec="$2"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS --max-time 1 "$url" >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - start >= timeout_sec )); then
      return 1
    fi
    sleep 1
  done
}

if [[ ! -f "$ROS_SETUP" ]]; then
  log "ROS setup not found: $ROS_SETUP"
  exit 1
fi

log "project: $PROJECT_DIR"
log "stopping project processes"
stop_pattern "${PROJECT_DIR}/ros_keyframe_recorder.py"
stop_pattern "${PROJECT_DIR}/semantic_nav_to_pose.py"
stop_pattern "${PROJECT_DIR}/launch/slam_nav2_launch.py"
stop_pattern "${PROJECT_DIR}/launch/localization_nav2_launch.py"
stop_pattern "${PROJECT_DIR}/launch/nav2_odom_launch.py"
stop_pattern "/slam_toolbox/async_slam_toolbox_node"
stop_pattern "/nav2_map_server/map_server"
stop_pattern "/nav2_amcl/amcl"
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

log "restarting Qdrant"
docker compose -f "$PROJECT_DIR/compose.qdrant.yml" down
docker compose -f "$PROJECT_DIR/compose.qdrant.yml" up -d

log "starting web panel: ${WEB_URL}"
mkdir -p "$(dirname "$WEB_LOG")"
: > "$WEB_LOG"
setsid bash -lc "cd '$PROJECT_DIR' && source '$ROS_SETUP' && exec python3 web_control_panel.py --host '$HOST' --port '$PORT'" \
  > "$WEB_LOG" 2>&1 < /dev/null &

if ! wait_http "${WEB_URL}/api/status" 20; then
  log "web panel did not become ready; last log lines:"
  tail -80 "$WEB_LOG" || true
  exit 1
fi

log "starting Map/Nav2"
curl -fsS --max-time 10 -X POST "${WEB_URL}/api/slam/start" >/dev/null

log "waiting for status"
sleep 3
curl -fsS --max-time 5 "${WEB_URL}/api/status"
printf '\n'

log "done"
log "open: ${WEB_URL}/"
log "web log: ${WEB_LOG}"
