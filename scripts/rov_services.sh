#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:-}

run_control() {
  exec python3 scripts/run_control_server.py
}

run_video() {
  exec python3 scripts/run_video_server.py
}

run_both() {
  python3 scripts/run_control_server.py &
  CONTROL_PID=$!
  python3 scripts/run_video_server.py &
  VIDEO_PID=$!

  trap 'kill ${CONTROL_PID} ${VIDEO_PID} 2>/dev/null || true; wait' INT TERM EXIT
  wait -n ${CONTROL_PID} ${VIDEO_PID}
}

case "$ACTION" in
  control) run_control ;;
  video) run_video ;;
  both) run_both ;;
  *)
    echo "Usage: $0 {control|video|both}"
    exit 1
    ;;
esac
