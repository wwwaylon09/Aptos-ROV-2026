"""Backward-compatible launcher for split ROV services."""

import signal
import subprocess
import sys
from subprocess import TimeoutExpired


def main():
    control = subprocess.Popen([sys.executable, "scripts/run_control_server.py"], start_new_session=True)
    video = subprocess.Popen([sys.executable, "scripts/run_video_server.py"], start_new_session=True)

    shutting_down = False

    def shutdown(*_):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True

        for proc in (control, video):
            if proc.poll() is None:
                proc.terminate()

        for proc in (control, video):
            if proc.poll() is not None:
                continue
            try:
                proc.wait(timeout=5)
            except TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    exited = None
    while exited is None:
        if control.poll() is not None:
            exited = "control"
        elif video.poll() is not None:
            exited = "video"

    shutdown()


if __name__ == "__main__":
    main()
