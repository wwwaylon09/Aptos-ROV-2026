"""Backward-compatible launcher for split ROV services."""

import signal
import subprocess
import sys


def main():
    control = subprocess.Popen([sys.executable, "scripts/run_control_server.py"])
    video = subprocess.Popen([sys.executable, "scripts/run_video_server.py"])

    def shutdown(*_):
        for proc in (control, video):
            if proc.poll() is None:
                proc.terminate()
        for proc in (control, video):
            proc.wait(timeout=10)
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
