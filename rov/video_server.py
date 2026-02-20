import logging
import os
import signal
import sys
import threading
import time

import cv2
from flask import Flask, Response, render_template_string

from rov.shared_config import (
    CAMERA_CONFIG,
    RECONNECT_INTERVAL,
    VIDEO_HEALTH_LOG_INTERVAL_SECONDS,
    WEBCAM_HOST,
    WEBCAM_PORT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [video] %(levelname)s: %(message)s")
LOGGER = logging.getLogger("rov.video")

app = Flask(__name__)


def camera_dom_id(name: str) -> str:
    return "cam-" + "".join(ch.lower() if ch.isalnum() else "-" for ch in name)


class Camera:
    def __init__(self, name, config):
        self.name = name
        self.device_path = config["device_path"]
        self.width = int(config.get("width", 640))
        self.height = int(config.get("height", 480))
        self.fps = max(1, int(config.get("fps", 30)))
        self.frame_interval = 1.0 / self.fps
        self.jpeg_quality = int(config.get("jpeg_quality", 55))
        self.aspect_ratio = f"{self.width} / {self.height}"

        self.cap = None
        self.online = False
        self.running = True

        self.lock = threading.Lock()
        self.jpeg_bytes = None
        self.last_frame_monotonic = None
        self.failures = 0

        threading.Thread(target=self.run, daemon=True).start()

    def open_camera(self):
        if not os.path.exists(self.device_path):
            return False

        cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.cap = cap
        self.online = True
        LOGGER.info("camera=%s online path=%s", self.name, self.device_path)
        return True

    def close_camera(self):
        with self.lock:
            self.jpeg_bytes = None
            self.online = False

        if self.cap:
            self.cap.release()
        self.cap = None

    def run(self):
        while self.running:
            loop_start = time.monotonic()

            if self.online and not os.path.exists(self.device_path):
                LOGGER.warning("camera=%s disappeared, reconnecting", self.name)
                self.close_camera()
                time.sleep(RECONNECT_INTERVAL)
                continue

            if not self.online:
                if not self.open_camera():
                    time.sleep(RECONNECT_INTERVAL)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                self.failures += 1
                LOGGER.warning("camera=%s capture failed (%s)", self.name, self.failures)
                self.close_camera()
                time.sleep(0.5)
                continue

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )

            with self.lock:
                self.jpeg_bytes = encoded.tobytes() if ok else None
                self.last_frame_monotonic = time.monotonic()

            elapsed = time.monotonic() - loop_start
            sleep_for = self.frame_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def get_jpeg(self):
        if not self.online:
            return None

        with self.lock:
            return self.jpeg_bytes

    def get_status(self):
        now = time.monotonic()
        age = None if self.last_frame_monotonic is None else round(now - self.last_frame_monotonic, 2)
        return {
            "online": self.online,
            "resolution": f"{self.width}x{self.height}",
            "fps": self.fps,
            "last_frame_age_seconds": age,
            "capture_failures": self.failures,
        }

    def stop(self):
        self.running = False
        self.close_camera()


cameras = {name: Camera(name, config) for name, config in CAMERA_CONFIG.items()}
START_TIME = time.time()


def generate_frames(cam: Camera):
    while True:
        frame = cam.get_jpeg()
        if frame is None:
            time.sleep(0.1)
            continue

        yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


def _service_status():
    return {
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "cameras_online": sum(1 for cam in cameras.values() if cam.online),
        "cameras_total": len(cameras),
    }


@app.route("/status")
def status():
    return {
        "service": _service_status(),
        "cameras": {name: cam.get_status() for name, cam in cameras.items()},
    }


@app.route("/")
def index():
    tiles = ""

    for name, cam in cameras.items():
        dom_id = camera_dom_id(name)
        tiles += f"""
        <article class="camera-tile" id="{dom_id}" data-camera-name="{name}" style="--tile-ratio: {cam.aspect_ratio};">
            <img src="/video/{name}" class="camera-img" alt="{name} feed" loading="lazy">
            <div class="no-signal-label">NO SIGNAL</div>
            <div class="camera-label">{name}</div>
            <div class="camera-meta">{cam.width}x{cam.height} @ {cam.fps} FPS</div>
        </article>
        """

    return render_template_string(
        f"""
    <html>
    <head>
        <title>ROV Cameras</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            :root {{ --gap: 8px; --tile-border: #2b2b2e; --label-bg: rgba(0,0,0,0.62); }}
            * {{ box-sizing: border-box; }}
            html, body {{ margin: 0; min-height: 100%; background: #111; color: #ddd; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
            .layout {{ display: grid; grid-template-columns: 1fr; min-height: 100vh; gap: var(--gap); padding: var(--gap); }}
            .camera-column {{ min-height: 0; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--gap); align-content: start; }}
            .camera-tile {{ position: relative; width: 100%; aspect-ratio: var(--tile-ratio, 4 / 3); background: #000; border: 1px solid var(--tile-border); border-radius: 6px; overflow: hidden; }}
            .camera-img {{ width: 100%; height: 100%; display: block; object-fit: cover; background: #000; }}
            .camera-label, .camera-meta {{ position: absolute; left: 8px; font-size: 13px; background: var(--label-bg); color: #f0f0f0; padding: 2px 7px; border-radius: 4px; }}
            .camera-label {{ bottom: 34px; font-weight: 600; }}
            .camera-meta {{ bottom: 8px; color: #b8b8ba; }}
            .no-signal-label {{ position: absolute; inset: 0; display: none; align-items: center; justify-content: center; font-size: clamp(18px, 2.8vw, 30px); font-weight: bold; color: #ff4141; background: rgba(0,0,0,0.76); }}
            .camera-tile.no-signal img {{ display: none; }}
            .camera-tile.no-signal .no-signal-label {{ display: flex; }}
            @media (max-width: 1100px) {{ .camera-column {{ grid-template-columns: 1fr; }} }}
        </style>

        <script>
        const previousState = new Map();

        async function updateStatus() {{
            const res = await fetch("/status");
            const data = await res.json();
            const cameraData = data.cameras || {{}};
            for (const [name, details] of Object.entries(cameraData)) {{
                const tile = document.querySelector(`[data-camera-name="${{name}}"]`);
                if (!tile) continue;
                const online = Boolean(details.online);
                const wasOnline = previousState.get(name);
                tile.classList.toggle("no-signal", !online);
                if (online && wasOnline === false) {{
                    const img = tile.querySelector("img");
                    const base = img.src.split("?")[0];
                    img.src = base + "?t=" + Date.now();
                }}
                previousState.set(name, online);
            }}
        }}
        updateStatus();
        setInterval(updateStatus, 1000);
        </script>
    </head>
    <body>
        <main class="layout">
            <section class="camera-column">{tiles}</section>
        </main>
    </body>
    </html>
    """
    )


@app.route("/video/<camera>")
def video_feed(camera):
    cam = cameras.get(camera)
    if not cam:
        return "Camera not found", 404

    return Response(generate_frames(cam), mimetype="multipart/x-mixed-replace; boundary=frame")


def stop_all():
    for cam in cameras.values():
        cam.stop()


def log_service_health(stop_event):
    while not stop_event.wait(VIDEO_HEALTH_LOG_INTERVAL_SECONDS):
        service = _service_status()
        LOGGER.info(
            "health cameras_online=%s/%s uptime_s=%s",
            service["cameras_online"],
            service["cameras_total"],
            service["uptime_seconds"],
        )


def main():
    stop_event = threading.Event()

    def handle_shutdown(sig, frame):
        LOGGER.info("Received signal %s, shutting down video service", sig)
        stop_event.set()
        stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    threading.Thread(target=log_service_health, args=(stop_event,), daemon=True).start()

    try:
        LOGGER.info("Video server listening on %s:%s", WEBCAM_HOST, WEBCAM_PORT)
        app.run(host=WEBCAM_HOST, port=WEBCAM_PORT, threaded=True)
    finally:
        stop_event.set()
        stop_all()


if __name__ == "__main__":
    main()
