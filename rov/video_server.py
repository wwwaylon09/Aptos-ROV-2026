import json
import logging
import os
import signal
import sys
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

from rov.shared_config import (
    CAMERA_CONFIG,
    HUD_STATE_PATH,
    RECONNECT_INTERVAL,
    VIDEO_HEALTH_LOG_INTERVAL_SECONDS,
    WEBCAM_HOST,
    WEBCAM_PORT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [video] %(levelname)s: %(message)s")
LOGGER = logging.getLogger("rov.video")

app = Flask(__name__)

# Detection JSON schema used by POST /detections.
DETECTION_JSON_SCHEMA = {
    "type": "object",
    "required": ["camera_id", "timestamp", "label", "confidence", "bbox"],
    "properties": {
        "camera_id": {"type": "string", "minLength": 1},
        "timestamp": {"type": "number", "description": "Unix epoch seconds"},
        "label": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "bbox": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "description": "[x1, y1, x2, y2] pixel coordinates",
        },
    },
}
DETECTION_TTL_SECONDS = 4.0
DETECTION_LOCK = threading.Lock()
LATEST_DETECTIONS = {}


def camera_dom_id(name: str) -> str:
    return "cam-" + "".join(ch.lower() if ch.isalnum() else "-" for ch in name)


def camera_url_id(name: str, config: dict) -> str:
    configured = str(config.get("id", "")).strip().lower()
    base = configured or "".join(ch.lower() if ch.isalnum() else "-" for ch in name)
    safe = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in base).strip("-")
    return safe or "camera"


class Camera:
    def __init__(self, name, config):
        self.name = name
        self.camera_id = camera_url_id(name, config)
        self.role = str(config.get("role", "viewer")).strip().lower() or "viewer"
        self.detection_enabled = self.role == "hq"
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
            "camera_id": self.camera_id,
            "name": self.name,
            "role": self.role,
            "detection_enabled": self.detection_enabled,
            "online": self.online,
            "resolution": f"{self.width}x{self.height}",
            "target_fps": self.fps,
            "last_frame_age_seconds": age,
            "capture_failures": self.failures,
        }

    def stop(self):
        self.running = False
        self.close_camera()


cameras = {name: Camera(name, config) for name, config in CAMERA_CONFIG.items()}
camera_ids = {cam.camera_id: cam for cam in cameras.values()}
if len(camera_ids) != len(cameras):
    raise ValueError("Camera IDs must be unique and URL-safe")
START_TIME = time.time()
DETECTION_CAMERA_IDS = {cam.camera_id for cam in cameras.values() if cam.detection_enabled}
DEFAULT_HUD_STATE = {
    "thrusters": [0.0] * 8,
    "claw_angle": 0.0,
    "claw_rotation": 0.0,
    "syringe_angle": 0.0,
    "camera_angle": 0.0,
    "stabilization_enabled": False,
    "mpu_pitch_deg": 0.0,
    "mpu_roll_deg": 0.0,
    "client_connected": False,
    "updated_at": 0.0,
}


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


def _hud_status():
    try:
        with open(HUD_STATE_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        payload = {}

    state = DEFAULT_HUD_STATE.copy()
    if isinstance(payload, dict):
        state.update(payload)
    return state


def _normalize_detection(detection):
    if not isinstance(detection, dict):
        raise ValueError("Each detection must be an object")

    camera_id = str(detection.get("camera_id", "")).strip().lower()
    label = str(detection.get("label", "")).strip()

    if not camera_id:
        raise ValueError("camera_id is required")
    if camera_id not in DETECTION_CAMERA_IDS:
        allowed = ", ".join(sorted(DETECTION_CAMERA_IDS)) or "none"
        raise ValueError(f"camera_id '{camera_id}' is not configured for detection (allowed: {allowed})")
    if not label:
        raise ValueError("label is required")

    try:
        timestamp = float(detection["timestamp"])
        confidence = float(detection["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("timestamp and confidence must be numbers") from exc

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    bbox = detection.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("bbox must be [x1, y1, x2, y2]")

    try:
        normalized_bbox = [int(round(float(value))) for value in bbox]
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox values must be numbers") from exc

    return {
        "camera_id": camera_id,
        "timestamp": timestamp,
        "label": label,
        "confidence": round(confidence, 4),
        "bbox": normalized_bbox,
    }


def _extract_detections(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("Payload must be an object or array")
    if isinstance(payload.get("detections"), list):
        return payload["detections"]
    return [payload]


def _latest_detections_snapshot(include_stale=False):
    now_monotonic = time.monotonic()
    snapshot = {}
    with DETECTION_LOCK:
        for camera_id, entry in LATEST_DETECTIONS.items():
            age_seconds = now_monotonic - entry["received_monotonic"]
            if not include_stale and age_seconds > DETECTION_TTL_SECONDS:
                continue
            snapshot[camera_id] = {
                "updated_at": entry["updated_at"],
                "age_seconds": round(age_seconds, 2),
                "detections": entry["detections"],
            }
    return snapshot


@app.route("/status")
def status():
    return {
        "service": _service_status(),
        "cameras": {name: cam.get_status() for name, cam in cameras.items()},
        "detections": _latest_detections_snapshot(),
        "hud": _hud_status(),
    }


@app.route("/detections", methods=["GET", "POST"])
def detections():
    if request.method == "GET":
        return {
            "schema": DETECTION_JSON_SCHEMA,
            "ttl_seconds": DETECTION_TTL_SECONDS,
            "latest": _latest_detections_snapshot(),
        }

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Expected JSON payload"}), 400

    try:
        normalized_detections = [_normalize_detection(item) for item in _extract_detections(payload)]
    except ValueError as exc:
        return jsonify({"error": str(exc), "schema": DETECTION_JSON_SCHEMA}), 400

    now_monotonic = time.monotonic()
    now_unix = time.time()

    grouped = {}
    for detection in normalized_detections:
        grouped.setdefault(detection["camera_id"], []).append(detection)

    with DETECTION_LOCK:
        for camera_id, camera_detections in grouped.items():
            LATEST_DETECTIONS[camera_id] = {
                "updated_at": now_unix,
                "received_monotonic": now_monotonic,
                "detections": camera_detections,
            }

    return {
        "ok": True,
        "received": len(normalized_detections),
        "cameras": sorted(grouped.keys()),
        "ttl_seconds": DETECTION_TTL_SECONDS,
    }


@app.route("/cv/status")
def cv_status():
    """CV API status endpoint.

    URL format: /cv/status
    Response content type: application/json
    """

    return {
        "service": _service_status(),
        "cameras": {
            cam_id: {
                "camera_id": cam.camera_id,
                "name": cam.name,
                "role": cam.role,
                "detection_enabled": cam.detection_enabled,
                "resolution": f"{cam.width}x{cam.height}",
                "target_fps": cam.fps,
                "online": cam.get_status()["online"],
            }
            for cam_id, cam in camera_ids.items()
        },
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
            <div class="detections-panel{' empty' if not cam.detection_enabled else ''}" aria-live="polite" {'data-detection-enabled="false"' if not cam.detection_enabled else ''}>{'No recent detections'}</div>
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
            .layout {{ display: grid; grid-template-columns: 1fr minmax(270px, 330px); height: 100vh; gap: var(--gap); padding: var(--gap); overflow: hidden; }}
            .hud {{ border: 1px solid #252527; border-radius: 6px; padding: 14px; background: rgba(15, 15, 18, 0.9); min-height: 0; overflow: auto; }}
            .hud h2 {{ margin: 0 0 10px; font-size: 18px; color: #e7e7ea; }}
            .hud-grid {{ display: grid; gap: 9px; }}
            .hud-row {{ display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px dashed #252529; padding-bottom: 6px; }}
            .hud-label {{ color: #9898a3; font-size: 13px; }}
            .hud-value {{ color: #f2f2f5; font-weight: 600; font-variant-numeric: tabular-nums; }}
            .connected-yes, .state-enabled {{ color: #39c66d; }}
            .connected-no, .state-disabled {{ color: #ff6868; }}
            .thruster-list {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }}
            .thruster-item {{ display: grid; grid-template-columns: auto 1fr auto; gap: 8px; align-items: center; }}
            .thruster-name {{ color: #9898a3; font-size: 12px; min-width: 64px; }}
            .thruster-bar {{ height: 8px; border-radius: 999px; background: #1f1f22; overflow: hidden; border: 1px solid #2b2b30; }}
            .thruster-fill {{ height: 100%; width: 50%; background: linear-gradient(90deg, #39c66d, #e9c23f, #e25a5a); transition: width .18s linear; }}
            .thruster-value {{ color: #dfdfe5; font-size: 12px; min-width: 58px; text-align: right; font-variant-numeric: tabular-nums; }}
            .camera-column {{ min-height: 0; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); grid-auto-rows: minmax(0, 1fr); gap: var(--gap); align-content: stretch; }}
            .camera-tile {{ position: relative; width: 100%; height: 100%; min-height: 0; background: #000; border: 1px solid var(--tile-border); border-radius: 6px; overflow: hidden; }}
            .camera-img {{ width: 100%; height: 100%; display: block; object-fit: cover; background: #000; }}
            .camera-label, .camera-meta {{ position: absolute; left: 8px; font-size: 13px; background: var(--label-bg); color: #f0f0f0; padding: 2px 7px; border-radius: 4px; }}
            .camera-label {{ bottom: 34px; font-weight: 600; }}
            .camera-meta {{ bottom: 8px; color: #b8b8ba; }}
            .detections-panel {{ position: absolute; top: 8px; right: 8px; min-width: 170px; max-width: calc(100% - 16px); max-height: 46%; overflow: auto; font-size: 12px; line-height: 1.35; padding: 6px 8px; border-radius: 6px; background: rgba(0,0,0,0.65); color: #e8f0e8; border: 1px solid rgba(160,160,160,0.3); }}
            .detections-panel ul {{ margin: 0; padding-left: 16px; }}
            .detections-panel.empty {{ color: #9f9fa6; }}
            .no-signal-label {{ position: absolute; inset: 0; display: none; align-items: center; justify-content: center; font-size: clamp(18px, 2.8vw, 30px); font-weight: bold; color: #ff4141; background: rgba(0,0,0,0.76); }}
            .camera-tile.no-signal img {{ display: none; }}
            .camera-tile.no-signal .no-signal-label {{ display: flex; }}
            @media (max-width: 1200px) {{ .layout {{ grid-template-columns: 1fr; height: auto; overflow: visible; }} .camera-column {{ grid-template-columns: 1fr; grid-auto-rows: auto; }} .camera-tile {{ aspect-ratio: var(--tile-ratio, 4 / 3); height: auto; }} }}
        </style>

        <script>
        const previousState = new Map();
        const latestDetectionByCamera = new Map();
        const DETECTION_UI_TTL_MS = {int(DETECTION_TTL_SECONDS * 1000)};
        const asNumber = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
        const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
        const thrusterPct = (value) => (clamp((asNumber(value) + 1) / 2, 0, 1) * 100);
        const formatInput = (value) => asNumber(value).toFixed(2);

        function updateHud(hud) {{
            if (!hud) return;
            const connected = Boolean(hud.client_connected);
            const connectedEl = document.getElementById("hud-client-connected");
            connectedEl.textContent = connected ? "Connected" : "Disconnected";
            connectedEl.classList.toggle("connected-yes", connected);
            connectedEl.classList.toggle("connected-no", !connected);

            const stabilizationEl = document.getElementById("hud-stabilization");
            const stabilizationEnabled = Boolean(hud.stabilization_enabled);
            stabilizationEl.textContent = stabilizationEnabled ? "Enabled" : "Disabled";
            stabilizationEl.classList.toggle("state-enabled", stabilizationEnabled);
            stabilizationEl.classList.toggle("state-disabled", !stabilizationEnabled);

            document.getElementById("hud-claw-angle").textContent = formatInput(hud.claw_angle) + "°";
            document.getElementById("hud-claw-rotation").textContent = formatInput(hud.claw_rotation) + "°";
            document.getElementById("hud-syringe-angle").textContent = formatInput(hud.syringe_angle) + "°";
            document.getElementById("hud-camera-angle").textContent = formatInput(hud.camera_angle) + "°";
            document.getElementById("hud-mpu-pitch").textContent = formatInput(hud.mpu_pitch_deg) + "°";
            document.getElementById("hud-mpu-roll").textContent = formatInput(hud.mpu_roll_deg) + "°";

            const thrusters = Array.isArray(hud.thrusters) ? hud.thrusters : [];
            for (let i = 0; i < 8; i++) {{
                const inputValue = thrusters[i] ?? 0;
                const fill = document.getElementById(`thruster-fill-${{i + 1}}`);
                const valueEl = document.getElementById(`thruster-value-${{i + 1}}`);
                if (!fill || !valueEl) continue;
                fill.style.width = `${{thrusterPct(inputValue)}}%`;
                valueEl.textContent = formatInput(inputValue);
            }}
        }}

        function normalizeDetectionTimestampMs(detectionBlock) {{
            if (!detectionBlock || !Array.isArray(detectionBlock.detections) || detectionBlock.detections.length === 0) return null;
            const latest = Math.max(...detectionBlock.detections.map((item) => Number(item.timestamp || 0)));
            if (!Number.isFinite(latest) || latest <= 0) return null;
            return latest * 1000;
        }}

        function renderDetections(tile, detections) {{
            const panel = tile.querySelector(".detections-panel");
            if (!panel) return;
            if (panel.dataset.detectionEnabled === "false") {{
                panel.classList.add("empty");
                panel.textContent = "No recent detections";
                return;
            }}
            if (!detections || detections.length === 0) {{
                panel.classList.add("empty");
                panel.textContent = "No recent detections";
                return;
            }}

            panel.classList.remove("empty");
            const entries = detections
                .map((item) => {{
                    const bbox = Array.isArray(item.bbox) ? item.bbox.join(",") : "n/a";
                    const confidence = Number(item.confidence);
                    const confidencePct = Number.isFinite(confidence) ? (confidence * 100).toFixed(0) : "?";
                    return `
<li>${{item.label || "Unknown"}} (${{confidencePct}}%) [${{bbox}}]</li>`;
                }})
                .join("");
            panel.innerHTML = `<ul>${{entries}}
</ul>`;
        }}

        async function updateStatus() {{
            let data;
            try {{
                const res = await fetch("/status");
                data = await res.json();
            }} catch (error) {{
                console.error("HUD status refresh failed", error);
                return;
            }}
            const cameraData = data.cameras || {{}};
            const detectionsByCamera = data.detections || {{}};
            updateHud(data.hud || {{}});
            for (const [name, details] of Object.entries(cameraData)) {{
                const tile = document.querySelector(`[data-camera-name="${{name}}"]`);
                if (!tile) continue;
                const online = Boolean(details.online);
                const wasOnline = previousState.get(name);
                const cameraId = details.camera_id;
                tile.classList.toggle("no-signal", !online);
                if (online && wasOnline === false) {{
                    const img = tile.querySelector("img");
                    const base = img.src.split("?")[0];
                    img.src = base + "?t=" + Date.now();
                }}

                const incoming = detectionsByCamera[cameraId];
                if (incoming) {{
                    latestDetectionByCamera.set(cameraId, incoming);
                }}

                const tracked = latestDetectionByCamera.get(cameraId);
                const detectionTimestampMs = normalizeDetectionTimestampMs(tracked);
                const isFresh = detectionTimestampMs !== null && (Date.now() - detectionTimestampMs) <= DETECTION_UI_TTL_MS;
                if (!isFresh) {{
                    latestDetectionByCamera.delete(cameraId);
                    renderDetections(tile, []);
                }} else {{
                    renderDetections(tile, tracked.detections || []);
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
            <aside class="hud">
                <h2>Vehicle Status</h2>
                <div class="hud-grid">
                    <div class="hud-row"><span class="hud-label">Client</span><span class="hud-value connected-no" id="hud-client-connected">Disconnected</span></div>
                    <div class="hud-row"><span class="hud-label">Stabilization</span><span class="hud-value" id="hud-stabilization">Disabled</span></div>
                    <div class="hud-row"><span class="hud-label">Claw angle</span><span class="hud-value" id="hud-claw-angle">0.00°</span></div>
                    <div class="hud-row"><span class="hud-label">Claw rotation</span><span class="hud-value" id="hud-claw-rotation">0.00°</span></div>
                    <div class="hud-row"><span class="hud-label">Syringe angle</span><span class="hud-value" id="hud-syringe-angle">0.00°</span></div>
                    <div class="hud-row"><span class="hud-label">Camera angle</span><span class="hud-value" id="hud-camera-angle">0.00°</span></div>
                    <div class="hud-row"><span class="hud-label">MPU pitch</span><span class="hud-value" id="hud-mpu-pitch">0.00°</span></div>
                    <div class="hud-row"><span class="hud-label">MPU roll</span><span class="hud-value" id="hud-mpu-roll">0.00°</span></div>
                    <div>
                        <div class="hud-label" style="margin-bottom: 6px;">Thruster power</div>
                        <ul class="thruster-list">
                            {''.join([f'<li class="thruster-item"><span class="thruster-name">Thruster {i}</span><div class="thruster-bar"><div class="thruster-fill" id="thruster-fill-{i}"></div></div><span class="thruster-value" id="thruster-value-{i}">0.00</span></li>' for i in range(1, 9)])}
                        </ul>
                    </div>
                </div>
            </aside>
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


@app.route("/cv/<camera_id>.mjpg")
def cv_feed(camera_id):
    """CV API MJPEG stream endpoint.

    URL format: /cv/<camera_id>.mjpg
    Response content type: multipart/x-mixed-replace; boundary=frame
    """

    cam = camera_ids.get(camera_id)
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
