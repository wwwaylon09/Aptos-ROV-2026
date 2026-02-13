from flask import Flask, Response, render_template_string
import cv2
import threading
import signal
import sys
import time
import os

app = Flask(__name__)

# ---------- Stable Camera Configuration ----------
CAMERA_CONFIG = {
    "Camera 1": "/dev/v4l/by-path/platform-xhci-hcd.1-usb-0:1:1.0-video-index0",
    "Camera 2": "/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:1:1.0-video-index0",
    "Camera 3": "/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:2:1.0-video-index0",
    "Camera 4": "/dev/v4l/by-path/platform-xhci-hcd.1-usb-0:2:1.0-video-index0",
}

RECONNECT_INTERVAL = 3.0


# ---------- Camera Class ----------
class Camera:
    def __init__(self, name, device_path):
        self.name = name
        self.device_path = device_path
        self.cap = None
        self.frame = None
        self.online = False
        self.running = True
        self.lock = threading.Lock()

        threading.Thread(target=self.run, daemon=True).start()

    def open_camera(self):
        if not os.path.exists(self.device_path):
            return False

        cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.cap = cap
        self.online = True
        return True

    def close_camera(self):
        with self.lock:
            self.frame = None
            self.online = False

        if self.cap:
            self.cap.release()
        self.cap = None

    def run(self):
        while self.running:
            if self.online and not os.path.exists(self.device_path):
                self.close_camera()
                time.sleep(RECONNECT_INTERVAL)
                continue
            
            if not self.online:
                if not self.open_camera():
                    time.sleep(RECONNECT_INTERVAL)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                self.close_camera()
                time.sleep(0.5)
                continue

            with self.lock:
                self.frame = frame
                
            time.sleep(0.03)

    def get_jpeg(self):
        if not self.online:
            return None

        with self.lock:
            if self.frame is None:
                return None

            ret, buf = cv2.imencode(
                ".jpg",
                self.frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            )
            return buf.tobytes() if ret else None

    def stop(self):
        self.running = False
        self.close_camera()


# ---------- Initialize Cameras ----------
cameras = {
    name: Camera(name, path)
    for name, path in CAMERA_CONFIG.items()
}


# ---------- MJPEG Generator ----------
def generate_frames(cam: Camera):
    while True:
        frame = cam.get_jpeg()
        if frame is None:
            time.sleep(0.1)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


# ---------- Routes ----------
@app.route("/status")
def status():
    return {
        name: cam.online
        for name, cam in cameras.items()
    }


@app.route("/")
def index():
    tiles = ""

    for name, cam in cameras.items():
        tiles += f"""
        <div class="camera-tile" id="{name}">
            <img src="/video/{name}" class="camera-img">
            <div class="no-signal-label">NO SIGNAL</div>
            <div class="camera-label">{name}</div>
        </div>
        """

    return render_template_string(f"""
    <html>
    <head>
        <title>ROV Cameras</title>
        <style>
            html, body {{
                margin: 0;
                height: 100%;
                background: #111;
                font-family: sans-serif;
            }}

            .layout {{
                display: grid;
                grid-template-columns: auto 1fr;
                height: 100vh;
            }}

            .camera-column {{
                height: 100vh;
                box-sizing: border-box;
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                grid-template-rows: repeat(2, 1fr);
                gap: 6px;
                padding: 6px;
            }}

            .camera-tile {{
                position: relative;
                background: #000;
                border: 1px solid #333;
                overflow: hidden;
                height: 100%;
                aspect-ratio: 4 / 3;
            }}

            .camera-img {{
                height: 100%;
                width: 100%;
                display: block;
                margin: 0 auto;
                object-fit: contain;
                background: black;
            }}

            .camera-label {{
                position: absolute;
                bottom: 6px;
                left: 6px;
                font-size: 14px;
                background: rgba(0,0,0,0.6);
                color: #eee;
                padding: 3px 6px;
            }}

            .no-signal-label {{
                position: absolute;
                inset: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 28px;
                font-weight: bold;
                color: red;
                background: #000;
                visibility: hidden;
            }}

            .camera-tile.no-signal img {{
                display: none;
            }}

            .camera-tile.no-signal .no-signal-label {{
                visibility: visible;
            }}

            .hud {{
                border-left: 2px solid #222;
                padding: 10px;
                color: #777;
            }}
        </style>

        <script>
        async function updateStatus() {{
            const res = await fetch("/status");
            const data = await res.json();

            for (const [name, online] of Object.entries(data)) {{
                const tile = document.getElementById(name);
                tile.classList.toggle("no-signal", !online);

                // Force reload when coming back online
                if (online) {{
                    const img = tile.querySelector("img");
                    const base = img.src.split("?")[0];
                    img.src = base + "?t=" + Date.now();
                }}
            }}
        }}

        setInterval(updateStatus, 1000);
        </script>
    </head>
    <body>
        <div class="layout">
            <div class="camera-column">
                {tiles}
            </div>
            <div class="hud">
                <h2>HUD (Reserved)</h2>
                <p>Status, telemetry, controls</p>
            </div>
        </div>
    </body>
    </html>
    """)


@app.route("/video/<name>")
def video_feed(name):
    cam = cameras.get(name)
    if not cam:
        return "Camera not found", 404

    return Response(
        generate_frames(cam),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------- Shutdown ----------
def shutdown(sig, frame):
    for cam in cameras.values():
        cam.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


# ---------- Run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
