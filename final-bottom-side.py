# Activate Python environment
# source my_env/bin/activate

import json
import math
import os
import signal
import socket
import struct
import sys
import threading
import time

import adafruit_mpu6050
import adafruit_pca9685
import board
import cv2
import gpiozero
from flask import Flask, Response, render_template_string

# ---------------- Control Server Configuration ----------------
HOST = ""
PORT = 5000
SOCKET_TIMEOUT_SECONDS = 1.0
MAX_FRAME_SIZE = 65536
NO_DATA_FAILSAFE_SECONDS = 0.5

# ---------------- Webcam Server Configuration ----------------
WEBCAM_HOST = os.getenv("ROV_WEBCAM_HOST", "0.0.0.0")
WEBCAM_PORT = int(os.getenv("ROV_WEBCAM_PORT", "5001"))

CAMERA_CONFIG = {
    "Camera 1": {
        "device_path": "/dev/v4l/by-path/platform-xhci-hcd.1-usb-0:1:1.0-video-index0",
        "width": 640,
        "height": 480,
        "fps": 30,
        "jpeg_quality": 55,
    },
    "Camera 2": {
        "device_path": "/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:1:1.0-video-index0",
        "width": 640,
        "height": 480,
        "fps": 30,
        "jpeg_quality": 55,
    },
    "Camera 3": {
        "device_path": "/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:2:1.0-video-index0",
        "width": 640,
        "height": 480,
        "fps": 30,
        "jpeg_quality": 55,
    },
    "Camera 4": {
        "device_path": "/dev/v4l/by-path/platform-xhci-hcd.1-usb-0:2:1.0-video-index0",
        "width": 1920,
        "height": 1080,
        "fps": 4,
        "jpeg_quality": 50,
    },
}

RECONNECT_INTERVAL = 3.0

# ---------------- Hardware Init ----------------
i2c = board.I2C()
pca = adafruit_pca9685.PCA9685(i2c)
mpu = adafruit_mpu6050.MPU6050(i2c)
relay_pin = gpiozero.LED(4)

pca.frequency = 50

# Motor and Servo configurations
motor_1 = 0
motor_2 = 1
motor_3 = 2
motor_4 = 3
motor_5 = 4
motor_6 = 5
motor_7 = 6
motor_8 = 7

claw_open = 15
claw_rotate = 14


# ---------------- Control Logic ----------------
def convert(x):
    throttle_multiplier = 0.2
    max_duty_cycle = 5240 + throttle_multiplier * 1640
    min_duty_cycle = 5240 - throttle_multiplier * 1640
    mapped_value = round((((x + 1) / 2) * (max_duty_cycle - min_duty_cycle)) + min_duty_cycle)
    return mapped_value


def calculate_orientation():
    accel_x, accel_y, accel_z = mpu.acceleration
    pitch = math.atan2(accel_x, math.sqrt(accel_y**2 + accel_z**2))
    roll = math.atan2(-accel_y, accel_z)
    return pitch, roll


def lerp(a, b, t):
    return a + (b - a) * t


def merge_inputs(joystick_input, mpu_input):
    if mpu_input < 0:
        mpu_input = -((-mpu_input) ** 0.5)
    else:
        mpu_input **= 0.5

    return lerp(mpu_input, joystick_input, math.fabs(joystick_input))


def set_neutral_thrusters():
    neutral = convert(0)
    pca.channels[motor_1].duty_cycle = neutral
    pca.channels[motor_2].duty_cycle = neutral
    pca.channels[motor_3].duty_cycle = neutral
    pca.channels[motor_4].duty_cycle = neutral
    pca.channels[motor_5].duty_cycle = neutral
    pca.channels[motor_6].duty_cycle = neutral
    pca.channels[motor_7].duty_cycle = neutral
    pca.channels[motor_8].duty_cycle = neutral


def apply_thrusters(inputs):
    pca.channels[motor_1].duty_cycle = convert(inputs[0])
    pca.channels[motor_2].duty_cycle = convert(inputs[1])
    pca.channels[motor_3].duty_cycle = convert(inputs[2])
    pca.channels[motor_4].duty_cycle = convert(inputs[3])
    pca.channels[motor_5].duty_cycle = convert(inputs[4])
    pca.channels[motor_6].duty_cycle = convert(inputs[5])
    pca.channels[motor_7].duty_cycle = convert(inputs[6])
    pca.channels[motor_8].duty_cycle = convert(inputs[7])


def read_exact(connection, length):
    buf = b""
    while len(buf) < length:
        try:
            chunk = connection.recv(length - len(buf))
        except socket.timeout:
            return None
        if not chunk:
            return None
        buf += chunk
    return buf


def receive_frame(connection):
    header = read_exact(connection, 4)
    if header is None:
        return None

    frame_length = struct.unpack("!I", header)[0]
    if frame_length <= 0 or frame_length > MAX_FRAME_SIZE:
        raise ValueError(f"Invalid frame size: {frame_length}")

    payload = read_exact(connection, frame_length)
    if payload is None:
        return None

    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, list) or len(decoded) != 13:
        raise ValueError("Expected control payload as list[13]")

    return decoded


def setup_server_socket():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    server.settimeout(1.0)
    print(f"Control server running on {HOST or '0.0.0.0'}:{PORT}. Waiting for client...")
    return server


def run_control_server(stop_event):
    print("Powering up ESCs")
    relay_pin.on()
    set_neutral_thrusters()

    server_socket = setup_server_socket()

    try:
        while not stop_event.is_set():
            try:
                connection, client_address = server_socket.accept()
            except socket.timeout:
                continue

            connection.settimeout(SOCKET_TIMEOUT_SECONDS)
            print(f"Control client connected: {client_address}")
            last_packet_time = time.monotonic()

            try:
                while not stop_event.is_set():
                    try:
                        inputs = receive_frame(connection)
                    except ValueError as exc:
                        print(f"Dropping invalid packet: {exc}")
                        continue

                    now = time.monotonic()

                    if inputs is None:
                        if now - last_packet_time > NO_DATA_FAILSAFE_SECONDS:
                            print("Failsafe: no valid data recently, setting neutral thrusters")
                            set_neutral_thrusters()
                        break

                    last_packet_time = now

                    if inputs[12]:
                        pitch, roll = calculate_orientation()
                        pitch, roll = pitch / math.pi, roll / math.pi

                        inputs[0] = merge_inputs(inputs[0], roll - pitch)
                        inputs[1] = merge_inputs(inputs[1], -roll - pitch)
                        inputs[2] = merge_inputs(inputs[2], -roll + pitch)
                        inputs[3] = merge_inputs(inputs[3], roll + pitch)
                        inputs[4] = merge_inputs(inputs[4], roll + pitch)
                        inputs[5] = merge_inputs(inputs[5], -roll + pitch)
                        inputs[6] = merge_inputs(inputs[6], -roll - pitch)
                        inputs[7] = merge_inputs(inputs[7], roll - pitch)

                    apply_thrusters(inputs)

            except OSError as exc:
                print(f"Control connection error: {exc}")
            finally:
                set_neutral_thrusters()
                connection.close()
                print("Control client disconnected. Waiting for reconnect...")
    finally:
        set_neutral_thrusters()
        server_socket.close()


# ---------------- Webcam Logic ----------------
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
        self.frame = None
        self.jpeg_bytes = None

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
        return True

    def close_camera(self):
        with self.lock:
            self.frame = None
            self.jpeg_bytes = None
            self.online = False

        if self.cap:
            self.cap.release()
        self.cap = None

    def run(self):
        while self.running:
            loop_start = time.monotonic()

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

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )

            with self.lock:
                self.frame = frame
                self.jpeg_bytes = encoded.tobytes() if ok else None

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
        return {
            "online": self.online,
            "resolution": f"{self.width}x{self.height}",
            "fps": self.fps,
        }

    def stop(self):
        self.running = False
        self.close_camera()


cameras = {name: Camera(name, config) for name, config in CAMERA_CONFIG.items()}


def generate_frames(cam: Camera):
    while True:
        frame = cam.get_jpeg()
        if frame is None:
            time.sleep(0.1)
            continue

        yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


@app.route("/status")
def status():
    return {name: cam.get_status() for name, cam in cameras.items()}


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

    return render_template_string(f"""
    <html>
    <head>
        <title>ROV Cameras</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            :root {{ --gap: 8px; --panel-bg: #0f0f10; --tile-border: #2b2b2e; --label-bg: rgba(0,0,0,0.62); }}
            * {{ box-sizing: border-box; }}
            html, body {{ margin: 0; min-height: 100%; background: #111; color: #ddd; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
            .layout {{ display: grid; grid-template-columns: minmax(0, 3fr) minmax(250px, 1fr); min-height: 100vh; gap: var(--gap); padding: var(--gap); }}
            .camera-column {{ min-height: 0; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--gap); align-content: start; }}
            .camera-tile {{ position: relative; width: 100%; aspect-ratio: var(--tile-ratio, 4 / 3); background: #000; border: 1px solid var(--tile-border); border-radius: 6px; overflow: hidden; }}
            .camera-img {{ width: 100%; height: 100%; display: block; object-fit: cover; background: #000; }}
            .camera-label, .camera-meta {{ position: absolute; left: 8px; font-size: 13px; background: var(--label-bg); color: #f0f0f0; padding: 2px 7px; border-radius: 4px; backdrop-filter: blur(1px); }}
            .camera-label {{ bottom: 34px; font-weight: 600; }}
            .camera-meta {{ bottom: 8px; color: #b8b8ba; }}
            .no-signal-label {{ position: absolute; inset: 0; display: none; align-items: center; justify-content: center; font-size: clamp(18px, 2.8vw, 30px); font-weight: bold; letter-spacing: 1px; color: #ff4141; background: rgba(0,0,0,0.76); }}
            .camera-tile.no-signal img {{ display: none; }}
            .camera-tile.no-signal .no-signal-label {{ display: flex; }}
            .hud {{ border: 1px solid #252527; border-radius: 6px; padding: 14px; background: var(--panel-bg); min-height: 0; }}
            .hud h2 {{ margin: 0 0 8px; font-size: 18px; color: #e7e7ea; }}
            .hud p {{ margin: 0; color: #9797a1; line-height: 1.4; }}
            @media (max-width: 1100px) {{ .layout {{ grid-template-columns: 1fr; }} .camera-column {{ grid-template-columns: 1fr; }} }}
        </style>

        <script>
        const previousState = new Map();
        async function updateStatus() {{
            const res = await fetch("/status");
            const data = await res.json();
            for (const [name, details] of Object.entries(data)) {{
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
            <aside class="hud">
                <h2>HUD (Reserved)</h2>
                <p>Reserved for merged telemetry + controls from new-bottom-side.py.</p>
            </aside>
        </main>
    </body>
    </html>
    """)


@app.route("/video/<name>")
def video_feed(name):
    cam = cameras.get(name)
    if not cam:
        return "Camera not found", 404

    return Response(generate_frames(cam), mimetype="multipart/x-mixed-replace; boundary=frame")


def stop_all(capture_only=False):
    for cam in cameras.values():
        cam.stop()
    set_neutral_thrusters()
    if not capture_only:
        relay_pin.off()


def main():
    stop_event = threading.Event()

    def handle_shutdown(sig, frame):
        print(f"Received signal {sig}, shutting down...")
        stop_event.set()
        stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    control_thread = threading.Thread(target=run_control_server, args=(stop_event,), daemon=True)
    control_thread.start()

    try:
        app.run(host=WEBCAM_HOST, port=WEBCAM_PORT, threaded=True)
    finally:
        stop_event.set()
        stop_all()


if __name__ == "__main__":
    main()
