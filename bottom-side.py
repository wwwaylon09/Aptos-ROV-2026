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
HUD_UPDATE_INTERVAL_SECONDS = float(os.getenv("ROV_HUD_UPDATE_INTERVAL_SECONDS", "0.2"))
MPU_DEADBAND_DEGREES = float(os.getenv("ROV_MPU_DEADBAND_DEGREES", "1.5"))

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


HUD_LOCK = threading.Lock()
HUD_STATE = {
    "thrusters": [0.0] * 8,
    "claw_angle": 0.0,
    "claw_rotation": 0.0,
    "stabilization_enabled": False,
    "mpu_pitch_deg": 0.0,
    "mpu_roll_deg": 0.0,
    "client_connected": False,
}


# ---------------- Control Logic ----------------
def convert(x):
    x = clamp(x)
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


def calculate_orientation_degrees():
    pitch_rad, roll_rad = calculate_orientation()
    return math.degrees(pitch_rad), math.degrees(roll_rad), pitch_rad, roll_rad


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(value, low=-1.0, high=1.0):
    return max(low, min(high, value))




def apply_deadband(value, threshold):
    if abs(value) < threshold:
        return 0.0
    return value

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


def set_client_connected(connected):
    with HUD_LOCK:
        HUD_STATE["client_connected"] = bool(connected)


def update_hud_state(inputs, pitch_deg, roll_deg):
    with HUD_LOCK:
        HUD_STATE["thrusters"] = [float(value) for value in inputs[:8]]
        HUD_STATE["claw_angle"] = float(inputs[8])
        HUD_STATE["claw_rotation"] = float(inputs[9])
        HUD_STATE["stabilization_enabled"] = bool(inputs[10])
        HUD_STATE["mpu_pitch_deg"] = float(pitch_deg)
        HUD_STATE["mpu_roll_deg"] = float(roll_deg)


def get_hud_state():
    with HUD_LOCK:
        return {
            "thrusters": [round(value, 3) for value in HUD_STATE["thrusters"]],
            "claw_angle": round(HUD_STATE["claw_angle"], 3),
            "claw_rotation": round(HUD_STATE["claw_rotation"], 3),
            "stabilization_enabled": HUD_STATE["stabilization_enabled"],
            "mpu_pitch_deg": round(HUD_STATE["mpu_pitch_deg"], 2),
            "mpu_roll_deg": round(HUD_STATE["mpu_roll_deg"], 2),
            "client_connected": HUD_STATE["client_connected"],
        }


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
    if not isinstance(decoded, list) or len(decoded) != 11:
        raise ValueError("Expected control payload as list[11]")

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
            set_client_connected(True)
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

                    try:
                        pitch_deg, roll_deg, pitch_rad, roll_rad = calculate_orientation_degrees()
                    except OSError as exc:
                        print(f"MPU read failed: {exc}")
                        pitch_deg, roll_deg, pitch_rad, roll_rad = 0.0, 0.0, 0.0, 0.0

                    if inputs[10]:
                        pitch = pitch_rad / math.pi
                        roll = roll_rad / math.pi

                        pitch = apply_deadband(pitch, MPU_DEADBAND_DEGREES / 180.0)
                        roll = apply_deadband(roll, MPU_DEADBAND_DEGREES / 180.0)

                        inputs[0] = merge_inputs(inputs[0], roll - pitch)
                        inputs[1] = merge_inputs(inputs[1], -roll - pitch)
                        inputs[2] = merge_inputs(inputs[2], -roll + pitch)
                        inputs[3] = merge_inputs(inputs[3], roll + pitch)
                        inputs[4] = merge_inputs(inputs[4], roll + pitch)
                        inputs[5] = merge_inputs(inputs[5], -roll + pitch)
                        inputs[6] = merge_inputs(inputs[6], -roll - pitch)
                        inputs[7] = merge_inputs(inputs[7], roll - pitch)

                    for i in range(8):
                        inputs[i] = clamp(inputs[i])

                    apply_thrusters(inputs)
                    update_hud_state(inputs, pitch_deg, roll_deg)

            except OSError as exc:
                print(f"Control connection error: {exc}")
            finally:
                set_neutral_thrusters()
                connection.close()
                set_client_connected(False)
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
    return {
        "cameras": {name: cam.get_status() for name, cam in cameras.items()},
        "hud": get_hud_state(),
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

    return render_template_string(f"""
    <html>
    <head>
        <title>ROV Interface</title>
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
            .hud h2 {{ margin: 0 0 10px; font-size: 18px; color: #e7e7ea; }}
            .hud-grid {{ display: grid; gap: 9px; }}
            .hud-row {{ display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px dashed #252529; padding-bottom: 6px; }}
            .hud-label {{ color: #9898a3; font-size: 13px; }}
            .hud-value {{ color: #f2f2f5; font-weight: 600; font-variant-numeric: tabular-nums; }}
            .thruster-list {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }}
            .thruster-item {{ display: grid; grid-template-columns: auto 1fr auto; gap: 8px; align-items: center; }}
            .thruster-name {{ color: #9898a3; font-size: 12px; min-width: 64px; }}
            .thruster-bar {{ height: 8px; border-radius: 999px; background: #1f1f22; overflow: hidden; border: 1px solid #2b2b30; }}
            .thruster-fill {{ height: 100%; width: 50%; background: linear-gradient(90deg, #39c66d, #e9c23f, #e25a5a); transition: width .18s linear; }}
            .thruster-value {{ color: #dfdfe5; font-size: 12px; min-width: 58px; text-align: right; font-variant-numeric: tabular-nums; }}
            .connected-yes {{ color: #5ad876; }}
            .connected-no {{ color: #ff6969; }}
            .state-enabled {{ color: #5ad876; }}
            .state-disabled {{ color: #ff6969; }}
            @media (max-width: 1100px) {{ .layout {{ grid-template-columns: 1fr; }} .camera-column {{ grid-template-columns: 1fr; }} }}
        </style>

        <script>
        const previousState = new Map();
        const clamp = (n, min, max) => Math.max(min, Math.min(max, n));
        const asNumber = (v) => Number.isFinite(Number(v)) ? Number(v) : 0;
        const formatInput = (value) => asNumber(value).toFixed(2);
        const thrusterPct = (value) => (clamp((asNumber(value) + 1) / 2, 0, 1) * 100);

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
            document.getElementById("hud-mpu-pitch").textContent = formatInput(hud.mpu_pitch_deg) + "°";
            document.getElementById("hud-mpu-roll").textContent = formatInput(hud.mpu_roll_deg) + "°";

            const thrusters = Array.isArray(hud.thrusters) ? hud.thrusters : [];
            for (let i = 0; i < 8; i++) {{
                const inputValue = thrusters[i] ?? 0;
                const fill = document.getElementById(`thruster-fill-${{i + 1}}`);
                const valueEl = document.getElementById(`thruster-value-${{i + 1}}`);
                fill.style.width = `${{thrusterPct(inputValue)}}%`;
                valueEl.textContent = formatInput(inputValue);
            }}
        }}

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

            updateHud(data.hud);
        }}
        updateStatus();
        setInterval(updateStatus, {max(50, int(HUD_UPDATE_INTERVAL_SECONDS * 1000))});
        </script>
    </head>
    <body>
        <main class="layout">
            <section class="camera-column">{tiles}</section>
            <aside class="hud">
                <h2>ROV HUD</h2>
                <div class="hud-grid">
                    <div class="hud-row"><span class="hud-label">Client</span><span class="hud-value connected-no" id="hud-client-connected">Disconnected</span></div>
                    <div class="hud-row"><span class="hud-label">Stabilization</span><span class="hud-value" id="hud-stabilization">Disabled</span></div>
                    <div class="hud-row"><span class="hud-label">Claw angle</span><span class="hud-value" id="hud-claw-angle">0.00°</span></div>
                    <div class="hud-row"><span class="hud-label">Claw rotation</span><span class="hud-value" id="hud-claw-rotation">0.00°</span></div>
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
