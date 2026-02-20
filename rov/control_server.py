import json
import logging
import math
import signal
import socket
import struct
import sys
import threading
import time

import adafruit_mpu6050
import adafruit_pca9685
import board
import gpiozero

from rov.shared_config import (
    CONTROL_HEALTH_LOG_INTERVAL_SECONDS,
    HOST,
    MAX_FRAME_SIZE,
    MPU_DEADBAND_DEGREES,
    NO_DATA_FAILSAFE_SECONDS,
    PORT,
    SOCKET_TIMEOUT_SECONDS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [control] %(levelname)s: %(message)s")
LOGGER = logging.getLogger("rov.control")

# ---------------- Hardware Init ----------------
i2c = board.I2C()
pca = adafruit_pca9685.PCA9685(i2c)
mpu = adafruit_mpu6050.MPU6050(i2c)
relay_pin = gpiozero.LED(4)

pca.frequency = 50

motor_1 = 0
motor_2 = 1
motor_3 = 2
motor_4 = 3
motor_5 = 4
motor_6 = 5
motor_7 = 6
motor_8 = 7

HEALTH_LOCK = threading.Lock()
HEALTH_STATE = {
    "client_connected": False,
    "last_packet_monotonic": None,
    "failsafe_events": 0,
    "packets_processed": 0,
}


def _set_health(**kwargs):
    with HEALTH_LOCK:
        HEALTH_STATE.update(kwargs)


def convert(x):
    x = clamp(x)
    throttle_multiplier = 0.2
    max_duty_cycle = 5240 + throttle_multiplier * 1640
    min_duty_cycle = 5240 - throttle_multiplier * 1640
    return round((((x + 1) / 2) * (max_duty_cycle - min_duty_cycle)) + min_duty_cycle)


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
    LOGGER.info("Control server listening on %s:%s", HOST or "0.0.0.0", PORT)
    return server


def health_logger(stop_event):
    while not stop_event.wait(CONTROL_HEALTH_LOG_INTERVAL_SECONDS):
        with HEALTH_LOCK:
            now = time.monotonic()
            last_packet_age = (
                None if HEALTH_STATE["last_packet_monotonic"] is None else round(now - HEALTH_STATE["last_packet_monotonic"], 2)
            )
            LOGGER.info(
                "health client_connected=%s packets=%s failsafe_events=%s last_packet_age_s=%s",
                HEALTH_STATE["client_connected"],
                HEALTH_STATE["packets_processed"],
                HEALTH_STATE["failsafe_events"],
                last_packet_age,
            )


def run_control_server(stop_event):
    LOGGER.info("Powering up ESC relay")
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
            LOGGER.info("Control client connected: %s", client_address)
            _set_health(client_connected=True, last_packet_monotonic=time.monotonic())
            last_packet_time = time.monotonic()

            try:
                while not stop_event.is_set():
                    try:
                        inputs = receive_frame(connection)
                    except ValueError as exc:
                        LOGGER.warning("Dropping invalid packet: %s", exc)
                        continue

                    now = time.monotonic()

                    if inputs is None:
                        if now - last_packet_time > NO_DATA_FAILSAFE_SECONDS:
                            LOGGER.warning("Failsafe: no valid control data, neutral thrusters")
                            set_neutral_thrusters()
                            with HEALTH_LOCK:
                                HEALTH_STATE["failsafe_events"] += 1
                        break

                    last_packet_time = now
                    with HEALTH_LOCK:
                        HEALTH_STATE["last_packet_monotonic"] = now
                        HEALTH_STATE["packets_processed"] += 1

                    try:
                        _, _, pitch_rad, roll_rad = calculate_orientation_degrees()
                    except OSError as exc:
                        LOGGER.warning("MPU read failed: %s", exc)
                        pitch_rad, roll_rad = 0.0, 0.0

                    if inputs[12]:
                        pitch = apply_deadband(pitch_rad / math.pi, MPU_DEADBAND_DEGREES / 180.0)
                        roll = apply_deadband(roll_rad / math.pi, MPU_DEADBAND_DEGREES / 180.0)

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

            except OSError as exc:
                LOGGER.error("Control connection error: %s", exc)
            finally:
                set_neutral_thrusters()
                connection.close()
                _set_health(client_connected=False)
                LOGGER.info("Control client disconnected. Waiting for reconnect")
    finally:
        set_neutral_thrusters()
        server_socket.close()


def shutdown_hardware():
    set_neutral_thrusters()
    relay_pin.off()


def main():
    stop_event = threading.Event()

    def handle_shutdown(sig, frame):
        LOGGER.info("Received signal %s, shutting down control service", sig)
        stop_event.set()
        shutdown_hardware()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    threading.Thread(target=health_logger, args=(stop_event,), daemon=True).start()

    try:
        run_control_server(stop_event)
    finally:
        stop_event.set()
        shutdown_hardware()


if __name__ == "__main__":
    main()
