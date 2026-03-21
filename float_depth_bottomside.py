#!/usr/bin/env python3
"""Bluetooth RFCOMM server for depth and stepper calibration testing."""

from __future__ import annotations

import json
import signal
import socket
import sys
import time
from typing import Callable

SERVER_BT_ADDRESS = getattr(socket, "BDADDR_ANY", "00:00:00:00:00:00")
RFCOMM_CHANNEL = 3
BUFFER_SIZE = 1024

STEPPER_STEP_ANGLE_DEGREES = 1.8
STEPPER_DIRECTION_PIN = 20
STEPPER_STEP_PIN = 21
STEPPER_PULSE_HIGH_SECONDS = 0.001
STEPPER_STEP_PERIOD_SECONDS = 0.003

SENSOR_INIT_RETRIES = 5
SENSOR_INIT_RETRY_DELAY_SECONDS = 1.0
SENSOR_READ_RETRIES = 3
SENSOR_READ_RETRY_DELAY_SECONDS = 0.25

try:
    import ms5837  # type: ignore
except ImportError:
    ms5837 = None

try:
    import RPi.GPIO as rpi_gpio  # type: ignore
except ImportError:
    rpi_gpio = None


class StepperController:
    def __init__(self) -> None:
        if rpi_gpio is None:
            raise RuntimeError(
                "RPi.GPIO backend is required and could not be imported. "
                "Install python3-rpi.gpio or python3-rpi-lgpio."
            )

        rpi_gpio.setwarnings(False)
        rpi_gpio.setmode(rpi_gpio.BCM)
        rpi_gpio.setup(STEPPER_DIRECTION_PIN, rpi_gpio.OUT, initial=rpi_gpio.LOW)
        rpi_gpio.setup(STEPPER_STEP_PIN, rpi_gpio.OUT, initial=rpi_gpio.LOW)

    def rotate(self, angle_degrees: float) -> int:
        steps = max(0, int(round(abs(angle_degrees) / STEPPER_STEP_ANGLE_DEGREES)))
        clockwise = angle_degrees >= 0.0

        rpi_gpio.output(STEPPER_DIRECTION_PIN, 1 if clockwise else 0)
        for _ in range(steps):
            rpi_gpio.output(STEPPER_STEP_PIN, 1)
            time.sleep(STEPPER_PULSE_HIGH_SECONDS)
            rpi_gpio.output(STEPPER_STEP_PIN, 0)
            time.sleep(max(0.0, STEPPER_STEP_PERIOD_SECONDS - STEPPER_PULSE_HIGH_SECONDS))

        return steps

    def shutdown(self) -> None:
        rpi_gpio.cleanup((STEPPER_DIRECTION_PIN, STEPPER_STEP_PIN))


class PressureSensor:
    def __init__(self) -> None:
        if ms5837 is None:
            raise RuntimeError("ms5837 library is required and could not be imported")

        self.sensor = ms5837.MS5837_02BA()

        for attempt in range(1, SENSOR_INIT_RETRIES + 1):
            try:
                if self.sensor.init():
                    self.sensor.setFluidDensity(ms5837.DENSITY_FRESHWATER)
                    return
            except OSError as exc:
                print(f"[SERVER] Sensor init attempt {attempt}/{SENSOR_INIT_RETRIES} failed: {exc}")

            if attempt < SENSOR_INIT_RETRIES:
                time.sleep(SENSOR_INIT_RETRY_DELAY_SECONDS)

        raise RuntimeError(f"MS5837 init failed after {SENSOR_INIT_RETRIES} attempts")

    def read(self) -> tuple[float, float]:
        for attempt in range(1, SENSOR_READ_RETRIES + 1):
            try:
                if self.sensor.read(ms5837.OSR_8192):
                    pressure_kpa = float(self.sensor.pressure(ms5837.UNITS_kPa))
                    depth_m = float(self.sensor.depth())
                    return pressure_kpa, depth_m
            except OSError as exc:
                print(f"[SERVER] Sensor read attempt {attempt}/{SENSOR_READ_RETRIES} failed: {exc}")

            if attempt < SENSOR_READ_RETRIES:
                time.sleep(SENSOR_READ_RETRY_DELAY_SECONDS)

        raise RuntimeError(f"MS5837 read failed after {SENSOR_READ_RETRIES} attempts")


class CalibrationController:
    def __init__(self) -> None:
        self.stepper = StepperController()
        self.sensor = PressureSensor()

    def rotate(self, angle_degrees: float) -> dict[str, float | int | str]:
        steps = self.stepper.rotate(angle_degrees)
        direction = "clockwise" if angle_degrees >= 0.0 else "counterclockwise"
        return {
            "status": "ok",
            "command": "rotate",
            "degrees": angle_degrees,
            "steps": steps,
            "direction": direction,
        }

    def depth(self) -> dict[str, float | str]:
        pressure_kpa, depth_m = self.sensor.read()
        return {
            "status": "ok",
            "command": "depth",
            "pressure_kpa": pressure_kpa,
            "depth_m": depth_m,
        }

    def shutdown(self) -> None:
        self.stepper.shutdown()


def recv_line(sock: socket.socket, carry: bytes) -> tuple[str | None, bytes]:
    while b"\n" not in carry:
        chunk = sock.recv(BUFFER_SIZE)
        if not chunk:
            return None, b""
        carry += chunk

    raw_line, carry = carry.split(b"\n", 1)
    return raw_line.decode("utf-8", errors="replace").strip(), carry


def send_json(sock: socket.socket, payload: dict[str, object]) -> None:
    sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))


def build_error(message: str) -> dict[str, str]:
    return {"status": "error", "message": message}


def handle_rotate(parts: list[str], controller: CalibrationController) -> dict[str, object]:
    if len(parts) != 2:
        return build_error("usage: rotate <degrees>")

    try:
        angle_degrees = float(parts[1])
    except ValueError:
        return build_error("degrees must be numeric")

    return controller.rotate(angle_degrees)


def handle_depth(_parts: list[str], controller: CalibrationController) -> dict[str, object]:
    return controller.depth()


def handle_help(_parts: list[str], _controller: CalibrationController) -> dict[str, object]:
    return {
        "status": "ok",
        "command": "help",
        "commands": {
            "rotate": "rotate <degrees> | positive=clockwise, negative=counterclockwise",
            "depth": "depth | returns current pressure and depth in meters",
            "quit": "quit | close current client connection",
        },
    }


def handle_client(client_sock: socket.socket, controller: CalibrationController) -> None:
    handlers: dict[str, Callable[[list[str], CalibrationController], dict[str, object]]] = {
        "rotate": handle_rotate,
        "depth": handle_depth,
        "help": handle_help,
    }
    carry = b""

    while True:
        message, carry = recv_line(client_sock, carry)
        if message is None:
            print("[SERVER] Client disconnected.")
            return

        normalized = message.strip()
        if not normalized:
            continue

        print(f"[SERVER] Received command: {normalized}")
        lowered = normalized.lower()
        if lowered == "quit":
            send_json(client_sock, {"status": "ok", "command": "quit"})
            return

        parts = normalized.split()
        command = parts[0].lower()
        handler = handlers.get(command)
        if handler is None:
            send_json(client_sock, build_error("unknown command"))
            continue

        try:
            response = handler(parts, controller)
        except Exception as exc:  # Hardware/runtime failures should be reported to client.
            response = build_error(str(exc))

        send_json(client_sock, response)


def main() -> None:
    if not hasattr(socket, "AF_BLUETOOTH"):
        print("[SERVER] Bluetooth sockets are not supported by this Python build.")
        sys.exit(1)

    controller = CalibrationController()
    server_sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    server_sock.bind((SERVER_BT_ADDRESS, RFCOMM_CHANNEL))
    server_sock.listen(1)

    def shutdown_handler(signum: int, _frame: object) -> None:
        print(f"\n[SERVER] Signal {signum} received. Shutting down...")
        controller.shutdown()
        server_sock.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    bound_addr, bound_channel = server_sock.getsockname()
    print(f"[SERVER] Listening on adapter={bound_addr}, channel={bound_channel}")
    print("[SERVER] Commands: rotate <degrees>, depth, help, quit")

    while True:
        print("[SERVER] Waiting for RFCOMM client...")
        client_sock, client_info = server_sock.accept()
        print(f"[SERVER] Client connected: {client_info}")
        try:
            handle_client(client_sock, controller)
        except OSError as exc:
            print(f"[SERVER] Connection error: {exc}")
        finally:
            client_sock.close()


if __name__ == "__main__":
    main()
