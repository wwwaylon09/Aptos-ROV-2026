#!/usr/bin/env python3
"""Bluetooth RFCOMM server for autonomous vertical profiling."""

from __future__ import annotations

import json
import math
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass, asdict
from typing import List

# =========================
# Runtime configuration
# =========================
SERVER_BT_ADDRESS = socket.BDADDR_ANY
RFCOMM_CHANNEL = 3
BUFFER_SIZE = 1024

DEVICE_NUMBER = "RA01"
PROFILE_START_DELAY_SECONDS = 60
TOP_SURFACE_DEPTH_METERS = 0.40
DEVICE_VERTICAL_LENGTH_METERS = 0.30
TARGET_SENSOR_DEPTH_METERS = 2.50
DEPTH_TOLERANCE_METERS = 0.33
HOLD_AT_TARGET_SECONDS = 30
HOLD_AT_SURFACE_SECONDS = 30
PROFILE_CYCLE_COUNT = 2
IGNORE_SENSOR_DEPTH_REQUIREMENT = False
DEPTH_WAIT_TIMEOUT_SECONDS = 45

STEPPER_STEP_ANGLE_DEGREES = 1.8
STEPPER_DESCENT_ANGLE_DEGREES = 360.0
STEPPER_DIRECTION_PIN = 20
STEPPER_STEP_PIN = 21
STEPPER_PULSE_HIGH_SECONDS = 0.001
STEPPER_STEP_PERIOD_SECONDS = 0.003

SENSOR_INIT_RETRIES = 5
SENSOR_INIT_RETRY_DELAY_SECONDS = 1.0
SENSOR_READ_RETRIES = 3
SENSOR_READ_RETRY_DELAY_SECONDS = 0.25

# Sensor depth when the top is 40 cm below surface.
SURFACE_SENSOR_DEPTH_METERS = TOP_SURFACE_DEPTH_METERS + DEVICE_VERTICAL_LENGTH_METERS

try:
    import ms5837  # type: ignore
except ImportError:
    ms5837 = None

try:
    import RPi.GPIO as rpi_gpio  # type: ignore
except ImportError:
    rpi_gpio = None


@dataclass
class DataPacket:
    number: str
    elapsed_seconds: int
    time_label: str
    pressure_kpa: float
    depth_m: float
    packet_type: str = "telemetry"
    stage: str = ""


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

    def rotate(self, angle_degrees: float, clockwise: bool) -> None:
        steps = max(0, int(round(abs(angle_degrees) / STEPPER_STEP_ANGLE_DEGREES)))

        rpi_gpio.output(STEPPER_DIRECTION_PIN, 1 if clockwise else 0)
        for _ in range(steps):
            rpi_gpio.output(STEPPER_STEP_PIN, 1)
            time.sleep(STEPPER_PULSE_HIGH_SECONDS)
            rpi_gpio.output(STEPPER_STEP_PIN, 0)
            time.sleep(max(0.0, STEPPER_STEP_PERIOD_SECONDS - STEPPER_PULSE_HIGH_SECONDS))

    def shutdown(self) -> None:
        rpi_gpio.cleanup((STEPPER_DIRECTION_PIN, STEPPER_STEP_PIN))


class PressureSensor:
    def __init__(self) -> None:
        if ms5837 is None:
            raise RuntimeError("ms5837 library is required and could not be imported")

        self.sensor = ms5837.MS5837_02BA()
        self.available = False

        for attempt in range(1, SENSOR_INIT_RETRIES + 1):
            try:
                if self.sensor.init():
                    self.sensor.setFluidDensity(ms5837.DENSITY_FRESHWATER)
                    self.available = True
                    break
            except OSError as exc:
                print(f"[SERVER] Sensor init attempt {attempt}/{SENSOR_INIT_RETRIES} failed: {exc}")

            if attempt < SENSOR_INIT_RETRIES:
                time.sleep(SENSOR_INIT_RETRY_DELAY_SECONDS)

        if not self.available:
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


class ProfilingController:
    def __init__(self) -> None:
        self.stepper = StepperController()
        self.sensor = PressureSensor()
        self.data_packets: List[DataPacket] = []
        self.data_lock = threading.Lock()
        self.profile_thread: threading.Thread | None = None
        self.profile_running = False
        self.arm_time_monotonic: float | None = None

    @staticmethod
    def _format_elapsed(seconds: int) -> str:
        minutes, remaining = divmod(max(0, seconds), 60)
        return f"{minutes}:{remaining:02d}"

    def _build_packet(self, packet_type: str = "telemetry", stage: str = "") -> DataPacket:
        elapsed = 0
        if self.arm_time_monotonic is not None:
            elapsed = int(time.monotonic() - self.arm_time_monotonic)

        pressure_kpa, depth_m = self.sensor.read()
        return DataPacket(
            number=DEVICE_NUMBER,
            elapsed_seconds=elapsed,
            time_label=self._format_elapsed(elapsed),
            pressure_kpa=pressure_kpa,
            depth_m=depth_m,
            packet_type=packet_type,
            stage=stage,
        )

    def _log_packet(self, stage_label: str, packet_type: str = "telemetry") -> DataPacket:
        packet = self._build_packet(packet_type=packet_type, stage=stage_label)

        with self.data_lock:
            self.data_packets.append(packet)

        print(
            f"[SERVER] {stage_label}: {packet.number}, {packet.time_label}, "
            f"{packet.pressure_kpa:.2f} kPa, {packet.depth_m:.2f} m"
        )
        return packet

    def arm(self) -> tuple[bool, DataPacket | None]:
        if self.profile_running:
            return False, None

        with self.data_lock:
            self.data_packets.clear()

        self.profile_running = True
        self.arm_time_monotonic = time.monotonic()

        initial_packet = self._build_packet(packet_type="stage", stage="PROFILE STARTED")

        with self.data_lock:
            self.data_packets.append(initial_packet)

        print(
            f"[SERVER] START PACKET: {initial_packet.number}, {initial_packet.time_label}, "
            f"{initial_packet.pressure_kpa:.2f} kPa, {initial_packet.depth_m:.2f} m"
        )

        self.profile_thread = threading.Thread(target=self._profile_worker, daemon=True)
        self.profile_thread.start()
        return True, initial_packet

    def _hold_and_log(self, seconds: int, stage_label: str) -> None:
        end_time = time.monotonic() + seconds
        while time.monotonic() < end_time:
            self._log_packet(stage_label)
            remaining = end_time - time.monotonic()
            if remaining > 0:
                time.sleep(min(1.0, remaining))

    def _wait_for_depth(self, target_depth_meters: float, stage_label: str) -> None:
        if IGNORE_SENSOR_DEPTH_REQUIREMENT:
            print(f"[SERVER] {stage_label}: depth wait bypassed by config.")
            return

        deadline = time.monotonic() + DEPTH_WAIT_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            packet = self._log_packet(f"{stage_label} SEEKING")
            if math.isnan(packet.depth_m):
                time.sleep(1.0)
                continue
            if abs(packet.depth_m - target_depth_meters) <= DEPTH_TOLERANCE_METERS:
                return
            time.sleep(1.0)

        print(f"[SERVER] WARNING: timed out waiting for target depth during {stage_label}.")

    def _profile_worker(self) -> None:
        print(f"[SERVER] Profiling armed. Waiting {PROFILE_START_DELAY_SECONDS}s before start.")
        time.sleep(PROFILE_START_DELAY_SECONDS)

        for cycle in range(1, PROFILE_CYCLE_COUNT + 1):
            print(f"[SERVER] DESCENT {cycle} STARTED")
            self._log_packet(f"DESCENT {cycle} STARTED", packet_type="stage")
            self.stepper.rotate(STEPPER_DESCENT_ANGLE_DEGREES, clockwise=False)

            self._wait_for_depth(TARGET_SENSOR_DEPTH_METERS, f"DESCENT {cycle}")
            print(f"[SERVER] DESCENT {cycle} HOLDING")
            self._log_packet(f"DESCENT {cycle} HOLDING", packet_type="stage")
            self._hold_and_log(HOLD_AT_TARGET_SECONDS, f"DESCENT {cycle} HOLDING")

            print(f"[SERVER] ASCENT {cycle} STARTED")
            self._log_packet(f"ASCENT {cycle} STARTED", packet_type="stage")
            self.stepper.rotate(STEPPER_DESCENT_ANGLE_DEGREES, clockwise=True)

            self._wait_for_depth(SURFACE_SENSOR_DEPTH_METERS, f"ASCENT {cycle}")
            print(f"[SERVER] ASCENT {cycle} HOLDING")
            self._log_packet(f"ASCENT {cycle} HOLDING", packet_type="stage")
            self._hold_and_log(HOLD_AT_SURFACE_SECONDS, f"ASCENT {cycle} HOLDING")

        self.profile_running = False
        print("[SERVER] Profiling complete. Data ready for collection.")

    def serialized_data(self) -> list[str]:
        with self.data_lock:
            packets = [asdict(packet) for packet in self.data_packets]
        return [json.dumps(packet) for packet in packets]

    def shutdown(self) -> None:
        self.stepper.shutdown()


def handle_client(client_sock: socket.socket, controller: ProfilingController) -> None:
    data = client_sock.recv(BUFFER_SIZE)
    if not data:
        return

    message = data.decode("utf-8", errors="replace").strip().lower()
    print(f"[SERVER] Received command: {message}")

    if message == "start":
        armed, initial_packet = controller.arm()
        if armed and initial_packet is not None:
            client_sock.sendall(b"OK profiling_armed\n")
            client_sock.sendall((json.dumps(asdict(initial_packet)) + "\n").encode("utf-8"))
        else:
            client_sock.sendall(b"ERR profiling_already_running\n")
        return

    if message == "collect":
        payload_lines = controller.serialized_data()
        client_sock.sendall(f"DATA_BEGIN {len(payload_lines)}\n".encode("utf-8"))
        for line in payload_lines:
            client_sock.sendall((line + "\n").encode("utf-8"))
        client_sock.sendall(b"DATA_END\n")
        return

    client_sock.sendall(b"ERR unknown_command\n")


def main() -> None:
    if not hasattr(socket, "AF_BLUETOOTH"):
        print("[SERVER] Bluetooth sockets are not supported by this Python build.")
        sys.exit(1)

    print(
        "[SERVER] Config: "
        f"device={DEVICE_NUMBER}, target={TARGET_SENSOR_DEPTH_METERS:.2f}m, "
        f"surface_top={TOP_SURFACE_DEPTH_METERS:.2f}m, "
        f"surface_sensor={SURFACE_SENSOR_DEPTH_METERS:.2f}m, "
        f"tolerance=±{DEPTH_TOLERANCE_METERS:.2f}m"
    )

    controller = ProfilingController()
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
