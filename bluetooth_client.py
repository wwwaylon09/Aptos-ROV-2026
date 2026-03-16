#!/usr/bin/env python3
"""Bluetooth RFCOMM client for arming and collecting profiler data."""

from __future__ import annotations

import argparse
import json
import socket
from dataclasses import dataclass
from typing import List

# =========================
# Runtime configuration
# =========================
SERVER_BT_MAC_ADDRESS = "00:00:00:00:00:00"
RFCOMM_CHANNEL = 3
CONNECT_TIMEOUT_SECONDS = 10.0
RECV_BUFFER_SIZE = 1024
PLOT_OUTPUT_PATH = "profile_depth_plot.png"


@dataclass
class DataPacket:
    number: str
    elapsed_seconds: int
    time_label: str
    pressure_kpa: float
    depth_m: float


def recv_line(sock: socket.socket, carry: bytes) -> tuple[str | None, bytes]:
    while b"\n" not in carry:
        chunk = sock.recv(RECV_BUFFER_SIZE)
        if not chunk:
            return None, b""
        carry += chunk

    raw_line, carry = carry.split(b"\n", 1)
    return raw_line.decode("utf-8", errors="replace").strip(), carry


def send_command(command: str) -> socket.socket:
    sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    sock.settimeout(CONNECT_TIMEOUT_SECONDS)
    sock.connect((SERVER_BT_MAC_ADDRESS, RFCOMM_CHANNEL))
    sock.sendall((command + "\n").encode("utf-8"))
    return sock


def decode_packet(payload_line: str) -> DataPacket:
    payload = json.loads(payload_line)
    return DataPacket(
        number=str(payload["number"]),
        elapsed_seconds=int(payload["elapsed_seconds"]),
        time_label=str(payload["time_label"]),
        pressure_kpa=float(payload["pressure_kpa"]),
        depth_m=float(payload["depth_m"]),
    )


def run_start() -> None:
    sock = send_command("start")
    carry = b""
    try:
        response, carry = recv_line(sock, carry)
        if response is None:
            print("[CLIENT] No response from server.")
            return
        print(f"[CLIENT] {response}")

        if response.startswith("OK"):
            packet_line, carry = recv_line(sock, carry)
            if packet_line is not None:
                packet = decode_packet(packet_line)
                print(f"[CLIENT] START PACKET -> {packet.number} {packet.time_label} {packet.pressure_kpa:.2f} kPa {packet.depth_m:.2f} meters")
    finally:
        sock.close()


def run_collect() -> None:
    sock = send_command("collect")
    carry = b""
    packets: List[DataPacket] = []

    try:
        start_line, carry = recv_line(sock, carry)
        if start_line is None or not start_line.startswith("DATA_BEGIN"):
            print(f"[CLIENT] Unexpected response: {start_line}")
            return

        while True:
            line, carry = recv_line(sock, carry)
            if line is None:
                break
            if line == "DATA_END":
                break
            packets.append(decode_packet(line))
    finally:
        sock.close()

    if not packets:
        print("[CLIENT] No profiling data received.")
        return

    print(f"{'Number':<10}{'Time':<10}{'Pressure':<18}{'Depth':<18}")
    for packet in packets:
        print(
            f"{packet.number:<10}"
            f"{packet.time_label:<10}"
            f"{packet.pressure_kpa:.2f} kPa{'':<8}"
            f"{packet.depth_m:.2f} meters"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[CLIENT] matplotlib not installed; skipping plot generation.")
        return

    x_seconds = [p.elapsed_seconds for p in packets]
    y_depth = [p.depth_m for p in packets]

    plt.figure(figsize=(10, 5))
    plt.plot(x_seconds, y_depth, marker="o", linewidth=1)
    plt.title("Vertical Profiler Depth vs Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Depth (meters)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=150)
    print(f"[CLIENT] Plot saved to {PLOT_OUTPUT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bluetooth profiler client")
    parser.add_argument("command", choices=["start", "collect"], help="Command to send to server")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "start":
        run_start()
    else:
        run_collect()


if __name__ == "__main__":
    main()
