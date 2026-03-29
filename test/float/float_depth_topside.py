#!/usr/bin/env python3
"""Bluetooth RFCOMM client for interactive float depth calibration."""

from __future__ import annotations

import argparse
import json
import shlex
import socket
import sys
import time

SERVER_BT_MAC_ADDRESS = "00:00:00:00:00:00"
RFCOMM_CHANNEL = 3
CONNECT_TIMEOUT_SECONDS = 10.0
CONNECT_RETRIES = 3
CONNECT_RETRY_DELAY_SECONDS = 1.0
RECV_BUFFER_SIZE = 1024


def recv_line(sock: socket.socket, carry: bytes) -> tuple[str | None, bytes]:
    while b"\n" not in carry:
        chunk = sock.recv(RECV_BUFFER_SIZE)
        if not chunk:
            return None, b""
        carry += chunk

    raw_line, carry = carry.split(b"\n", 1)
    return raw_line.decode("utf-8", errors="replace").strip(), carry


def connect() -> socket.socket:
    last_error: OSError | None = None

    for attempt in range(1, CONNECT_RETRIES + 1):
        sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        sock.settimeout(CONNECT_TIMEOUT_SECONDS)
        try:
            sock.connect((SERVER_BT_MAC_ADDRESS, RFCOMM_CHANNEL))
            sock.settimeout(None)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
            if attempt < CONNECT_RETRIES:
                print(f"[CLIENT] Connect attempt {attempt}/{CONNECT_RETRIES} failed: {exc}. Retrying...")
                time.sleep(CONNECT_RETRY_DELAY_SECONDS)

    raise RuntimeError(f"unable to connect after {CONNECT_RETRIES} attempts: {last_error}")


def send_and_receive(sock: socket.socket, carry: bytes, command: str) -> tuple[dict[str, object] | None, bytes]:
    sock.sendall((command + "\n").encode("utf-8"))
    line, carry = recv_line(sock, carry)
    if line is None:
        return None, carry
    return json.loads(line), carry


def print_response(response: dict[str, object]) -> None:
    status = response.get("status", "unknown")
    command = response.get("command", "")

    if status != "ok":
        print(f"[CLIENT] ERROR: {response.get('message', 'unknown error')}")
        return

    if command == "rotate":
        print(
            "[CLIENT] Rotated "
            f"{float(response['degrees']):.2f}° "
            f"({response['direction']}, {int(response['steps'])} steps)"
        )
        return

    if command == "depth":
        print(
            "[CLIENT] Current depth: "
            f"{float(response['depth_m']):.3f} m "
            f"at {float(response['pressure_kpa']):.2f} kPa"
        )
        return

    if command == "help":
        print("[CLIENT] Available commands:")
        commands = response.get("commands", {})
        if isinstance(commands, dict):
            for name, description in commands.items():
                print(f"  - {name}: {description}")
        return

    if command == "quit":
        print("[CLIENT] Server closed the active calibration session.")
        return

    print(f"[CLIENT] {response}")


def run_command(sock: socket.socket, command: str, carry: bytes) -> tuple[bool, bytes]:
    response, carry = send_and_receive(sock, carry, command)
    if response is None:
        print("[CLIENT] Server disconnected.")
        return False, carry

    print_response(response)
    return command.split()[0].lower() != "quit", carry


def run_repl(sock: socket.socket) -> None:
    carry = b""
    print("[CLIENT] Connected. Enter commands: rotate <degrees>, depth, help, quit")

    while True:
        try:
            raw_command = input("float-depth> ").strip()
        except EOFError:
            raw_command = "quit"
        except KeyboardInterrupt:
            print()
            raw_command = "quit"

        if not raw_command:
            continue

        try:
            parts = shlex.split(raw_command)
        except ValueError as exc:
            print(f"[CLIENT] Invalid command syntax: {exc}")
            continue

        if not parts:
            continue

        command_name = parts[0].lower()
        if command_name == "rotate" and len(parts) != 2:
            print("[CLIENT] Usage: rotate <degrees>")
            continue
        if command_name == "depth" and len(parts) != 1:
            print("[CLIENT] Usage: depth")
            continue

        keep_running, carry = run_command(sock, raw_command, carry)
        if not keep_running:
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bluetooth float depth calibration client")
    parser.add_argument(
        "--mac-address",
        default=SERVER_BT_MAC_ADDRESS,
        help="Bluetooth MAC address of the bottomside server",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("shell", help="Open a persistent interactive calibration session")
    rotate_parser = subparsers.add_parser("rotate", help="Rotate the stepper motor by the requested degrees")
    rotate_parser.add_argument("degrees", type=float, help="Degrees to rotate; positive or negative")
    subparsers.add_parser("depth", help="Read and print the current depth in meters")

    return parser.parse_args()


def main() -> None:
    global SERVER_BT_MAC_ADDRESS

    if not hasattr(socket, "AF_BLUETOOTH"):
        print("[CLIENT] Bluetooth sockets are not supported by this Python build.")
        sys.exit(1)

    args = parse_args()
    SERVER_BT_MAC_ADDRESS = args.mac_address

    sock = connect()
    try:
        if args.command in (None, "shell"):
            run_repl(sock)
            return

        carry = b""
        if args.command == "rotate":
            run_command(sock, f"rotate {args.degrees}", carry)
        else:
            run_command(sock, "depth", carry)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
