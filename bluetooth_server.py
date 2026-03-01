#!/usr/bin/env python3
"""Bluetooth RFCOMM server for Raspberry Pi (no PyBluez required).

This script uses Python's built-in Linux Bluetooth socket support, so it works
on Raspberry Pi OS with Python 3.11 without installing `pybluez`.

It optionally registers an SPP service record via `sdptool` so Windows can find
it as a serial service while pairing/connecting.

Typical use on Pi:
    sudo python3 bluetooth_server.py --channel 3 --register-sdp

Typical use on Windows:
    Pair with the Pi, create/use an Outgoing COM port for the Pi Serial service,
    then run bluetooth_client.py against that COM port.
"""

from __future__ import annotations

import argparse
import signal
import socket
import subprocess
import sys
from typing import Optional

DEFAULT_CHANNEL = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Bluetooth RFCOMM server on Linux/Raspberry Pi.")
    parser.add_argument(
        "--channel",
        type=int,
        default=DEFAULT_CHANNEL,
        help=f"RFCOMM channel to bind (default: {DEFAULT_CHANNEL}).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1024,
        help="Receive buffer size in bytes.",
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Echo received payload back to the client.",
    )
    parser.add_argument(
        "--register-sdp",
        action="store_true",
        help="Register Serial Port Profile service with sdptool on the selected channel.",
    )
    parser.add_argument(
        "--sdp-name",
        default="Pi5 Bluetooth Server",
        help="Human-friendly service name for SDP registration.",
    )
    return parser.parse_args()


def register_sdp_service(channel: int, service_name: str) -> None:
    """Register an SPP SDP record via sdptool if available.

    On many Linux systems this requires root privileges.
    """
    cmd = ["sdptool", "add", "--channel", str(channel), "SP"]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[SERVER] SDP service registered on channel {channel}: {service_name}")
        if result.stdout.strip():
            print(f"[SERVER] sdptool: {result.stdout.strip()}")
    except FileNotFoundError:
        print("[WARN] 'sdptool' not found. Install bluez package if SDP registration is needed.")
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] Failed to register SDP service ({exc.returncode}).")
        if exc.stderr.strip():
            print(f"[WARN] sdptool stderr: {exc.stderr.strip()}")
        print("[WARN] Server will continue, but Windows discovery may require manual COM setup.")


def run_server(channel: int, buffer_size: int, echo: bool, register_sdp: bool, sdp_name: str) -> None:
    if not hasattr(socket, "AF_BLUETOOTH"):
        raise RuntimeError("This Python build does not support Bluetooth sockets.")

    server_sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    server_sock.bind((socket.BDADDR_ANY, channel))
    server_sock.listen(1)

    actual_channel = server_sock.getsockname()[1]
    print(f"[SERVER] Listening on RFCOMM channel {actual_channel}")

    if register_sdp:
        register_sdp_service(actual_channel, sdp_name)

    client_sock: Optional[socket.socket] = None

    def shutdown_handler(signum: int, _frame: object) -> None:
        nonlocal client_sock
        print(f"\n[SERVER] Received signal {signum}, shutting down...")
        if client_sock is not None:
            try:
                client_sock.close()
            except OSError:
                pass
        server_sock.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    while True:
        print("[SERVER] Waiting for client connection...")
        client_sock, client_info = server_sock.accept()
        print(f"[SERVER] Client connected: {client_info}")

        try:
            while True:
                data = client_sock.recv(buffer_size)
                if not data:
                    print("[SERVER] Client disconnected.")
                    break

                decoded = data.decode("utf-8", errors="replace").rstrip("\r\n")
                print(f"[RECV] {decoded}")

                if echo:
                    client_sock.sendall(data)
        except OSError as exc:
            print(f"[SERVER] Connection error: {exc}")
        finally:
            try:
                client_sock.close()
            except OSError:
                pass
            client_sock = None


def main() -> None:
    args = parse_args()
    try:
        run_server(
            channel=args.channel,
            buffer_size=args.buffer_size,
            echo=args.echo,
            register_sdp=args.register_sdp,
            sdp_name=args.sdp_name,
        )
    except PermissionError:
        print("[ERROR] Permission denied opening Bluetooth RFCOMM socket.")
        print("[ERROR] Try running with sudo, or grant CAP_NET_ADMIN/CAP_NET_RAW as needed.")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
