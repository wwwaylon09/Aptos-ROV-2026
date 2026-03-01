#!/usr/bin/env python3
"""Bluetooth RFCOMM server for Raspberry Pi.

This script starts a Bluetooth Classic SPP (Serial Port Profile) server.
Use it on the Raspberry Pi 5 side, then connect from Windows via the paired
Bluetooth COM port.

Dependencies:
    pip install pybluez
"""

from __future__ import annotations

import argparse
import signal
import sys
from typing import Optional

import bluetooth

# Standard Serial Port Profile UUID.
SPP_UUID = "00001101-0000-1000-8000-00805F9B34FB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Bluetooth RFCOMM server.")
    parser.add_argument(
        "--name",
        default="Pi5 Bluetooth Server",
        help="Service name advertised to clients.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="RFCOMM channel (0 = auto-assign).",
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
    return parser.parse_args()


def run_server(name: str, channel: int, buffer_size: int, echo: bool) -> None:
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", channel))
    server_sock.listen(1)

    actual_channel = server_sock.getsockname()[1]

    bluetooth.advertise_service(
        server_sock,
        name,
        service_id=SPP_UUID,
        service_classes=[SPP_UUID, bluetooth.SERIAL_PORT_CLASS],
        profiles=[bluetooth.SERIAL_PORT_PROFILE],
    )

    print(f"[SERVER] Listening on RFCOMM channel {actual_channel}")
    print(f"[SERVER] Advertising service '{name}' with UUID {SPP_UUID}")

    client_sock: Optional[bluetooth.BluetoothSocket] = None

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
                    client_sock.send(data)
        except (OSError, bluetooth.BluetoothError) as exc:
            print(f"[SERVER] Connection error: {exc}")
        finally:
            try:
                client_sock.close()
            except OSError:
                pass
            client_sock = None


def main() -> None:
    args = parse_args()
    run_server(
        name=args.name,
        channel=args.channel,
        buffer_size=args.buffer_size,
        echo=args.echo,
    )


if __name__ == "__main__":
    main()
