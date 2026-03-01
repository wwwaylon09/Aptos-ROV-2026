#!/usr/bin/env python3
"""Bluetooth RFCOMM server for Raspberry Pi (no PyBluez required).

This script uses Python's built-in Linux Bluetooth socket support, so it works
on Raspberry Pi OS with Python 3.11 without installing `pybluez`.

It can optionally register an SPP service record via `sdptool` for Windows COM mapping.
For reliability, you can skip SDP and use direct RFCOMM client mode on Windows
(`bluetooth_client.py --bt-address ... --channel ...`).

Typical use on Pi:
    sudo python3 bluetooth_server.py --channel 3 --echo
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
        help="Attempt SDP registration via sdptool (needed only for Windows COM mode).",
    )
    parser.add_argument(
        "--sdp-name",
        default="Pi5 Bluetooth Server",
        help="Human-friendly service name for SDP registration (informational).",
    )
    parser.add_argument(
        "--adapter",
        default=socket.BDADDR_ANY,
        help="Local Bluetooth adapter address to bind, default is ANY.",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    try:
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return res.returncode, res.stdout.strip(), res.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"


def register_sdp_service(channel: int, service_name: str) -> bool:
    """Register an SPP SDP record via sdptool if available."""
    cmd = ["sdptool", "add", "--channel", str(channel), "SP"]
    code, out, err = run_cmd(cmd)
    if code == 0:
        print(f"[SERVER] SDP service registered on channel {channel}: {service_name}")
        if out:
            print(f"[SERVER] sdptool: {out}")
        return True

    print(f"[WARN] Failed to register SDP service (exit={code}).")
    if err:
        print(f"[WARN] sdptool stderr: {err}")
    print("[WARN] Windows may not map the correct COM port unless service/channel match.")
    return False


def print_local_adapter_info() -> None:
    code, out, err = run_cmd(["bluetoothctl", "show"])
    if code == 0 and out:
        print("[SERVER] bluetoothctl show:")
        for line in out.splitlines():
            print(f"[SERVER]   {line}")
    elif err:
        print(f"[WARN] Could not query adapter state: {err}")


def run_server(adapter: str, channel: int, buffer_size: int, echo: bool, register_sdp: bool, sdp_name: str) -> None:
    if not hasattr(socket, "AF_BLUETOOTH"):
        raise RuntimeError("This Python build does not support Bluetooth sockets.")

    server_sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    server_sock.bind((adapter, channel))
    server_sock.listen(1)

    actual_adapter, actual_channel = server_sock.getsockname()
    print(f"[SERVER] Listening on adapter {actual_adapter}, RFCOMM channel {actual_channel}")
    print_local_adapter_info()

    if register_sdp:
        sdp_ok = register_sdp_service(actual_channel, sdp_name)
        if not sdp_ok:
            print("[WARN] Continuing without SDP. Use Windows RFCOMM socket mode to bypass COM mapping.")
    else:
        print("[SERVER] SDP registration not requested. This is fine for direct RFCOMM socket mode.")

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
        print("[SERVER] Waiting for client RFCOMM connection...")
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
            adapter=args.adapter,
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
