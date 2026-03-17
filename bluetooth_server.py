#!/usr/bin/env python3
"""Raspberry Pi Bluetooth RFCOMM server (direct channel mode).

Configuration is done via LOUD_SNAKE_CASE constants below.
No argparse and no SDP tooling are used.
"""

from __future__ import annotations

import signal
import socket
import sys

# =========================
# Runtime configuration
# =========================
SERVER_BT_ADDRESS = socket.BDADDR_ANY  # Use local adapter automatically
RFCOMM_CHANNEL = 3
BUFFER_SIZE = 1024
ECHO_BACK_TO_CLIENT = True


def main() -> None:
    if not hasattr(socket, "AF_BLUETOOTH"):
        print("[SERVER] Bluetooth sockets are not supported by this Python build.")
        sys.exit(1)

    server_sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    server_sock.bind((SERVER_BT_ADDRESS, RFCOMM_CHANNEL))
    server_sock.listen(1)

    bound_addr, bound_channel = server_sock.getsockname()
    print(f"[SERVER] Listening on adapter={bound_addr}, channel={bound_channel}")

    client_sock: socket.socket | None = None

    def shutdown_handler(signum: int, _frame: object) -> None:
        nonlocal client_sock
        print(f"\n[SERVER] Signal {signum} received. Shutting down...")
        if client_sock is not None:
            client_sock.close()
        server_sock.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    while True:
        print("[SERVER] Waiting for RFCOMM client...")
        client_sock, client_info = server_sock.accept()
        print(f"[SERVER] Client connected: {client_info}")

        try:
            while True:
                data = client_sock.recv(BUFFER_SIZE)
                if not data:
                    print("[SERVER] Client disconnected.")
                    break

                message = data.decode("utf-8", errors="replace").rstrip("\r\n")
                print(f"[RECV] {message}")

                if ECHO_BACK_TO_CLIENT:
                    client_sock.sendall(data)
        except OSError as exc:
            print(f"[SERVER] Connection closed with error: {exc}")
        finally:
            client_sock.close()
            client_sock = None


if __name__ == "__main__":
    main()
