#!/usr/bin/env python3
"""Windows Bluetooth RFCOMM client (direct MAC+channel mode).

Configuration is done via LOUD_SNAKE_CASE constants below.
No argparse and no COM/SDP infrastructure are used.
"""

from __future__ import annotations

import socket
import time

# =========================
# Runtime configuration
# =========================
SERVER_BT_MAC_ADDRESS = "00:00:00:00:00:00"  # Set Raspberry Pi Bluetooth MAC
RFCOMM_CHANNEL = 3
READ_TIMEOUT_SECONDS = 0.2
PING_MODE = True
PING_INTERVAL_SECONDS = 1.0
EXPECT_RESPONSE = True


def read_line(sock: socket.socket, buffer: bytes) -> tuple[bytes, bytes]:
    while b"\n" not in buffer:
        try:
            chunk = sock.recv(1024)
            if not chunk:
                return b"", b""
            buffer += chunk
        except TimeoutError:
            return b"", buffer

    line, buffer = buffer.split(b"\n", 1)
    return line + b"\n", buffer


def main() -> None:
    sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
    sock.settimeout(10.0)
    sock.connect((SERVER_BT_MAC_ADDRESS, RFCOMM_CHANNEL))
    sock.settimeout(READ_TIMEOUT_SECONDS)

    print(f"[CLIENT] Connected to {SERVER_BT_MAC_ADDRESS} channel {RFCOMM_CHANNEL}")

    rx_buffer = b""
    counter = 0

    try:
        if PING_MODE:
            print("[CLIENT] Ping mode started. Ctrl+C to stop.")
            while True:
                counter += 1
                line = f"ping {counter} @ {time.time():.3f}"
                sock.sendall((line + "\n").encode("utf-8"))
                print(f"[SEND] {line}")

                if EXPECT_RESPONSE:
                    response, rx_buffer = read_line(sock, rx_buffer)
                    if response:
                        print(f"[RECV] {response.decode('utf-8', errors='replace').rstrip()}")

                time.sleep(PING_INTERVAL_SECONDS)
        else:
            print("[CLIENT] Interactive mode started. Ctrl+C to stop.")
            while True:
                line = input("> ").strip()
                if not line:
                    continue

                sock.sendall((line + "\n").encode("utf-8"))
                print(f"[SEND] {line}")

                if EXPECT_RESPONSE:
                    response, rx_buffer = read_line(sock, rx_buffer)
                    if response:
                        print(f"[RECV] {response.decode('utf-8', errors='replace').rstrip()}")

    except KeyboardInterrupt:
        print("\n[CLIENT] Stopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
