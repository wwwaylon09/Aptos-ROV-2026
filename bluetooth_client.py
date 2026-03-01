#!/usr/bin/env python3
"""Bluetooth client for Windows 11.

Two connection modes are supported:
1) COM mode (default): connect to Windows Bluetooth COM port via pyserial.
2) RFCOMM socket mode: connect directly to Pi MAC + channel using Bluetooth socket.

RFCOMM socket mode avoids Windows COM/SDP mapping issues entirely.

Dependencies:
    pip install pyserial
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from typing import Optional

import serial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Connect to Raspberry Pi via Bluetooth COM port or RFCOMM socket.")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--port", help="Windows COM port (e.g., COM7).")
    mode.add_argument("--bt-address", help="Pi Bluetooth MAC address for direct RFCOMM socket mode.")

    parser.add_argument("--channel", type=int, default=3, help="RFCOMM channel for --bt-address mode (default: 3).")
    parser.add_argument("--baudrate", type=int, default=115200, help="Nominal serial baudrate (COM mode only).")
    parser.add_argument("--timeout", type=float, default=0.2, help="Read timeout in seconds.")
    parser.add_argument("--interval", type=float, default=1.0, help="Delay between messages in ping mode.")
    parser.add_argument("--ping", action="store_true", help="Send periodic ping messages instead of reading stdin.")
    parser.add_argument(
        "--expect-response",
        action="store_true",
        help="After each send, wait for a response line (useful when server echoes/replies).",
    )
    return parser.parse_args()


class IOAdapter:
    def write_line(self, text: str) -> None:
        raise NotImplementedError

    def read_line(self) -> bytes:
        raise NotImplementedError

    def read_available(self) -> list[bytes]:
        raise NotImplementedError


class SerialAdapter(IOAdapter):
    def __init__(self, ser: serial.Serial) -> None:
        self.ser = ser

    def write_line(self, text: str) -> None:
        self.ser.write((text + "\n").encode("utf-8"))
        self.ser.flush()

    def read_line(self) -> bytes:
        return self.ser.readline()

    def read_available(self) -> list[bytes]:
        lines: list[bytes] = []
        while self.ser.in_waiting > 0:
            chunk = self.ser.readline()
            if not chunk:
                break
            lines.append(chunk)
        return lines


class RFCOMMAdapter(IOAdapter):
    def __init__(self, sock: socket.socket, timeout: float) -> None:
        self.sock = sock
        self.sock.settimeout(timeout)
        self._buffer = b""

    def write_line(self, text: str) -> None:
        self.sock.sendall((text + "\n").encode("utf-8"))

    def read_line(self) -> bytes:
        while b"\n" not in self._buffer:
            try:
                data = self.sock.recv(1024)
                if not data:
                    return b""
                self._buffer += data
            except TimeoutError:
                return b""
            except OSError:
                return b""

        line, self._buffer = self._buffer.split(b"\n", 1)
        return line + b"\n"

    def read_available(self) -> list[bytes]:
        out: list[bytes] = []
        while True:
            line = self.read_line()
            if not line:
                break
            out.append(line)
        return out


def print_recv(data: bytes) -> None:
    print(f"[RECV] {data.decode('utf-8', errors='replace').rstrip()}")


def run_interactive(ioa: IOAdapter, expect_response: bool) -> None:
    print("[CLIENT] Connected. Type messages and press Enter. Ctrl+C to exit.")
    while True:
        line = input("> ").strip()
        if not line:
            continue

        ioa.write_line(line)
        print(f"[SEND] {line}")

        if expect_response:
            response = ioa.read_line()
            if response:
                print_recv(response)
        else:
            for response in ioa.read_available():
                print_recv(response)


def run_ping(ioa: IOAdapter, interval: float, expect_response: bool) -> None:
    print("[CLIENT] Ping mode started. Ctrl+C to exit.")
    counter = 0
    while True:
        counter += 1
        line = f"ping {counter} @ {time.time():.3f}"
        ioa.write_line(line)
        print(f"[SEND] {line}")

        if expect_response:
            response = ioa.read_line()
            if response:
                print_recv(response)
        else:
            for response in ioa.read_available():
                print_recv(response)

        time.sleep(interval)


def main() -> None:
    args = parse_args()

    try:
        if args.port:
            with serial.Serial(args.port, baudrate=args.baudrate, timeout=args.timeout, write_timeout=2.0) as ser:
                print(f"[CLIENT] Opened {ser.portstr} (COM mode).")
                adapter: IOAdapter = SerialAdapter(ser)
                if args.ping:
                    run_ping(adapter, args.interval, args.expect_response)
                else:
                    run_interactive(adapter, args.expect_response)
        else:
            if not hasattr(socket, "AF_BLUETOOTH"):
                print("[CLIENT] This Python build does not support Bluetooth sockets.", file=sys.stderr)
                sys.exit(1)
            sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
            try:
                sock.settimeout(10.0)
                sock.connect((args.bt_address, args.channel))
                print(f"[CLIENT] Connected to {args.bt_address} channel {args.channel} (RFCOMM socket mode).")
                adapter = RFCOMMAdapter(sock, args.timeout)
                if args.ping:
                    run_ping(adapter, args.interval, args.expect_response)
                else:
                    run_interactive(adapter, args.expect_response)
            finally:
                sock.close()
    except serial.SerialTimeoutException as exc:
        print(f"[CLIENT] Write timeout: {exc}", file=sys.stderr)
        sys.exit(1)
    except serial.SerialException as exc:
        print(f"[CLIENT] Serial error: {exc}", file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        print(f"[CLIENT] Bluetooth socket error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[CLIENT] Stopped.")


if __name__ == "__main__":
    main()
