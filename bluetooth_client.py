#!/usr/bin/env python3
"""Bluetooth client for Windows 11 using a paired COM port.

Use this script on the Windows laptop after pairing it with the Raspberry Pi.
Windows exposes the Pi Serial Port Profile service as a COM port.

Dependencies:
    pip install pyserial
"""

from __future__ import annotations

import argparse
import sys
import time

import serial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Connect to Raspberry Pi via Bluetooth COM port.")
    parser.add_argument(
        "--port",
        required=True,
        help="Windows COM port assigned to the Raspberry Pi SPP service (e.g., COM7).",
    )
    parser.add_argument("--baudrate", type=int, default=115200, help="Nominal serial baudrate.")
    parser.add_argument("--timeout", type=float, default=1.0, help="Read timeout in seconds.")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Delay between messages in ping mode.",
    )
    parser.add_argument(
        "--ping",
        action="store_true",
        help="Send periodic ping messages instead of reading stdin.",
    )
    return parser.parse_args()


def run_interactive(ser: serial.Serial) -> None:
    print("[CLIENT] Connected. Type messages and press Enter. Ctrl+C to exit.")
    while True:
        line = input("> ").strip()
        if not line:
            continue

        payload = (line + "\n").encode("utf-8")
        ser.write(payload)

        response = ser.readline()
        if response:
            print(f"[RECV] {response.decode('utf-8', errors='replace').rstrip()}")


def run_ping(ser: serial.Serial, interval: float) -> None:
    print("[CLIENT] Ping mode started. Ctrl+C to exit.")
    counter = 0
    while True:
        counter += 1
        payload = f"ping {counter} @ {time.time():.3f}\n".encode("utf-8")
        ser.write(payload)
        print(f"[SEND] {payload.decode().rstrip()}")

        response = ser.readline()
        if response:
            print(f"[RECV] {response.decode('utf-8', errors='replace').rstrip()}")

        time.sleep(interval)


def main() -> None:
    args = parse_args()

    try:
        with serial.Serial(args.port, baudrate=args.baudrate, timeout=args.timeout) as ser:
            if args.ping:
                run_ping(ser, args.interval)
            else:
                run_interactive(ser)
    except serial.SerialException as exc:
        print(f"[CLIENT] Serial error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[CLIENT] Stopped.")


if __name__ == "__main__":
    main()
