# Activate Python environment
# source my_env/bin/activate

import json
import socket
import struct
import time
import board
import adafruit_pca9685
import adafruit_mpu6050
import gpiozero
import math

HOST = ""
PORT = 5000
SOCKET_TIMEOUT_SECONDS = 1.0
MAX_FRAME_SIZE = 65536
NO_DATA_FAILSAFE_SECONDS = 0.5

# Initialize hardware components
i2c = board.I2C()
pca = adafruit_pca9685.PCA9685(i2c)
mpu = adafruit_mpu6050.MPU6050(i2c)
relay_pin = gpiozero.LED(4)

pca.frequency = 50

# Motor and Servo configurations
# Change these based on which port each motor is pluged in to
motor_1 = 0
motor_2 = 1
motor_3 = 2
motor_4 = 3
motor_5 = 4
motor_6 = 5
motor_7 = 6
motor_8 = 7

claw_open = 15
claw_rotate = 14


# Define function to convert joystick inputs to PWM values
def convert(x):
    throttle_multiplier = 0.2  # Number between 0 and 1
    max_duty_cycle = 5240 + throttle_multiplier * 1640
    min_duty_cycle = 5240 - throttle_multiplier * 1640
    mapped_value = round((((x + 1) / 2) * (max_duty_cycle - min_duty_cycle)) + min_duty_cycle)
    return mapped_value


# Calculate pitch and roll angles (radians)
def calculate_orientation():
    accel_x, accel_y, accel_z = mpu.acceleration
    pitch = math.atan2(accel_x, math.sqrt(accel_y**2 + accel_z**2))
    roll = math.atan2(-accel_y, accel_z)
    return pitch, roll


# Interpolates a to b by fraction t
def lerp(a, b, t):
    return a + (b - a) * t


# Merge joystick and MPU inputs, but prioritize joystick inputs
def merge_inputs(joystick_input, mpu_input):
    # Avoid imaginary numbers if MPU input is negative
    if mpu_input < 0:
        mpu_input = -((-mpu_input) ** 0.5)
    else:
        mpu_input **= 0.5

    return lerp(mpu_input, joystick_input, math.fabs(joystick_input))


def set_neutral_thrusters():
    """Set all thrusters to neutral throttle."""
    neutral = convert(0)
    pca.channels[motor_1].duty_cycle = neutral
    pca.channels[motor_2].duty_cycle = neutral
    pca.channels[motor_3].duty_cycle = neutral
    pca.channels[motor_4].duty_cycle = neutral
    pca.channels[motor_5].duty_cycle = neutral
    pca.channels[motor_6].duty_cycle = neutral
    pca.channels[motor_7].duty_cycle = neutral
    pca.channels[motor_8].duty_cycle = neutral


def apply_thrusters(inputs):
    """Apply decoded control values to all thruster channels."""
    pca.channels[motor_1].duty_cycle = convert(inputs[0])
    pca.channels[motor_2].duty_cycle = convert(inputs[1])
    pca.channels[motor_3].duty_cycle = convert(inputs[2])
    pca.channels[motor_4].duty_cycle = convert(inputs[3])
    pca.channels[motor_5].duty_cycle = convert(inputs[4])
    pca.channels[motor_6].duty_cycle = convert(inputs[5])
    pca.channels[motor_7].duty_cycle = convert(inputs[6])
    pca.channels[motor_8].duty_cycle = convert(inputs[7])


def read_exact(connection, length):
    """Read exactly length bytes or return None on disconnect/timeout."""
    buf = b""
    while len(buf) < length:
        try:
            chunk = connection.recv(length - len(buf))
        except socket.timeout:
            return None
        if not chunk:
            return None
        buf += chunk
    return buf


def receive_frame(connection):
    """Receive one length-prefixed JSON frame."""
    header = read_exact(connection, 4)
    if header is None:
        return None

    frame_length = struct.unpack("!I", header)[0]
    if frame_length <= 0 or frame_length > MAX_FRAME_SIZE:
        raise ValueError(f"Invalid frame size: {frame_length}")

    payload = read_exact(connection, frame_length)
    if payload is None:
        return None

    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, list) or len(decoded) != 13:
        raise ValueError("Expected control payload as list[13]")

    return decoded


def setup_server_socket():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print("Server is now running. Waiting for client...")
    return server


# Send a signal to power the ESC relay
print("Powering up ESCs")
relay_pin.on()
set_neutral_thrusters()

server_socket = setup_server_socket()

while True:
    connection, client_address = server_socket.accept()
    connection.settimeout(SOCKET_TIMEOUT_SECONDS)
    print(f"Client connected: {client_address}")
    last_packet_time = time.monotonic()

    try:
        while True:
            try:
                inputs = receive_frame(connection)
            except ValueError as exc:
                print(f"Dropping invalid packet: {exc}")
                continue

            now = time.monotonic()

            if inputs is None:
                if now - last_packet_time > NO_DATA_FAILSAFE_SECONDS:
                    print("Failsafe: no valid data recently, setting neutral thrusters")
                    set_neutral_thrusters()
                break

            last_packet_time = now

            # Check if stabilization is enabled
            if inputs[12]:
                # Get angle inputs from MPU
                pitch, roll = calculate_orientation()
                pitch, roll = pitch / math.pi, roll / math.pi

                # Merge inputs
                inputs[0] = merge_inputs(inputs[0], roll - pitch)
                inputs[1] = merge_inputs(inputs[1], -roll - pitch)
                inputs[2] = merge_inputs(inputs[2], -roll + pitch)
                inputs[3] = merge_inputs(inputs[3], roll + pitch)
                inputs[4] = merge_inputs(inputs[4], roll + pitch)
                inputs[5] = merge_inputs(inputs[5], -roll + pitch)
                inputs[6] = merge_inputs(inputs[6], -roll - pitch)
                inputs[7] = merge_inputs(inputs[7], roll - pitch)

            print(inputs)
            apply_thrusters(inputs)

    except OSError as exc:
        print(f"Connection error: {exc}")
    finally:
        set_neutral_thrusters()
        connection.close()
        print("Client disconnected. Waiting for reconnect...")
