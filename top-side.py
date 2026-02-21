import json
import socket
import struct
import pygame
import time
from typing import Optional

SERVER_IP = "192.168.42.42"  # Change to the server's IP address
SERVER_PORT = 5000
SEND_HZ = 50
DEADBAND = 0.05
RECONNECT_DELAY_SECONDS = 1.0
SOCKET_TIMEOUT_SECONDS = 2.0


PS3_LAYOUT = {
    "claw_angle_increase": 9,
    "claw_angle_decrease": 8,
    "claw_rotate_increase": 11,
    "claw_rotate_decrease": 10,
    "pitch_positive": 4,
    "pitch_negative": 6,
    "roll_positive": 5,
    "roll_negative": 7,
    "claw_angle_preset_low": 14,
    "claw_angle_preset_high": 13,
    "syringe_open": 12,
    "syringe_close": 15,
    "camera_zero": 1,
    "camera_max": 2,
    "stabilization_toggle": 3,
}

# Xbox One maps D-pad to the first hat on most drivers instead of button indices.
XBOX_ONE_LAYOUT = {
    "claw_angle_increase": 1,  # B
    "claw_angle_decrease": 0,  # A
    "claw_rotate_increase": 3,  # Y
    "claw_rotate_decrease": 2,  # X
    "pitch_positive": 4,  # LB
    "pitch_negative": 6,  # View/Back
    "roll_positive": 5,  # RB
    "roll_negative": 7,  # Menu/Start
    "syringe_open": "dpad_up",
    "syringe_close": "dpad_down",
    "claw_angle_preset_high": "dpad_left",
    "claw_angle_preset_low": "dpad_right",
    "camera_zero": 9,  # Left stick press
    "camera_max": 10,  # Right stick press
    "stabilization_toggle": 8,  # Xbox/Guide
}


def get_controller_layout(joystick_name: str) -> dict:
    """Select a button map based on connected controller name."""
    normalized_name = joystick_name.lower()
    if "xbox" in normalized_name:
        print("Using Xbox One button mapping")
        return XBOX_ONE_LAYOUT

    print("Using PS3 button mapping")
    return PS3_LAYOUT


# Initialize Pygame and joystick
pygame.init()
pygame.joystick.init()

# Variables to control the main loop and joystick state
got_joystick = False
stabilization_debounce = True
joystick: Optional[pygame.joystick.Joystick] = None
controller_layout = PS3_LAYOUT

# Initial input states
claw_angle = 50
claw_rotate = 90
syringe_angle = 90
camera_angle = 90
enable_stabilization = False

# Input array to store joystick inputs
control_input = [0] * 13
control_input[8] = claw_angle
control_input[9] = claw_rotate
control_input[10] = syringe_angle
control_input[11] = camera_angle
control_input[12] = enable_stabilization
print(control_input)


def clamp(value):
    """Clamp a value between -1.0 and 1.0."""
    return max(-1.0, min(1.0, value))


def connect_with_retry() -> socket.socket:
    """Continuously attempt to connect to the server."""
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(SOCKET_TIMEOUT_SECONDS)
            sock.connect((SERVER_IP, SERVER_PORT))
            print(f"Connected to {SERVER_IP}:{SERVER_PORT}")
            return sock
        except OSError as exc:
            print(f"Socket connection failed: {exc}. Retrying in {RECONNECT_DELAY_SECONDS}s...")
            time.sleep(RECONNECT_DELAY_SECONDS)


def send_frame(sock: socket.socket, payload: list):
    """Send a length-prefixed JSON payload over TCP."""
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    header = struct.pack("!I", len(data))
    sock.sendall(header + data)


s = connect_with_retry()

try:
    while True:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEADDED:
                joystick = pygame.joystick.Joystick(event.device_index)
                controller_layout = get_controller_layout(joystick.get_name())
                got_joystick = True
                print(f"Joystick connected: {joystick.get_name()}")
            elif event.type == pygame.JOYDEVICEREMOVED:
                got_joystick = False
                joystick = None
                print("Joystick disconnected")

        if got_joystick and joystick is not None:
            # Initialize pitch and roll
            pitch = 0
            roll = 0

            # Need this value outside of the loop to prevent it from being falsely set by other buttons
            stabilization_pressed = False

            # Check joystick buttons
            for button in range(joystick.get_numbuttons()):
                pressed = joystick.get_button(button)
                if pressed:
                    if button == controller_layout["claw_angle_increase"] and claw_angle <= 179:
                        claw_angle += 1
                    elif button == controller_layout["claw_angle_decrease"] and claw_angle >= 1:
                        claw_angle -= 1
                    if button == controller_layout["claw_rotate_increase"] and claw_rotate <= 179:
                        claw_rotate += 1
                    elif button == controller_layout["claw_rotate_decrease"] and claw_rotate >= 1:
                        claw_rotate -= 1
                    if button == controller_layout["pitch_positive"]:
                        pitch = 1
                    elif button == controller_layout["pitch_negative"]:
                        pitch = -1
                    if button == controller_layout["roll_positive"]:
                        roll = 1
                    elif button == controller_layout["roll_negative"]:
                        roll = -1
                    if button == controller_layout["claw_angle_preset_low"]:
                        claw_angle = 65
                    if button == controller_layout["claw_angle_preset_high"]:
                        claw_angle = 100
                    if button == controller_layout["syringe_open"]:
                        syringe_angle = 180
                    if button == controller_layout["syringe_close"]:
                        syringe_angle = 0
                    if button == controller_layout["camera_zero"]:
                        camera_angle = 0
                    if button == controller_layout["camera_max"]:
                        camera_angle = 180
                    if button == controller_layout["stabilization_toggle"]:
                        stabilization_pressed = True

                    control_input[8] = claw_angle
                    control_input[9] = claw_rotate
                    control_input[10] = syringe_angle
                    control_input[11] = camera_angle
                    control_input[12] = enable_stabilization

            if joystick.get_numhats() > 0:
                dpad_x, dpad_y = joystick.get_hat(0)
                if controller_layout["claw_angle_preset_low"] == "dpad_right" and dpad_x == 1:
                    claw_angle = 65
                if controller_layout["claw_angle_preset_high"] == "dpad_left" and dpad_x == -1:
                    claw_angle = 100
                if controller_layout["syringe_open"] == "dpad_up" and dpad_y == 1:
                    syringe_angle = 180
                if controller_layout["syringe_close"] == "dpad_down" and dpad_y == -1:
                    syringe_angle = 0

                control_input[8] = claw_angle
                control_input[9] = claw_rotate
                control_input[10] = syringe_angle
                control_input[11] = camera_angle
                control_input[12] = enable_stabilization

            # Check for stabilization button pressed
            if stabilization_pressed:
                # Using a debounce ensures stabilization isn't constantly toggled while the button is pressed
                if stabilization_debounce:
                    stabilization_debounce = False
                    enable_stabilization = not enable_stabilization
            else:
                stabilization_debounce = True

            # Ensure latest stabilization value is always transmitted
            control_input[12] = enable_stabilization

            # Check joystick axes
            yaw = joystick.get_axis(0)
            forward_backward = -joystick.get_axis(3)
            left_right = joystick.get_axis(2)
            up_down = joystick.get_axis(1)

            # Apply deadband to joystick input
            forward_backward = 0 if abs(forward_backward) < DEADBAND else forward_backward
            left_right = 0 if abs(left_right) < DEADBAND else left_right
            yaw = 0 if abs(yaw) < DEADBAND else yaw
            up_down = 0 if abs(up_down) < DEADBAND else up_down

            # Calculate motor speeds from joystick inputs
            control_input[0] = round(clamp(-forward_backward + left_right + up_down - pitch + yaw + roll), 3)
            control_input[1] = round(clamp(-forward_backward - left_right + up_down - pitch - yaw - roll), 3)
            control_input[2] = round(clamp(-forward_backward + left_right - up_down + pitch + yaw - roll), 3)
            control_input[3] = round(clamp(-forward_backward - left_right - up_down + pitch - yaw + roll), 3)
            control_input[4] = round(clamp(forward_backward + left_right + up_down + pitch - yaw + roll), 3)
            control_input[5] = round(clamp(forward_backward - left_right + up_down + pitch + yaw - roll), 3)
            control_input[6] = round(clamp(forward_backward + left_right - up_down - pitch - yaw - roll), 3)
            control_input[7] = round(clamp(forward_backward - left_right - up_down - pitch + yaw + roll), 3)
            print(control_input)

            # Send input data to the server using framed messages and reconnect on failure
            try:
                send_frame(s, control_input)
            except (OSError, TimeoutError) as exc:
                print(f"Send failed: {exc}. Reconnecting...")
                try:
                    s.close()
                except OSError:
                    pass
                s = connect_with_retry()

            time.sleep(1 / SEND_HZ)

except KeyboardInterrupt:
    # Handle cleanup on exit
    print("\nCtrl + C pressed. Cleaning up")
finally:
    if joystick is not None:
        joystick.quit()
    pygame.quit()
    try:
        s.close()
    except OSError:
        pass
