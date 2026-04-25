import json
import socket
import struct
import pygame
import time
from typing import Optional

SERVER_IP = "192.168.42.42"  # Change to the server's IP address
SERVER_PORT = 5000
SEND_HZ = 50
DEADBAND = 0.1
RECONNECT_DELAY_SECONDS = 1.0
SOCKET_TIMEOUT_SECONDS = 2.0
TARGET_ADJUST_RATE_DEG_PER_SEC = 30.0
TARGET_ANGLE_LIMIT_DEG = 60.0


PS3_LAYOUT = {
    "claw_angle_increase": 14,
    "claw_angle_decrease": 12,
    "claw_rotate_increase": 13,
    "claw_rotate_decrease": 15,
    "pitch_positive": 4,
    "pitch_negative": 6,
    "roll_positive": 5,
    "roll_negative": 7,
    "stabilization_toggle": 3,
    "gyro_reset": 0,
}

# Xbox One maps D-pad to the first hat on most drivers instead of button indices.
XBOX_ONE_LAYOUT = {
    "claw_angle_increase": 0,
    "claw_angle_decrease": 3,
    "claw_rotate_increase": 1,
    "claw_rotate_decrease": 2,
    "pitch_positive": "dpad_up",
    "pitch_negative": "dpad_down",
    "roll_positive": "dpad_right",
    "roll_negative": "dpad_left",
    "stabilization_toggle": 7,
    "gyro_reset": 6,
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
enable_stabilization = False

# Input array to store joystick inputs
# [0:8] thrusters, [8] claw angle step command, [9] claw rotate step command, [10] stabilization
# [11] target pitch deg, [12] target roll deg, [13] gyro reset request
control_input = [0] * 14
control_input[10] = enable_stabilization
print(control_input)

target_pitch_deg = 0.0
target_roll_deg = 0.0


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
            # Initialize pitch, roll, and claw step commands
            pitch = 0
            roll = 0
            claw_angle_command = 0
            claw_rotate_command = 0

            # Need these values outside of the loop to prevent them from being falsely set by other buttons
            stabilization_pressed = False
            gyro_reset_pressed = False

            dpad_x = 0
            dpad_y = 0

            if joystick.get_numhats() > 0:
                dpad_x, dpad_y = joystick.get_hat(0)

            def control_active(control_name: str, button: Optional[int] = None) -> bool:
                mapping = controller_layout[control_name]
                if isinstance(mapping, int):
                    return button == mapping if button is not None else False
                if mapping == "dpad_left":
                    return dpad_x == -1
                if mapping == "dpad_right":
                    return dpad_x == 1
                if mapping == "dpad_up":
                    return dpad_y == 1
                if mapping == "dpad_down":
                    return dpad_y == -1
                return False

            # Check joystick buttons
            for button in range(joystick.get_numbuttons()):
                pressed = joystick.get_button(button)
                if pressed:
                    if control_active("claw_angle_increase", button):
                        claw_angle_command = 1
                    elif control_active("claw_angle_decrease", button):
                        claw_angle_command = -1
                    if control_active("claw_rotate_increase", button):
                        claw_rotate_command = 1
                    elif control_active("claw_rotate_decrease", button):
                        claw_rotate_command = -1
                    if control_active("pitch_positive", button):
                        pitch = 1
                    elif control_active("pitch_negative", button):
                        pitch = -1
                    if control_active("roll_positive", button):
                        roll = 1
                    elif control_active("roll_negative", button):
                        roll = -1
                    if control_active("stabilization_toggle", button):
                        stabilization_pressed = True
                    if control_active("gyro_reset", button):
                        gyro_reset_pressed = True

            if control_active("pitch_positive"):
                pitch = 1
            elif control_active("pitch_negative"):
                pitch = -1

            if control_active("roll_positive"):
                roll = 1
            elif control_active("roll_negative"):
                roll = -1

            control_input[8] = claw_angle_command
            control_input[9] = claw_rotate_command
            control_input[10] = enable_stabilization

            # Check for stabilization button pressed
            if stabilization_pressed:
                # Using a debounce ensures stabilization isn't constantly toggled while the button is pressed
                if stabilization_debounce:
                    stabilization_debounce = False
                    enable_stabilization = not enable_stabilization
            else:
                stabilization_debounce = True

            # Ensure latest stabilization value is always transmitted
            control_input[10] = enable_stabilization

            frame_dt = 1.0 / SEND_HZ
            if enable_stabilization:
                target_pitch_deg += pitch * TARGET_ADJUST_RATE_DEG_PER_SEC * frame_dt
                target_roll_deg += roll * TARGET_ADJUST_RATE_DEG_PER_SEC * frame_dt
                target_pitch_deg = max(-TARGET_ANGLE_LIMIT_DEG, min(TARGET_ANGLE_LIMIT_DEG, target_pitch_deg))
                target_roll_deg = max(-TARGET_ANGLE_LIMIT_DEG, min(TARGET_ANGLE_LIMIT_DEG, target_roll_deg))
                pitch = 0
                roll = 0

            if gyro_reset_pressed:
                target_pitch_deg = 0.0
                target_roll_deg = 0.0

            control_input[11] = round(target_pitch_deg, 2)
            control_input[12] = round(target_roll_deg, 2)
            control_input[13] = bool(gyro_reset_pressed)

            # Check joystick axes
            yaw = joystick.get_axis(2)
            forward_backward = -joystick.get_axis(1)
            left_right = joystick.get_axis(0)
            up_down = joystick.get_axis(3)

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
