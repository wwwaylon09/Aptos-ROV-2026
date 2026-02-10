import socket
import pickle
import pygame
import time

# Initialize socket connection
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("192.168.42.42", 5000))  # Change the IP address to the server's IP address

# Initialize Pygame and joystick
pygame.init()
pygame.joystick.init()

# Variables to control the main loop and joystick state
got_joystick = False
stabilization_debounce = True

# Initial input states
claw_angle = 50
claw_rotate = 90
syringe_angle = 90
camera_angle = 90
enable_stabilization = False

# Input array to store joystick inputs
input = [0] * 13
input[8] = claw_angle
input[9] = claw_rotate
input[10] = syringe_angle
input[11] = camera_angle
input[12] = enable_stabilization
print(input)

def clamp(value):
    """Clamp a value between -1.0 and 1.0."""
    return max(-1.0, min(1.0, value))

try:
    while True:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEADDED:
                joystick = pygame.joystick.Joystick(event.device_index)
                got_joystick = True
                print(joystick)

        if got_joystick:
            # Initialize pitch and roll
            pitch = 0
            roll = 0

            # Need this value outside of the loop to prevent it from being falsely set by other buttons
            stabilization_pressed = False

            # Check joystick buttons
            for button in range(17):
                pressed = joystick.get_button(button)
                if pressed:
                    if button == 9 and claw_angle <= 179:
                        claw_angle += 1
                    elif button == 8 and claw_angle >= 1:
                        claw_angle -= 1
                    if button == 11 and claw_rotate <= 179:
                        claw_rotate += 1
                    elif button == 10 and claw_rotate >= 1:
                        claw_rotate -= 1
                    if button == 4:
                        pitch = 1
                    elif button == 6:
                        pitch = -1
                    if button == 5:
                        roll = 1
                    elif button == 7:
                        roll = -1
                    if button == 14:
                        claw_angle = 65
                    if button == 13:
                        claw_angle = 100
                    if button == 12:
                        syringe_angle = 180
                    if button == 15:
                        syringe_angle = 0
                    if button == 1:
                        camera_angle = 0
                    if button == 2:
                        camera_angle = 180
                    if button == 3:
                        stabilization_pressed = True

                    input[8] = claw_angle
                    input[9] = claw_rotate
                    input[10] = syringe_angle
                    input[11] = camera_angle
                    input[12] = enable_stabilization

            # Check for stabilization button pressed
            if stabilization_pressed:
                # Using a debounce ensures stabilization isn't constantly toggled while the button is pressed
                if stabilization_debounce:
                    stabilization_debounce = False
                    enable_stabilization = not enable_stabilization
            else:
                stabilization_debounce = True

            # Check joystick axes
            yaw = joystick.get_axis(0)
            forward_backward = -joystick.get_axis(3)
            left_right = joystick.get_axis(2)
            up_down = joystick.get_axis(1)

            # Apply deadband to joystick input
            deadband = 0.05
            forward_backward = 0 if abs(forward_backward) < deadband else forward_backward
            left_right = 0 if abs(left_right) < deadband else left_right
            yaw = 0 if abs(yaw) < deadband else yaw
            up_down = 0 if abs(up_down) < deadband else up_down

            # Calculate motor speeds from joystick inputs
            input[0] = round(clamp(-forward_backward + left_right + up_down - pitch + yaw + roll), 3)
            input[1] = round(clamp(-forward_backward - left_right + up_down - pitch - yaw - roll), 3)
            input[2] = round(clamp(-forward_backward + left_right - up_down + pitch + yaw - roll), 3)
            input[3] = round(clamp(-forward_backward - left_right - up_down + pitch - yaw + roll), 3)
            input[4] = round(clamp(forward_backward + left_right + up_down + pitch - yaw + roll), 3)
            input[5] = round(clamp(forward_backward - left_right + up_down + pitch + yaw - roll), 3)
            input[6] = round(clamp(forward_backward + left_right - up_down - pitch - yaw - roll), 3)
            input[7] = round(clamp(forward_backward - left_right - up_down - pitch + yaw + roll), 3)
            print(input)

            # Send input data to the server
            data = pickle.dumps(input)
            s.send(data)
            time.sleep(0.02)

except KeyboardInterrupt:
    # Handle cleanup on exit
    print("\nCtrl + C pressed. Cleaning Up")
    joystick.quit()
    pygame.quit()
