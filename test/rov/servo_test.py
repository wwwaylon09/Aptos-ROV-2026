from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16)

channel = 0 # Change this to the channel you want to control (0-15)

while True:

    kit.servo[channel].angle = 90  # Set the servo to 90 degrees
    time.sleep(2)
    kit.servo[channel].angle = 0   # Set the servo to 0 degrees
    time.sleep(2)