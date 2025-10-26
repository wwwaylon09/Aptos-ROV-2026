# Activate Python environment
# source my_env/bin/activate

import socket
import pickle
import board
import adafruit_pca9685
import adafruit_mpu6050
import math

# Initialize socket server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 5000))
s.listen(1)
print("Server is now running")
connection, client_address = s.accept()

# Initialize hardware components
i2c = board.I2C()
pca = adafruit_pca9685.PCA9685(i2c)
mpu = adafruit_mpu6050.MPU6050(i2c)

pca.frequency = 50

# Motor and Servo configurations
# Change these based on which port each motor is pluged in to
motor_1 = 4
motor_2 = 6
motor_3 = 0
motor_4 = 2
motor_5 = 5
motor_6 = 3
motor_7 = 7
motor_8 = 1

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

# Merge joystick and gyro inputs, but prioritize joystick inputs
def merge_inputs(joystick, gyro):
    return lerp(gyro, joystick, math.fabs(joystick * 2))

# Main server loop to receive and process data
while True:
    data = connection.recv(4096)

    if not data:
        break
    else:
        # Get joystick inputs from server
        inputs = pickle.loads(data)
        # Check if stabilization is enabled
        if inputs[12]:
            # Get angle inputs from gyro
            pitch, roll = calculate_orientation()
            pitch, roll = pitch / math.pi, roll / math.pi
            
            # Merge inputs
            inputs[2] = merge_inputs(inputs[2], roll)
        
        print(inputs)
        pca.channels[motor_1].duty_cycle = convert(inputs[0])
        pca.channels[motor_2].duty_cycle = convert(inputs[1])
        pca.channels[motor_3].duty_cycle = convert(inputs[2])
        pca.channels[motor_4].duty_cycle = convert(inputs[3])
        pca.channels[motor_5].duty_cycle = convert(inputs[4])
        pca.channels[motor_5].duty_cycle = convert(inputs[5])
        pca.channels[motor_5].duty_cycle = convert(inputs[6])
        pca.channels[motor_5].duty_cycle = convert(inputs[7])
        pca.channels[motor_5].duty_cycle = convert(inputs[8])
        

