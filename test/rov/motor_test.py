import board
import adafruit_pca9685
import time

i2c = board.I2C()
pca = adafruit_pca9685.PCA9685(i2c)
pca.frequency = 50

channel = 0 #Number between 0 and 15

while True:

    pca.channels[channel].duty_cycle = 5240
    time.sleep(2)
    pca.channels[channel].duty_cycle = 5500
    time.sleep(2)
