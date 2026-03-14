import time
import board
from adafruit_lsm6ds.lsm6dsox import LSM6DSOX as LSM6DS
from adafruit_lis3mdl import LIS3MDL
from git import Repo
import cv2
from picamera2 import Picamera2

i2c = board.I2C()
accel_gyro = LSM6DS(i2c)
mag = LIS3MDL(i2c)


picam2 = Picamera2()
WIDTH = 1024
HEIGHT = 768
config = picam2.create_preview_configuration({'format': 'YUV420', 'size': (WIDTH, HEIGHT)})
picam2.configure(config)
picam2.start()

def take_grayscale_photo():
    """
    Function that takes a grayscale photo when called.
    """
    time.sleep(2)
    yuv = picam2.capture_array()
    grey = yuv[:HEIGHT, :WIDTH]