import cv2
import dxcam
from datetime import datetime
import time
from test_vjoy import VJoyController

camera = dxcam.create()
vjoy_controller = VJoyController()

period = 5
i = 13

vjoy_controller.set_controls(0,1,0)

time.sleep(period)

while True:
    frame = camera.grab()
    cv2.imwrite(f"./images/raw/unsorted/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{i}.png", frame)
    time.sleep(period)

    i += 1