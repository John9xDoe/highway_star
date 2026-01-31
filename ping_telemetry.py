import time
import truck_telemetry
from test_vjoy import VJoyController

controller = VJoyController()

controller.set_controls(0, 0, 0)

time.sleep(10)

truck_telemetry.init()

while True:
    data = truck_telemetry.get_data()
    if data:
        print("speed:", data.get("speed"))
    time.sleep(0.05)  # 20 Hz
