import time
import truck_telemetry
from test_vjoy import VJoyController

controller = VJoyController()

controller.set_controls(0, 0, 0)

time.sleep(10)

truck_telemetry.init()

def ping_speed(vis=True, convert_const=2.236936):

    data = truck_telemetry.get_data()
    if data and vis:
        print("speed:", data.get("speed") * convert_const)
    time.sleep(0.05)  # 20 Hz

    return data.get("speed") * convert_const

if __name__ == "__main__":
    while True:
        ping_speed()