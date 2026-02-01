from control_gearbox import GearboxController
from control_speed import SpeedController
from test_vjoy import VJoyController
from ping_telemetry import Telemetry
import time

if __name__ == "__main__":

    vjoy_controller = VJoyController()
    speed_controller = SpeedController()
    gearbox_controller = GearboxController()
    telemetry_pinger = Telemetry()

    vjoy_controller.reset()

    i = 0

    print("Starting in 10 seconds...")
    time.sleep(10)

    start_time = time.time()
    while True:
        vis = True if i % 100 == 0 else False
        speed, rpm, rpm_max = telemetry_pinger.ping_data(vis=vis)
        throttle = speed_controller.update_speed(v=speed, vis=vis)
        gearbox_shift = gearbox_controller.update_gear(rpm=rpm, rpm_max=rpm_max, throttle=throttle, now=time.time() - start_time)

        # Later -> add-on controller:
        if gearbox_shift == 'up':
            vjoy_controller.tap_button(1, 0.1)
            gearbox_controller.last_shift_t = time.time() - start_time
        elif gearbox_shift == 'down':
            vjoy_controller.tap_button(2, 0.1)
            gearbox_controller.last_shift_t = time.time() - start_time

        if vis:
            print(gearbox_shift)
            print()

        vjoy_controller.set_controls(0, throttle, 0)

        i += 1

        time.sleep(0.05) # 20 Hz