from control_gearbox import GearboxController
from control_speed import SpeedController
from control_steer import SteerController
from test_vjoy import VJoyController
from ping_telemetry import Telemetry

import time
from datetime import datetime

import cv2

if __name__ == "__main__":
    vjoy_controller = VJoyController()
    steer_controller = SteerController()
    speed_controller = SpeedController()
    gearbox_controller = GearboxController()
    telemetry_pinger = Telemetry()

    vjoy_controller.reset()

    i = 0

    prev_steer = None
    grub_per = 1
    prev_grub = 0.0

    print("Starting in 10 seconds...")
    time.sleep(10)

    start_time = time.time()
    while True:
        vis = True if i % 10 == 0 else False
        if vis: print(f"{i} it")
        speed, rpm, rpm_max = telemetry_pinger.ping_data(vis=vis)
        throttle = speed_controller.update_speed(v=speed, vis=vis)
        gearbox_shift = gearbox_controller.update_gear(rpm=rpm, rpm_max=rpm_max, throttle=throttle, now=time.time() - start_time)

        steer, e = steer_controller.calculate_steering_wheel_angle(time.time() - start_time)

        # steer -> target speed
        if steer is not None:
            s = abs(steer)
            v_target = max(steer_controller.V_MIN, steer_controller.V_MAX * (1.0 - steer_controller.K * s))
            speed_controller.desired_speed = v_target
            if vis:
                print(f"v_target={v_target} | s={s}\n")
        else:
            if prev_steer is not None:
                steer = prev_steer
            else:
                steer = 0


        # Later -> add-on controller:
        if gearbox_shift == 'up':
            vjoy_controller.tap_button(1, 0.1)
            gearbox_controller.last_shift_t = time.time() - start_time

            #print()
            #print(f"{i}: {gearbox_shift} ({speed}, {rpm}, {throttle}, {time.time() - start_time} s.)")
        elif gearbox_shift == 'down':
            vjoy_controller.tap_button(2, 0.1)
            gearbox_controller.last_shift_t = time.time() - start_time

            #print()
            #print(f"{i}: {gearbox_shift} ({speed}, {rpm}, {throttle}, {time.time() - start_time} s.)")

        vjoy_controller.set_controls(steer, throttle, 0)

        prev_steer = steer

        if time.monotonic() - prev_grub >= grub_per:
            frame = None
            while frame is None:
                frame = steer_controller.camera.grab()
            cv2.imwrite(f"./images/raw/unsorted/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{i}.png", frame)
            prev_grub = time.monotonic()

        i += 1

        time.sleep(0.05) # 20 Hz