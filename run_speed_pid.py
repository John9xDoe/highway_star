from test_vjoy import VJoyController, clamp
from ping_telemetry import ping_speed
import logging
import time

class PID:
    def __init__(self, desired_speed=25):
        self.controller = VJoyController()
        self.controller.set_controls(0,0,0)
        self.desired_speed = desired_speed
        self.throttle = 0
        self.scale_error = 0.05
        self.deadband = 0.3
        self.v_filter_prev = 0.0
        self.v_filter_prev_init = False
        self.a = 0.2
        logging.info("PID start")

    def calculate_filter(self):
        v = ping_speed(vis=True)
        if not self.v_filter_prev_init:
            self.v_filter_prev = v
            self.v_filter_prev_init = True

        v_filter = self.a * v + (1 - self.a) * self.v_filter_prev
        self.v_filter_prev = v_filter
        return v_filter

    def calculate_error(self):
        return self.desired_speed - self.calculate_filter()

    def update_speed(self):
        error = self.calculate_error()

        if abs(error) < self.deadband:
            error = 0

        self.throttle += self.scale_error * error
        self.throttle = clamp(self.throttle, 0, 1)
        print(f"error={error:.3f} throttle={self.throttle:.3f}")

        self.controller.set_controls(0, self.throttle,0)

def test_pid():
    pid = PID()
    print("PID start")
    time.sleep(5)

    while True:
        pid.update_speed()
        time.sleep(0.05)

if __name__ == "__main__":
    test_pid()