from test_vjoy import VJoyController, clamp
from ping_telemetry import ping_speed
import logging
import time

class PID:
    def __init__(self, desired_speed=15):
        self.controller = VJoyController()
        self.controller.set_controls(0,0,0)
        self.desired_speed = desired_speed
        self.throttle = 0
        self.scale_error = 0.05
        self.deadband = 0.3
        logging.info("PID start")

    def calculate_error(self):
        return self.desired_speed - ping_speed(vis=True)

    def update_speed(self):
        error = self.calculate_error()

        if abs(error) < self.deadband:
            error = 0

        self.throttle += self.scale_error * error
        self.throttle = clamp(self.throttle, 0, 1)
        print("error=%.3f throttle=%.3f", error, self.throttle)

        self.controller.set_controls(0, self.throttle,0)

def test_pid():
    pid = PID()
    while True:
        pid.update_speed()
        time.sleep(0.05)

if __name__ == "__main__":
    test_pid()