from test_vjoy import clamp
import logging
import time

class SpeedController:
    def __init__(self, desired_speed=25):
        #self.controller = VJoyController()
        #self.controller.set_controls(0,0,0)

        self.desired_speed = desired_speed
        self.throttle = 0
        self.scale_error = 0.05
        self.deadband = 0.3
        self.v_filter_prev = 0.0
        self.v_filter_prev_init = False
        self.a = 0.2

        self.t_prev = None

        logging.info("PID start")

    def _get_dt(self):
        now = time.time()
        if self.t_prev is None:
            self.t_prev = now
            return 0
        dt = now - self.t_prev
        self.t_prev = now

        '''
        if dt <= 0.0:
            return None
        if dt > 0.5:
            dt = 0.5
        '''

        return dt


    def _calculate_filter(self, v):
        if not self.v_filter_prev_init:
            self.v_filter_prev = v
            self.v_filter_prev_init = True

        v_filter = self.a * v + (1 - self.a) * self.v_filter_prev
        self.v_filter_prev = v_filter
        return v_filter

    def _calculate_error(self, v):
        return self.desired_speed - self._calculate_filter(v=v)

    def update_speed(self, v, vis):
        error = self._calculate_error(v=v)

        if abs(error) < self.deadband:
            error = 0

        self.throttle += self.scale_error * error
        self.throttle = clamp(self.throttle, 0, 1)

        if vis:
            print(f"error={error:.3f} | throttle={self.throttle:.3f} | dt = {self._get_dt():.3f}")

        #self.controller.set_controls(0, self.throttle,0)
        return self.throttle

def test_pid():
    from ping_telemetry import Telemetry
    telemetry_pinger = Telemetry()
    controller = SpeedController()

    print("PID start")
    time.sleep(5)

    while True:
        v, _, _ = telemetry_pinger.ping_data(vis=False)
        controller.update_speed(v=v, vis=True)
        time.sleep(0.05) # 20 Hz


if __name__ == "__main__":
    test_pid()