from test_vjoy import clamp
import logging
import time

class SpeedController:
    def __init__(self, desired_speed=25):
        #self.controller = VJoyController()
        #self.controller.set_controls(0,0,0)

        self.desired_speed = desired_speed
        self.throttle = 0
        self.Ki = 0.1
        self.Kp = 0.5
        self.deadband = 0.3
        self.v_filter_prev = 0.0
        self.v_filter_prev_init = False
        self.a = 0.2

        self.U_p = 0
        self.U_i = 0
        self.U = 0

        self.t_prev = None

        logging.info("PID start")

    def _get_dt(self):
        now = time.monotonic()
        if self.t_prev is None:
            self.t_prev = now
            return 1e-3
        dt = now - self.t_prev
        self.t_prev = now

        dt = clamp(dt, 1e-3, 0.2)

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
        dt = self._get_dt()
        eI = self._calculate_error(v=v)

        if abs(eI) < self.deadband:
            eP = 0
        else:
            eP = eI

        self.U_p = self.Kp * eP
        self.U = self.U_p + self.U_i
        self.throttle = clamp(self.U, 0, 1)

        if (self.U == self.throttle) or (self.U == 1.0 and eP < 1.0) or (self.U == 1.0 and eP > 0.0):
            self.U_i += self.Ki * eI * dt

        if vis:
            print(f"eP={eP:.3f} | eI={eI:.3f} throttle={self.throttle:.3f} | P={self.U_p:.3f} | I={self.U_i:.3f} | dt = {dt:.3f}")

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