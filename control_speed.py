from test_vjoy import clamp
import logging
import time

class SpeedController:
    def __init__(self, desired_speed=25):
        #self.controller = VJoyController()
        #self.controller.set_controls(0,0,0)

        self.desired_speed = desired_speed
        self.throttle = 0
        self.deadband = 0.3
        self.v_filter_prev = 0.0
        self.v_filter_prev_init = False
        self.a = 0.2

        self.Ki = 0.1
        self.Kp = 0.5
        self.Kd = 0.1 # 0.02 - 0.1, freeze Ki for search

        self.U_p = 0
        self.U_i = 0
        self.U_d = 0
        self.U = 0

        self.v_prev = None
        self.dv_prev = None
        self.t_prev = None
        self.dt = 0.0

        self.T = 0.3

        logging.info("PID start")

    def _update_dt(self):
        now = time.monotonic()
        if self.t_prev is None:
            self.t_prev = now
            return 1e-3
        dt = now - self.t_prev
        self.t_prev = now

        self.dt = clamp(dt, 1e-3, 0.2)

        #return dt

    def _update_filter_coef(self):
        self.a = self.dt / (self.dt + self.T)


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
        self._update_dt()
        self._update_filter_coef()


        if self.v_prev is None:
            self.v_prev = v

        dv = (v - self.v_prev) / self.dt
        if self.dv_prev is None:
            self.dv_prev = self.U_d

        dv_filter = self.a * self.U_d + (1 - self.a) * self.dv_prev

        eI = self._calculate_error(v=v)


        if abs(eI) < self.deadband:
            eP = 0
        else:
            eP = eI

        self.U_p = self.Kp * eP
        self.U_d = -self.Kd * dv_filter

        self.U = self.U_p + self.U_i + self.U_d
        self.throttle = clamp(self.U, 0, 1)

        if (self.U == self.throttle) or (self.U == 1.0 and eP < 1.0) or (self.U == 1.0 and eP > 0.0):
            self.U_i += self.Ki * eI * self.dt

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