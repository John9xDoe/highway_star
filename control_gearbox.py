class GearboxController:
    def __init__(self, coef_up=0.8, coef_down=0.4, cooldown_s=2.0):
        self.coef_up = coef_up
        self.coef_down = coef_down
        self.cooldown_s = cooldown_s
        self.last_shift_t = 0.0

    def update_gear(self, rpm, rpm_max, throttle, now):
        if rpm_max is None or rpm is None:
            return None

        if now - self.last_shift_t < self.cooldown_s:
            return None

        if rpm > self.coef_up * rpm_max and throttle > 0.2:
            return "up"
        if rpm < self.coef_down * rpm_max and throttle > 0.4:
            return "down"

        return None