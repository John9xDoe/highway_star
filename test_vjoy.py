import pyvjoy
import time

MAX_AXIS = 0x8000  # 32768

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

class VJoyController:
    def __init__(self, device_id=1):
        self.joystick = pyvjoy.VJoyDevice(device_id)
        print(f"vJoy device {device_id} initialized")

    def set_controls(self, steering, throttle, brake):
        # steering: -1..1 -> 0..32768
        s = clamp(steering, -1.0, 1.0)
        self.joystick.data.wAxisX = int((s + 1.0) * (MAX_AXIS / 2))

        # throttle/brake: 0..1 -> 0..32768
        t = clamp(throttle, 0.0, 1.0)
        b = clamp(brake, 0.0, 1.0)
        self.joystick.data.wAxisY = int(t * MAX_AXIS)
        self.joystick.data.wAxisZ = int(b * MAX_AXIS)

        self.joystick.update()

    def reset(self):
        self.set_controls(0, 0, 0)

    def tap_button(self, button_id: int, hold_s: float = 0.1):
        bit = 1 << (button_id - 1)

        self.joystick.data.lButtons |= bit
        self.joystick.update()

        time.sleep(hold_s)

        self.joystick.data.lButtons &= ~bit
        self.joystick.update()

def test_vjoy():
    controller = VJoyController()
    try:
        time.sleep(10)
        '''
        print("Steering sweep")
        for steering in [-1, -0.5, 0, 0.5, 1, 0]:
            controller.set_controls(steering, 0, 0)
            time.sleep(0.5)

        print("Throttle sweep")
        for throttle in [0, 0.25, 0.5, 0.75, 1.0, 0]:
            controller.set_controls(0, throttle, 0)
            time.sleep(0.5)

        print("Brake sweep")
        for brake in [0, 0.5, 1.0, 0]:
            controller.set_controls(0, 0, brake)
            time.sleep(0.5)
        '''
        #time.sleep(10)
        #controller.set_controls(0, 0, 1)
        print("Button test")
        time.sleep(5)
        controller.tap_button(button_id=1, hold_s=0.5)
        time.sleep(5)
        controller.tap_button(button_id=2, hold_s=0.5)
        time.sleep(1000)
    finally:
        controller.reset()
        print("done")

if __name__ == "__main__":
    test_vjoy()
