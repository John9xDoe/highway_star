import time
import cv2
import dxcam
import numpy as np
from datetime import datetime

class SteerController:
    def __init__(self):
        self.camera = dxcam.create()

        self.W, self.H = 1920, 1080
        self.base_shift = 0
        self.high_center = 270
        self.width_frame_center = 960
        self.width_road_center = self.width_frame_center - self.base_shift
        self.width_shift_line_lookahead = 15

        self.Kp = 0.5
        self.steer_max = 0.5
        self.rate = 1
        self.last_turn_t = 0.0

    def find_derivation_err(self, vis=False, save=False):
        #time.sleep(5)
        frame = self.camera.grab(region=(0, 540, 1920, 1080))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame = cv2.GaussianBlur(frame, (5,5), 0)

        lookahead_area = frame[self.high_center - 5: self.high_center + 5, :]
        lookahead = np.mean(lookahead_area, axis=0)
        v_channel = lookahead[:, 2]
        grad = np.abs(np.diff(v_channel))
        left_part, right_part = grad[:int(0.5 * 1920)], grad[int(0.5 * 1920):]
        right_border, left_border = np.argmax(right_part), np.argmax(left_part)

        x_mid = (right_border + self.W // 2 + left_border) // 2
        e_y = x_mid - self.width_frame_center

        e = e_y / (self.W // 2)

        if vis:
            cv2.circle(frame, (x_mid, self.high_center), 3, (255, 0, 0), -1)

            cv2.line(frame, (right_border + self.W // 2, 0 * self.high_center), (right_border  + self.W // 2, 2 * self.high_center), (255, 0, 0), 2)
            cv2.line(frame, (left_border, 0 * self.high_center), (left_border, 2 * self.high_center), (255, 0, 0), 2)

            cv2.circle(frame, (self.width_frame_center, self.high_center), 3, (0, 255, 0), -1)
            cv2.line(frame, (self.width_road_center, self.high_center * 0), (self.width_road_center, self.high_center * 2), (0, 0, 255), 1)

            cv2.line(frame, (0, self.high_center), (self.W, self.high_center), (0, 0, 0), 1)
            #cv2.line(frame, (width_road_center + width_shift_line_lookahead, high_center), (width_road_center + width_shift_line_lookahead + 500, high_center), (0, 0, 0), 1)

            cv2.imshow('frame', frame)
            cv2.waitKey(1)

        if save:
            cv2.imwrite(f'./images/e_y/run_3/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{abs(e_y)}.png', frame)

        return e

    def calculate_steering_wheel_angle(self, now):

        if now - self.last_turn_t > self.rate:
            return None

        e_norm = self.find_derivation_err(vis=False, save=False)

        steer_raw = self.Kp * e_norm
        steer_raw = np.clip(steer_raw, -self.steer_max, self.steer_max)

        return steer_raw, e_norm

if __name__ == "__main__":

    from test_vjoy import VJoyController
    vjoy_controller = VJoyController()

    steer_controller = SteerController()
    #steer_controller.find_derivation_err(save=True)
    i = 0

    print("Starting in 10 seconds...")
    time.sleep(10)

    start_time = time.time()
    while True:
        steer, e = steer_controller.calculate_steering_wheel_angle(time.time() - start_time)
        vjoy_controller.set_controls(steer,1,0)

        if i % 10 == 0:
            print(f"{i} it: steer = {steer} | e = {e * (steer_controller.W // 2)}")

        steer_controller.last_turn_t = time.time()
        time.sleep(1)

        i += 1