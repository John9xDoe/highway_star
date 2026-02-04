import time
import cv2
import dxcam
import numpy as np
from datetime import datetime

from numpy.ma.extras import median


class SteerController:
    def __init__(self):
        self.camera = dxcam.create()

        self.W, self.H = 1920, 1080
        self.ROI = (0, (self.H * 2 // 3), self.W, self.H)
        self.W_ROI = self.ROI[2] - self.ROI[0]
        self.H_ROI = self.ROI[3] - self.ROI[1]
        self.base_shift = 0
        self.high_center = self.H_ROI // 2
        self.width_frame_center = self.W_ROI // 2
        self.width_road_center = self.width_frame_center - self.base_shift
        self.width_shift_line_lookahead = 15

        self.lane_width = 800 # 2600 # average without extremums (experimental)

        self.Kp = 0.5
        self.steer_max = 0.5
        self.rate = 1
        self.last_turn_t = 0.0

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.sigma = 0.33

    def prepare_frame(self, i=0):
        raw_frame = None

        while raw_frame is None:
            raw_frame = self.camera.grab(region=self.ROI)

        prep_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2HSV)
        #cv2.imshow('prep_frame', prep_frame)
        #prep_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        prep_frame = cv2.GaussianBlur(prep_frame, (7, 7), 0)


        #prep_frame = np.clip(self.clahe.apply(prep_frame), 0, 255).astype(np.uint8)

        #lower = np.array([85, 60, 120], np.uint8)
        #upper = np.array([105, 180, 255], np.uint8)

        lower_y = np.array([18, 80, 120], np.uint8)
        upper_y = np.array([35, 255, 255], np.uint8)

        #lower_w = np.array([0, 0, 180], np.uint8)
        #upper_w = np.array([179, 60, 255], np.uint8)

        lower_w = np.array([0, 0, 200], np.uint8)
        upper_w = np.array([179, 40, 255], np.uint8)

        mask_w = cv2.inRange(prep_frame, lower_w, upper_w) # mask
        mask_y = cv2.inRange(prep_frame, lower_y, upper_y) # mask

        prep_frame = cv2.bitwise_or(mask_y, mask_w)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(prep_frame, cv2.MORPH_OPEN, kernel, iterations=1)
        prep_frame = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        #h_coef = 0.326
        #prep_frame, H = self.make_bird_eye(prep_frame, h_coef, 1 - h_coef)

        cv2.imwrite(f'./images/e_y/run_7/{i}_mask.png', prep_frame)

        return prep_frame, raw_frame

    def make_bird_eye(self, frame, left_percent, right_percent):
        tl, tr = [int(self.W_ROI * left_percent), int(self.H_ROI * 0)], [int(self.W_ROI * right_percent), int(self.H_ROI * 0)]
        bl, br = [int(self.W_ROI * 0), int(self.H_ROI - 1)], [int(self.W_ROI - 1), int(self.H_ROI - 1)]


        src = np.array([tl, tr, br, bl], np.float32)

        dst = np.array([
            [0, 0],  # TL
            [self.W_ROI - 1, 0],  # TR
            [self.W_ROI - 1, self.H_ROI - 1],  # BR
            [0, self.H_ROI - 1],  # BL
        ], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src, dst)  # 3x3 homography (dst_p = H * src_p)

        bev = cv2.warpPerspective(
            frame, H, (self.W_ROI, self.H_ROI),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return bev, H


    def find_lane_borders_v0(self, vis=True):
        frame, raw_frame = self.prepare_frame()

        lookahead_area = frame[self.high_center - 5: self.high_center + 5, :]
        lookahead = np.mean(lookahead_area, axis=0)
        v_channel = lookahead[:, 2]
        grad = np.abs(np.diff(v_channel))
        left_part, right_part = grad[:int(0.5 * 1920)], grad[int(0.5 * 1920):]
        right_border, left_border = np.argmax(right_part), np.argmax(left_part)

        return frame, left_border, right_border

    def find_lane_borders_v1(self, vis=True):
        frame, raw_frame = self.prepare_frame()

        lookahead_area = frame[self.high_center - 5: self.high_center + 5, :]
        lookahead = np.mean(lookahead_area, axis=0)
        v_channel = lookahead[:, 2]
        grad = np.abs(np.diff(v_channel))
        batch = 320
        peaks = []
        for i in range(self.W // batch):
            peak = np.argsort(grad[i * batch:(i + 1) * batch])[::-1][0] + 240 * i
            peaks.append(peak)

            if vis:
                cv2.line(frame, (peak, 0 * self.high_center), (peak, 2 * self.high_center), (0, 0, 255), 2)

        err_lane_borders = float('inf')
        base_meaning = 1100
        left_boarder, right_border = None, None
        peaks.sort()

        for i in range(len(peaks)):
            for j in range(i + 1, len(peaks)):
                if abs(base_meaning - abs(peaks[i] - peaks[j])) < err_lane_borders:
                    err_lane_borders = base_meaning - abs(peaks[i] - peaks[j])
                    left_boarder, right_border = peaks[i], peaks[j]

        if vis:
            cv2.line(frame, (0, self.high_center), (self.W, self.high_center), (0, 0, 0), 1)
            cv2.line(frame, (left_boarder, 0 * self.high_center), (left_boarder, 2 * self.high_center), (0, 255, 0),2)
            cv2.line(frame, (right_border, 0 * self.high_center), (right_border, 2 * self.high_center), (0, 255, 0), 2)
            cv2.imshow('frame', frame)
            cv2.waitKey(0)

        return frame, right_border + self.W // 2, left_boarder

        if vis:
            print(peaks)
            cv2.imshow('frame', frame)
            cv2.waitKey(0)

    def find_lane_borders_v2(self, vis=True, i=0):
        frame, raw_frame = self.prepare_frame(i=i)

        v = median(frame)
        t_low = max(0, (1 - self.sigma) * v)
        t_high = min(255, (1 + self.sigma) * v)

        edges = cv2.Canny(frame, t_low, t_high, apertureSize=3, L2gradient=True)

        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        segments = cv2.HoughLinesP(closing, rho=1, theta=np.pi/180, threshold=50, minLineLength=240, maxLineGap=20)

        if segments is None:
            return None, None, None

        left_sum_aw, right_sum_aw = 0, 0
        left_sum_bw, right_sum_bw = 0, 0
        left_sum_w, right_sum_w = 0, 0

        eps = 10
        for x1, y1, x2, y2 in segments.reshape(-1, 4):
            if abs(x2 - x1) < eps:
                continue

            cv2.line(raw_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
            L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if a < 0:
                left_sum_aw += a * L
                left_sum_bw += b * L
                left_sum_w += L
            elif a > 0:
                right_sum_aw += a * L
                right_sum_bw += b * L
                right_sum_w += L


        if left_sum_w == 0 and right_sum_w == 0:
            return None, None, None
        elif left_sum_w == 0:
            ar_avg = right_sum_aw / right_sum_w
            br_avg = right_sum_bw / right_sum_w

            r_b = ((-int(br_avg / ar_avg), 0), (int((self.H - br_avg) / ar_avg), self.H))

            if vis:
                cv2.line(raw_frame, r_b[0], r_b[1], (0, 255, 0), 2)
                cv2.line(raw_frame, (r_b[0][0] - self.lane_width, r_b[0][1]), (r_b[1][0] - self.lane_width, r_b[1][1]), (0, 255, 0), 2)

            return raw_frame, r_b[0][0] - self.lane_width, r_b

        elif right_sum_w == 0:
            al_avg = left_sum_aw / left_sum_w
            bl_avg = left_sum_bw / left_sum_w

            l_b = ((-int(bl_avg / al_avg), 0), (int((self.H - bl_avg) / al_avg),self.H))

            if vis:
                cv2.line(raw_frame, l_b[0], l_b[1], (0, 255, 0), 2)
                cv2.line(raw_frame, (l_b[0][0] + self.lane_width, l_b[0][1]), (l_b[1][0] + self.lane_width, l_b[1][1]), (0, 255, 0), 2)

            return raw_frame, l_b, l_b[0][0] + self.lane_width

        else:
            al_avg, ar_avg = left_sum_aw / left_sum_w, right_sum_aw / right_sum_w
            bl_avg, br_avg = left_sum_bw / left_sum_w, right_sum_bw / right_sum_w

            l_b = ((-int(bl_avg / al_avg), 0), (int((self.H - bl_avg) / al_avg),self.H))
            r_b = ((-int(br_avg / ar_avg), 0), (int((self.H - br_avg) / ar_avg),self.H))

            if vis:
                cv2.line(raw_frame, l_b[0], l_b[1], (0, 255, 0), 2)
                cv2.line(raw_frame, r_b[0], r_b[1], (0, 255, 0), 2)

                #cv2.imshow('frame', raw_frame)
                #cv2.waitKey(0)

            cv2.imwrite(f'./images/e_y/run_7/{i}_raw_frame.png', raw_frame)
            cv2.imwrite(f'./images/e_y/run_7/{i}_prep_frame.png', frame)
            #cv2.imshow('frame', raw_frame)
            #cv2.waitKey(0)

            return raw_frame, int(((self.H // 2 - bl_avg) / al_avg)),  int(((self.H // 2 - br_avg) / ar_avg))

    def find_derivation_err(self, vis=True, save=True):
        #time.sleep(5)
        #frame = self._prepare_frame()

        frame, left_border, right_border = self.find_lane_borders_v2(vis=vis)

        if frame is None or left_border is None or right_border is None:
            return None

        print(abs(right_border - left_border))

        x_mid = (right_border + left_border) // 2
        e_y = x_mid - self.width_frame_center

        e = e_y / (self.W // 2)

        if vis:
            cv2.circle(frame, (x_mid, self.high_center), 3, (255, 0, 0), -1)

            cv2.line(frame, (right_border, 0 * self.high_center), (right_border, 2 * self.high_center), (255, 0, 0), 2)
            cv2.line(frame, (left_border, 0 * self.high_center), (left_border, 2 * self.high_center), (255, 0, 0), 2)

            cv2.circle(frame, (self.width_frame_center, self.high_center), 3, (0, 255, 0), -1)
            cv2.line(frame, (self.width_road_center, self.high_center * 0), (self.width_road_center, self.high_center * 2), (0, 0, 255), 1)

            cv2.line(frame, (0, self.high_center), (self.W, self.high_center), (0, 0, 0), 1)
            #cv2.line(frame, (width_road_center + width_shift_line_lookahead, high_center), (width_road_center + width_shift_line_lookahead + 500, high_center), (0, 0, 0), 1)

            #cv2.imshow('frame', frame)
            #cv2.waitKey(0)

        if save:
            cv2.imwrite(f'./images/e_y/run_6/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{abs(e_y)}.png', frame)

        return e

    def calculate_steering_wheel_angle(self, now):

        if now - self.last_turn_t > self.rate:
            return None, None

        e_norm = self.find_derivation_err(vis=True, save=True)

        if e_norm is None:
            return None, None

        steer_raw = self.Kp * e_norm
        steer_raw = np.clip(steer_raw, -self.steer_max, self.steer_max)

        return steer_raw, e_norm

if __name__ == "__main__":
    #print(np.argsort([1, 2, 5, 3, 4, 7, 6])[::-1][:4])

    from test_vjoy import VJoyController

    vjoy_controller = VJoyController()
    vjoy_controller.set_controls(0, 1, 0)

    #print("Starting in 5 seconds...")
    #time.sleep(5)
    time.sleep(3)

    steer_controller = SteerController()

    i = 112

    while True:
        steer_controller.find_lane_borders_v2(vis=True, i=i)
        i += 1
        time.sleep(5)


    #test_frame = cv2.imread('test_prep_photo.png')
    #cv2.imshow('test', test_frame)
    #cv2.waitKey(0)

    #for h_coef in range(320, 330): # 0.326
    #    h_frame, _ = steer_controller.make_bird_eye(test_frame, h_coef / 1000, 1 - h_coef / 1000)
    #    cv2.imshow(f'{h_coef/100}', h_frame)
    #    cv2.waitKey(0)

    #h_frame, _ = steer_controller.make_bird_eye(test_frame, 0.326, 0.674)
    #cv2.imshow('test_prep_photo', h_frame)
    #cv2.waitKey(0)

    #steer_controller.find_lane_borders_v2()

    #steering, e_norm = steer_controller.calculate_steering_wheel_angle(0.0)
    #print(steering, e_norm, e_norm * (steer_controller.W // 2))
    #steer_controller.find_derivation_err(save=True)

    #prep, raw = steer_controller.prepare_frame()
    #cv2.imshow("Prepare Frame", prep)
    #cv2.waitKey(0)

    #steer_controller.find_derivation_err(vis=True)

    '''
    i = 0
    while True:
        e = steer_controller.find_derivation_err()

        if i % 1 == 0:
            print(f"e-{i}={e}")

        i += 1

        time.sleep(10)
    '''

    #steer_controller.find_lane_borders_v2()
    '''
    i = 0
    start_time = time.time()
    while True:
        steer, e = steer_controller.calculate_steering_wheel_angle(time.time() - start_time)

        if steer is None or e is None:
            #print('НИ-ХУ-Я!!!')
            continue

        vjoy_controller.set_controls(steer,1,0)
        
        #if i % 1 == 0:
            #print(f"{i} it: steer = {steer} | e = {e * (steer_controller.W // 2)}")

        steer_controller.last_turn_t = time.time()
        time.sleep(1)

        i += 1
    #'''