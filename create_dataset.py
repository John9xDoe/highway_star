import cv2

import dxcam
from datetime import datetime
import time
from test_vjoy import VJoyController

import os
import shutil
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

camera = dxcam.create()
vjoy_controller = VJoyController()

def start_recording():
    period = 5
    i = 13

    vjoy_controller.set_controls(0,1,0)

    time.sleep(period)

    while True:
        frame = camera.grab()
        cv2.imwrite(f"./images/raw/unsorted/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{i}.png", frame)
        time.sleep(period)

        i += 1

SRC_DIR = r"./images/raw/unsorted"
DST_DIR = r"./images/raw/unique/val"
SSIM_THRESHOLD = 0.99 # 0.95 - 3, 0.8 - 11, 0.7 - 76
DOWNSCALE_W = 320

ROI_Y0_FRAC = 0.5

def load_preprocess(path):
    os.makedirs(DST_DIR, exist_ok=True)

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read: {path}")
    h, w = img.shape[:2]

    y0 = int(h * ROI_Y0_FRAC)
    roi = img[y0:h, :, :]

    # downscale
    scale = DOWNSCALE_W / roi.shape[1]
    new_h = int(roi.shape[0] * scale)
    roi_small = cv2.resize(roi, (DOWNSCALE_W, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(roi_small, cv2.COLOR_BGR2GRAY)
    return gray

if __name__ == '__main__':
    '''
    files = sorted([f for f in os.listdir(SRC_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not files:
        raise RuntimeError("No images found")

    for i, f in enumerate(tqdm(files)):
        if i % 5 == 0:
            shutil.copy2(os.path.join(SRC_DIR, f), os.path.join(DST_DIR, f))
    '''
    '''
    kept = 0
    dropped = 0

    last_path = os.path.join(SRC_DIR, files[0])
    last_img = load_preprocess(last_path)
    shutil.copy2(last_path, os.path.join(DST_DIR, files[0]))
    kept += 1

    for f in tqdm(files[1:]):
        path = os.path.join(SRC_DIR, f)
        cur_img = load_preprocess(path)

        score = ssim(last_img, cur_img, data_range=cur_img.max() - cur_img.min())
        if score >= SSIM_THRESHOLD:
            dropped += 1
            continue

        shutil.copy2(path, os.path.join(DST_DIR, f))
        last_img = cur_img
        kept += 1

    print(f"Kept: {kept}, Dropped: {dropped}, Total: {len(files)}")
    '''

    #frame = cv2.imread(os.path.join(DST_DIR, '2026-02-10_11-49-00_0.png'))
    #cv2.imshow(f"frame", frame)
    #cv2.imshow(f"cropped_frame", frame[:frame.shape[0] - 30, :, :])
    #cv2.waitKey(0)


    for f in tqdm(os.listdir(DST_DIR)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(os.path.join(DST_DIR, f))
            #img_fixed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_fixed = img[img.shape[0] // 2 :, :, :]
            cv2.imwrite(os.path.join(DST_DIR, f), img_fixed)
