# train_seg.py
# Short, practical PyTorch segmentation trainer (CE+Dice), good baseline quality.
# Uses segmentation_models_pytorch (SMP) for robust, pretrained backbones.
#
# Install:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   (or cu118)
#   pip install segmentation-models-pytorch albumentations opencv-python tqdm numpy
#
# Expected dataset (VOC export already OK):
#   dataset/train/JPEGImages/*.jpg
#   dataset/train/SegmentationClass/*.png   (paletted/colored ok)
#   dataset/val/JPEGImages/*.jpg
#   dataset/val/SegmentationClass/*.png
#   dataset/*/labelmap   (optional but recommended)
#
# You must set LABELS below to match CVAT labelmap order/colors.
#
# Run:
#   python train_seg.py

import os, glob
import time

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
import segmentation_models_pytorch as smp

from test_seg import load_model
# -------------------- CONFIG --------------------
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
ROOT = r".\dataset\SegmentationMask_1_1\ds_AS" # contains train/, val/
IMG_SIZE = (512, 288)                 # (W,H)
BATCH = 8
#EPOCHS = 20
LR = 3e-4
WD = 1e-2
#WORKDIR = "./runs/run_3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# IMPORTANT: map colors -> class index.
# OpenCV reads PNG as BGR.
# Example below; replace with your real labelmap colors.
LABELS = [
    ("background", (0, 0, 0)),        # idx 0
    ("drivable",   (114, 183, 71)),      # idx 1
    #("lane_mark",  (0, 255, 255)),    # idx 2 (yellow in BGR)
]
NUM_CLASSES = len(LABELS)


# -------------------- DATA --------------------
def list_pairs(subset: str):
    img_dir = os.path.join(ROOT, "JPEGImages", subset)
    msk_dir = os.path.join(ROOT, "SegmentationClass", subset)
    imgs = sorted(sum([glob.glob(os.path.join(img_dir, f"*{e}")) for e in [".jpg", ".jpeg", ".png"]], []))
    if not imgs:
        raise RuntimeError(f"No images in {img_dir}")
    pairs = []
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = os.path.join(msk_dir, stem + ".png")
        if not os.path.exists(m):
            raise RuntimeError(f"Missing mask for {p}: {m}")
        pairs.append((p, m))
    return pairs


COLOR2IDX = {tuple(bgr): i for i, (_, bgr) in enumerate(LABELS)}

def mask_to_index(mask_bgr: np.ndarray) -> np.ndarray:
    # mask_bgr: HxWx3
    out = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
    for bgr, idx in COLOR2IDX.items():
        m = np.all(mask_bgr == np.array(bgr, dtype=np.uint8), axis=2)
        out[m] = idx
    return out


class SegDS(Dataset):
    def __init__(self, pairs, aug=None):
        self.pairs = pairs
        self.aug = aug

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"bad img: {ip}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mp, cv2.IMREAD_COLOR)  # paletted -> BGR ok
        if mask is None:
            raise RuntimeError(f"bad mask: {mp}")

        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        mask_idx = mask_to_index(mask)

        if self.aug:
            out = self.aug(image=img, mask=mask_idx)
            img, mask_idx = out["image"], out["mask"]

        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)  # CHW
        mask_idx = torch.from_numpy(mask_idx.astype(np.int64))                   # HW
        return img, mask_idx


train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.Affine(translate_percent=0.02, scale=(0.95, 1.05), rotate=(-2, 2), fit_output=False, p=0.5,),
    #A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=2,
    #                  border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
])

val_aug = None


# -------------------- MODEL / LOSS / METRIC --------------------
class CEDice(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    def forward(self, logits, y):
        return self.ce(logits, y) + self.dice(logits, y)

@torch.no_grad()
def miou(logits, y, ncls):
    p = torch.argmax(logits, dim=1)
    ious = []
    for c in range(ncls):
        pc = (p == c); yc = (y == c)
        inter = (pc & yc).sum().item()
        union = (pc | yc).sum().item()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


# -------------------- TRAIN --------------------
def main(ckpt_path, WORKDIR, EPOCHS):
    os.makedirs(WORKDIR, exist_ok=True)

    tr = SegDS(list_pairs("train"), aug=train_aug)
    va = SegDS(list_pairs("val"), aug=val_aug)

    trl = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val = DataLoader(va, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    '''
    model = smp.DeepLabV3Plus(
        encoder_name="resnet18", #"mobilenet_v3_large",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(DEVICE)
    '''

    model, _ = load_model(ckpt_path='./runs/run_2/best.pt', num_classes=2).to(DEVICE)

    loss_fn = CEDice()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best = -1.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        tl, tm = 0.0, 0.0
        for x, y in tqdm(trl, desc=f"train {ep}/{EPOCHS}", leave=False):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tl += loss.item()
            tm += miou(logits, y, NUM_CLASSES)

        model.eval()
        vl, vm = 0.0, 0.0
        for x, y in tqdm(val, desc=f"val {ep}/{EPOCHS}", leave=False):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            logits = model(x)
            vl += loss_fn(logits, y).item()
            vm += miou(logits, y, NUM_CLASSES)

        tl /= max(1, len(trl)); tm /= max(1, len(trl))
        vl /= max(1, len(val)); vm /= max(1, len(val))
        print(f"ep {ep:02d} | train loss {tl:.4f} miou {tm:.4f} | val loss {vl:.4f} miou {vm:.4f}")

        torch.save({"model": model.state_dict(), "labels": LABELS}, os.path.join(WORKDIR, "last.pt"))
        if vm > best:
            best = vm
            torch.save({"model": model.state_dict(), "labels": LABELS}, os.path.join(WORKDIR, "best.pt"))
            print(f"  -> best updated: {best:.4f}")

    print("done. best miou:", best)


if __name__ == "__main__":
    main(ckpt_path='./runs/run_2/last.pt', WORKDIR='./runs/run_3', EPOCHS=20)
    print("done for 25 epochs")
    time.sleep(5)

    main(ckpt_path='./runs/run_3/last.pt', WORKDIR='./runs/run_4', EPOCHS=25)
    print("done for 50 epochs")
    time.sleep(5)

    main(ckpt_path='./runs/run_4/last.pt', WORKDIR='./runs/run_5', EPOCHS=50)
    print("done for 100 epochs")
    time.sleep(5)

    print("Finished")

