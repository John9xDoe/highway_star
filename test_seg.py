import os
import sys
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

IMG_SIZE = (512, 288)  # must match training (W,H)

def load_model(ckpt_path: str, num_classes: int):
    # must match training architecture/encoder
    model = smp.DeepLabV3Plus(
        encoder_name="resnet18",      # <-- поставь тот, который реально учил
        encoder_weights=None,         # веса уже в чекпоинте
        in_channels=3,
        classes=num_classes,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt.get("labels", None)

def preprocess(img_bgr: np.ndarray):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return x, img_rgb

@torch.no_grad()
def predict(model, x, device):
    x = x.to(device)
    logits = model(x)            # [1,C,H,W]
    p = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)  # [H,W]
    return p

def colorize(pred: np.ndarray, labels):
    # labels: [("name",(b,g,r)), ...] from checkpoint, or fallback
    if labels is None:
        # fallback colors for 2 classes
        palette = {
            0: (0, 0, 0),
            1: (114, 183, 71),
            #2: (0, 255, 255),
        }
    else:
        palette = {i: tuple(bgr) for i, (_, bgr) in enumerate(labels)}  # BGR
    vis = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for k, bgr in palette.items():
        vis[pred == k] = bgr
    return vis

def main():
    img_path = 'test_photo_0.jpg'
    ckpt_path = './runs/run_3/best.pt'
    out_mask = 'out_mask.png'
    out_overlay = 'out_overlay.png'

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load checkpoint to know num_classes
    tmp = torch.load(ckpt_path, map_location="cpu")
    labels = tmp.get("labels", None)
    if labels is not None:
        num_classes = len(labels)
    else:
        # if you trained only drivable+bg, set 2; if +lane_mark set 3
        num_classes = 2

    model, labels = load_model(ckpt_path, num_classes)
    model.to(device)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read {img_path}")

    x, img_rgb = preprocess(img)
    pred = predict(model, x, device)

    u, c = np.unique(pred, return_counts=True)
    print("unique:", dict(zip(u.tolist(), c.tolist())))

    # save raw index mask (0..C-1)
    cv2.imwrite(out_mask, pred * 255)

    # optional overlay
    if out_overlay:
        pred_col = colorize(pred, labels)            # BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.7, pred_col, 0.3, 0)
        cv2.imwrite(out_overlay, overlay)

    print("saved:", out_mask, ("and "+out_overlay if out_overlay else ""))

if __name__ == "__main__":
    main()
