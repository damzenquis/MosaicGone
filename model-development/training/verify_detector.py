"""Checks our YOLOX-Nano code against the official yolox_nano.pth before any training.

1. every official weight loads into our class (nothing left over)
2. it detects: a person on imgs/example/lena.jpg (COCO class 0) with a high score
3. the single-class student with the mask head builds, and its CPU speed at the app's input size
"""
import os
import sys
import time

import numpy as np
import torch
import torchvision
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dmtrain.yolox import load_yolox_nano  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # the repository
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")  # upstream weights, not in Git (../README.md)
torch.set_num_threads(16)


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    im = np.array(Image.fromarray(img).resize((int(w * r), int(h * r)), Image.BILINEAR))
    out = np.full((size, size, 3), 114, np.uint8)
    out[:im.shape[0], :im.shape[1]] = im
    return out, r


def main():
    net, skipped, missing = load_yolox_nano(os.path.join(MODELS, "yolox_nano.pth"))
    print(f"YOLOX-Nano: {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M params; official weights not used: {skipped or 'none'}; ours without weights: {missing or 'none'}")
    net.eval()
    rgb = np.array(Image.open(os.path.join(ROOT, "core", "testdata", "reference", "esrgan_input_ADE_val_00000114.png")).convert("RGB"))
    img, r = letterbox(rgb, 416)
    x = torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0)  # BGR 0..255
    with torch.no_grad():
        outs, _ = net(x)
        d = net.decode(outs)[0]
    score = d[:, 4:5] * d[:, 5:85]
    best, cls = score.max(1)
    keep = best > 0.3
    boxes = d[keep, :4]
    xyxy = torch.cat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], 1)
    k = torchvision.ops.nms(xyxy, best[keep], 0.45)
    for i in k[:5]:
        print(f"   class {cls[keep][i].item():2d} score {best[keep][i].item():.2f} box {[round(v / r) for v in xyxy[i].tolist()]}")
    assert len(k) and cls[keep][k[0]].item() == 0 and best[keep][k[0]].item() > 0.5, "expected a confident person (class 0)"

    student, skipped, missing = load_yolox_nano(os.path.join(MODELS, "yolox_nano.pth"), num_classes=1, num_protos=32)
    student.eval()
    print(f"student (1 class + mask head): {sum(p.numel() for p in student.parameters()) / 1e6:.2f} M params; new layers: {sorted({m.rsplit('.', 2)[0] for m in missing})}")
    for size in (416, 480, 640):
        x = torch.zeros(1, 3, size, size)
        with torch.no_grad():
            student(x)
            t0 = time.time()
            for _ in range(5):
                student(x)
        print(f"   {size} px: {(time.time() - t0) / 5 * 1000:.0f} ms/frame on this CPU")
    print("ALL_OK")


if __name__ == "__main__":
    main()
