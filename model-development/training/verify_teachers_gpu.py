"""Runs both teachers on the GPU on our ground-truth clips (core/testdata/gt) and reports what they see.

- mosaic detection teacher (Lada v4 accurate): should find the mosaic in the mosaiced frames, nothing in the clean ones
- restoration teacher (Lada generic v1.2): PSNR in the mosaic area + speed, fp32 vs fp16
"""
import glob
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dmtrain.basicvsrpp import load_lada_teacher  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # the repository
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")  # upstream weights, not in Git (../README.md)
GT = os.path.join(ROOT, "core", "testdata", "gt")


def frames(clip, kind, n=None):
    files = sorted(glob.glob(os.path.join(GT, clip, kind, "*.png")))
    return [np.array(Image.open(f).convert("RGB")) for f in files[:n]]


def detect(model, imgs, conf=0.15):
    found, confs, areas = 0, [], []
    for img in imgs:
        r = model.predict(img[..., ::-1].copy(), imgsz=640, conf=conf, verbose=False, device=0)[0]  # Ultralytics wants BGR arrays
        if r.boxes is not None and len(r.boxes):
            found += 1
            confs.append(float(r.boxes.conf.max()))
            b = r.boxes.xyxy[r.boxes.conf.argmax()].tolist()
            areas.append(max(b[2] - b[0], b[3] - b[1]))
    return found, (np.mean(confs) if confs else 0.0), (np.median(areas) if areas else 0)


def main():
    from ultralytics import YOLO
    dev = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    det = YOLO(os.path.join(MODELS, "lada_mosaic_detection_model_v4_accurate.pt"))
    print("mosaic teacher classes:", det.names)
    for clip in ("onepiece", "mid", "small", "tiny", "shapes"):
        if not os.path.isdir(os.path.join(GT, clip)):
            continue
        mz, gt = frames(clip, "mosaic"), frames(clip, "gt")
        t0 = time.time()
        f, c, a = detect(det, mz)
        dt = (time.time() - t0) / len(mz) * 1000
        fc, _, _ = detect(det, gt)
        print(f"{clip:9s} mosaic teacher: found in {f}/{len(mz)} mosaiced frames (mean conf {c:.2f}, median box {a:.0f} px), "
              f"{fc}/{len(gt)} clean frames (false alarms) · {dt:.0f} ms/frame")

    # restoration teacher on the GPU
    teacher = load_lada_teacher(os.path.join(ROOT, "Releases", "lada_mosaic_restoration_model_generic_v1.2.pth"), device=dev)
    gt = np.stack(frames("onepiece", "gt", 30))[:, 72:328, 62:318]
    mz = np.stack(frames("onepiece", "mosaic", 30))[:, 72:328, 62:318]
    x = torch.from_numpy(mz).permute(0, 3, 1, 2).float().div(255).unsqueeze(0).to(dev)
    box = (slice(8, 248), slice(23, 233))
    for half in (False, True):
        m = teacher.half() if half else teacher.float()
        inp = x.half() if half else x
        with torch.no_grad():
            m(inp[:, :4])
            torch.cuda.synchronize()
            t0 = time.time()
            y = m(inp).float().clamp(0, 1)
            torch.cuda.synchronize()
        out = (y[0].permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
        mse = np.mean((out[:, box[0], box[1]].astype(float) - gt[:, box[0], box[1]].astype(float)) ** 2)
        print(f"restoration teacher {'fp16' if half else 'fp32'}: PSNR {10 * np.log10(255 ** 2 / mse):.2f} dB, "
              f"{(time.time() - t0) / 30 * 1000:.0f} ms/frame (30-frame clip, 256 px), GPU memory peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    print("ALL_OK")


if __name__ == "__main__":
    main()
