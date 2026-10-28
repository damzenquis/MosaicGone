"""Saves contact sheets of generated training samples (training/data/preview_*.jpg) and times their generation."""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dmtrain import data  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "data")


def main():
    t0 = time.time()
    idx = data.load_index()
    print({k: len(v) for k, v in idx.items()}, f"index in {time.time() - t0:.1f} s")
    toy = "--toy" in sys.argv

    ds = data.RestorerClips("train", t=16, toy=toy)
    rows = []
    t0 = time.time()
    for i in range(6):
        s = ds[i]
        lq, gt, m = s["lq"].permute(0, 2, 3, 1).numpy(), s["gt"].permute(0, 2, 3, 1).numpy(), s["mask"][:, 0].numpy()
        k = [0, 8, 15]
        row = np.concatenate([np.concatenate([lq[j], gt[j], np.repeat(m[j][..., None] * 255, 3, 2).astype(np.uint8)], 1) for j in k], 1)
        cv2.putText(row, s["kind"], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        rows.append(row)
    print(f"restorer: {(time.time() - t0) / 6:.2f} s per 16-frame clip")
    cv2.imwrite(os.path.join(OUT, "preview_restorer.jpg"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])

    ds = data.DetectorImages("train", toy=toy)
    tiles = []
    t0 = time.time()
    for i in range(12):
        s = ds[i]
        img = s["img"].permute(1, 2, 0).numpy().copy()
        for j in range(s["n"]):
            x0, y0, x1, y1 = s["boxes"][j].int().tolist()
            cnt, _ = cv2.findContours(s["masks"][j].numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnt, -1, (0, 255, 0), 1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 1)
        tiles.append(img)
    print(f"detector: {(time.time() - t0) / 12 * 1000:.0f} ms per image")
    grid = np.concatenate([np.concatenate(tiles[r * 4:(r + 1) * 4], 1) for r in range(3)], 0)
    cv2.imwrite(os.path.join(OUT, "preview_detector.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("ALL_OK")


if __name__ == "__main__":
    main()
