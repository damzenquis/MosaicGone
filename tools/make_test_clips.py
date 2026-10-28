"""Make the synthetic "shapes" test clip: core/testdata/gt/shapes/{gt,mosaic}/NNN.png and core/testdata/images/shapes.gif.

Run:  python -I tools/make_test_clips.py   (from the repository root; needs numpy, opencv-python and pillow)

Every pixel is drawn here from a fixed random seed, so the clip has no third-party content and the same script always
makes the same frames. The scene is a 640 x 300 picture of random shapes, gradients and stripes; the clip pans across it
(21 frames of 384 x 214). The mosaic is a square-block average (10 px blocks) over an ellipse in the middle of each frame.
The GIF puts the clean frames (left) and the mosaicked frames (right) side by side, 2 frames per second.
"""
import os

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GT = os.path.join(ROOT, 'core', 'testdata', 'gt', 'shapes')
GIF = os.path.join(ROOT, 'core', 'testdata', 'images', 'shapes.gif')
W, H, N, STEP = 384, 214, 21, 6
BLOCK = 10


def scene(rng):
    sw, sh = W + STEP * (N - 1) + 16, 300
    # soft colour background: blurred noise
    bg = rng.random((sh // 20, sw // 20, 3)).astype(np.float32)
    img = cv2.resize(bg, (sw, sh), interpolation=cv2.INTER_CUBIC) * 180 + 40
    yy, xx = np.mgrid[0:sh, 0:sw]
    for _ in range(45):
        c = rng.uniform(20, 235, 3)
        kind = rng.integers(0, 4)
        x, y = rng.uniform(0, sw), rng.uniform(0, sh)
        if kind == 0:  # disc with a radial shade
            r = rng.uniform(10, 45)
            d = np.hypot(xx - x, yy - y)
            m = d < r
            shade = (1.15 - 0.35 * d / r)[..., None]
            img[m] = np.clip(c * shade[m], 0, 255)
        elif kind == 1:  # rectangle with a linear gradient
            w, h = rng.uniform(15, 80), rng.uniform(15, 60)
            m = (abs(xx - x) < w / 2) & (abs(yy - y) < h / 2)
            g = ((xx - x) / w + 0.5)[..., None]
            c2 = rng.uniform(20, 235, 3)
            img[m] = (c * (1 - g) + c2 * g)[m]
        elif kind == 2:  # striped patch
            w, h = rng.uniform(20, 70), rng.uniform(20, 70)
            m = (abs(xx - x) < w / 2) & (abs(yy - y) < h / 2)
            period = rng.uniform(4, 10)
            s = ((np.sin((xx + yy * rng.uniform(-1, 1)) * 2 * np.pi / period) > 0)[..., None])
            img[m] = np.where(s, c, c * 0.45)[m]
        else:  # polygon outline
            pts = np.stack([rng.uniform(x - 40, x + 40, 5), rng.uniform(y - 40, y + 40, 5)], 1).astype(np.int32)
            layer = img.copy()
            cv2.polylines(layer, [pts], True, c.tolist(), int(rng.integers(2, 5)), cv2.LINE_AA)
            img = layer
    img += rng.normal(0, 3, img.shape).astype(np.float32)  # light sensor-like noise
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    return np.clip(img, 0, 255).astype(np.uint8)


def add_mosaic(frame, mask):
    out = frame.copy()
    for y in range(0, H, BLOCK):
        for x in range(0, W, BLOCK):
            cy, cx = min(y + BLOCK // 2, H - 1), min(x + BLOCK // 2, W - 1)
            if mask[cy, cx]:
                blk = out[y:y + BLOCK, x:x + BLOCK]
                blk[:] = blk.reshape(-1, 3).mean(0).round().astype(np.uint8)
    return out


def main():
    rng = np.random.default_rng(20261028)
    big = scene(rng)
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (W // 2, H // 2), (62, 72), 0, 0, 360, 255, -1)
    gt = [big[8:8 + H, 8 + k * STEP:8 + k * STEP + W] for k in range(N)]
    mz = [add_mosaic(f, mask > 0) for f in gt]
    for name, frames in (('gt', gt), ('mosaic', mz)):
        d = os.path.join(GT, name)
        os.makedirs(d, exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(d, '%03d.png' % i), f)  # the arrays are BGR, as cv2 expects
    panels = [Image.fromarray(np.concatenate([g, m], 1)[..., ::-1]) for g, m in zip(gt, mz)]
    panels[0].save(GIF, save_all=True, append_images=panels[1:], duration=500, loop=0, optimize=False)
    print(f'{N} frames {W}x{H} -> {GT}, {GIF}')


if __name__ == '__main__':
    main()
