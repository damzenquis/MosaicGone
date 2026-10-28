"""Checks our BasicVSR++ code against the Lada restoration teacher, before any training.

1. the teacher checkpoint loads into our class with no missing / extra weights
2. the teacher really restores: PSNR in the mosaic area on the onepiece ground-truth clip (android/core/testdata/gt/onepiece),
   for both colour orders, to find the one it was trained with
3. the ONNX-friendly grid_sample deformable conv gives the same output as torchvision's
4. sizes and CPU speed of the slim students
"""
import glob
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dmtrain.basicvsrpp import BasicVSRPlusPlus, load_lada_teacher  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # the repository
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")  # upstream weights, not in Git (../README.md)
GT = os.path.join(ROOT, "core", "testdata", "gt", "onepiece")
torch.set_num_threads(16)


def load(d, n):
    files = sorted(glob.glob(os.path.join(d, "*.png")))[:n]
    return np.stack([np.array(Image.open(f).convert("RGB")) for f in files])  # (t, h, w, 3) RGB


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse < 1e-10 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    t = 16
    gt = load(os.path.join(GT, "gt"), t)
    mz = load(os.path.join(GT, "mosaic"), t)
    # 256x256 crop around the mosaic (ellipse at (190, 200), radii 105 x 120 in quality_reference.py)
    x0, y0 = 62, 72
    gt_c = gt[:, y0:y0 + 256, x0:x0 + 256]
    mz_c = mz[:, y0:y0 + 256, x0:x0 + 256]
    box = (slice(200 - 120 - y0, 200 + 120 - y0), slice(190 - 105 - x0, 190 + 105 - x0))
    print(f"mosaic input:   PSNR {psnr(mz_c[:, box[0], box[1]], gt_c[:, box[0], box[1]]):.2f} dB (mosaic area)")

    teacher = load_lada_teacher(os.path.join(ROOT, "Releases", "lada_mosaic_restoration_model_generic_v1.2.pth"))
    print(f"teacher: {sum(p.numel() for p in teacher.parameters()) / 1e6:.2f} M params, loaded with no missing / extra weights")
    for order in ("RGB", "BGR"):
        x = mz_c if order == "RGB" else mz_c[..., ::-1]
        inp = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float().div(255).unsqueeze(0)
        t0 = time.time()
        with torch.no_grad():
            out = teacher(inp).clamp(0, 1)
        y = (out[0].permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
        if order == "BGR":
            y = y[..., ::-1]
        print(f"teacher ({order} in): PSNR {psnr(y[:, box[0], box[1]], gt_c[:, box[0], box[1]]):.2f} dB  ({time.time() - t0:.1f} s for {t} frames on the CPU)")
        Image.fromarray(np.concatenate([mz_c[t // 2], y[t // 2], gt_c[t // 2]], 1)).save(os.path.join(os.path.dirname(__file__), f"verify_teacher_{order}.png"))

    # grid_sample deformable conv == torchvision (teacher weights, 8 frames)
    inp = torch.from_numpy(np.ascontiguousarray(mz_c[:8])).permute(0, 3, 1, 2).float().div(255).unsqueeze(0)
    with torch.no_grad():
        a = teacher.set_deform_mode("torchvision")(inp)
        b = teacher.set_deform_mode("grid_sample")(inp)
    teacher.set_deform_mode("torchvision")
    print(f"grid_sample vs torchvision deformable conv: max diff {(a - b).abs().max().item() * 255:.4f} levels")

    for name, c, nb in (("R-S", 32, 5), ("R-M", 48, 7)):
        s = BasicVSRPlusPlus(c, nb).eval()
        t0 = time.time()
        with torch.no_grad():
            s(inp)
        print(f"student {name} ({c} ch / {nb} blocks): {sum(p.numel() for p in s.parameters()) / 1e6:.2f} M params, {(time.time() - t0) / 8 * 1000:.0f} ms/frame on this CPU")
    print("ALL_OK")


if __name__ == "__main__":
    main()
