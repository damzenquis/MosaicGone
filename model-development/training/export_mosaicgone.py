"""Exports MosaicGone's two models to ONNX for the Android app and checks them against PyTorch.

Run with a Python that has torch + onnx + onnxruntime (the app tools' one), from training/:
  python export_mosaicgone.py --det runs/det_v2/best.pt --res runs/res_s2/final.pt [--out ../../app/src/main/assets/models]

  mosaicgone_detector.onnx     input  (1, 3, 416, 416) BGR 0..255, the frame letterboxed (long side 416, pad 114)
                               output (1, 1, 104, 104) "obscured here" probability (stride 4), the mask the app thresholds
  mosaicgone_restorer.onnx     input  (1, 16, 3, 256, 256) RGB 0..1: 16 frames of the square crop around the mosaic
                               output (1, 16, 3, 256, 256) RGB 0..1, the restored crops
  mosaicgone_restorer_t4.onnx  the same for 4 frames (photos: one frame repeated, 4x less work than 16)
The restorer uses the grid_sample deformable conv (same maths as torchvision's, see dmtrain/basicvsrpp.py).
"""
import argparse
import os
import time

import numpy as np
import onnxruntime as ort
import torch

from dmtrain.basicvsrpp import BasicVSRPlusPlus
from dmtrain.yolox import YOLOX

DET_SIZE = 416
CROP = 256


# wrappers must be in eval mode too: torch.onnx.export restores the wrapper's mode afterwards, recursively
class Detector(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        _, (_, sem) = self.net(x)
        return torch.sigmoid(sem)


class Restorer(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).clamp(0, 1)


def load_restorer(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = BasicVSRPlusPlus(ck.get("channels", 32), ck.get("blocks", 5), deform_mode="grid_sample")
    net.load_state_dict(ck["ema"] if "ema" in ck else ck["student"])
    return net.eval()


def load_detector(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = YOLOX(num_classes=1, num_protos=32)
    net.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
    return net.eval()


def export(model, x, path, name_in, name_out):
    torch.onnx.export(model, (x,), path, input_names=[name_in], output_names=[name_out], opset_version=17,
                      do_constant_folding=True, dynamo=False)


def check(model, x, path, name_in, threads=4, runs=3):
    with torch.no_grad():
        ref = model(x).numpy()
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    out = s.run(None, {name_in: x.numpy()})[0]
    t0 = time.perf_counter()
    for _ in range(runs):
        s.run(None, {name_in: x.numpy()})
    dt = (time.perf_counter() - t0) / runs
    diff = float(np.abs(out - ref).max())
    print(f"{os.path.basename(path)}: {os.path.getsize(path) / 1e6:.1f} MB, max diff vs PyTorch {diff:.2e}, "
          f"{dt * 1000:.0f} ms per call on {threads} PC CPU threads")
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="runs/det_v2/best.pt")
    ap.add_argument("--res", default="runs/res_s2/best.pt")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "app", "src", "main", "assets", "models"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)

    det = Detector(load_detector(a.det)).eval()
    x = torch.rand(1, 3, DET_SIZE, DET_SIZE) * 255
    p = os.path.join(a.out, "mosaicgone_detector.onnx")
    export(det, x, p, "input", "mask")
    assert check(det, x, p, "input") < 1e-3

    res = Restorer(load_restorer(a.res)).eval()
    for t, name in ((16, "mosaicgone_restorer.onnx"), (4, "mosaicgone_restorer_t4.onnx")):
        # smooth random clips (pure noise makes the motion estimator's flows meaningless)
        base = torch.nn.functional.interpolate(torch.rand(t, 3, 32, 32), size=(CROP, CROP), mode="bicubic").clamp(0, 1)
        x = base[None]
        p = os.path.join(a.out, name)
        export(res, x, p, "frames", "restored")
        assert check(res, x, p, "frames", runs=1) < 2e-3


if __name__ == "__main__":
    main()
