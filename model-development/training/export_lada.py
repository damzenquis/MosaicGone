"""Lada's restoration model (BasicVSR++ 64 ch / 15 blocks, its EMA weights) exported exactly like MosaicGone's restorer
(same input/output names and shapes): lada_restorer.onnx (T=16) and lada_restorer_t4.onnx.

Run from this folder with the export Python (torch + onnx + onnxruntime):
  python export_lada.py path/to/lada_mosaic_restoration_model_generic_v1.2.pth [out_dir]
Then copy both files into app/src/main/assets/models/ (in this repository).

The ONNX files shipped in MosaicGone v0.2 were made by an earlier copy of this script that found the .pth in the
project's model folder and wrote to a temporary folder; this copy only takes the paths as arguments instead.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_mosaicgone as E  # noqa: E402
from dmtrain.basicvsrpp import load_lada_teacher  # noqa: E402

pth = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "lada_onnx"
os.makedirs(OUT, exist_ok=True)
print("weights:", pth)
net = load_lada_teacher(pth).set_deform_mode("grid_sample").eval()
print("params %.2f M" % (sum(p.numel() for p in net.parameters()) / 1e6))
res = E.Restorer(net).eval()
torch.manual_seed(0)
for t, name in ((16, "lada_restorer.onnx"), (4, "lada_restorer_t4.onnx")):
    base = torch.nn.functional.interpolate(torch.rand(t, 3, 32, 32), size=(256, 256), mode="bicubic").clamp(0, 1)
    x = base[None]
    p = os.path.join(OUT, name)
    E.export(res, x, p, "frames", "restored")
    assert E.check(res, x, p, "frames", runs=1) < 2e-3
print("->", OUT)
