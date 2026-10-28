"""Restorer distillation, stage 1: slim BasicVSR++ (student) learns to undo synthetic mosaics and blur on real video, from
  - the true clean frames (ground truth)                        "gt"   Charbonnier, obscured area weighted x10
  - the Lada restoration teacher's output on the same clip      "out"  Charbonnier (teacher run every step, frozen)
  - the teacher's last propagation features (via a 1x1 adapter) "feat" MSE
Stage 2 (GAN + perceptual) comes after this run, from its best checkpoint.

Usage:  python train_restorer.py --run runs/res_s1 [--channels 32 --blocks 5] [--iters 80000] [--t 16] [--batch 2] [--toy]
Writes runs/<name>/{log.csv, val.csv, last.pt, best.pt, val_*.jpg}.
"""
import argparse
import csv
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dmtrain import data  # noqa: E402
from dmtrain.basicvsrpp import BasicVSRPlusPlus, load_lada_teacher, load_spynet  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # the repository
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")  # upstream weights, not in Git (../README.md)
GT = os.path.join(ROOT, "core", "testdata", "gt")
SPYNET = os.path.join(os.path.dirname(__file__), "models", "helpers", "spynet_20210409-c6c1bd09.pth")
LOSS = dict(gt=1.0, out=0.5, feat=0.1)


def charbonnier(a, b, w=None, eps=1e-3):
    d = torch.sqrt((a - b) ** 2 + eps * eps)
    return d.mean() if w is None else (d * w).sum() / (w.sum() * a.shape[2] / w.shape[2]).clamp(min=1)


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in ("lq", "gt", "mask")}


def psnr_masked(a, b, m):
    """a, b: (t, 3, h, w) 0..1; m: (t, 1, h, w) bool. PSNR inside the obscured area."""
    mse = (((a - b) ** 2) * m).sum() / (m.sum() * 3).clamp(min=1)
    return float(10 * torch.log10(1 / mse.clamp(min=1e-10)))


@torch.no_grad()
def evaluate(student, teacher, val, out_dir, step, t_ref):
    student.eval()
    sp, tp, ip, sheet = [], [], [], []
    for k, s in enumerate(val):
        lq = s["lq"][None].cuda().float() / 255
        gt = s["gt"].cuda().float() / 255
        m = s["mask"].cuda()
        y = student(lq)[0].clamp(0, 1)
        sp.append(psnr_masked(y, gt, m))
        ip.append(psnr_masked(lq[0], gt, m))
        if t_ref is None:
            tp.append(psnr_masked(teacher(lq)[0].clamp(0, 1), gt, m))
        if k < 4:
            j = s["lq"].shape[0] // 2
            row = [s["lq"][j], (y[j] * 255).byte().cpu(), s["gt"][j]]
            sheet.append(np.concatenate([r.permute(1, 2, 0).numpy() for r in row], 1))
    res = {"val_psnr": float(np.mean(sp)), "val_input_psnr": float(np.mean(ip)),
           "val_teacher_psnr": float(np.mean(tp)) if t_ref is None else t_ref}
    # the onepiece ground-truth clip (same crop as verify_restorer.py; teacher 22.66 dB there)
    d = os.path.join(GT, "onepiece")
    if os.path.isdir(d):
        files = sorted(os.listdir(os.path.join(d, "gt")))[:30]
        g = np.stack([data.read_rgb(os.path.join(d, "gt", f))[72:328, 62:318] for f in files])
        z = np.stack([data.read_rgb(os.path.join(d, "mosaic", f))[72:328, 62:318] for f in files])
        y = student(torch.from_numpy(z).permute(0, 3, 1, 2)[None].cuda().float() / 255)[0].clamp(0, 1)
        out = (y.permute(0, 2, 3, 1).cpu().numpy() * 255).round()
        box = (slice(8, 248), slice(23, 233))
        mse = np.mean((out[:, box[0], box[1]] - g[:, box[0], box[1]]) ** 2)
        res["onepiece_psnr"] = float(10 * np.log10(255 ** 2 / mse))
    if sheet:
        cv2.imwrite(os.path.join(out_dir, f"val_{step:06d}.jpg"), cv2.cvtColor(np.concatenate(sheet, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    student.train()
    return res


def forever(loader):
    """Keeps drawing batches (new random samples) until the step count is reached."""
    while True:
        yield from loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/res_s1")
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--iters", type=int, default=80000)
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--fix_spynet", type=int, default=5000, help="keep the motion estimator frozen for the first N steps")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out_dir = os.path.join(os.path.dirname(__file__), a.run)
    os.makedirs(out_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    student = load_spynet(BasicVSRPlusPlus(a.channels, a.blocks), SPYNET).cuda().train()
    teacher = load_lada_teacher(os.path.join(ROOT, "Releases", "lada_mosaic_restoration_model_generic_v1.2.pth"), device="cuda")
    adapter = nn.Conv2d(a.channels, 64, 1).cuda()
    params_main = [p for n, p in student.named_parameters() if not n.startswith("spynet.")]
    opt = torch.optim.Adam([{"params": params_main + list(adapter.parameters()), "lr": a.lr},
                            {"params": student.spynet.parameters(), "lr": a.lr * 0.25}], betas=(0.9, 0.99))
    start, best = 0, -1e9
    last = os.path.join(out_dir, "last.pt")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location="cuda", weights_only=False)
        student.load_state_dict(ck["student"]); adapter.load_state_dict(ck["adapter"]); opt.load_state_dict(ck["opt"])
        start, best = ck["step"], ck["best"]
    train = data.RestorerClips("train", t=a.t, toy=a.toy)
    val = [data.RestorerClips("val", t=a.t, toy=a.toy, seed=1)[i] for i in range(8 if a.toy else 32)]
    loader = DataLoader(train, batch_size=a.batch, num_workers=a.workers, collate_fn=collate, persistent_workers=a.workers > 0, worker_init_fn=data.worker_init,
                        prefetch_factor=4 if a.workers else None, pin_memory=True, drop_last=True)
    log = open(os.path.join(out_dir, "log.csv"), "a", newline="")
    vlog = open(os.path.join(out_dir, "val.csv"), "a", newline="")
    writer = vwriter = None
    t_ref = None
    t0, it = time.time(), start
    for batch in forever(loader):
        if it >= a.iters:
            break
        lr_scale = min(1.0, (it + 1) / 1000) * (0.5 * (1 + math.cos(math.pi * it / a.iters)) * 0.98 + 0.02)
        opt.param_groups[0]["lr"] = a.lr * lr_scale
        opt.param_groups[1]["lr"] = 0.0 if it < a.fix_spynet else a.lr * 0.25 * lr_scale
        lq = batch["lq"].cuda(non_blocking=True).float() / 255
        gt = batch["gt"].cuda(non_blocking=True).float() / 255
        w = (batch["mask"].cuda().float() * 9 + 1)  # obscured area x10
        with torch.no_grad():
            t_out, t_feat = teacher(lq, return_feats=True)
        y, s_feat = student(lq, return_feats=True)
        n, t, c, h, wd = s_feat.shape
        f = adapter(s_feat.reshape(n * t, c, h, wd)).view(n, t, 64, h, wd)
        L = {"gt": charbonnier(y, gt, w), "out": charbonnier(y, t_out.clamp(0, 1), w), "feat": F.mse_loss(f, t_feat)}
        total = sum(LOSS[k] * v for k, v in L.items())
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(adapter.parameters()), 1.0)
        opt.step()
        it += 1
        if it % 50 == 0:
            row = {"step": it, "lr": a.lr * lr_scale, "loss": float(total.detach()), **{k: float(v.detach()) for k, v in L.items()}, "s_per_it": (time.time() - t0) / 50}
            if writer is None:
                writer = csv.DictWriter(log, fieldnames=list(row.keys()))
                if log.tell() == 0:
                    writer.writeheader()
            writer.writerow(row)
            log.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            if it % 500 == 0:  # frequent resume point (a lost window costs at most ~10 min)
                torch.save({"student": student.state_dict(), "adapter": adapter.state_dict(), "opt": opt.state_dict(), "step": it,
                            "best": best, "channels": a.channels, "blocks": a.blocks}, last)
            if it % a.val_every == 0 or it == a.iters:
                v = {"step": it, **evaluate(student, teacher, val, out_dir, it, t_ref)}
                t_ref = v["val_teacher_psnr"]
                score = v["val_psnr"]
                torch.save({"student": student.state_dict(), "adapter": adapter.state_dict(), "opt": opt.state_dict(), "step": it,
                            "best": max(best, score), "channels": a.channels, "blocks": a.blocks}, last)
                if score > best:
                    best = score
                    torch.save({"student": student.state_dict(), "step": it, "metrics": v, "channels": a.channels, "blocks": a.blocks}, os.path.join(out_dir, "best.pt"))
                if vwriter is None:
                    vwriter = csv.DictWriter(vlog, fieldnames=list(v.keys()))
                    if vlog.tell() == 0:
                        vwriter.writeheader()
                vwriter.writerow(v)
                vlog.flush()
                print("VAL " + " ".join(f"{k}={x:.4g}" if isinstance(x, float) else f"{k}={x}" for k, x in v.items()), flush=True)
            t0 = time.time()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
