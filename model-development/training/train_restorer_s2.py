"""Restorer stage 2: GAN + perceptual fine-tune of the stage-1 student (sharper, less averaged detail) with the
animation stills (TRAINING DATA/Animation: manga, classic) mixed in. Starts from runs/res_s1/best.pt.
  - "gt"    Charbonnier to the true frames, obscured area weighted x10 (keeps it faithful)
  - "out"   Charbonnier to the Lada teacher's output (the teacher was trained this way itself); on animation clips only
            where the teacher beats stage 1 there (decided from base.json at start)
  - "perc"  VGG19 feature distance to the true frames (texture / edges)
  - "gan"   fool a U-Net discriminator that sees true frames vs ours (per-pixel)
The model that is evaluated and kept is the EMA (slow moving average) of the student's weights, as in Lada.

Checks, every --val_every steps, on: real val clips, held-out manga stills, held-out classic stills, the onepiece clip,
and (overfitting watch only) animation stills it trains on. Step 0 = stage 1 and the teacher on the same checks (base.json).
  psnr    obscured area                                    accuracy
  edge    PSNR on line / edge pixels                       line art
  seam    PSNR on a band across the region border          visible patch edges
  colour  mean colour offset in the obscured area (0-255)  tint
  sharp   detail energy out / true (1 = right, <1 blur)    blur (reported)
  grain   our detail on flat true areas (grey levels)      invented texture (GAN on cel shading)
  flick   frame-to-frame change out vs true                temporal shimmer
  vgg     perceptual distance (lower = looks closer)       overall look
  aperc   anime perceptual distance (Danbooru ResNet50,    anime look (animation sets only; when the weights are in
          x ANIME_SCALE onto VGG's scale)                  models/helpers)
best.pt is picked by score = look (vgg relative to stage 1, every domain) - FLOOR_PENALTY per broken floor -
TARGET_PENALTY per dB / 10% the animation domains are still behind the teacher (target: catch up to it).
Floors, every domain, against stage 1 (s1) and the teacher (t):
  psnr >= s1 - PSNR_SLACK (real also >= t)   edge, seam >= s1 - PSNR_SLACK   colour <= s1 + COLOUR_SLACK
  flick <= s1 x FLICK_MAX                    grain <= s1 x GRAIN_MAX           vgg <= s1
and, with --ref (checkpoints of earlier runs, checked at start on the same val sets), against their record (best of
them, per check) with only measurement slack (REF_*). best.pt is only written for a check that holds every floor.
Animation-only extras: --line_w (line-art pixels weigh more in the gt loss), --out_gated (teacher loss only on pixels
where the teacher is closer to the truth than the student), --line_sharp (APISR-style target: the true frame's ink lines
mildly unsharp-masked, inside the obscured region only), --tv (share of the New_Dataset tv frames among animation clips); the
animation clips' codec damage is in dmtrain.synth.AnimeDamage. Real-footage clips are made exactly as before.
--chroma (animation clips): colour loss (Cb/Cr) inside the obscured region, per pixel + the area's mean tint, against the true
frames. --line_mask (animation clips): that share of obscured regions sits on the densest ink lines (faces / hair) - the
onepiece-style case of a whole face hidden. --info_sets: val sets checked and reported, but not blocking best.pt.
--anime_perc f: on animation frames the perceptual loss is (1 - f) VGG + f anime ResNet50 (APISR's twin perceptual loss;
same total strength as VGG alone, real frames VGG only).
--keep (all clips): pull towards the --keep_ref model (default: the first --ref) only where it is closer to the truth
than we are, so a fine-tune can improve on it but not drift below it.
--init may be an earlier stage-2 run (its EMA weights and discriminator are loaded); stage 1 for the floors is --stage1.
Guards: non-finite losses skip the step (ABORT after MAX_BAD in a row); the discriminator skips its update while it
wins too easily (D_GAP_MAX); warnings for train / val gap on animation (overfitting) and floors broken 3 checks running.

Usage:  python train_restorer_s2.py --run runs/res_s2 [--iters 40000] [--anime 0.35] [--gan 0.05 --perc 0.2]
Pilots: --iters 3000 --run runs/s2_pilot_real --anime 0   /   --iters 3000 --run runs/s2_pilot_anime --anime 0.35
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dmtrain import data  # noqa: E402
from dmtrain.basicvsrpp import BasicVSRPlusPlus, load_lada_teacher  # noqa: E402
from dmtrain.discriminator import AnimeFeatures, UNetDiscriminatorSN, VGGFeatures, anime_perceptual, perceptual  # noqa: E402
from train_restorer import GT, MODELS, ROOT, charbonnier, psnr_masked  # noqa: E402

VGG = os.path.join(os.path.dirname(__file__), "models", "helpers", "vgg19-dcbb9e9d.pth")
ANIME_NET = os.path.join(os.path.dirname(__file__), "models", "helpers", "danbooru_resnet50-13306192.pth")
ANIME_SCALE = 14.0  # VGG / anime perceptual distance on G's animation val outputs (measured): puts the two on one scale
ANIME_PERC_FRAMES = 4  # --anime_perc: animation frames per step through the ResNet50 (GPU memory)
ANIME_SETS = ("manga", "classic", "tv", "onepiece", "opapp")
FLOOR_PENALTY = 10.0
TARGET_PENALTY = 2.0
PSNR_SLACK = 0.3
COLOUR_SLACK = 0.5
GRAIN_MAX = 1.25
FLICK_MAX = 1.15
# against the best earlier run (--ref): tighter, only measurement slack
REF_DB = 0.1
REF_VGG = 1.01
REF_COLOUR = 0.2
REF_GRAIN = 1.1
REF_FLICK = 1.05
LINE_T = 0.2      # dark-stroke Laplacian of the true frame above this = line art (animation clips, --line_w)
XDOG_T = 0.02     # --line_sharp: DoG (sigma 0.8 - 1.6) of the true frame below -this = ink line
SHARP_SIGMA = 1.0  # --line_sharp: unsharp-mask radius
TV_VAL = 24       # tv val clips (held-out source videos)
LOWER_BETTER = ("colour", "grain", "flick", "vgg", "aperc")
D_GAP_MAX = 0.7   # discriminator: mean P(true) on true frames - on ours; above it, it skips its update
MAX_BAD = 20


def collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in ("lq", "gt", "mask")}
    out["anime"] = torch.tensor([b["anime"] for b in batch])
    return out


def forever(loader):
    while True:
        yield from loader


# ---------------------------------------------------------------- checks


def detail(x):
    """|Laplacian| of the grey image, (t, 1, h, w); x: (t, 3, h, w) 0..1."""
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    return F.conv2d(x.mean(1, keepdim=True), k, padding=1).abs()


def dark_lines(x):
    """Line art: pixels darker than their 4 neighbours by > LINE_T (ink strokes, not screentone), thickened 1 px."""
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    return F.max_pool2d((F.conv2d(x.mean(1, keepdim=True), k, padding=1) > LINE_T).float(), 3, 1, 1)


def gauss(x, sigma):
    """Separable Gaussian blur, edges replicated; x: (n, c, h, w)."""
    r = max(1, int(sigma * 3 + 0.5))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def chroma(x):
    """Cb, Cr (BT.601) of 0..1 RGB, (..., 3, h, w) -> (..., 2, h, w)."""
    r, g, b = x.unbind(-3)
    return torch.stack([-0.1687 * r - 0.3313 * g + 0.5 * b, 0.5 * r - 0.4187 * g - 0.0813 * b], -3)


def line_sharpened(gt, m, amount):
    """Training target for animation clips, after APISR's line-enhanced pseudo-GT but mild and local: the true frame's
    ink lines (DoG line map; isolated specks dropped, 1 px dilation) unsharp-masked by [amount] (APISR: 3 full passes),
    only inside the obscured region and fading out over its last ~8 px, so the restored lines are not crisper than the frame
    around them right at the border (that would show as a seam). gt: (n, 3, h, w) 0..1; m: (n, 1, h, w) 0/1."""
    g = gt.mean(1, keepdim=True)
    lines = ((gauss(g, 0.8) - gauss(g, 1.6)) < -XDOG_T).float()
    lines = lines * (F.avg_pool2d(lines, 3, 1, 1) > 0.34).float()
    lines = F.avg_pool2d(F.max_pool2d(lines, 3, 1, 1), 3, 1, 1)
    inner = F.avg_pool2d(1 - F.max_pool2d(1 - m, 9, 1, 4), 9, 1, 4)
    sharp = gt + amount * (gt - gauss(gt, SHARP_SIGMA))
    return (gt + lines * inner * (sharp - gt)).clamp(0, 1), lines * inner


def checks(vgg, y, gt, m, anet=None):
    """y, gt: (t, 3, h, w) 0..1; m: (t, 1, h, w) bool obscured area. anet: also the anime perceptual distance."""
    mf = m.float()
    dg, dy = detail(gt), detail(y)
    gt_u8 = (gt.mean(1) * 255).byte().cpu().numpy()
    edges = torch.from_numpy(np.stack([cv2.dilate(cv2.Canny(f, 60, 150), np.ones((3, 3), np.uint8)) > 0 for f in gt_u8]))
    edges = edges.to(y.device).unsqueeze(1) & m
    flat = (dg < 0.01) & m
    # band across the border: dilated minus eroded mask (4 px each side)
    grow = F.max_pool2d(mf, 9, 1, 4)
    shrink = 1 - F.max_pool2d(1 - mf, 9, 1, 4)
    band = (grow - shrink) > 0.5
    eps = 1e-6
    res = {
        "psnr": psnr_masked(y, gt, m),
        "edge": psnr_masked(y, gt, edges) if edges.any() else float("nan"),
        "seam": psnr_masked(y, gt, band) if band.any() else float("nan"),
        "colour": float((((y - gt) * mf).sum(dim=(0, 2, 3)) / (mf.sum() + eps)).abs().mean() * 255),
        # sharp: undefined when the true obscured area has no detail at all (a flat patch)
        "sharp": float((dy * mf).sum() / (dg * mf).sum()) if (dg * mf).sum() > 1e-3 * mf.sum() else float("nan"),
        # grain: detail we put on flat true areas, in grey levels (flat true areas have ~none, so not a ratio)
        "grain": float((dy * flat).sum() / flat.sum() * 255) if flat.any() else float("nan"),
        "vgg": float(torch.cat([perceptual(vgg, y[i:i + 8], gt[i:i + 8]) for i in range(0, len(y), 8)]).mean()),
    }
    dy_t, dg_t = (y[1:] - y[:-1]).abs(), (gt[1:] - gt[:-1]).abs()
    mm = (m[1:] | m[:-1]).float()
    res["flick"] = float(((dy_t - dg_t).abs() * mm).sum() / (mm.sum() * 3 + eps))
    if anet is not None:  # anime look, on VGG's scale (ANIME_SCALE)
        res["aperc"] = float(torch.cat([anime_perceptual(anet, y[i:i + 8], gt[i:i + 8]) for i in range(0, len(y), 8)]).mean()) * ANIME_SCALE
    return res


def onepiece():
    d = os.path.join(GT, "onepiece")
    if not os.path.isdir(d):
        return None
    files = sorted(os.listdir(os.path.join(d, "gt")))[:30]
    g = np.stack([data.read_rgb(os.path.join(d, "gt", f))[72:328, 62:318] for f in files])
    z = np.stack([data.read_rgb(os.path.join(d, "mosaic", f))[72:328, 62:318] for f in files])
    m = np.zeros(g.shape[:3] + (1,), bool)
    m[:, 8:248, 23:233] = True  # same box as verify_restorer.py
    t = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2)  # noqa: E731
    return {"lq": t(z), "gt": t(g), "mask": t(m)}


def onepiece_app():
    """The onepiece clip as the app crops it: a square 1.5x the mosaic's size around it (Options.exMultValue), resized
    to 256. The onepiece() box hides 94% of the crop's width, more than training (<= 87%) or the app (67%) ever does."""
    d = os.path.join(GT, "onepiece")
    if not os.path.isdir(d):
        return None
    files = sorted(os.listdir(os.path.join(d, "gt")))[:30]
    x0, y0, s = 10, 20, 360  # the onepiece() box in the full frame: x 85..295, y 80..320 (240 px) -> 360 px around it
    crop = lambda a, i=cv2.INTER_AREA: cv2.resize(a[y0:y0 + s, x0:x0 + s], (256, 256), interpolation=i)  # noqa: E731
    g = np.stack([crop(data.read_rgb(os.path.join(d, "gt", f))) for f in files])
    z = np.stack([crop(data.read_rgb(os.path.join(d, "mosaic", f))) for f in files])
    full = np.zeros(data.read_rgb(os.path.join(d, "gt", files[0])).shape[:2], np.uint8)
    full[80:320, 85:295] = 1
    m = np.repeat(crop(full, cv2.INTER_NEAREST)[None, ..., None] > 0, len(files), 0)
    t = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2)  # noqa: E731
    return {"lq": t(z), "gt": t(g), "mask": t(m)}


def sobel(x):
    """Edge strength (Sobel gradient magnitude) of the luminance; x: (n, 3, h, w) -> (n, 1, h, w)."""
    k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device) / 8
    g = F.pad(x.mean(1, keepdim=True), (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(g, k.view(1, 1, 3, 3)), F.conv2d(g, k.t().reshape(1, 1, 3, 3))
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def val_sets(t):
    sets = {"real": [data.RestorerClips("val", t=t, seed=1)[i] for i in range(16)]}
    for k, style in enumerate(("manga", "classic")):
        ds = data.RestorerClips("val", t=t, seed=2 + k, weights=(0, 0, 0, 1), style=style)
        if ds.weights[3] > 0:
            sets[style] = [ds[i] for i in range(16)]
    # tv: held-out source videos of New_Dataset (so animation is not judged on the onepiece clip alone)
    ds = data.RestorerClips("val", t=t, seed=4, weights=(0, 0, 0, 1), style="tv")
    if ds.weights[3] > 0:
        sets["tv"] = [ds[i] for i in range(TV_VAL)]
    op = onepiece()
    if op is not None:
        sets["onepiece"] = [op]
        sets["opapp"] = [onepiece_app()]
    # stills it trains on, same way: a train / val gap that keeps growing = memorising the stills
    ds = data.RestorerClips("train", t=t, seed=7, weights=(0, 0, 0, 1))
    if ds.weights[3] > 0:
        ds.fixed = True
        sets["anime_train"] = [ds[i] for i in range(8)]
    return sets


@torch.no_grad()
def evaluate(net, vgg, sets, out_dir=None, step=0, tag="", anet=None):
    net.eval()
    res, sheet = {}, []
    for name, clips in sets.items():
        rows = []
        for k, s in enumerate(clips):
            lq = s["lq"][None].cuda().float() / 255
            gt = s["gt"].cuda().float() / 255
            y = net(lq)[0].clamp(0, 1)
            rows.append(checks(vgg, y, gt, s["mask"].cuda(), anet if name != "real" else None))
            if out_dir and k < 2 and name != "anime_train":
                j = s["lq"].shape[0] // 2
                row = [s["lq"][j], (y[j] * 255).byte().cpu(), s["gt"][j]]
                sheet.append(np.concatenate([r.permute(1, 2, 0).numpy() for r in row], 1))
        for key in rows[0]:
            res[f"{name}_{key}"] = float(np.nanmean([r[key] for r in rows]))
    if out_dir and sheet:
        cv2.imwrite(os.path.join(out_dir, f"val{tag}_{step:06d}.jpg"), cv2.cvtColor(np.concatenate(sheet, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    net.train()
    return res


def score(v, base, info=()):
    """(score, broken floors, animation shortfall vs the teacher). Domains in [info] are still checked against every
    floor, but only reported (returned in broken with an "(info)" tag): they neither block best.pt nor count in the
    score."""
    s1, te = base["stage1"], base["teacher"]
    broken, looks, short = [], [], 0.0
    for d in ["real"] + [d for d in ANIME_SETS if f"{d}_psnr" in v and d not in info]:
        g = lambda k: v[f"{d}_{k}"]  # noqa: E731
        b = lambda k: s1[f"{d}_{k}"]  # noqa: E731
        if g("psnr") < b("psnr") - PSNR_SLACK or (d == "real" and g("psnr") < te["real_psnr"]):
            broken.append(f"{d}_psnr")
        for k in ("edge", "seam"):
            if not math.isnan(b(k)) and g(k) < b(k) - PSNR_SLACK:
                broken.append(f"{d}_{k}")
        if g("colour") > b("colour") + COLOUR_SLACK:
            broken.append(f"{d}_colour")
        if g("flick") > b("flick") * FLICK_MAX:
            broken.append(f"{d}_flick")
        if not math.isnan(b("grain")) and g("grain") > b("grain") * GRAIN_MAX:
            broken.append(f"{d}_grain")
        if g("vgg") > b("vgg"):
            broken.append(f"{d}_vgg")
        if not math.isnan(v.get(f"{d}_aperc", float("nan"))) and g("aperc") > s1.get(f"{d}_aperc", float("inf")):
            broken.append(f"{d}_aperc")
        if "ref" in base:  # high-water mark: no worse than the best earlier run (--ref), within measurement slack
            r = lambda k: base["ref"].get(f"{d}_{k}", float("nan"))  # noqa: E731
            for k, bad_ in (("psnr", g("psnr") < r("psnr") - REF_DB), ("edge", g("edge") < r("edge") - REF_DB),
                            ("seam", g("seam") < r("seam") - REF_DB), ("vgg", g("vgg") > r("vgg") * REF_VGG),
                            ("colour", g("colour") > r("colour") + REF_COLOUR), ("grain", g("grain") > r("grain") * REF_GRAIN),
                            ("flick", g("flick") > r("flick") * REF_FLICK),
                            ("aperc", v.get(f"{d}_aperc", float("nan")) > r("aperc") * REF_VGG)):
                if bad_ and f"{d}_{k}" not in broken:
                    broken.append(f"{d}_{k}~ref")
        looks.append(g("vgg") / b("vgg"))
        if d != "real":  # target: at least the teacher's accuracy and look on animation
            short += max(0.0, te[f"{d}_psnr"] - g("psnr")) + 10 * max(0.0, g("vgg") / te[f"{d}_vgg"] - 1)
    sc = -float(np.mean(looks)) - FLOOR_PENALTY * len(broken) - TARGET_PENALTY * short
    if info:  # report-only domains: their floors, checked the same way, tagged
        _, ib, _ = score({k: x for k, x in v.items() if k.split("_")[0] in info or k.startswith("real_")}, base)
        broken = broken + [f"{b}(info)" for b in ib if b.split("_")[0] in info]
    return sc, broken, short


# ---------------------------------------------------------------- training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/res_s2")
    ap.add_argument("--init", default="runs/res_s1/best.pt")
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr_d", type=float, default=1e-4)
    ap.add_argument("--anime", type=float, default=0.35, help="share of animation clips (0 = real footage only)")
    ap.add_argument("--gt", type=float, default=1.0)
    ap.add_argument("--out", type=float, default=0.25)
    ap.add_argument("--perc", type=float, default=0.2)
    ap.add_argument("--gan", type=float, default=0.05)
    ap.add_argument("--gan_anime", type=float, default=1.0, help="GAN weight multiplier on animation clips")
    ap.add_argument("--d_frames", type=int, default=4, help="frames per clip shown to VGG and the discriminator")
    ap.add_argument("--d_warmup", type=int, default=1000, help="steps the discriminator trains before the GAN loss is used")
    ap.add_argument("--line_w", type=float, default=1.0, help="gt-loss weight on line-art pixels of animation clips (1 = off)")
    ap.add_argument("--out_gated", type=float, default=0.0,
                    help="teacher loss on animation clips, only on pixels where the teacher is closer to the truth than we are")
    ap.add_argument("--ref", nargs="*", default=[],
                    help="checkpoints of earlier runs: checked at start; their record (best per check) is a floor too")
    ap.add_argument("--stage1", default="runs/res_s1/best.pt", help="stage 1 (floors, look baseline) when --init is another run")
    ap.add_argument("--tv", type=float, default=0.0, help="share of tv frames (New_Dataset) among animation clips (0 = by count)")
    ap.add_argument("--line_sharp", type=float, default=0.0, help="animation clips: unsharp amount on the target's ink lines (0 = off)")
    ap.add_argument("--keep", type=float, default=0.0, help="pull towards --keep_ref where it is closer to the truth (0 = off)")
    ap.add_argument("--keep_ref", default="", help="model for --keep (default: the first --ref)")
    ap.add_argument("--chroma", type=float, default=0.0,
                    help="animation clips: colour (Cb/Cr) loss inside the obscured region, per pixel + the area's mean tint (0 = off)")
    ap.add_argument("--line_mask", type=float, default=0.0,
                    help="share of animation clips whose obscured region sits on the densest ink lines (faces / hair)")
    ap.add_argument("--info_sets", nargs="*", default=[],
                    help="val sets only reported: checked against the floors but not blocking best.pt nor scored (e.g. onepiece)")
    ap.add_argument("--anime_perc", type=float, default=0.0,
                    help="animation frames: share of the perceptual loss from the anime ResNet50 instead of VGG (APISR twin loss)")
    ap.add_argument("--tone", type=float, default=0.0,
                    help="every clip: the obscured area's mean offset per channel against the true frames (brightness + tint)")
    ap.add_argument("--out_gate_real", action="store_true",
                    help="teacher loss only where the teacher is closer to the truth than the student (all clips it applies to)")
    ap.add_argument("--edge", type=float, default=0.0,
                    help="animation clips: penalty when the edge strength near the true ink lines inside the obscured region is weaker "
                         "than the truth's (one-sided, not tied to the exact line position: punishes blur, not a shifted line)")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out_dir = os.path.join(os.path.dirname(__file__), a.run)
    os.makedirs(out_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    here = os.path.dirname(__file__)

    def load_net(path):
        ck = torch.load(os.path.join(here, path), map_location="cuda", weights_only=False)
        net = BasicVSRPlusPlus(ck["channels"], ck["blocks"]).cuda().eval()
        net.load_state_dict(ck["ema"] if "ema" in ck else ck["student"])  # a stage-2 run: its EMA is the evaluated model
        return net.requires_grad_(False)

    init = torch.load(os.path.join(here, a.init), map_location="cuda", weights_only=False)
    student = BasicVSRPlusPlus(init["channels"], init["blocks"]).cuda()
    student.load_state_dict(init["ema"] if "ema" in init else init["student"])
    student.train()
    ema = copy.deepcopy(student).eval().requires_grad_(False)
    teacher = load_lada_teacher(os.path.join(ROOT, "Releases", "lada_mosaic_restoration_model_generic_v1.2.pth"), device="cuda")
    disc = UNetDiscriminatorSN().cuda().train()
    if "disc" in init:
        disc.load_state_dict(init["disc"])
    vgg = VGGFeatures(VGG).cuda()
    # anime perceptual network: a check on the animation sets whenever its weights are there, a loss with --anime_perc
    anet = AnimeFeatures(ANIME_NET).cuda() if os.path.exists(ANIME_NET) else None
    if a.anime_perc > 0 and anet is None:
        raise SystemExit(f"--anime_perc needs {ANIME_NET}")
    params_main = [p for n, p in student.named_parameters() if not n.startswith("spynet.")]
    opt_g = torch.optim.Adam([{"params": params_main, "lr": a.lr}, {"params": student.spynet.parameters(), "lr": a.lr * 0.25}], betas=(0.9, 0.99))
    opt_d = torch.optim.Adam(disc.parameters(), lr=a.lr_d, betas=(0.9, 0.99))

    sets = val_sets(a.t)
    print("val sets: " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()), flush=True)
    base_path = os.path.join(out_dir, "base.json")
    if os.path.exists(base_path):
        with open(base_path) as fh:
            base = json.load(fh)
    else:
        s1_net = ema if os.path.normpath(a.init) == os.path.normpath(a.stage1) else load_net(a.stage1)
        base = {"stage1": evaluate(s1_net, vgg, sets, out_dir, 0, "_stage1", anet),
                "teacher": evaluate(teacher, vgg, sets, out_dir, 0, "_teacher", anet)}
        if s1_net is not ema:
            del s1_net
        with open(base_path, "w") as fh:
            json.dump(base, fh, indent=1)
    for who, v in base.items():
        if who != "refs":
            print(f"BASE {who} " + " ".join(f"{k}={x:.4g}" for k, x in v.items()), flush=True)
    # teacher loss on animation clips only if the teacher is the better model there (else it would pull us down)
    anime_doms = [d for d in ("manga", "classic") if f"{d}_psnr" in base["teacher"]]
    out_anime = a.out if anime_doms and all(base["teacher"][f"{d}_psnr"] > base["stage1"][f"{d}_psnr"] for d in anime_doms) else 0.0
    print(f"teacher loss weight: real {a.out}, animation {out_anime}, animation gated {a.out_gated}; line weight {a.line_w}", flush=True)
    if a.ref and "refs" not in base:
        # earlier runs, checked here on the same val sets (their own val.csv may come from other val clips); the record
        # per check = the best any of them reached (higher psnr / edge / seam / sharp, lower LOWER_BETTER)
        base["refs"] = {}
        for k, path in enumerate(a.ref):
            ref_net = load_net(path)
            base["refs"][path] = evaluate(ref_net, vgg, sets, out_dir, 0, f"_ref{k}", anet)
            del ref_net
            print(f"BASE ref ({path}) " + " ".join(f"{q}={x:.4g}" for q, x in base["refs"][path].items()), flush=True)
        keys = next(iter(base["refs"].values())).keys()
        pick = lambda q, xs: float(np.nanmin(xs) if q.rsplit("_", 1)[1] in LOWER_BETTER else np.nanmax(xs))  # noqa: E731
        base["ref"] = {q: pick(q, [r[q] for r in base["refs"].values()]) for q in keys}
        with open(base_path, "w") as fh:
            json.dump(base, fh, indent=1)
    missing = {k: v for k, v in sets.items() if f"{k}_psnr" not in base["stage1"]}
    if missing:
        # val sets added after the run started (e.g. opapp): stage 1, teacher and --ref values on them, the same way
        print("BASE adding val sets: " + ", ".join(missing), flush=True)
        s1_net = load_net(a.stage1)
        base["stage1"].update(evaluate(s1_net, vgg, missing, anet=anet))
        del s1_net
        base["teacher"].update(evaluate(teacher, vgg, missing, anet=anet))
        if "refs" in base:
            for path in base["refs"]:
                ref_net = load_net(path)
                base["refs"][path].update(evaluate(ref_net, vgg, missing, anet=anet))
                del ref_net
            keys = next(iter(base["refs"].values())).keys()
            pick = lambda q, xs: float(np.nanmin(xs) if q.rsplit("_", 1)[1] in LOWER_BETTER else np.nanmax(xs))  # noqa: E731
            base["ref"] = {q: pick(q, [r[q] for r in base["refs"].values()]) for q in keys}
        with open(base_path, "w") as fh:
            json.dump(base, fh, indent=1)
    if "ref" in base:
        print("BASE record (best of --ref) " + " ".join(f"{k}={x:.4g}" for k, x in base["ref"].items()), flush=True)
    keep_net = None
    if a.keep > 0:
        keep_path = a.keep_ref or (a.ref[0] if a.ref else "")
        if not keep_path:
            raise SystemExit("--keep needs --keep_ref or --ref")
        keep_net = load_net(keep_path)
        print(f"keep: weight {a.keep}, towards {keep_path} where it is closer to the truth", flush=True)
    print(f"animation: tv share {a.tv or 'by count'}, line sharpening {a.line_sharp}", flush=True)

    torch.cuda.reset_peak_memory_stats()  # gpu_gb in log.csv = training's peak, not the step-0 checks'
    start, best = 0, -1e9
    last = os.path.join(out_dir, "last.pt")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location="cuda", weights_only=False)
        student.load_state_dict(ck["student"]); ema.load_state_dict(ck["ema"]); disc.load_state_dict(ck["disc"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        start, best = ck["step"], ck["best"]

    w_real = 1.0 - a.anime
    train = data.RestorerClips("train", t=a.t, weights=(0.55 * w_real, 0.38 * w_real, 0.07 * w_real, a.anime),
                               styles={"tv": a.tv} if a.tv > 0 else None, line_mask=a.line_mask)
    if a.anime > 0 and train.weights[3] == 0:
        print("WARNING: --anime > 0 but no images in TRAINING DATA/Animation; training on real footage only", flush=True)
    loader = DataLoader(train, batch_size=a.batch, num_workers=a.workers, collate_fn=collate, persistent_workers=a.workers > 0,
                        worker_init_fn=data.worker_init, prefetch_factor=4 if a.workers else None, pin_memory=True, drop_last=True)
    log = open(os.path.join(out_dir, "log.csv"), "a", newline="")
    vlog = open(os.path.join(out_dir, "val.csv"), "a", newline="")
    writer = vwriter = None
    t0, it = time.time(), start
    stats, bad, d_gap, d_skips, broken_run = {}, 0, 0.0, 0, 0

    def save(path):
        torch.save({"student": student.state_dict(), "ema": ema.state_dict(), "disc": disc.state_dict(), "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(), "step": it, "best": best, "channels": init["channels"], "blocks": init["blocks"]}, path)

    for batch in forever(loader):
        if it >= a.iters:
            break
        lr_scale = min(1.0, (it + 1) / 500) * (0.5 * (1 + math.cos(math.pi * it / a.iters)) * 0.9 + 0.1)
        opt_g.param_groups[0]["lr"] = a.lr * lr_scale
        opt_g.param_groups[1]["lr"] = a.lr * 0.25 * lr_scale
        opt_d.param_groups[0]["lr"] = a.lr_d * lr_scale
        lq = batch["lq"].cuda(non_blocking=True).float() / 255
        gt = batch["gt"].cuda(non_blocking=True).float() / 255
        m = batch["mask"].cuda().float()
        w = m * 9 + 1
        is_anime = batch["anime"].cuda()
        n, t = lq.shape[:2]
        if a.line_w != 1.0 and is_anime.any():
            # line art of the true frames counts line_w x in the gt loss, animation clips only
            lines = dark_lines(gt.flatten(0, 1)).view(n, t, 1, *gt.shape[-2:])
            w = w * (1 + (a.line_w - 1) * lines * is_anime.view(n, 1, 1, 1, 1).float())
        # what the gt and perceptual losses aim at: the true frames; animation clips with --line_sharp: their ink lines
        # mildly sharpened inside the obscured region (the checks, the gates and the discriminator keep the true frames)
        tgt = gt
        if a.line_sharp > 0 and is_anime.any():
            sel = is_anime.view(n, 1, 1, 1, 1)
            ls, lw = line_sharpened(gt.flatten(0, 1), m.flatten(0, 1), a.line_sharp)
            tgt = torch.where(sel, ls.view_as(gt), gt)
            stats.setdefault("ls_px", []).append(float((lw.view(n, t, 1, *gt.shape[-2:]) * sel.float()).sum()
                                                       / (m * sel.float()).sum().clamp(min=1)))
        with torch.no_grad():
            t_out = teacher(lq).clamp(0, 1)
            k_out = keep_net(lq).clamp(0, 1) if keep_net is not None else None
        y = student(lq)
        # a few frames per clip for VGG and the discriminator (memory)
        idx = torch.stack([torch.randperm(t, device=lq.device)[:a.d_frames] for _ in range(n)])
        pick = lambda v: torch.stack([v[i, idx[i]] for i in range(n)]).flatten(0, 1)  # noqa: E731
        y_d, gt_d, tgt_d = pick(y), pick(gt), pick(tgt)
        frame_anime = is_anime.repeat_interleave(a.d_frames).float()
        gan_w = 1.0 + (a.gan_anime - 1.0) * frame_anime
        out_w = torch.where(is_anime, torch.full_like(is_anime, out_anime, dtype=torch.float), torch.full_like(is_anime, a.out, dtype=torch.float))
        use_gan = it >= a.d_warmup

        # generator
        disc.requires_grad_(False)
        perc = perceptual(vgg, y_d, tgt_d)
        if a.anime_perc > 0 and frame_anime.any():
            # animation frames: (1 - anime_perc) VGG + anime_perc anime ResNet50 (on VGG's scale); real frames: VGG only.
            # At most ANIME_PERC_FRAMES frames per step (memory)
            sel = frame_anime.nonzero().flatten()[:ANIME_PERC_FRAMES]
            pa = anime_perceptual(anet, y_d[sel], tgt_d[sel]) * ANIME_SCALE
            perc = perc.clone()
            perc[sel] = (1 - a.anime_perc) * perc[sel] + a.anime_perc * pa
            stats.setdefault("perc_anime", []).append(float(pa.detach().mean()))
        L = {"gt": charbonnier(y, tgt, w), "perc": perc.mean()}
        if a.chroma > 0:
            L["chroma"] = y.new_zeros(())
        if a.chroma > 0 and is_anime.any():
            # colour of animation clips inside the obscured region, against the true frames: per pixel, plus the area's mean
            # offset per clip (the tint the "colour" check measures; e.g. colour put into black-and-white manga)
            sel = is_anime.view(n, 1, 1, 1, 1).float()
            dc = chroma(y) - chroma(gt)
            mm = m * sel
            px = (torch.sqrt(dc ** 2 + 1e-6) * mm).sum() / (mm.sum() * 2).clamp(min=1)
            tint = ((dc * mm).sum(dim=(1, 3, 4)) / (mm.sum(dim=(1, 3, 4)) + 1).view(n, 1)).abs()  # (n, 2)
            L["chroma"] = a.chroma * (px + tint[is_anime].mean())
            stats.setdefault("tint", []).append(float(tint[is_anime].detach().mean() * 255))
        if k_out is not None:
            # no drift below the --keep_ref model: follow it only where it is closer to the truth than we are right now
            with torch.no_grad():
                kgate = ((k_out - gt).abs().mean(2, keepdim=True) < (y - gt).abs().mean(2, keepdim=True)).float()
            L["keep"] = a.keep * (torch.sqrt((y - k_out) ** 2 + 1e-6) * w * kgate).sum() / (w.sum() * 3).clamp(min=1)
            stats.setdefault("keep_gate", []).append(float(kgate.mean()))
        if a.out_gate_real:
            # teacher loss only where the teacher is closer to the truth than we are right now: on real footage it is the
            # weaker model overall (brighter, grainier, lower psnr than us), so an ungated pull drags colour and grain
            with torch.no_grad():
                tg = ((t_out - gt).abs().mean(2, keepdim=True) < (y - gt).abs().mean(2, keepdim=True)).float()
            per = (torch.sqrt((y - t_out) ** 2 + 1e-6) * w * tg).sum(dim=(1, 2, 3, 4)) / (w.sum(dim=(1, 2, 3, 4)) * 3).clamp(min=1)
            L["out"] = (out_w * per).mean()
            stats.setdefault("teacher_gate", []).append(float(tg.mean()))
        else:
            L["out"] = sum(out_w[i] * charbonnier(y[i:i + 1], t_out[i:i + 1], w[i:i + 1]) for i in range(n)) / n
        if a.tone > 0:
            # mean offset of the obscured area per clip and channel against the true frames, every clip: exactly what the
            # "colour" check measures (brightness + tint). The model had drifted darker there (manga -1.7 levels)
            off = ((y - gt) * m).sum(dim=(1, 3, 4)) / m.sum(dim=(1, 3, 4)).clamp(min=1)  # (n, 3)
            L["tone"] = a.tone * off.abs().mean()
            stats.setdefault("tone_lv", []).append(float(off.detach().abs().mean() * 255))
        if a.edge > 0:
            L["edge"] = y.new_zeros(())
        if a.edge > 0 and is_anime.any():
            # animation: edge strength within ~3 px of the true ink lines inside the obscured region, ours vs the truth's, per clip;
            # only a shortfall counts (a sharp line a little off its true place is not punished, a blurred one is)
            sel = is_anime.nonzero().flatten()
            ya, ga = y[sel].flatten(0, 1), gt[sel].flatten(0, 1)
            zone = (F.max_pool2d(dark_lines(ga), 7, 1, 3) * m[sel].flatten(0, 1)).view(len(sel), t, 1, *gt.shape[-2:])
            ey = (sobel(ya).view_as(zone) * zone).sum(dim=(1, 2, 3, 4))
            eg = (sobel(ga).view_as(zone) * zone).sum(dim=(1, 2, 3, 4))
            has = zone.sum(dim=(1, 2, 3, 4)) > 64
            if has.any():
                ratio = ey[has] / eg[has].clamp(min=1e-6)
                L["edge"] = a.edge * F.relu(1 - ratio).mean()
                stats.setdefault("edge_ratio", []).append(float(ratio.detach().mean()))
        if a.out_gated > 0:
            L["out_g"] = y.new_zeros(())
        if a.out_gated > 0 and is_anime.any():
            # follow the teacher only inside the obscured region and only where it is closer to the truth than we are right now
            with torch.no_grad():
                gate = (((t_out - gt).abs().mean(2, keepdim=True) < (y - gt).abs().mean(2, keepdim=True)).float() * m
                        * is_anime.view(n, 1, 1, 1, 1).float())
            d = torch.sqrt((y - t_out) ** 2 + 1e-6)
            L["out_g"] = a.out_gated * (d * w * gate).sum() / (w.sum() * 3).clamp(min=1)
            stats.setdefault("out_g", []).append(float(L["out_g"].detach()))
            stats.setdefault("gate", []).append(float(gate.sum() / (m * is_anime.view(n, 1, 1, 1, 1).float()).sum().clamp(min=1)))
        if use_gan:
            logit = disc(y_d)
            L["gan"] = (F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit), reduction="none").mean(dim=(1, 2, 3)) * gan_w).mean()
        weights = {"gt": a.gt, "out": 1.0, "out_g": 1.0, "keep": 1.0, "chroma": 1.0, "tone": 1.0, "edge": 1.0, "perc": a.perc, "gan": a.gan}  # own weights
        total = sum(weights[k] * v for k, v in L.items())
        if not torch.isfinite(total):
            bad += 1
            print(f"WARN step {it}: non-finite loss ({bad} in a row), step skipped", flush=True)
            if bad >= MAX_BAD:
                print("ABORT: losses keep going non-finite; last.pt / best.pt are from before", flush=True)
                break
            continue
        bad = 0
        opt_g.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt_g.step()

        # discriminator: true frames -> 1, ours -> 0 (skips its update while it is winning too easily)
        disc.requires_grad_(True)
        lr_ = disc(gt_d)
        lf_ = disc(y_d.detach().clamp(0, 1))
        d_loss = F.binary_cross_entropy_with_logits(lr_, torch.ones_like(lr_)) + F.binary_cross_entropy_with_logits(lf_, torch.zeros_like(lf_))
        with torch.no_grad():
            pr, pf = torch.sigmoid(lr_).mean(dim=(1, 2, 3)), torch.sigmoid(lf_).mean(dim=(1, 2, 3))
            d_gap = 0.98 * d_gap + 0.02 * float(pr.mean() - pf.mean())
        if d_gap < D_GAP_MAX and torch.isfinite(d_loss):
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
        else:
            d_skips += 1

        with torch.no_grad():
            for pe, ps in zip(ema.parameters(), student.parameters()):
                pe.lerp_(ps, 1 - a.ema)
            for be, bs in zip(ema.buffers(), student.buffers()):
                be.copy_(bs)
            # what the discriminator thinks, per domain (real-looking = high): watch animation vs real drift apart
            for dom, sel in (("real", frame_anime == 0), ("anime", frame_anime == 1)):
                if sel.any():
                    stats.setdefault(f"d_{dom}_true", []).append(float(pr[sel].mean()))
                    stats.setdefault(f"d_{dom}_ours", []).append(float(pf[sel].mean()))
        it += 1
        if it % 50 == 0:
            row = {"step": it, "lr": a.lr * lr_scale, "loss": float(total.detach()), **{k: float(v.detach()) for k, v in L.items()},
                   "d_loss": float(d_loss.detach())}
            # gated teacher loss + gate share: averages over the steps that had animation clips
            for k in (("d_real_true", "d_real_ours", "d_anime_true", "d_anime_ours") + (("out_g", "gate") if a.out_gated > 0 else ())
                      + (("keep_gate",) if keep_net is not None else ()) + (("ls_px",) if a.line_sharp > 0 else ())
                      + (("perc_anime",) if a.anime_perc > 0 else ()) + (("tint",) if a.chroma > 0 else ())
                      + (("tone_lv",) if a.tone > 0 else ()) + (("teacher_gate",) if a.out_gate_real else ())):
                row[k] = float(np.mean(stats[k])) if stats.get(k) else float("nan")
            row["d_skips"] = d_skips
            row["s_per_it"] = (time.time() - t0) / 50
            row["gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
            stats, d_skips = {}, 0
            if writer is None:
                if log.tell() > 0:
                    with open(log.name, newline="") as fh:
                        old = next(csv.reader(fh), [])
                    if old != list(row.keys()):
                        # resumed with other losses (new columns): a new file instead of rows under the old header
                        log.close()
                        log = open(os.path.join(out_dir, f"log_from_{it}.csv"), "a", newline="")
                writer = csv.DictWriter(log, fieldnames=list(row.keys()))
                if log.tell() == 0:
                    writer.writeheader()
            writer.writerow({k: row.get(k, float("nan")) for k in writer.fieldnames})
            log.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            if it % 250 == 0:  # a sudden power-off loses at most ~8 min
                save(last)
            if it % a.val_every == 0 or it == a.iters:
                v = {"step": it, **evaluate(ema, vgg, sets, out_dir, it, anet=anet)}
                sc, broken, short = score(v, base, tuple(a.info_sets))
                blocking = [b for b in broken if not b.endswith("(info)")]
                v["score"], v["anime_shortfall"], v["floors_broken"] = sc, short, ";".join(broken)
                if not blocking and sc > best:  # a check that breaks any floor is never kept: stage 1 / --ref stay the fallback
                    best = sc
                    torch.save({"student": ema.state_dict(), "step": it, "metrics": v, "channels": init["channels"], "blocks": init["blocks"]},
                               os.path.join(out_dir, "best.pt"))
                save(last)
                if vwriter is None:
                    if vlog.tell() > 0:
                        with open(vlog.name, newline="") as fh:
                            old = next(csv.reader(fh), [])
                        if old != list(v.keys()):
                            # resumed with other val sets (new columns): a new file, as for log.csv
                            vlog.close()
                            vlog = open(os.path.join(out_dir, f"val_from_{it}.csv"), "a", newline="")
                    vwriter = csv.DictWriter(vlog, fieldnames=list(v.keys()))
                    if vlog.tell() == 0:
                        vwriter.writeheader()
                vwriter.writerow(v)
                vlog.flush()
                print("VAL " + " ".join(f"{k}={x:.4g}" if isinstance(x, float) else f"{k}={x}" for k, x in v.items()), flush=True)
                broken_run = broken_run + 1 if blocking else 0
                if broken_run >= 3:
                    print(f"WARN: floors broken at 3 checks running ({v['floors_broken']}); best.pt is still the last good one", flush=True)
                if "anime_train_psnr" in v:
                    doms = [d for d in ("manga", "classic", "tv") if f"{d}_psnr" in v]
                    gap = v["anime_train_psnr"] - np.nanmean([v[f"{d}_psnr"] for d in doms])
                    gap0 = base["stage1"]["anime_train_psnr"] - np.nanmean([base["stage1"][f"{d}_psnr"] for d in doms])
                    if gap - gap0 > 1.0:
                        print(f"WARN: animation train / val gap grew by {gap - gap0:.2f} dB (memorising the stills)", flush=True)
            t0 = time.time()
    if not os.path.exists(os.path.join(out_dir, "best.pt")):
        print("NOTE: no check held every floor, so there is no best.pt (nothing here beats stage 1 / --ref)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
