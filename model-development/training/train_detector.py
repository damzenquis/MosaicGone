"""Detector distillation: YOLOX-Nano + mask head (student) learns "obscured area" from synthetic ground truth and from
the Lada mosaic-detection teacher (YOLO11-seg, run on the same augmented images every step).

Losses per image (weights in LOSS):
  obj / cls   objectness + "obscured" score at the grid cells assigned to each region (centre sampling, as FCOS/YOLOX)
  box         GIoU of the predicted box at those cells
  mask        instance mask from the prototypes, inside the region's box
  sem         whole-frame obscured map (BCE + Dice) against the true union mask
  kd          the same map against the teacher's union mask (confidence-weighted): the "penalise when it deviates" part
  keep        (fine-tune only, --ref) the same map against the previous detector's, at the pixels it already got right
              (so a fine-tune can fix old mistakes but not forget what worked)
Small regions are weighted up in the sem BCE (clamp(sqrt(SMALL_REF / area), 1, 8)) so tiny obscured regions count.

Usage:  python train_detector.py --run runs/det_v1 [--iters 40000] [--batch 32] [--toy] [--workers 12]
Writes runs/<name>/{log.csv, last.pt, best.pt, val_*.jpg, report.md}.
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
import torchvision
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dmtrain import data  # noqa: E402
from dmtrain.yolox import load_yolox_nano  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # the repository
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")  # upstream weights, not in Git (../README.md)
GT = os.path.join(ROOT, "core", "testdata", "gt")
LOSS = dict(obj=1.0, cls=1.0, box=5.0, mask=2.0, sem=2.0, kd=1.0, keep=1.0)
SMALL_REF = 0.01  # region area (share of the frame) from which the sem loss starts weighting a region up
FLOOR_PENALTY = 10.0  # subtracted from the score per broken floor (see score())
STRIDES = (8, 16, 32)
LEVEL_RANGE = ((0, 96), (48, 192), (96, 1e9))  # which box sizes (max side, px) each stride handles


def collate(batch):
    return {"img": torch.stack([b["img"] for b in batch]), "masks": torch.stack([b["masks"] for b in batch]),
            "boxes": torch.stack([b["boxes"] for b in batch]), "n": torch.tensor([b["n"] for b in batch]),
            "anime": torch.tensor([b.get("anime", False) for b in batch])}


class EMA:
    def __init__(self, model, decay=0.9998):
        self.m = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 2000))
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                e.copy_(p)


def anchors(size, device):
    pts, strides = [], []
    for s in STRIDES:
        n = size // s
        gy, gx = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing="ij")
        pts.append(torch.stack([(gx.flatten() + 0.5) * s, (gy.flatten() + 0.5) * s], 1))
        strides.append(torch.full((n * n,), s, device=device, dtype=torch.float32))
    return torch.cat(pts), torch.cat(strides)


def assign(boxes, n, centres, strides):
    """For each image: index of the region each grid cell learns (-1 = background)."""
    B = boxes.shape[0]
    out = torch.full((B, centres.shape[0]), -1, dtype=torch.long, device=boxes.device)
    for b in range(B):
        k = int(n[b])
        if k == 0:
            continue
        bx = boxes[b, :k]  # (k, 4) xyxy
        cx, cy = (bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2
        side = torch.maximum(bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1])
        px, py = centres[:, 0:1], centres[:, 1:2]
        inside = (px > bx[:, 0]) & (px < bx[:, 2]) & (py > bx[:, 1]) & (py < bx[:, 3])
        near = ((px - cx).abs() < 2.5 * strides[:, None]) & ((py - cy).abs() < 2.5 * strides[:, None])
        lvl_ok = torch.zeros_like(inside)
        for s, (lo, hi) in zip(STRIDES, LEVEL_RANGE):
            lvl_ok |= (strides[:, None] == s) & (side[None] >= lo) & (side[None] < hi)
        pos = inside & near & lvl_ok
        area = ((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]))[None].expand_as(pos)
        cost = torch.where(pos, area, torch.full_like(area, float("inf")))
        best, idx = cost.min(1)
        out[b] = torch.where(torch.isfinite(best), idx, torch.full_like(idx, -1))
        # tiny regions without any cell: the nearest stride-8 cell
        for j in range(k):
            if not (out[b] == j).any():
                d = (px[:, 0] - cx[j]) ** 2 + (py[:, 0] - cy[j]) ** 2 + (strides != 8) * 1e12
                out[b, d.argmin()] = j
    return out


def giou_loss(p, g):
    ix0, iy0 = torch.maximum(p[:, 0], g[:, 0]), torch.maximum(p[:, 1], g[:, 1])
    ix1, iy1 = torch.minimum(p[:, 2], g[:, 2]), torch.minimum(p[:, 3], g[:, 3])
    inter = (ix1 - ix0).clamp(min=0) * (iy1 - iy0).clamp(min=0)
    ap = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    union = ap + ag - inter + 1e-6
    iou = inter / union
    cx0, cy0 = torch.minimum(p[:, 0], g[:, 0]), torch.minimum(p[:, 1], g[:, 1])
    cx1, cy1 = torch.maximum(p[:, 2], g[:, 2]), torch.maximum(p[:, 3], g[:, 3])
    hull = (cx1 - cx0) * (cy1 - cy0) + 1e-6
    return 1 - (iou - (hull - union) / hull)


def dice(logits, target):
    p = logits.sigmoid().flatten(1)
    t = target.flatten(1)
    return (1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).mean()


def teacher_maps(teacher, imgs_rgb, size, out):
    """Teacher's union mask at stride 4, each instance weighted by its confidence (0 where it sees nothing)."""
    bgr = [np.ascontiguousarray(i.permute(1, 2, 0).cpu().numpy()[..., ::-1]) for i in imgs_rgb]
    with torch.autocast("cuda", enabled=False):
        res = teacher.predict(bgr, imgsz=size, conf=0.1, verbose=False, half=True, retina_masks=True)
    maps = torch.zeros(len(bgr), 1, out, out, device=imgs_rgb.device)
    for b, r in enumerate(res):
        if r.masks is None or r.boxes is None or len(r.boxes) == 0:
            continue
        m = r.masks.data.float() * r.boxes.conf.float()[:, None, None]
        u = m.max(0).values[None, None]
        maps[b] = F.interpolate(u, size=(out, out), mode="area")[0]
    return maps


def losses(model, batch, teacher, size, centres, strides, ref=None):
    img = batch["img"].cuda(non_blocking=True)
    x = img.flip(1).float()  # RGB -> BGR, 0..255 (YOLOX convention)
    outs, (protos, sem) = model(x)
    pred = model.decode(outs)  # (B, A, 4+1+1+P)
    raw = torch.cat([o.flatten(2).permute(0, 2, 1) for o in outs], 1)
    B, A, _ = pred.shape
    boxes, n = batch["boxes"].cuda(), batch["n"].cuda()
    masks = batch["masks"].cuda().float()
    tgt = assign(boxes, n, centres, strides)
    pos = tgt >= 0
    npos = max(1, int(pos.sum()))
    L = {}
    L["obj"] = F.binary_cross_entropy_with_logits(raw[..., 4], pos.float(), reduction="sum") / npos
    if pos.any():
        b_idx, a_idx = pos.nonzero(as_tuple=True)
        g = boxes[b_idx, tgt[b_idx, a_idx]]
        pb = pred[b_idx, a_idx, :4]
        pxy = torch.cat([pb[:, :2] - pb[:, 2:] / 2, pb[:, :2] + pb[:, 2:] / 2], 1)
        L["box"] = giou_loss(pxy, g).mean()
        L["cls"] = F.binary_cross_entropy_with_logits(raw[b_idx, a_idx, 5], torch.ones_like(raw[b_idx, a_idx, 5]))
        # instance masks: at most 256 cells per batch
        sel = torch.randperm(len(b_idx), device=img.device)[:256]
        coef = raw[b_idx[sel], a_idx[sel], 6:]
        pr = protos[b_idx[sel]]  # (k, P, h, w)
        ml = (coef[:, :, None, None] * pr).sum(1)  # (k, h, w)
        h4 = ml.shape[-1]
        gm = F.interpolate(masks[b_idx[sel], tgt[b_idx[sel], a_idx[sel]]][:, None], size=(h4, h4), mode="area")[:, 0]
        gb = boxes[b_idx[sel], tgt[b_idx[sel], a_idx[sel]]] / 4
        yy, xx = torch.meshgrid(torch.arange(h4, device=img.device), torch.arange(h4, device=img.device), indexing="ij")
        inbox = ((xx[None] >= gb[:, 0, None, None] - 1) & (xx[None] < gb[:, 2, None, None] + 1) &
                 (yy[None] >= gb[:, 1, None, None] - 1) & (yy[None] < gb[:, 3, None, None] + 1)).float()
        bce = F.binary_cross_entropy_with_logits(ml, (gm > 0.5).float(), reduction="none")
        L["mask"] = ((bce * inbox).sum((1, 2)) / inbox.sum((1, 2)).clamp(min=1)).mean()
    else:
        zero = raw.sum() * 0
        L["box"] = L["cls"] = L["mask"] = zero
    union = F.interpolate(masks.max(1, keepdim=True).values, size=sem.shape[-2:], mode="area")
    union_t = (union > 0.5).float()
    # per-pixel weight: each region's pixels count clamp(sqrt(SMALL_REF / area), 1, 8) times (background 1)
    share = masks.flatten(2).mean(2)  # (B, MAX_OBJ) region area as a share of the frame
    rw = torch.where(share > 0, (SMALL_REF / share.clamp(min=1e-8)).sqrt().clamp(1, 8), torch.zeros_like(share))
    small = F.interpolate(masks, size=sem.shape[-2:], mode="area") > 0.25
    pw = torch.maximum((small * rw[:, :, None, None]).max(1, keepdim=True).values, torch.ones_like(union_t))
    L["sem"] = (F.binary_cross_entropy_with_logits(sem, union_t, reduction="none") * pw).mean() * 10 + dice(sem, union_t)
    if teacher is not None:
        tm = teacher_maps(teacher, img, size, sem.shape[-1])
        # animation images: the teacher misses many obscured regions there (35-43% found vs our 55-58%), so it only teaches
        # where it agrees with the label
        agree = torch.where(batch["anime"].cuda()[:, None, None, None], ((tm > 0.5).float() == union_t).float(), torch.ones_like(tm))
        L["kd"] = (F.binary_cross_entropy_with_logits(sem, tm, reduction="none") * agree).sum() / agree.sum().clamp(min=1) * 10
    else:
        L["kd"] = sem.sum() * 0
    if ref is not None:
        with torch.no_grad():
            rp = ref(x)[1][1].float().sigmoid()
        right = ((rp > 0.5).float() == union_t).float()  # where the previous detector agreed with this batch's label
        # BCE minus the reference's own entropy (= KL divergence): 0 when the student matches the reference
        q = rp.clamp(1e-6, 1 - 1e-6)
        kl = F.binary_cross_entropy_with_logits(sem.float(), rp, reduction="none") + q * q.log() + (1 - q) * (1 - q).log()
        L["keep"] = (kl * right).sum() / right.sum().clamp(min=1) * 10
    total =sum(LOSS[k] * v for k, v in L.items())
    return total, {k: float(v.detach()) for k, v in L.items()}


FA_BLOB = 24
GT_CLIPS = ("onepiece", "mid", "small", "tiny", "hand")
_gt_cache = {}


def gt_clips(size):
    """{clip: [(letterboxed mosaic frame, true mosaic area (bool, letterboxed), letterboxed clean frame)]}, loaded once."""
    if size not in _gt_cache:
        clips = {}
        for clip in GT_CLIPS:
            d = os.path.join(GT, clip)
            if not os.path.isdir(d):
                continue
            frames = []
            for f in sorted(os.listdir(os.path.join(d, "mosaic"))):
                m, c = data.read_rgb(os.path.join(d, "mosaic", f)), data.read_rgb(os.path.join(d, "gt", f))
                # compared lightly smoothed: some clips' clean and mosaic frames carry different encoding noise (hand: ~8
                # levels everywhere), which would make the whole frame count as mosaic
                ms, cs = (cv2.GaussianBlur(x.astype(np.float32), (0, 0), 2) for x in (m, c))
                diff = (np.abs(ms - cs).max(2) > 6).astype(np.uint8) * 255
                diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                true, _ = data.letterbox(diff, size)
                frames.append((data.letterbox(m, size)[0], true > 127, data.letterbox(c, size)[0]))
            clips[clip] = frames
        _gt_cache[size] = clips
    return _gt_cache[size]


ANIME_FA_MAX = 1  # held-out animation stills (clean) a checkpoint may flag at most
_anime_cache = {}


def anime_val(size):
    """Held-out animation stills (never trained on), each as (clean letterboxed, obscured letterboxed, true mask)."""
    if size not in _anime_cache:
        out = []
        rng = np.random.default_rng(5)
        for s in data.anime_split(data.anime_stills(), True):
            img = data.read_still(s["path"])
            if min(img.shape[:2]) > 280:
                img = data.resize_short(img, int(rng.uniform(280, min(560, min(img.shape[:2])))))
            h, w = img.shape[:2]
            for _ in range(20):
                m = data.synth.grow(data.synth.random_blob(h, w, rng, float(np.exp(rng.uniform(np.log(0.03), np.log(0.4))))), rng)
                ys, xs = np.nonzero(m)
                if len(xs) == 0:
                    continue
                cen = data.synth.Obscurer(rng, min(h, w), max(xs.max() - xs.min(), ys.max() - ys.min()))
                cz, vis = cen.seen_mask(img, m)
                if vis.any():
                    cz = data.synth.Damage(rng, 0.8)([cz])[0]
                    out.append((data.letterbox(np.ascontiguousarray(img), size)[0], data.letterbox(np.ascontiguousarray(cz), size)[0],
                                data.letterbox(vis.astype(np.uint8), size)[0] > 0))
                    break
        _anime_cache[size] = out
    return _anime_cache[size]


def predict_map(model, lb, size):
    _, (_, sem) = model(torch.from_numpy(lb).permute(2, 0, 1)[None].cuda().flip(1).float())
    return (F.interpolate(sem.sigmoid(), size=(size, size), mode="bilinear")[0, 0] > 0.5).cpu().numpy()


def score(v, base):
    """Checkpoint score. Floors (each broken one costs FLOOR_PENALTY, so best.pt only comes from checkpoints that pass all):
    found on every clip the reference got 100% must stay 100%; clean-frame false alarms no worse than the reference on
    any clip; synthetic false alarms no worse than the reference + 0.01. Above the floors: overlap + recall - false
    alarms, real clean-frame false alarms x3, tiny-clip finds x1."""
    gt_fa = [x for k, x in v.items() if k.startswith("gt_") and k.endswith("_fa")]
    s = v["val_iou"] + v["val_recall"] - v["val_false_alarm"] - 3 * (float(np.mean(gt_fa)) if gt_fa else 0.0) + v.get("gt_tiny", 0.0)
    broken = []
    if "anime_found" in v:  # animation stills: clean ones flagged at most ANIME_FA_MAX; found no lower than the start
        s += v["anime_found"] + v["anime_iou"] - 3 * v["anime_fa"]
        if v["anime_fa_n"] > ANIME_FA_MAX:
            broken.append("anime_fa")
        if base and "anime_found" in base and v["anime_found"] < base["anime_found"]:
            broken.append("anime_found")
    if base:
        for clip in GT_CLIPS:
            k = f"gt_{clip}"
            if k in v and base.get(k, 0) >= 1.0 and v[k] < 1.0:
                broken.append(k)
            if k + "_fa" in v and v[k + "_fa"] > base.get(k + "_fa", 1.0):
                broken.append(k + "_fa")
        if v["val_false_alarm"] > base["val_false_alarm"] + 0.01:
            broken.append("val_false_alarm")
    return s - FLOOR_PENALTY * len(broken), broken


@torch.no_grad()
def evaluate(model, val, size, out_dir, step):
    """Synthetic validation (held-out sources) + our ground-truth clips (android/core/testdata/gt)."""
    model.eval()
    ious, hits, objs, fa, negs = [], 0, 0, 0, 0
    sheet = []
    for k, s in enumerate(val):
        x = s["img"][None].cuda().flip(1).float()
        _, (_, sem) = model(x)
        p = F.interpolate(sem.sigmoid(), size=(size, size), mode="bilinear")[0, 0] > 0.5
        gt = s["masks"].cuda().max(0).values > 0
        if s["n"] == 0:
            negs += 1
            fa += int(p.float().mean() > 0.002)
        else:
            ious.append(float((p & gt).sum() / (p | gt).sum().clamp(min=1)))
            for j in range(s["n"]):
                g = s["masks"][j].cuda() > 0
                objs += 1
                hits += int((p & g).sum() >= 0.5 * g.sum())
        if k < 8:
            im = s["img"].permute(1, 2, 0).numpy().copy()
            ov = im.copy()
            ov[p.cpu().numpy()] = (0.5 * ov[p.cpu().numpy()] + [0, 127, 0]).astype(np.uint8)
            cnt, _ = cv2.findContours(gt.cpu().numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, cnt, -1, (255, 0, 0), 1)
            sheet.append(ov)
    res = {"val_iou": float(np.mean(ious)) if ious else 0.0, "val_recall": hits / max(1, objs), "val_false_alarm": fa / max(1, negs)}
    # ground-truth clips (the exam, never trained on), every frame:
    #   found       the predicted map covers at least half of that frame's true mosaic (where mosaic and clean frames differ)
    #   false alarm a clean frame with any predicted blob of FA_BLOB px or more (at the model's input size)
    for clip, frames in gt_clips(size).items():
        found = flagged = 0
        for lb_m, true, lb_c in frames:
            p = predict_map(model, lb_m, size)
            found += int((p & true).sum() >= 0.5 * true.sum())
            q = predict_map(model, lb_c, size).astype(np.uint8)
            n, _, st, _ = cv2.connectedComponentsWithStats(q, connectivity=8)
            flagged += int(n > 1 and st[1:, cv2.CC_STAT_AREA].max() >= FA_BLOB)
        res[f"gt_{clip}"] = found / len(frames)
        res[f"gt_{clip}_fa"] = flagged / len(frames)
    stills = anime_val(size)
    if stills:
        found = flagged = 0
        ious = []
        for lb_c, lb_m, true in stills:
            p = predict_map(model, lb_m, size)
            found += int((p & true).sum() >= 0.5 * true.sum())
            ious.append(float((p & true).sum() / max(1, (p | true).sum())))
            q = predict_map(model, lb_c, size).astype(np.uint8)
            n, _, st, _ = cv2.connectedComponentsWithStats(q, connectivity=8)
            flagged += int(n > 1 and st[1:, cv2.CC_STAT_AREA].max() >= FA_BLOB)
        res.update(anime_found=found / len(stills), anime_iou=float(np.mean(ious)), anime_fa=flagged / len(stills), anime_fa_n=flagged)
    if sheet:
        cv2.imwrite(os.path.join(out_dir, f"val_{step:06d}.jpg"), cv2.cvtColor(np.concatenate(sheet, 1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    model.train()
    return res


def forever(loader):
    """Keeps drawing batches (new random samples) until the step count is reached."""
    while True:
        yield from loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/det_v1")
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--no_teacher", action="store_true")
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", help="fine-tune: start from this checkpoint's EMA weights (fresh optimizer and schedule)")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ref", help="fine-tune: frozen previous detector (its checkpoint's EMA) for the keep loss and the floors")
    ap.add_argument("--anime", type=float, default=0.0, help="share of animation stills (TRAINING DATA/Animation) in training")
    a = ap.parse_args()
    out_dir = os.path.join(os.path.dirname(__file__), a.run)
    os.makedirs(out_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    model, _, _ = load_yolox_nano(os.path.join(MODELS, "yolox_nano.pth"), num_classes=1, num_protos=32)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["ema"])
    model.cuda().train()
    ref = None
    if a.ref:
        ref, _, _ = load_yolox_nano(os.path.join(MODELS, "yolox_nano.pth"), num_classes=1, num_protos=32)
        ref.load_state_dict(torch.load(a.ref, map_location="cpu", weights_only=False)["ema"])
        ref = ref.cuda().eval().requires_grad_(False)
    teacher = None
    if not a.no_teacher:
        os.environ.setdefault("YOLO_OFFLINE", "1")
        from ultralytics import YOLO
        teacher = YOLO(os.path.join(MODELS, "lada_mosaic_detection_model_v4_accurate.pt"))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=5e-4)
    ema = EMA(model)
    start, best = 0, -1e9
    last = os.path.join(out_dir, "last.pt")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"]); ema.m.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        start, best, ema.updates = ck["step"], ck["best"], ck["step"]
    scaler = torch.amp.GradScaler()
    wr = 1.0 - a.anime
    train = data.DetectorImages("train", size=a.size, toy=a.toy, weights=(0.4 * wr, 0.4 * wr, 0.2 * wr, a.anime))
    val = [data.DetectorImages("val", size=a.size, toy=a.toy, seed=1)[i] for i in range(60 if a.toy else 400)]
    loader = DataLoader(train, batch_size=a.batch, num_workers=a.workers, collate_fn=collate, persistent_workers=a.workers > 0, worker_init_fn=data.worker_init,
                        prefetch_factor=4 if a.workers else None, pin_memory=True, drop_last=True)
    centres, strides = anchors(a.size, "cuda")
    log = open(os.path.join(out_dir, "log.csv"), "a", newline="")
    vlog = open(os.path.join(out_dir, "val.csv"), "a", newline="")
    writer = vwriter = None

    def write_val(v):
        nonlocal vwriter
        if vwriter is None:
            vwriter = csv.DictWriter(vlog, fieldnames=list(v.keys()))
            if vlog.tell() == 0:
                vwriter.writeheader()
        vwriter.writerow(v)
        vlog.flush()
        print("VAL " + " ".join(f"{k}={x:.4g}" if isinstance(x, float) else f"{k}={x}" for k, x in v.items()), flush=True)

    # the reference for the floors: the starting weights measured before any fine-tuning (step-0 row in val.csv)
    base_path, base = os.path.join(out_dir, "base.json"), None
    if os.path.exists(base_path):
        with open(base_path) as fh:
            base = json.load(fh)
    elif a.init and start == 0:
        base = {"step": 0, **evaluate(model, val, a.size, out_dir, 0)}
        model.train()
        base["score"], _ = score(base, None)
        base["floors_broken"] = ""
        with open(base_path, "w") as fh:
            json.dump(base, fh, indent=1)
        write_val(base)
    t0, it = time.time(), start
    for batch in forever(loader):
        if it >= a.iters:
            break
        lr = a.lr * min(1.0, (it + 1) / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * it / a.iters)))
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.float16):
            total, parts = losses(model, batch, teacher, a.size, centres, strides, ref)
        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt)
        scaler.update()
        ema.update(model)
        it += 1
        if it % 50 == 0:
            row = {"step": it, "lr": lr, "loss": float(total), **parts, "s_per_it": (time.time() - t0) / 50}
            if writer is None:
                writer = csv.DictWriter(log, fieldnames=list(row.keys()))
                if log.tell() == 0:
                    writer.writeheader()
            writer.writerow(row)
            log.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            if it % 500 == 0:  # frequent resume point (a lost window costs at most ~4 min)
                torch.save({"model": model.state_dict(), "ema": ema.m.state_dict(), "opt": opt.state_dict(), "step": it, "best": best}, last)
            if it % a.val_every == 0 or it == a.iters:
                v = {"step": it, **evaluate(ema.m, val, a.size, out_dir, it)}
                sc, broken = score(v, base)
                v["score"], v["floors_broken"] = sc, " ".join(broken)
                torch.save({"model": model.state_dict(), "ema": ema.m.state_dict(), "opt": opt.state_dict(), "step": it, "best": max(best, sc)}, last)
                if sc > best:
                    best = sc
                    torch.save({"ema": ema.m.state_dict(), "step": it, "metrics": v}, os.path.join(out_dir, "best.pt"))
                write_val(v)
            t0 = time.time()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
