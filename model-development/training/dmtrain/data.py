"""Training samples made on the fly from the clean datasets in training/data/TRAINING DATA (not in Git).

Sources (indexed once into training/data/index.json):
  reds   REDS train_sharp: 240 clips x 100 frames, 1280x720 (CC BY 4.0)
  ytvos  YouTube-VOS 2019 train: 3471 videos (every 5th frame, 720p) with per-object masks
  srb    super-resolution benchmark stills (BSD, DIV2K, manga109, urban100, ...), animated by a smooth camera move
  toy    REDS VSR toy crops (dry runs only)
  anime  animation stills (restorer stage 2), animated like srb; see anime_stills. Styles: TRAINING DATA/Animation/<style>
         (manga, classic) and TRAINING DATA/New_Dataset (style "tv": 720p TV / film anime frames, ~10 per source video)
Held out for validation (never trained on): REDS clips 000-011, the first 120 YouTube-VOS videos, the srb sets
Set5 / Set14 / Kodak24, every 10th duplicate group of manga / classic, and every 10th source video of tv (plus any
train still that is a near-duplicate of a held-out one).

RestorerClips   -> (lq, gt, mask): T frames of a 256x256 crop around an obscured region (lq = obscured + damaged)
DetectorImages  -> (image, semantic mask, instance boxes/masks) at S x S, ~25% without any obscured region (negatives)
"""
import glob
import json
import math
import os
import re

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import synth

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "training", "data", "TRAINING DATA")
INDEX = os.path.join(ROOT, "training", "data", "index.json")
VAL_SRB = {"Set5", "Set14", "Kodak24"}


def worker_init(_):
    """DataLoader workers: one CPU thread each. Otherwise every worker's OpenCV / torch starts a thread per core and
    8 workers x 32 threads fight over the CPU (seen: 100% CPU, GPU idle, 6 s per step)."""
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def build_index():
    idx = {"reds": [], "ytvos": [], "srb": [], "toy": []}
    for d in sorted(glob.glob(os.path.join(DATA, "train_sharp", "*"))):
        idx["reds"].append({"name": os.path.basename(d), "frames": sorted(glob.glob(os.path.join(d, "*.png")))})
    for d in sorted(glob.glob(os.path.join(DATA, "YTVOS", "JPEGImages", "*"))):
        v = os.path.basename(d)
        fr = sorted(glob.glob(os.path.join(d, "*.jpg")))
        an = [os.path.join(DATA, "YTVOS", "Annotations", v, os.path.basename(f)[:-4] + ".png") for f in fr]
        if all(os.path.exists(a) for a in an):
            idx["ytvos"].append({"name": v, "frames": fr, "masks": an})
    for f in sorted(glob.glob(os.path.join(DATA, "SuperResolutionBenchmark", "**", "*.*"), recursive=True)):
        if f.lower().endswith((".png", ".jpg", ".bmp")):
            idx["srb"].append({"name": os.path.relpath(f, os.path.join(DATA, "SuperResolutionBenchmark")).split(os.sep)[0], "path": f})
    for d in sorted(glob.glob(os.path.join(DATA, "REDSVSRTOY", "hr", "*"))):
        idx["toy"].append({"name": os.path.basename(d), "frames": sorted(glob.glob(os.path.join(d, "*.png")))})
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    with open(INDEX, "w") as fh:
        json.dump(idx, fh)
    return idx


ANIM_DIR = "Animation"  # TRAINING DATA/Animation/<style>/*  (style = sub-folder: manga, classic)
TV_DIR = "New_Dataset"  # TRAINING DATA/New_Dataset/<source>_<n>[_<m>].png  (style tv; group = source video)
ANIM_INDEX = os.path.join(ROOT, "training", "data", "animation_index.json")
ANIM_MIN_SHORT = 250    # px; camera_move needs short / 1.25 >= 200
ANIM_MAX_JPEG = 1.5     # 8x8 grid strength; above it the JPEG blocks would be learnt as content
# frame selection by image complexity (tv only; manga / classic were picked by hand). The tv frames were already
# ranked by an image-complexity model when the set was made (APISR: top 10 I-frames per video), so frames are not
# dropped by rank (that removed good character shots): only near-empty ones are skipped, and the rest are drawn more
# often the more detailed they are (edge density + ink-line share at 720p, our own measure; see RestorerClips).
TV_MIN_EDGE = 0.015     # Canny edge share below this = near-empty frame (sky, fade, title card), skipped
TV_PICK = (0.5, 2.0)    # draw weight = complexity / median complexity, clipped to this range
LINE_T_U8 = 51          # dark-stroke Laplacian > this (0.2 x 255) = ink line (as train_restorer_s2.dark_lines)


def read_still(path):
    """RGB; transparency composited on white; works with any characters in the path."""
    raw = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(path)
    if raw.dtype == np.uint16:  # 16-bit PNGs
        raw = (raw >> 8).astype(np.uint8)
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    elif raw.shape[2] == 4:
        a = raw[..., 3:].astype(np.float32) / 255
        raw = (raw[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


def _dhash(g):
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits((s[:, 1:] > s[:, :-1]).flatten()).tobytes(), "big")


def _jpeg_grid(g):
    d = np.abs(np.diff(g.astype(np.float32), axis=1))
    return float(d[:, 7::8].mean() / (np.delete(d, np.s_[7::8], axis=1).mean() + 1e-6))


CX_VERSION = 2  # bump when _complexity changes (the index is rebuilt)


def _complexity(g):
    """(edge share, ink-line share) of a grey still at 720p short side. Edges after stretching the still's own 1-99%
    grey range, so a detailed night / dark scene is not taken for a simple one (obscured scenes are often dark)."""
    g = resize_short(g, 720) if min(g.shape) > 720 else g
    lo, hi = np.percentile(g, (1, 99))
    gs = np.clip((g.astype(np.float32) - lo) * 255 / max(hi - lo, 16), 0, 255).astype(np.uint8)
    edge = float((cv2.Canny(gs, 60, 150) > 0).mean())
    lap = cv2.filter2D(gs.astype(np.float32), -1, np.float32([[0, 1, 0], [1, -4, 1], [0, 1, 0]]))
    return edge, float((lap > LINE_T_U8).mean())


def _tv_source(name):
    """anime104_3_0.png -> anime104, your_name.mkv_79.png -> your_name.mkv (the video the frame came from)."""
    return re.sub(r"_\d+(_\d+)?\.[a-zA-Z]+$", "", name)


def _select_tv(stills):
    """Frame selection by complexity (tv): near-empty frames are skipped."""
    for s in stills:
        if s["style"] == "tv" and s["keep"] and s["edge"] < TV_MIN_EDGE:
            s["keep"], s["why"] = False, "simple"
    return stills


def anime_stills():
    """Animation stills with quality checks, duplicate groups and complexity, cached in animation_index.json (rebuilt
    when the file list changes; the complexity selection is applied on load, so its thresholds can change freely).
    manga / classic: near-duplicates (same picture re-saved / copied) share a group, so a group is wholly train or
    wholly val. tv: the group is the source video. Skipped: short side < ANIM_MIN_SHORT, JPEG grid > ANIM_MAX_JPEG,
    simple tv frames (_select_tv)."""
    roots = [(os.path.join(DATA, ANIM_DIR), None), (os.path.join(DATA, TV_DIR), "tv")]
    files = []
    for root, style in roots:
        files += [(f, root, style) for f in sorted(glob.glob(os.path.join(root, "**", "*.*"), recursive=True))
                  if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))]
    if not files:
        return []
    if os.path.exists(ANIM_INDEX):
        with open(ANIM_INDEX) as fh:
            cached = json.load(fh)
        if [s["path"] for s in cached] == [f for f, _, _ in files] and all(s.get("cxv") == CX_VERSION for s in cached):
            return [s for s in _select_tv(cached) if s["keep"]]
    out, hashes = [], []
    for f, root, style in files:
        rel = os.path.relpath(f, root)
        if style is None:
            style = rel.split(os.sep)[0] if os.sep in rel else "misc"
        try:
            g = cv2.cvtColor(read_still(f), cv2.COLOR_RGB2GRAY)
        except IOError:
            out.append({"path": f, "name": os.path.basename(f), "style": style, "keep": False, "why": "unreadable", "group": -1,
                        "hash": 0, "edge": 0.0, "line": 0.0, "cxv": CX_VERSION})
            continue
        h = _dhash(g)
        if style == "tv":
            group = _tv_source(os.path.basename(f))
        else:
            group = next((grp for hh, grp in hashes if bin(hh ^ h).count("1") <= 4), None)
            if group is None:
                group = len(hashes)
            hashes.append((h, group))
        edge, line = _complexity(g)
        why = "small" if min(g.shape) < ANIM_MIN_SHORT else "jpeg" if _jpeg_grid(g) > ANIM_MAX_JPEG else ""
        out.append({"path": f, "name": os.path.basename(f), "style": style, "keep": not why, "why": why, "group": group,
                    "hash": h, "edge": edge, "line": line, "cxv": CX_VERSION})
    os.makedirs(os.path.dirname(ANIM_INDEX), exist_ok=True)
    with open(ANIM_INDEX, "w") as fh:
        json.dump(out, fh, indent=0)
    return [s for s in _select_tv(out) if s["keep"]]


def load_index():
    if not os.path.exists(INDEX):
        idx = build_index()
    else:
        with open(INDEX) as fh:
            idx = json.load(fh)
    idx["anime"] = anime_stills()
    return idx


def anime_split(stills, val):
    """Every 10th group of each style (duplicate group; tv: source video) is held out for validation. A train still
    that is a near-duplicate of a held-out one (the same shot in two sources) is dropped from train."""
    out, held_hashes = [], []
    for style in sorted({s["style"] for s in stills}):
        groups = sorted({s["group"] for s in stills if s["style"] == style})  # one type per style (int / source name)
        held = set(groups[::10])
        mine = [s for s in stills if s["style"] == style]
        held_hashes += [s.get("hash", 0) for s in mine if s["group"] in held]
        out += [s for s in mine if (s["group"] in held) == val]
    if not val:
        out = [s for s in out if not any(bin(s.get("hash", 0) ^ h).count("1") <= 4 for h in held_hashes)]
    return out


def split(idx, part):
    """part = train | val"""
    val = part == "val"
    return {
        "reds": [c for c in idx["reds"] if (int(c["name"]) < 12) == val],
        "ytvos": idx["ytvos"][:120] if val else idx["ytvos"][120:],
        "srb": [s for s in idx["srb"] if (s["name"] in VAL_SRB) == val],
        "toy": idx["toy"][:8] if val else idx["toy"][8:],
        "anime": anime_split(idx.get("anime", []), val),
    }


def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_short(img, short, interp=cv2.INTER_AREA):
    h, w = img.shape[:2]
    s = short / min(h, w)
    return img if abs(s - 1) < 1e-3 else cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


def camera_move(img, t, rng, out_hw):
    """A still turned into t frames by a smooth random pan / zoom / slight rotation (srb stills)."""
    h, w = img.shape[:2]
    oh, ow = out_hw
    a0 = np.array([rng.uniform(-0.1, 0.1) * w, rng.uniform(-0.1, 0.1) * h, rng.uniform(1.0, 1.15), rng.uniform(-3, 3)])
    a1 = a0 + np.array([rng.uniform(-0.15, 0.15) * w, rng.uniform(-0.15, 0.15) * h, rng.uniform(-0.1, 0.1), rng.uniform(-3, 3)])
    mats = []
    for i in range(t):
        k = i / max(1, t - 1)
        dx, dy, z, ang = a0 * (1 - k) + a1 * k
        s = z * max(oh / h, ow / w)
        # pan only as far as the zoom leaves picture beyond the frame (minus what the rotation needs), no mirrored edges
        r = abs(math.sin(math.radians(ang)))
        room_x = max(0.0, (w * s - ow) / 2 - r * oh / 2)
        room_y = max(0.0, (h * s - oh) / 2 - r * ow / 2)
        dx, dy = float(np.clip(dx, -room_x, room_x)), float(np.clip(dy, -room_y, room_y))
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
        m[0, 2] += (ow - w) / 2 + dx
        m[1, 2] += (oh - h) / 2 + dy
        mats.append(m)
    return [cv2.warpAffine(img, m, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for m in mats], mats


def ytvos_object(masks_ids, rng, min_frac=0.003):
    """Picks an object visible in the first frame (YouTube-VOS masks hold one id per object)."""
    ids, counts = np.unique(masks_ids[0], return_counts=True)
    ok = [i for i, c in zip(ids, counts) if i != 0 and c > min_frac * masks_ids[0].size]
    if not ok:
        return None
    oid = rng.choice(ok)
    return [((m == oid).astype(np.uint8) * 255) for m in masks_ids]


# ---------------------------------------------------------------- restorer


def crop_box(masks, rng, ex_range=(1.15, 1.6)):
    """One square crop per frame around the region, centre / size smoothed over the clip (as the app does)."""
    boxes = []
    for m in masks:
        ys, xs = np.nonzero(m)
        boxes.append(None if len(xs) == 0 else (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    known = [b for b in boxes if b is not None]
    if not known:
        return None
    last = known[0]
    filled = []
    for b in boxes:
        last = b or last
        filled.append(last)
    c = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in filled])
    s = np.array([max(b[2] - b[0], b[3] - b[1]) for b in filled])
    k = 5
    cs = np.stack([np.convolve(np.pad(c[:, j], k // 2, mode="edge"), np.ones(k) / k, mode="valid") for j in range(2)], 1)
    size = max(32, float(np.median(s)) * rng.uniform(*ex_range))
    return cs, size


def take_crop(img, cx, cy, size, out=256, interp=None):
    h, w = img.shape[:2]
    half = size / 2
    m = np.float32([[out / size, 0, -(cx - half) * out / size], [0, out / size, -(cy - half) * out / size]])
    if interp is None:
        interp = cv2.INTER_AREA if size > out else cv2.INTER_CUBIC
    if interp == cv2.INTER_AREA:
        # warpAffine has no area filter: shrink first, then place
        s = out / size
        small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        m = np.float32([[1, 0, -(cx - half) * s], [0, 1, -(cy - half) * s]])
        return cv2.warpAffine(small, m, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.warpAffine(img, m, (out, out), flags=interp, borderMode=cv2.BORDER_REFLECT)


class RestorerClips(Dataset):
    """[t] frames per clip, 256x256 crops. weights: share of reds / ytvos / srb / anime clips.
    styles: share of each animation style among the anime clips, e.g. {"tv": 0.5} (styles not named split the rest
    by their still counts); None = every still equally likely (the old behaviour).
    line_mask: share of anime clips whose obscured region sits on the densest ink lines (faces / hair) instead of anywhere."""

    KINDS = ("reds", "ytvos", "srb", "anime")

    def __init__(self, part="train", t=16, length=200_000, weights=(0.55, 0.38, 0.07, 0.0), seed=0, toy=False, style=None,
                 styles=None, line_mask=0.0):
        self.line_mask = line_mask
        self.src = split(load_index(), part)
        if style:  # one animation style only (validation per style)
            self.src["anime"] = [s for s in self.src["anime"] if s["style"] == style]
        w = np.array(list(weights) + [0.0] * (4 - len(weights)), float)
        if not self.src["anime"]:
            w[3] = 0.0
        self.t, self.length, self.weights, self.seed, self.toy = t, length, w / w.sum() if w.sum() > 0 else w, seed, toy
        self.fixed = part == "val"
        self.by_style = None
        if styles and self.src["anime"]:
            self.by_style = {}
            for s in self.src["anime"]:
                self.by_style.setdefault(s["style"], []).append(s)
            named = {k: v for k, v in styles.items() if k in self.by_style}
            rest = [k for k in self.by_style if k not in named]
            left = max(0.0, 1.0 - sum(named.values())) if rest else 0.0
            n_rest = sum(len(self.by_style[k]) for k in rest)
            sw = {**named, **{k: left * len(self.by_style[k]) / n_rest for k in rest}}
            self.style_names = sorted(sw)
            p = np.array([sw[k] for k in self.style_names], float)
            self.style_p = p / p.sum()
            # tv: detailed, line-heavy frames are drawn more often (TV_PICK); other styles: every still alike
            self.pick_p = {}
            for k, pool in self.by_style.items():
                if k == "tv":
                    cx = np.array([s.get("edge", 0.0) + s.get("line", 0.0) for s in pool])
                    q = np.clip(cx / max(np.median(cx), 1e-6), *TV_PICK)
                    self.pick_p[k] = q / q.sum()

    def anime_still(self, rng):
        if self.by_style is None:
            return self.src["anime"][rng.integers(len(self.src["anime"]))]
        k = self.style_names[rng.choice(len(self.style_names), p=self.style_p)]
        pool = self.by_style[k]
        return pool[rng.choice(len(pool), p=self.pick_p[k])] if k in self.pick_p else pool[rng.integers(len(pool))]

    def __len__(self):
        return self.length

    def frames_for(self, rng):
        t = self.t
        if self.toy:
            c = self.src["toy"][rng.integers(len(self.src["toy"]))]
            s = int(rng.integers(0, len(c["frames"]) - t + 1))
            fr = [resize_short(read_rgb(p), 384, cv2.INTER_CUBIC) for p in c["frames"][s:s + t]]
            return fr, None, "toy"
        kind = rng.choice(self.KINDS, p=self.weights)
        if kind == "anime":
            # stills: never enlarged past their own size (an upscaled still would teach blur)
            img = read_still(self.anime_still(rng)["path"])
            h, w = img.shape[:2]
            # camera_move zooms in up to 1.25x, so the frame is at most 1/1.25 of the still
            short = int(min(min(h, w) / 1.25, rng.uniform(256, 560)))
            if short < 200:
                raise ValueError("anime still too small")
            # big stills: area-shrink first (camera_move's bilinear warp would alias screentone / halftone into moire)
            if min(h, w) > short * 1.4:
                img = resize_short(img, int(short * 1.3))
                h, w = img.shape[:2]
            long = int(max(short, min(short * rng.uniform(1.0, 1.78), max(h, w) / 1.25)))
            fr, mats = camera_move(img, t, rng, (short, long) if w >= h else (long, short))
            return fr, mats, kind
        if kind == "ytvos":
            for _ in range(10):
                v = self.src["ytvos"][rng.integers(len(self.src["ytvos"]))]
                if len(v["frames"]) < t:
                    continue
                s = int(rng.integers(0, len(v["frames"]) - t + 1))
                ids = [np.array(Image.open(p)) for p in v["masks"][s:s + t]]
                obj = ytvos_object(ids, rng)
                if obj is None:
                    continue
                return [read_rgb(p) for p in v["frames"][s:s + t]], obj, kind
            kind = "reds"
        if kind == "srb":
            img = read_rgb(self.src["srb"][rng.integers(len(self.src["srb"]))]["path"])
            short = int(rng.uniform(360, 720))
            fr, mats = camera_move(img, t, rng, (short, int(short * rng.uniform(1.0, 1.78))))
            return fr, mats, kind
        c = self.src["reds"][rng.integers(len(self.src["reds"]))]
        s = int(rng.integers(0, len(c["frames"]) - t + 1))
        return [read_rgb(p) for p in c["frames"][s:s + t]], None, "reds"

    def __getitem__(self, i):
        for attempt in range(20):
            rng = np.random.default_rng((self.seed * 1_000_003 + i * 97 + attempt) if self.fixed else None)
            try:
                frames, extra, src = self.frames_for(rng)
                # vary the resolution (so the mosaic's block size relative to the content varies too)
                if not self.toy and src != "anime" and rng.random() < 0.5:
                    short = int(rng.uniform(360, min(frames[0].shape[:2])))
                    frames = [resize_short(f, short) for f in frames]
                h, w = frames[0].shape[:2]
                # ytvos: extra = its object masks; srb / anime: extra = camera_move's 2x3 matrices (not masks)
                if src == "ytvos":
                    masks = [m if m.shape == (h, w) else cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) for m in extra]
                else:
                    frac = float(np.exp(rng.uniform(np.log(0.08), np.log(0.6))))
                    centre = None
                    if src == "anime" and self.line_mask > 0 and rng.random() < self.line_mask:
                        centre = synth.line_dense_centre(frames[0], rng, frac * min(h, w))
                    m0 = synth.random_blob(h, w, rng, frac, centre)
                    if isinstance(extra, list):  # srb: the mask moves with the camera move
                        inv0 = cv2.invertAffineTransform(extra[0])
                        masks = []
                        for mat in extra:
                            full = np.vstack([mat, [0, 0, 1]]) @ np.vstack([inv0, [0, 0, 1]])
                            masks.append(cv2.warpAffine(m0, full[:2], (w, h), flags=cv2.INTER_NEAREST))
                    else:
                        masks = synth.track_mask(frames, m0)
                masks = [synth.grow(m, rng) for m in masks]
                ys, xs = np.nonzero(masks[0])
                if len(xs) == 0:
                    continue
                region = max(xs.max() - xs.min(), ys.max() - ys.min())
                cen = synth.Obscurer(rng, min(h, w), region)
                lq, vis = [], []
                for f, m in zip(frames, masks):
                    c, v = cen.visible_mask(f, m)
                    lq.append(c)
                    vis.append(v)
                anime = src == "anime"
                if not anime:  # real footage: unchanged (same damage, same random draws as before the animation branch)
                    lq = synth.Damage(rng)(lq)
                box = crop_box(vis, rng)
                if box is None:
                    continue
                cs, size = box
                # slide the crop inside the frame where it fits (mirrored padding = invented, symmetric content)
                cs = np.stack([np.clip(cs[:, 0], size / 2, w - size / 2) if size <= w else cs[:, 0],
                               np.clip(cs[:, 1], size / 2, h - size / 2) if size <= h else cs[:, 1]], 1)
                if anime:
                    # animation: broadcast / web codec damage (synth.AnimeDamage), on the part of the frame the crops
                    # cover plus a margin (video codecs on whole frames would be ~4x slower); 16 px grid kept
                    x0 = max(0, int(cs[:, 0].min() - size / 2) - 24) // 16 * 16
                    y0 = max(0, int(cs[:, 1].min() - size / 2) - 24) // 16 * 16
                    x1, y1 = min(w, int(cs[:, 0].max() + size / 2) + 24), min(h, int(cs[:, 1].max() + size / 2) + 24)
                    dmg = synth.AnimeDamage(rng)([f[y0:y1, x0:x1] for f in lq])
                    lq = [f.copy() for f in lq]
                    for f, d in zip(lq, dmg):
                        f[y0:y1, x0:x1] = d
                gt_c = np.stack([take_crop(f, x, y, size) for f, (x, y) in zip(frames, cs)])
                lq_c = np.stack([take_crop(f, x, y, size) for f, (x, y) in zip(lq, cs)])
                m_c = np.stack([take_crop(m, x, y, size, interp=cv2.INTER_NEAREST) for m, (x, y) in zip(vis, cs)])
                if rng.random() < 0.5:
                    gt_c, lq_c, m_c = gt_c[:, :, ::-1], lq_c[:, :, ::-1], m_c[:, :, ::-1]
                return {
                    "lq": torch.from_numpy(np.ascontiguousarray(lq_c)).permute(0, 3, 1, 2),
                    "gt": torch.from_numpy(np.ascontiguousarray(gt_c)).permute(0, 3, 1, 2),
                    "mask": torch.from_numpy(np.ascontiguousarray(m_c > 0)).unsqueeze(1),
                    "kind": cen.kind,
                    "anime": src == "anime",
                }
            except (IOError, ValueError, cv2.error):
                continue
        raise RuntimeError("could not make a clip")


# ---------------------------------------------------------------- detector


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    out = np.full((size, size) + img.shape[2:], 114 if img.ndim == 3 else 0, img.dtype)
    out[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else (cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST))
    return out, r


class DetectorImages(Dataset):
    """S x S images: up to 3 obscured regions each (masks + boxes), or none (negatives with blur / look-alikes)."""

    MAX_OBJ = 4

    def __init__(self, part="train", size=416, length=400_000, weights=(0.4, 0.4, 0.2, 0.0), seed=0, toy=False, neg=0.25, neg_anime=0.4):
        self.src = split(load_index(), part)
        w = np.array(list(weights) + [0.0] * (4 - len(weights)), float)
        if not self.src["anime"]:
            w[3] = 0.0
        self.size, self.length, self.weights, self.seed, self.toy, self.neg = size, length, w / w.sum(), seed, toy, neg
        # animation stills are clean (no obscured region) more often: halftone / screentone / flat fills must read as clean
        self.neg_anime = neg_anime
        self.fixed = part == "val"

    def __len__(self):
        return self.length

    def image(self, rng):
        self.last_kind = "toy"
        if self.toy:
            c = self.src["toy"][rng.integers(len(self.src["toy"]))]
            return resize_short(read_rgb(c["frames"][rng.integers(len(c["frames"]))]), 384, cv2.INTER_CUBIC), None
        kind = rng.choice(["reds", "ytvos", "srb", "anime"], p=self.weights)
        self.last_kind = kind
        if kind == "anime":
            return read_still(self.src["anime"][rng.integers(len(self.src["anime"]))]["path"]), None
        if kind == "reds":
            c = self.src["reds"][rng.integers(len(self.src["reds"]))]
            return read_rgb(c["frames"][rng.integers(len(c["frames"]))]), None
        if kind == "ytvos":
            v = self.src["ytvos"][rng.integers(len(self.src["ytvos"]))]
            k = int(rng.integers(len(v["frames"])))
            return read_rgb(v["frames"][k]), np.array(Image.open(v["masks"][k]))
        return read_rgb(self.src["srb"][rng.integers(len(self.src["srb"]))]["path"]), None

    @staticmethod
    def tiny(img, inst, rng):
        h, w = img.shape[:2]
        long = rng.uniform(12, 40)
        short = long * rng.uniform(0.55, 1.0)
        cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
        m = np.zeros((h, w), np.uint8)
        if rng.random() < 0.5:
            cv2.ellipse(m, (int(cx), int(cy)), (max(2, int(long / 2)), max(2, int(short / 2))), rng.uniform(0, 180), 0, 360, 255, -1)
        else:
            box = cv2.boxPoints(((cx, cy), (long, short), rng.uniform(-30, 30))).astype(np.int32)
            cv2.fillPoly(m, [box], 255)
        cen = synth.Obscurer(rng, min(h, w), long)
        cen.kind, cen.block = "mosaic", int(rng.integers(3, 9))
        cen.rect = min(cen.rect, 1.3)
        cen.feather = 0
        img2, vis = cen.seen_mask(img, m)
        if not vis.any():
            return img, inst
        return img2, [(o & ~vis) for o in inst] + [vis]

    def __getitem__(self, i):
        for attempt in range(20):
            rng = np.random.default_rng((self.seed * 1_000_003 + i * 97 + attempt) if self.fixed else None)
            try:
                img, ids = self.image(rng)
                # work near the model's input size (faster, and real videos come in many sizes): short side 280-560 px
                if min(img.shape[:2]) > 280:
                    img = resize_short(img, int(rng.uniform(280, min(560, min(img.shape[:2])))))
                    if ids is not None:
                        ids = cv2.resize(ids, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                h, w = img.shape[:2]
                inst = []
                # negative: an out-of-focus or soft shot (whole frame), so smooth, textureless areas read as clean
                anime = self.last_kind == "anime"
                if rng.random() < 0.12:
                    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(1.5, 5.0))
                if rng.random() >= (self.neg_anime if anime else self.neg):
                    n = int(rng.choice([1, 1, 1, 2, 2, 3]))
                    objs = [o for o in np.unique(ids) if o != 0] if ids is not None else []
                    for _ in range(n):
                        if objs and rng.random() < 0.6:
                            m = (ids == rng.choice(objs)).astype(np.uint8) * 255
                            if m.sum() < 0.0005 * 255 * h * w:
                                continue
                        else:
                            m = synth.random_blob(h, w, rng, float(np.exp(rng.uniform(np.log(0.03), np.log(0.6)))))
                        m = synth.grow(m, rng)
                        ys, xs = np.nonzero(m)
                        if len(xs) == 0:
                            continue
                        region = max(xs.max() - xs.min(), ys.max() - ys.min())
                        cen = synth.Obscurer(rng, min(h, w), region)
                        # a blur over most of the picture looks like an out-of-focus shot, not an obscured region: big regions get a mosaic
                        if cen.kind != "mosaic" and region > 0.35 * min(h, w):
                            cen.kind = "mosaic"
                        img, vis = cen.seen_mask(img, m)
                        if not vis.any():
                            continue
                        # a later obscured region can cover an earlier one
                        inst = [(o & ~vis) for o in inst]
                        inst.append(vis)
                    # one extra tiny mosaic on top of the usual mix (12-40 px long side, 3-8 px blocks)
                    if rng.random() < 0.25:
                        img, inst = self.tiny(img, inst, rng)
                # negatives inside every image: natural blur and things that look like mosaics but are not obscured regions
                if rng.random() < 0.15:
                    img = synth.motion_blur(img, rng)
                if rng.random() < 0.15:
                    look = synth.lookalike(img, rng)
                    keep = np.zeros(img.shape[:2], bool)
                    for o in inst:
                        keep |= o > 0
                    img = np.where(keep[..., None], img, look)
                img = synth.Damage(rng, 0.8)([img])[0]
                if rng.random() < 0.5:
                    img = img[:, ::-1]
                    inst = [o[:, ::-1] for o in inst]
                lb, r = letterbox(np.ascontiguousarray(img), self.size)
                masks = np.zeros((self.MAX_OBJ, self.size, self.size), np.uint8)
                boxes = np.zeros((self.MAX_OBJ, 4), np.float32)
                n = 0
                for o in inst:
                    om, _ = letterbox(np.ascontiguousarray(o), self.size)
                    ys, xs = np.nonzero(om)
                    if len(xs) < 9 or n >= self.MAX_OBJ:
                        continue
                    masks[n] = om > 0
                    boxes[n] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
                    n += 1
                return {
                    "img": torch.from_numpy(lb).permute(2, 0, 1),  # RGB uint8
                    "masks": torch.from_numpy(masks),
                    "boxes": torch.from_numpy(boxes),
                    "n": n,
                    "anime": anime,
                }
            except (IOError, ValueError, cv2.error):
                continue
        raise RuntimeError("could not make an image")
