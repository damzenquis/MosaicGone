"""Synthetic obscuring and real-video damage, written for this project (no code copied from other projects).

Everything works on uint8 RGB numpy frames (h, w, 3) and uint8 masks (h, w) with 255 = obscured area.
Randomness comes from the `rng` (numpy Generator) passed in, so a sample can be reproduced.

Obscuring types (one per region, the same parameters on every frame of a clip, as real pixelation or blur would be):
  mosaic   square or rectangular blocks; block colour = block average, block centre pixel, or a random pixel;
           block grid fixed to the frame (as video editors do) or to the region; optional feathered edge
  blur     strong Gaussian or box blur inside the region, optional feathered edge
  pixel+blur  mosaic followed by a light blur (common in re-encoded uploads)
Damage (whole clip): resize down/up, Gaussian blur, noise, JPEG at a random quality per clip.
AnimeDamage (animation clips, restorer stage 2): video / web codecs, steps in a random order (see the class).
"""
import io
import os
import tempfile

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------- region masks


def random_blob(h, w, rng, size_frac, centre=None):
    """A random filled shape (ellipse, rounded rectangle or polygon) about [size_frac] of the short side, at a random
    place (or at [centre] = (x, y))."""
    m = np.zeros((h, w), np.uint8)
    s = size_frac * min(h, w)
    if centre is None:
        cx = rng.uniform(0.15, 0.85) * w
        cy = rng.uniform(0.15, 0.85) * h
    else:
        cx, cy = centre
    kind = rng.choice(["ellipse", "rect", "poly"], p=[0.45, 0.3, 0.25])
    if kind == "ellipse":
        cv2.ellipse(m, (int(cx), int(cy)), (max(2, int(s * rng.uniform(0.35, 0.7))), max(2, int(s * rng.uniform(0.35, 0.7)))),
                    rng.uniform(0, 180), 0, 360, 255, -1)
    elif kind == "rect":
        rw, rh = s * rng.uniform(0.5, 1.2), s * rng.uniform(0.5, 1.2)
        box = cv2.boxPoints(((cx, cy), (rw, rh), rng.uniform(-30, 30))).astype(np.int32)
        cv2.fillPoly(m, [box], 255)
    else:
        n = rng.integers(5, 10)
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        rad = s * 0.5 * rng.uniform(0.5, 1.0, n)
        pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).astype(np.int32)
        cv2.fillPoly(m, [pts], 255)
    return m


def line_dense_centre(img, rng, size):
    """The centre of an obscured region on drawn characters (faces, hair, hands) of an animation frame: where dark ink outlines are
    dense AND the areas between them are flat cel fills (painted backgrounds have edges too, but textured ones).
    Drawn at random with chance ~ that score^2 over [size]-wide windows, inside the 15-85% band random_blob uses."""
    h, w = img.shape[:2]
    f = 4  # work at 1/4 size
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (max(8, w // f), max(8, h // f)), interpolation=cv2.INTER_AREA)
    gf = g.astype(np.float32)
    lap = cv2.filter2D(gf, -1, np.float32([[0, 1, 0], [1, -4, 1], [0, 1, 0]]))
    k = max(3, int(size / f) | 1)
    # ink: a dark centre (area-shrunk lines are fainter: half the usual threshold) on a dark pixel
    ink = ((lap > 51 / 2) & (g < 110)).astype(np.float32)
    # texture between the lines: detail that is not ink (painted stone, foliage, clutter)
    tex = np.abs(lap) * (1 - cv2.dilate(ink, np.ones((3, 3), np.uint8)))
    dens = cv2.blur(ink, (k, k)) / (1 + cv2.blur(tex, (k, k)) / 4)
    gh, gw = dens.shape
    band = np.zeros_like(dens)
    band[int(0.15 * gh):int(0.85 * gh) + 1, int(0.15 * gw):int(0.85 * gw) + 1] = 1
    p = (dens ** 2) * band
    if p.sum() <= 0:
        return None
    i = int(rng.choice(p.size, p=(p / p.sum()).ravel()))
    return ((i % gw + 0.5) * w / gw, (i // gw + 0.5) * h / gh)


def track_mask(frames, mask0):
    """Moves a mask through a clip along the scene's own motion (DIS optical flow), like an obscured region that follows an object."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    masks = [mask0]
    prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    h, w = mask0.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    for f in frames[1:]:
        cur = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        flow = dis.calc(cur, prev, None)  # for each pixel of the current frame: where it was in the previous one
        m = cv2.remap(masks[-1], gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_NEAREST)
        # keep the shape solid (flow is noisy at edges): close small holes, keep the largest blob
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        if m.max() == 0:
            m = masks[-1]
        masks.append(m)
        prev = cur
    return masks


def grow(mask, rng, max_frac=0.25):
    """Real pixelation overshoots the area it covers: dilate by up to [max_frac] of the region size."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return mask
    size = max(xs.max() - xs.min(), ys.max() - ys.min())
    k = int(size * rng.uniform(0, max_frac))
    if k < 2:
        return mask
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


# ---------------------------------------------------------------- obscured regions


class Obscurer:
    """One obscured region's parameters, drawn once and applied identically to every frame of a clip."""

    def __init__(self, rng, frame_short, region_size):
        self.kind = rng.choice(["mosaic", "blur", "pixel_blur"], p=[0.7, 0.18, 0.12])
        # block size: like real mosaics, roughly 1/100 of the frame (Pixiv guideline) up to large blocks, never < 3 px
        base = max(3.0, frame_short / 100.0)
        self.block = int(np.clip(base * rng.uniform(0.7, 3.5), 3, max(4, region_size / 3)))
        self.rect = rng.uniform(1.0, 1.8) if rng.random() < 0.25 else 1.0
        self.colour = rng.choice(["avg", "centre", "random"], p=[0.6, 0.3, 0.1])
        self.grid_from_frame = rng.random() < 0.7
        self.feather = int(self.block * rng.uniform(0.3, 2.0)) if rng.random() < 0.6 else 0
        self.blur_sigma = region_size * rng.uniform(0.04, 0.15)
        self.box_blur = rng.random() < 0.3
        self.seed = int(rng.integers(1 << 30))

    def mosaic(self, img, mask):
        h, w = mask.shape
        bh, bw = self.block, max(1, int(self.block * self.rect))
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return img
        oy, ox = (0, 0) if self.grid_from_frame else (ys.min() % bh, xs.min() % bw)
        y0 = max(0, (ys.min() - oy) // bh * bh + oy)
        x0 = max(0, (xs.min() - ox) // bw * bw + ox)
        ny = (ys.max() + 1 - y0 + bh - 1) // bh
        nx = (xs.max() + 1 - x0 + bw - 1) // bw
        # the block grid over the region, padded at the frame edge by repeating the last row/column
        area = img[y0:y0 + ny * bh, x0:x0 + nx * bw]
        area = np.pad(area, ((0, ny * bh - area.shape[0]), (0, nx * bw - area.shape[1]), (0, 0)), mode="edge")
        blocks = area.reshape(ny, bh, nx, bw, 3)
        if self.colour == "avg":
            col = blocks.mean((1, 3))
        elif self.colour == "centre":
            col = blocks[:, bh // 2, :, bw // 2]
        else:
            r = np.random.default_rng(self.seed)
            col = blocks[np.arange(ny)[:, None], r.integers(bh, size=(ny, nx)), np.arange(nx)[None], r.integers(bw, size=(ny, nx))]
        # a block is obscured when its centre lies in the mask
        cyy = np.minimum(y0 + np.arange(ny) * bh + bh // 2, h - 1)
        cxx = np.minimum(x0 + np.arange(nx) * bw + bw // 2, w - 1)
        on = mask[cyy[:, None], cxx[None, :]] > 0
        up = np.repeat(np.repeat(col.astype(np.uint8), bh, 0), bw, 1)
        upm = np.repeat(np.repeat(on, bh, 0), bw, 1)
        out = img.copy()
        hh, ww = min(h - y0, ny * bh), min(w - x0, nx * bw)
        sub = out[y0:y0 + hh, x0:x0 + ww]
        sub[upm[:hh, :ww]] = up[:hh, :ww][upm[:hh, :ww]]
        return out

    def apply(self, img, mask):
        if self.kind == "blur":
            k = int(self.blur_sigma * 3) * 2 + 1
            cen = cv2.blur(img, (k, k)) if self.box_blur else cv2.GaussianBlur(img, (k, k), self.blur_sigma)
        else:
            cen = self.mosaic(img, mask)
            if self.kind == "pixel_blur":
                cen = cv2.GaussianBlur(cen, (0, 0), self.block * 0.35)
        # where the obscuring shows: the mask, optionally with a soft edge; for mosaics the blocks themselves set the edge
        if self.kind == "blur" or self.feather:
            a = mask.astype(np.float32) / 255
            if self.feather:
                f = self.feather * 2 + 1
                a = cv2.blur(a, (f, f))
            a = a[..., None]
            return (img * (1 - a) + cen * a).astype(np.uint8)
        changed = (cen != img).any(2) | (mask > 0)
        out = img.copy()
        out[changed] = cen[changed]
        return out

    def visible_mask(self, img, mask):
        """The area that really looks obscured (mosaic blocks snap to the grid, so they can poke out of the mask)."""
        mask = (np.asarray(mask) > 0).astype(np.uint8) * 255  # any mask type in, 0/255 uint8 inside
        c = self.apply(img, mask)
        diff = (np.abs(c.astype(np.int16) - img.astype(np.int16)).max(2) > 6).astype(np.uint8) * 255
        m = cv2.morphologyEx(diff | mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return c, m

    def seen_mask(self, img, mask, min_share=0.3):
        """Detector label: only the part of the region where the obscuring visibly changed the picture.
        A blur or mosaic over a flat area (a plain wall, sky, a desk) leaves it looking clean, so it must not be labelled
        obscured, or the detector learns "flat = obscured". Returns (image, label): the whole region (plus any block
        overshoot) when at least [min_share] of it visibly changed, otherwise empty."""
        mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
        c = self.apply(img, mask)
        diff = (np.abs(c.astype(np.int16) - img.astype(np.int16)).max(2) > 6).astype(np.uint8) * 255
        # fill between the changed pixels (a block's flat inside differs only at its edges) but stay inside the region
        k = max(5, self.block | 1) if self.kind != "blur" else 9
        seen = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        seen = cv2.dilate(seen, np.ones((3, 3), np.uint8)) & cv2.dilate(mask, np.ones((k, k), np.uint8))
        if (seen > 0).sum() < min_share * max(1, (mask > 0).sum()):
            return c, np.zeros_like(mask)
        return c, cv2.morphologyEx(seen | mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


# ---------------------------------------------------------------- damage


class Damage:
    """Real-video damage with parameters drawn once per clip."""

    def __init__(self, rng, strength=1.0):
        self.scale = rng.uniform(0.5, 1.0) if rng.random() < 0.4 * strength else 1.0
        self.blur = rng.uniform(0.3, 1.5) if rng.random() < 0.3 * strength else 0
        self.noise = rng.uniform(1, 6) if rng.random() < 0.3 * strength else 0
        self.jpeg = int(rng.uniform(30, 92)) if rng.random() < 0.8 * strength else 0
        self.seed = int(rng.integers(1 << 30))

    def __call__(self, frames):
        r = np.random.default_rng(self.seed)
        out = []
        for f in frames:
            h, w = f.shape[:2]
            g = f
            if self.blur:
                g = cv2.GaussianBlur(g, (0, 0), self.blur)
            if self.scale < 1:
                g = cv2.resize(cv2.resize(g, (max(8, int(w * self.scale)), max(8, int(h * self.scale))), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
            if self.noise:
                g = np.clip(g + r.normal(0, self.noise, g.shape), 0, 255).astype(np.uint8)
            if self.jpeg:
                ok, enc = cv2.imencode(".jpg", cv2.cvtColor(g, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg])
                g = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            out.append(g)
        return out


def _pad16(img):
    h, w = img.shape[:2]
    return np.pad(img, ((0, -h % 16), (0, -w % 16), (0, 0)), mode="edge")


def _video_codec(frames, codec):
    """frames (RGB uint8, same size) through a video codec and back; None if this OpenCV build cannot encode it."""
    fourcc, ext = {"mpeg2": ("mpg2", "mkv"), "mpeg4": ("mp4v", "mp4"), "vp8": ("VP80", "mkv"), "vp9": ("VP90", "mkv")}[codec]
    h, w = frames[0].shape[:2]
    fd, path = tempfile.mkstemp(suffix="." + ext)
    os.close(fd)
    try:
        wr = cv2.VideoWriter(path, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*fourcc), 25, (w, h))
        if not wr.isOpened():
            return None
        for f in frames:
            wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        wr.release()
        cap = cv2.VideoCapture(path)
        out = []
        for _ in frames:
            ok, g = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(g, cv2.COLOR_BGR2RGB))
        cap.release()
        return out if len(out) == len(frames) else None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _image_codec(img, codec, q):
    if codec == "avif":
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "AVIF", quality=q, speed=10)
        return np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    ext, flag = {"jpeg": (".jpg", cv2.IMWRITE_JPEG_QUALITY), "webp": (".webp", cv2.IMWRITE_WEBP_QUALITY)}[codec]
    ok, enc = cv2.imencode(ext, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [flag, q])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class AnimeDamage:
    """Damage on animation clips, after APISR (Wang et al., CVPR 2024: "prediction-oriented compression" + "shuffled
    resize"), reimplemented for a video restorer. Parameters drawn once per clip; every frame gets the same steps.
      codec   1-2 passes: JPEG / WebP / AVIF (quality 25-90), or a video codec - MPEG-2, MPEG-4 part 2, VP8 (H.264
              class: 4x4 transforms, intra prediction, deblocking), VP9 (H.265 class). Video codecs run either on each
              frame alone (intra-only, as APISR) or on the clip as one video (inter-frame prediction too, as real
              releases; APISR is single-image, we restore video). This OpenCV build has no bitrate setting, so their
              strength comes from the resolution they run at (the downscale below).
      down/up a downscale (0.45-1.0, random filter) at a random point among the steps and the upscale back at a random
              later point (APISR: resizing is not a fixed stage of how anime gets degraded)
      blur    slight (0.3-1.0 px), sometimes;  grain  slight (0.5-2 levels), rarely (film-era anime)
    Real footage keeps Damage."""

    CODECS = ("jpeg", "webp", "avif", "mpeg2", "mpeg4", "vp8", "vp9")
    CODEC_P = (0.2, 0.1, 0.1, 0.1, 0.15, 0.2, 0.15)

    def __init__(self, rng):
        steps = []
        for _ in range(1 if rng.random() < 0.6 else 2):
            steps.append(("codec", str(rng.choice(self.CODECS, p=self.CODEC_P)), int(rng.uniform(25, 90)), bool(rng.random() < 0.5)))
        if rng.random() < 0.25:
            steps.append(("blur", float(rng.uniform(0.3, 1.0))))
        if rng.random() < 0.1:
            steps.append(("grain", float(rng.uniform(0.5, 2.0))))
        steps = [steps[i] for i in rng.permutation(len(steps))]
        if rng.random() < 0.6:
            i = int(rng.integers(0, len(steps) + 1))
            j = int(rng.integers(i, len(steps) + 1))
            downs = (cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC)
            steps.insert(j, ("up", int(rng.choice((cv2.INTER_LINEAR, cv2.INTER_CUBIC)))))
            steps.insert(i, ("down", float(rng.uniform(0.45, 1.0)), int(rng.choice(downs))))
        self.steps = steps
        self.seed = int(rng.integers(1 << 30))

    def __call__(self, frames):
        r = np.random.default_rng(self.seed)
        h, w = frames[0].shape[:2]
        fr = list(frames)
        for st in self.steps:
            kind = st[0]
            if kind == "down":
                fr = [cv2.resize(f, (max(16, round(w * st[1])), max(16, round(h * st[1]))), interpolation=st[2]) for f in fr]
            elif kind == "up":
                fr = [cv2.resize(f, (w, h), interpolation=st[1]) for f in fr]
            elif kind == "blur":
                fr = [cv2.GaussianBlur(f, (0, 0), st[1]) for f in fr]
            elif kind == "grain":
                fr = [np.clip(f + r.normal(0, st[1], f.shape), 0, 255).astype(np.uint8) for f in fr]
            else:
                _, codec, q, whole_clip = st
                fh, fw = fr[0].shape[:2]
                if codec in ("jpeg", "webp", "avif"):
                    fr = [_image_codec(f, codec, q) for f in fr]
                    continue
                padded = [_pad16(f) for f in fr]
                out = _video_codec(padded, codec) if whole_clip else [(_video_codec([f], codec) or [None])[0] for f in padded]
                if out is None or any(o is None for o in out):  # codec missing in this build: JPEG instead
                    fr = [_image_codec(f, "jpeg", q) for f in fr]
                else:
                    fr = [o[:fh, :fw] for o in out]
        if fr[0].shape[:2] != (h, w):  # a downscale with no upscale after it (cannot happen; kept safe)
            fr = [cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in fr]
        return [np.ascontiguousarray(f) for f in fr]


def motion_blur(img, rng):
    """Natural-looking camera/object motion blur (a negative: the detector must not call this an obscured region)."""
    k = int(rng.integers(5, 25))
    kern = np.zeros((k, k), np.float32)
    kern[k // 2] = 1
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
    kern = cv2.warpAffine(kern, rot, (k, k))
    kern /= max(kern.sum(), 1e-6)
    return cv2.filter2D(img, -1, kern)


def lookalike(img, rng):
    """Pastes something that resembles a mosaic but is not an obscured region: a checkerboard, a pixel-art patch, a tiled texture."""
    h, w = img.shape[:2]
    s = int(min(h, w) * rng.uniform(0.1, 0.4))
    x, y = int(rng.integers(0, max(1, w - s))), int(rng.integers(0, max(1, h - s)))
    kind = rng.choice(["checker", "pixelart", "tiles"])
    cell = int(rng.integers(4, 24))
    yy, xx = np.mgrid[0:s, 0:s]
    if kind == "checker":
        a, b = rng.integers(0, 255, 3), rng.integers(0, 255, 3)
        patch = np.where((((yy // cell) + (xx // cell)) % 2)[..., None] == 0, a, b).astype(np.uint8)
    elif kind == "pixelart":
        small = rng.integers(0, 255, (s // cell + 1, s // cell + 1, 3)).astype(np.uint8)
        small = (small // 64 * 64).astype(np.uint8)  # few colours, like sprites
        patch = cv2.resize(small, (s, s), interpolation=cv2.INTER_NEAREST)
    else:
        tile = rng.integers(0, 255, (cell, cell, 3)).astype(np.uint8)
        patch = np.tile(tile, (s // cell + 1, s // cell + 1, 1))[:s, :s]
    out = img.copy()
    out[y:y + s, x:x + s] = patch
    return out
