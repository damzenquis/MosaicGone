"""YOLOX-Nano (Megvii YOLOX, Apache-2.0 architecture) written for this project, plus a mask head.

Parameter names match Megvii's `YOLOX(YOLOPAFPN, YOLOXHead)`, so the official `yolox_nano.pth` loads into it.
The mask head (prototype masks + per-detection coefficients, the same idea as YOLACT / YOLO-seg) is ours and
starts untrained; it gives the app the *shape* of each obscured area, not just a box.

Input: (n, 3, H, W) BGR, values 0..255 (YOLOX convention), H and W multiples of 32.
Training output: per level raw maps; `decode()` turns them into boxes / scores / mask coefficients.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseConv(nn.Module):
    def __init__(self, cin, cout, k, stride, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, (k - 1) // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=1e-3, momentum=0.03)

    def forward(self, x):
        return F.silu(self.bn(self.conv(x)))


class DWConv(nn.Module):
    def __init__(self, cin, cout, k, stride=1):
        super().__init__()
        self.dconv = BaseConv(cin, cin, k, stride, groups=cin)
        self.pconv = BaseConv(cin, cout, 1, 1)

    def forward(self, x):
        return self.pconv(self.dconv(x))


class Bottleneck(nn.Module):
    def __init__(self, cin, cout, shortcut=True):
        super().__init__()
        self.conv1 = BaseConv(cin, cout, 1, 1)
        self.conv2 = DWConv(cout, cout, 3, 1)
        self.use_add = shortcut and cin == cout

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y + x if self.use_add else y


class CSPLayer(nn.Module):
    def __init__(self, cin, cout, n=1, shortcut=True):
        super().__init__()
        h = int(cout * 0.5)
        self.conv1 = BaseConv(cin, h, 1, 1)
        self.conv2 = BaseConv(cin, h, 1, 1)
        self.conv3 = BaseConv(2 * h, cout, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(h, h, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), 1))


class SPPBottleneck(nn.Module):
    def __init__(self, cin, cout, ks=(5, 9, 13)):
        super().__init__()
        h = cin // 2
        self.conv1 = BaseConv(cin, h, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(k, 1, k // 2) for k in ks])
        self.conv2 = BaseConv(h * (len(ks) + 1), cout, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.conv = BaseConv(cin * 4, cout, k, 1)

    def forward(self, x):
        tl, tr = x[..., ::2, ::2], x[..., ::2, 1::2]
        bl, br = x[..., 1::2, ::2], x[..., 1::2, 1::2]
        return self.conv(torch.cat((tl, bl, tr, br), 1))


class CSPDarknet(nn.Module):
    def __init__(self, depth=0.33, width=0.25):
        super().__init__()
        c = int(width * 64)
        d = max(round(depth * 3), 1)
        self.stem = Focus(3, c, 3)
        self.dark2 = nn.Sequential(DWConv(c, c * 2, 3, 2), CSPLayer(c * 2, c * 2, d))
        self.dark3 = nn.Sequential(DWConv(c * 2, c * 4, 3, 2), CSPLayer(c * 4, c * 4, d * 3))
        self.dark4 = nn.Sequential(DWConv(c * 4, c * 8, 3, 2), CSPLayer(c * 8, c * 8, d * 3))
        self.dark5 = nn.Sequential(DWConv(c * 8, c * 16, 3, 2), SPPBottleneck(c * 16, c * 16), CSPLayer(c * 16, c * 16, d, shortcut=False))

    def forward(self, x):
        x = self.dark2(self.stem(x))
        d3 = self.dark3(x)
        d4 = self.dark4(d3)
        return d3, d4, self.dark5(d4)


class YOLOPAFPN(nn.Module):
    def __init__(self, depth=0.33, width=0.25):
        super().__init__()
        c3, c4, c5 = int(256 * width), int(512 * width), int(1024 * width)
        n = round(3 * depth)
        self.backbone = CSPDarknet(depth, width)
        self.lateral_conv0 = BaseConv(c5, c4, 1, 1)
        self.C3_p4 = CSPLayer(2 * c4, c4, n, False)
        self.reduce_conv1 = BaseConv(c4, c3, 1, 1)
        self.C3_p3 = CSPLayer(2 * c3, c3, n, False)
        self.bu_conv2 = DWConv(c3, c3, 3, 2)
        self.C3_n3 = CSPLayer(2 * c3, c4, n, False)
        self.bu_conv1 = DWConv(c4, c4, 3, 2)
        self.C3_n4 = CSPLayer(2 * c4, c5, n, False)

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)
        fpn0 = self.lateral_conv0(x0)
        f0 = self.C3_p4(torch.cat([F.interpolate(fpn0, scale_factor=2, mode="nearest"), x1], 1))
        fpn1 = self.reduce_conv1(f0)
        p2 = self.C3_p3(torch.cat([F.interpolate(fpn1, scale_factor=2, mode="nearest"), x2], 1))
        p1 = self.C3_n3(torch.cat([self.bu_conv2(p2), fpn1], 1))
        p0 = self.C3_n4(torch.cat([self.bu_conv1(p1), fpn0], 1))
        return p2, p1, p0  # strides 8, 16, 32


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=0.25, num_protos=0):
        super().__init__()
        cins = [int(c * width) for c in (256, 512, 1024)]
        h = int(256 * width)
        self.num_classes = num_classes
        self.num_protos = num_protos
        self.strides = (8, 16, 32)
        self.stems = nn.ModuleList([BaseConv(c, h, 1, 1) for c in cins])
        self.cls_convs = nn.ModuleList([nn.Sequential(DWConv(h, h, 3), DWConv(h, h, 3)) for _ in cins])
        self.reg_convs = nn.ModuleList([nn.Sequential(DWConv(h, h, 3), DWConv(h, h, 3)) for _ in cins])
        self.cls_preds = nn.ModuleList([nn.Conv2d(h, num_classes, 1) for _ in cins])
        self.reg_preds = nn.ModuleList([nn.Conv2d(h, 4, 1) for _ in cins])
        self.obj_preds = nn.ModuleList([nn.Conv2d(h, 1, 1) for _ in cins])
        for conv in list(self.cls_preds) + list(self.obj_preds):
            nn.init.constant_(conv.bias, -4.6)  # prior 0.01, as YOLOX initialises them
        if num_protos:
            # ours: mask coefficients per location (from the box branch) + prototypes at stride 4, and a whole-frame
            # "obscured here" map at stride 4 (the mask the app thresholds)
            self.coef_preds = nn.ModuleList([nn.Conv2d(h, num_protos, 1) for _ in cins])
            self.proto = nn.Sequential(BaseConv(cins[0], h, 3, 1), nn.Upsample(scale_factor=2, mode="nearest"), BaseConv(h, h, 3, 1))
            self.proto_out = nn.Conv2d(h, num_protos, 1)
            self.sem_out = nn.Conv2d(h, 1, 1)
            nn.init.constant_(self.sem_out.bias, -4.0)

    def forward(self, feats):
        outs = []
        for k, x in enumerate(feats):
            x = self.stems[k](x)
            c = self.cls_convs[k](x)
            r = self.reg_convs[k](x)
            parts = [self.reg_preds[k](r), self.obj_preds[k](r), self.cls_preds[k](c)]
            if self.num_protos:
                parts.append(self.coef_preds[k](r))
            outs.append(torch.cat(parts, 1))  # (n, 4 + 1 + classes [+ protos], h, w)
        if not self.num_protos:
            return outs, None
        p = self.proto(feats[0])
        return outs, (self.proto_out(p), self.sem_out(p))  # prototypes (n, P, H/4, W/4), semantic logits (n, 1, H/4, W/4)


class YOLOX(nn.Module):
    """Nano: depth 0.33, width 0.25. [num_protos] > 0 adds the mask head (ours)."""

    def __init__(self, num_classes=80, depth=0.33, width=0.25, num_protos=0):
        super().__init__()
        self.backbone = YOLOPAFPN(depth, width)
        self.head = YOLOXHead(num_classes, width, num_protos)

    def forward(self, x):
        return self.head(self.backbone(x))

    def decode(self, outs):
        """Raw maps -> (n, anchors, 4 + 1 + classes [+ protos]) with boxes as cx, cy, w, h in input pixels and
        sigmoid scores; also the grid cell centres and strides (for training)."""
        rows = []
        for out, s in zip(outs, self.head.strides):
            n, ch, h, w = out.shape
            gy, gx = torch.meshgrid(torch.arange(h, device=out.device), torch.arange(w, device=out.device), indexing="ij")
            grid = torch.stack((gx, gy), -1).view(1, h * w, 2).to(out.dtype)
            o = out.flatten(2).permute(0, 2, 1)
            xy = (o[..., :2] + grid) * s
            wh = torch.exp(o[..., 2:4].clamp(max=10)) * s
            rows.append(torch.cat([xy, wh, o[..., 4:5].sigmoid(), o[..., 5:5 + self.head.num_classes].sigmoid(), o[..., 5 + self.head.num_classes:]], -1))
        return torch.cat(rows, 1)


def load_yolox_nano(path, num_classes=80, num_protos=0):
    """Loads the official COCO weights. With a different [num_classes] the class layer starts fresh;
    the mask head (num_protos > 0) is always new."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck.get("model", ck)
    net = YOLOX(num_classes=num_classes, num_protos=num_protos)
    own = net.state_dict()
    use = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(set(sd) - set(use))
    missing = sorted(set(own) - set(use))
    net.load_state_dict(use, strict=False)
    return net, skipped, missing
