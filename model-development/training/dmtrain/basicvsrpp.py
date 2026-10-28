"""BasicVSR++ for mosaic restoration (same-size output), written for this project.

Architecture as in the BasicVSR++ paper (Chan et al., CVPR 2022) and OpenMMLab mmagic (Apache-2.0): SPyNet flow,
two rounds of backward/forward propagation with second-order flow-guided deformable alignment, reconstruction.
Parameter names match mmagic's `BasicVSRPlusPlusNet`, so the same class loads the Lada restoration teacher
(64 channels, 15 blocks) and builds our slim students (e.g. 32 channels, 5 blocks).

Input / output: (n, t, 3, h, w), values 0..1, h and w multiples of 4 (and h/4, w/4 >= 64 for SPyNet), same size out.

`deform_mode`:
  "torchvision"  torchvision.ops.deform_conv2d (training, fast on GPU)
  "grid_sample"  the same maths built from grid_sample + a 1x1-equivalent conv, for ONNX / ONNX Runtime on Android
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_warp(x, flow, padding_mode="zeros"):
    """Warp x (n,c,h,w) by flow (n,h,w,2) in pixels (x, y)."""
    n, _, h, w = x.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype), torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
    gx = gx.unsqueeze(0) + flow[..., 0]
    gy = gy.unsqueeze(0) + flow[..., 1]
    grid = torch.stack((2.0 * gx / max(w - 1, 1) - 1.0, 2.0 * gy / max(h - 1, 1) - 1.0), dim=3)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1)
        for m in (self.conv1, self.conv2):  # scaled init for residual paths (ESRGAN / mmagic default)
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            m.weight.data *= 0.1
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, cin, c, num_blocks):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(cin, c, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                  nn.Sequential(*[ResidualBlockNoBN(c) for _ in range(num_blocks)]))

    def forward(self, x):
        return self.main(x)


class PixelShufflePack(nn.Module):
    def __init__(self, cin, cout, scale):
        super().__init__()
        self.scale = scale
        self.upsample_conv = nn.Conv2d(cin, cout * scale * scale, 3, 1, 1)

    def forward(self, x):
        return F.pixel_shuffle(self.upsample_conv(x), self.scale)


class _ConvModule(nn.Module):
    def __init__(self, cin, cout, relu):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 7, 1, 3)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x) if self.relu else x


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(_ConvModule(8, 32, True), _ConvModule(32, 64, True), _ConvModule(64, 32, True),
                                          _ConvModule(32, 16, True), _ConvModule(16, 2, False))

    def forward(self, x):
        return self.basic_module(x)


class SPyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        n, _, h, w = ref.shape
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref.append(F.avg_pool2d(ref[-1], 2, 2, count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], 2, 2, count_include_pad=False))
        ref, supp = ref[::-1], supp[::-1]
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            up = flow if level == 0 else F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            flow = up + self.basic_module[level](torch.cat([ref[level], flow_warp(supp[level], up.permute(0, 2, 3, 1), "border"), up], 1))
        return flow

    def forward(self, ref, supp):
        h, w = ref.shape[2:]
        hu, wu = (h + 31) // 32 * 32, (w + 31) // 32 * 32
        ref = F.interpolate(ref, size=(hu, wu), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(hu, wu), mode="bilinear", align_corners=False)
        flow = F.interpolate(self.compute_flow(ref, supp), size=(h, w), mode="bilinear", align_corners=False)
        return torch.stack((flow[:, 0] * (w / wu), flow[:, 1] * (h / hu)), dim=1)


def deform_conv_grid_sample(x, offset, mask, weight, bias, groups):
    """Modulated deformable 3x3 conv (stride 1, pad 1) from grid_sample, matching torchvision.ops.deform_conv2d.

    offset: (n, 2*9*groups, h, w) as (dy, dx) pairs per kernel point, grouped like torchvision; mask (n, 9*groups, h, w).
    """
    n, c, h, w = x.shape
    cg = c // groups
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype), torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
    off = offset.view(n, groups, 9, 2, h, w)
    m = mask.view(n, groups, 9, 1, h, w)
    xg = x.view(n * groups, cg, h, w)
    cols = []
    for k in range(9):
        ky, kx = k // 3 - 1, k % 3 - 1
        py = gy + ky + off[:, :, k, 0]  # (n, g, h, w)
        px = gx + kx + off[:, :, k, 1]
        grid = torch.stack((2.0 * px / max(w - 1, 1) - 1.0, 2.0 * py / max(h - 1, 1) - 1.0), dim=-1).view(n * groups, h, w, 2)
        s = F.grid_sample(xg, grid, mode="bilinear", padding_mode="zeros", align_corners=True).view(n, groups, cg, h, w)
        cols.append(s * m[:, :, k])
    # (n, 9, c, h, w) -> conv as a sum over kernel points: weight (cout, c, 3, 3)
    col = torch.stack(cols, dim=1).view(n, 9, c, h, w)
    wk = weight.view(weight.shape[0], c, 9).permute(2, 0, 1)  # (9, cout, c)
    out = sum(torch.einsum("oc,nchw->nohw", wk[k], col[:, k]) for k in range(9))
    return out + bias.view(1, -1, 1, 1)


class SecondOrderDeformableAlignment(nn.Module):
    def __init__(self, cin, c, deform_groups, max_residue_magnitude=10, mode="torchvision"):
        super().__init__()
        self.deform_groups = deform_groups
        self.max_residue_magnitude = max_residue_magnitude
        self.mode = mode
        self.weight = nn.Parameter(torch.empty(c, cin, 3, 3))
        self.bias = nn.Parameter(torch.zeros(c))
        nn.init.uniform_(self.weight, -1 / (cin * 9) ** 0.5, 1 / (cin * 9) ** 0.5)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * c + 4, c, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, c, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, c, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, 27 * deform_groups, 3, 1, 1))
        nn.init.zeros_(self.conv_offset[-1].weight)
        nn.init.zeros_(self.conv_offset[-1].bias)

    def forward(self, x, extra, flow_1, flow_2):
        out = self.conv_offset(torch.cat([extra, flow_1, flow_2], 1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), 1))
        off1, off2 = torch.chunk(offset, 2, dim=1)
        off1 = off1 + flow_1.flip(1).repeat(1, off1.size(1) // 2, 1, 1)
        off2 = off2 + flow_2.flip(1).repeat(1, off2.size(1) // 2, 1, 1)
        offset = torch.cat([off1, off2], 1)
        mask = torch.sigmoid(mask)
        if self.mode == "grid_sample":
            # offset = [o1 | o2] covers deform_groups groups over x = [prop | feat2]
            return deform_conv_grid_sample(x, offset, mask, self.weight, self.bias, self.deform_groups)
        import torchvision
        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, padding=(1, 1), mask=mask)


class BasicVSRPlusPlus(nn.Module):
    """[mid_channels]/[num_blocks]: teacher 64/15; slim students e.g. 32/5 (R-S) or 48/7 (R-M)."""

    MODULES = ["backward_1", "forward_1", "backward_2", "forward_2"]

    def __init__(self, mid_channels=64, num_blocks=15, deform_groups=16, deform_mode="torchvision"):
        super().__init__()
        c = mid_channels
        self.mid_channels = c
        self.spynet = SPyNet()
        self.feat_extract = nn.Sequential(nn.Conv2d(3, c, 3, 2, 1), nn.LeakyReLU(0.1, inplace=True),
                                          nn.Conv2d(c, c, 3, 2, 1), nn.LeakyReLU(0.1, inplace=True),
                                          ResidualBlocksWithInputConv(c, c, 5))
        self.deform_align = nn.ModuleDict({m: SecondOrderDeformableAlignment(2 * c, c, deform_groups, mode=deform_mode) for m in self.MODULES})
        self.backbone = nn.ModuleDict({m: ResidualBlocksWithInputConv((2 + i) * c, c, num_blocks) for i, m in enumerate(self.MODULES)})
        self.reconstruction = ResidualBlocksWithInputConv(5 * c, c, 5)
        self.upsample1 = PixelShufflePack(c, c, 2)
        self.upsample2 = PixelShufflePack(c, 64, 2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

    def set_deform_mode(self, mode):
        for m in self.deform_align.values():
            m.mode = mode
        return self

    def compute_flow(self, lqs):
        n, t, c, h, w = lqs.shape
        a = lqs[:, :-1].reshape(-1, c, h, w)
        b = lqs[:, 1:].reshape(-1, c, h, w)
        return self.spynet(b, a).view(n, t - 1, 2, h, w), self.spynet(a, b).view(n, t - 1, 2, h, w)  # forward, backward

    def propagate(self, feats, flows, name):
        n, t, _, h, w = flows.shape
        frame_idx = list(range(0, t + 1))
        flow_idx = list(range(-1, t))
        mapping = list(range(len(feats["spatial"])))
        mapping += mapping[::-1]
        if "backward" in name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
        prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            cur = feats["spatial"][mapping[idx]]
            if i > 0:
                f1 = flows[:, flow_idx[i]]
                cond1 = flow_warp(prop, f1.permute(0, 2, 3, 1))
                feat2 = torch.zeros_like(prop)
                f2 = torch.zeros_like(f1)
                cond2 = torch.zeros_like(cond1)
                if i > 1:
                    feat2 = feats[name][-2]
                    f2 = flows[:, flow_idx[i - 1]]
                    f2 = f1 + flow_warp(f2, f1.permute(0, 2, 3, 1))
                    cond2 = flow_warp(feat2, f2.permute(0, 2, 3, 1))
                prop = self.deform_align[name](torch.cat([prop, feat2], 1), torch.cat([cond1, cur, cond2], 1), f1, f2)
            feat = torch.cat([cur] + [feats[k][idx] for k in feats if k not in ("spatial", name)] + [prop], 1)
            prop = prop + self.backbone[name](feat)
            feats[name].append(prop)
        if "backward" in name:
            feats[name] = feats[name][::-1]
        return feats

    def forward(self, lqs, return_feats=False):
        n, t, c, h, w = lqs.shape
        low = F.interpolate(lqs.view(-1, c, h, w), scale_factor=0.25, mode="bicubic").view(n, t, c, h // 4, w // 4)
        f = self.feat_extract(lqs.view(-1, c, h, w))
        fh, fw = f.shape[2:]
        f = f.view(n, t, -1, fh, fw)
        feats = {"spatial": [f[:, i] for i in range(t)]}
        flows_forward, flows_backward = self.compute_flow(low)
        for it in (1, 2):
            for d in ("backward", "forward"):
                name = f"{d}_{it}"
                feats[name] = []
                feats = self.propagate(feats, flows_backward if d == "backward" else flows_forward, name)
        # features of the last propagation branch, for feature distillation
        last = torch.stack(feats["forward_2"], 1) if return_feats else None
        outs = []
        mapping = list(range(t))
        mapping += mapping[::-1]
        for i in range(t):
            hr = torch.cat([feats["spatial"][mapping[i]]] + [feats[k][i] for k in self.MODULES], 1)
            hr = self.reconstruction(hr)
            hr = F.leaky_relu(self.upsample1(hr), 0.1)
            hr = F.leaky_relu(self.upsample2(hr), 0.1)
            hr = F.leaky_relu(self.conv_hr(hr), 0.1)
            outs.append(self.conv_last(hr) + lqs[:, i])
        out = torch.stack(outs, 1)
        return (out, last) if return_feats else out


def load_spynet(net, path):
    """Starts a student's motion estimator from OpenMMLab's SPyNet (Apache-2.0, spynet_20210409-c6c1bd09.pth)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck.get("state_dict", ck)
    net.spynet.load_state_dict({k: v for k, v in sd.items() if k.startswith("basic_module.")}, strict=False)
    missing = [k for k in net.spynet.state_dict() if k.startswith("basic_module.") and k not in sd]
    if missing:
        raise RuntimeError(f"SPyNet weights do not fit: {missing[:5]}")
    return net


def load_lada_teacher(path, use_ema=True, device="cpu"):
    """The Lada restoration checkpoint holds `generator.*` and `generator_ema.*`; Lada's app uses the EMA weights."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    prefix = "generator_ema." if use_ema and any(k.startswith("generator_ema.") for k in sd) else "generator."
    weights = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    net = BasicVSRPlusPlus(64, 15)
    missing, unexpected = net.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"teacher weights do not fit: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return net.eval().to(device).requires_grad_(False)
