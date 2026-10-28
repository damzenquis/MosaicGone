"""Restorer stage 2 helpers: the GAN discriminator and the VGG19 perceptual loss.

UNetDiscriminatorSN is Real-ESRGAN's U-Net discriminator with spectral norm (BSD 3-Clause, (c) 2021 Xintao Wang,
github.com/xinntao/Real-ESRGAN), the same design Lada's stage 2 uses. Copied without the basicsr registry.
VGG19 weights: torchvision's ImageNet vgg19-dcbb9e9d.pth (training only, never shipped in the app).
AnimeFeatures: ResNet50 trained on Danbooru2018 anime illustrations (top 6000 tags) by M. Baas, "danbooru-pretrained"
(github.com/RF5/danbooru-pretrained, MIT with attribution: credit M Baas), weights resnet50-13306192.pth; training only,
never shipped in the app. Used as APISR's anime perceptual loss (Wang et al., CVPR 2024), reimplemented here.

UNetDiscriminatorSN: modified for MosaicGone, 2026 (basicsr registry removed), from Real-ESRGAN
(realesrgan/archs/discriminator_arch.py), under the following license:

BSD 3-Clause License

Copyright (c) 2021, Xintao Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import torch
import torchvision
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class UNetDiscriminatorSN(nn.Module):
    """Per-pixel real / fake logits, (n, 1, h, w)."""

    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        return self.conv9(out)


# torchvision vgg19.features indices (the layer outputs Lada's stage 2 uses) and their weights
VGG_LAYERS = {2: 0.1, 7: 0.1, 16: 1.0, 25: 1.0, 34: 1.0}


class VGGFeatures(nn.Module):
    """Frozen VGG19 trunk; returns the VGG_LAYERS outputs of 0..1 RGB input (ImageNet-normalised inside)."""

    def __init__(self, path):
        super().__init__()
        layers = []
        cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
        c = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(c, v, 3, padding=1), nn.ReLU(inplace=False)]
                c = v
        self.features = nn.Sequential(*layers[:max(VGG_LAYERS) + 1])
        sd = torch.load(path, map_location="cpu", weights_only=True)
        self.features.load_state_dict({k[len("features."):]: v for k, v in sd.items()
                                       if k.startswith("features.") and int(k.split(".")[1]) <= max(VGG_LAYERS)})
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in VGG_LAYERS:
                out[i] = x
        return out


def perceptual(vgg, a, b):
    """L1 between VGG features of a (trained) and b (target), summed over VGG_LAYERS with their weights; (n,) per image."""
    fa = vgg(a)
    with torch.no_grad():
        fb = vgg(b)
    return sum(w * (fa[i] - fb[i]).abs().mean(dim=(1, 2, 3)) for i, w in VGG_LAYERS.items())


# APISR's layers (conv outputs before their BatchNorm: the stem conv, then the last conv3 of a block in each stage) and
# the paper's weights (early layers scaled up so the stages weigh about alike)
ANIME_LAYERS = {"conv1": 0.1, "layer1.2.conv3": 20.0, "layer2.3.conv3": 25.0, "layer3.5.conv3": 1.0, "layer4.2.conv3": 1.0}


class AnimeFeatures(nn.Module):
    """Frozen Danbooru ResNet50 body; returns the ANIME_LAYERS outputs of 0..1 RGB input (normalised inside with the
    mean / std it was trained with)."""

    def __init__(self, path):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        sd = torch.load(path, map_location="cpu", weights_only=True)  # "0.<i>.*" = body (fastai: body, head)
        self.body.load_state_dict({k[2:]: v for k, v in sd.items() if k.startswith("0.")})
        named = {"conv1": r.conv1, **{f"layer{s}.{b}.conv3": getattr(r, f"layer{s}")[b].conv3 for s, b in ((1, 2), (2, 3), (3, 5), (4, 2))}}
        self._out = {}
        for name, mod in named.items():
            mod.register_forward_hook(lambda _m, _i, o, name=name: self._out.__setitem__(name, o))
        self.register_buffer("mean", torch.tensor([0.7137, 0.6628, 0.6519]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.2970, 0.3017, 0.2979]).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        self._out = {}
        self.body((x - self.mean) / self.std)
        return self._out


def anime_perceptual(net, a, b):
    """As perceptual(), with AnimeFeatures and ANIME_LAYERS; (n,) per image."""
    fa = net(a)
    with torch.no_grad():
        fb = net(b)
    return sum(w * (fa[k] - fb[k]).abs().mean(dim=(1, 2, 3)) for k, w in ANIME_LAYERS.items())
