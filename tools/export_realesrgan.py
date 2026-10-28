"""Convert the bundled Real-ESRGAN models to ONNX and make Python reference outputs.

Run:  TORCH_FORCE_WEIGHTS_ONLY_LOAD=1 python -I tools/export_realesrgan.py path/to/Real-ESRGAN   (from the repository root)
Needs a checkout of https://github.com/xinntao/Real-ESRGAN with its weights/ folder filled.

Exports (all SRVGGNetCompact, x4, dynamic H/W, input RGB 0..1 NCHW -> output RGB 0..1, not clamped):
  realesr-general-x4v3_dn{000,050,100}.onnx  denoise strength 0 / 0.5 / 1 baked in with the same
                                            network interpolation (dni) as inference_realesrgan.py
  realesr-animevideov3.onnx
References (real RealESRGANer code, fp32, CPU) -> core/testdata/reference/esrgan_*.png
"""
import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))  # the repository
RE = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else sys.exit(__doc__)
OUT = os.path.join(ROOT, 'app', 'src', 'main', 'assets', 'models')
REF = os.path.join(ROOT, 'core', 'testdata', 'reference')
W = os.path.join(RE, 'weights')

import numpy as np
import torch
import cv2
import onnxruntime as ort

# --- load Real-ESRGAN's own code without installing basicsr (only two tiny helpers are needed) ---
class _Registry:
    def register(self, *a, **k):
        return lambda cls: cls

for name in ['basicsr', 'basicsr.utils']:
    sys.modules.setdefault(name, types.ModuleType(name))
reg = types.ModuleType('basicsr.utils.registry'); reg.ARCH_REGISTRY = _Registry(); sys.modules['basicsr.utils.registry'] = reg
dl = types.ModuleType('basicsr.utils.download_util'); dl.load_file_from_url = None; sys.modules['basicsr.utils.download_util'] = dl


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SRVGGNetCompact = load_file(os.path.join(RE, 'realesrgan', 'archs', 'srvgg_arch.py'), 'srvgg_arch').SRVGGNetCompact
RealESRGANer = load_file(os.path.join(RE, 'realesrgan', 'utils.py'), 'realesrgan_utils').RealESRGANer


def net(num_conv):
    return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=num_conv, upscale=4, act_type='prelu')


def params(path):
    d = torch.load(path, map_location='cpu', weights_only=True)
    return d['params_ema'] if 'params_ema' in d else d['params']


def dni_params(dn):
    # RealESRGANer.dni: dni_weight = [dn, 1 - dn] over (general-x4v3, general-wdn-x4v3), key 'params'
    a = torch.load(os.path.join(W, 'realesr-general-x4v3.pth'), map_location='cpu', weights_only=True)['params']
    b = torch.load(os.path.join(W, 'realesr-general-wdn-x4v3.pth'), map_location='cpu', weights_only=True)['params']
    return {k: dn * v + (1 - dn) * b[k] for k, v in a.items()}


def export(model, name):
    model.eval()
    path = os.path.join(OUT, name + '.onnx')
    x = torch.rand(1, 3, 64, 80)
    with torch.no_grad():
        torch.onnx.export(model, (x,), path, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {2: 'h', 3: 'w'}, 'output': {2: 'h4', 3: 'w4'}},
                          opset_version=17, do_constant_folding=True, dynamo=False)
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    for shape in [(1, 3, 64, 80), (1, 3, 37, 129)]:
        t = torch.rand(*shape)
        with torch.no_grad():
            ref = model(t).numpy()
        got = sess.run(None, {'input': t.numpy()})[0]
        err = float(np.abs(ref - got).max())
        print(f'  {name}.onnx {os.path.getsize(path) / 1e6:.1f} MB  {shape}->{got.shape}  max err {err:.2e}')
        assert err < 1e-4
    return path


def reference(model_path, dni_weight, num_conv, img_name, outscale, tile, tag):
    up = RealESRGANer(scale=4, model_path=model_path, dni_weight=dni_weight, model=net(num_conv),
                      tile=tile, tile_pad=10, pre_pad=0, half=False, device=torch.device('cpu'))
    img = cv2.imread(os.path.join(RE, 'inputs', img_name), cv2.IMREAD_COLOR)
    # the decoded pixels, so the Kotlin test starts from exactly the same input (JPEG decoders differ)
    cv2.imwrite(os.path.join(REF, 'esrgan_input_' + os.path.splitext(img_name)[0] + '.png'), img)
    out, _ = up.enhance(img, outscale=outscale)
    cv2.imwrite(os.path.join(REF, f'esrgan_{tag}.png'), out)
    print(f'  reference esrgan_{tag}.png {img.shape[1]}x{img.shape[0]} -> {out.shape[1]}x{out.shape[0]}')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(REF, exist_ok=True)
    torch.manual_seed(0)
    for dn, tag in [(0.0, '000'), (0.5, '050'), (1.0, '100')]:
        m = net(32)
        m.load_state_dict(params(os.path.join(W, 'realesr-general-x4v3.pth')) if dn == 1.0 else dni_params(dn), strict=True)
        export(m, f'realesr-general-x4v3_dn{tag}')
    m = net(16)
    m.load_state_dict(params(os.path.join(W, 'realesr-animevideov3.pth')), strict=True)
    export(m, 'realesr-animevideov3')

    gen = os.path.join(W, 'realesr-general-x4v3.pth')
    wdn = os.path.join(W, 'realesr-general-wdn-x4v3.pth')
    # same model/denoise handling as inference_realesrgan.py (-dn 0.5 default, dni over the two files)
    reference([gen, wdn], [0.5, 0.5], 32, '0014.jpg', 4, 0, 'general050_0014_x4')
    reference([gen, wdn], [0.5, 0.5], 32, '0014.jpg', 4, 64, 'general050_0014_x4_tile64')
    reference(gen, None, 32, '0030.jpg', 2, 0, 'general100_0030_x2')
    reference(os.path.join(W, 'realesr-animevideov3.pth'), None, 16, 'ADE_val_00000114.jpg', 4, 0, 'anime_ADE_x4')
    print('ALL_OK')
