# Models shipped in MosaicGone

All models are in `app/src/main/assets/models/` and run on the phone through ONNX Runtime (CPU).
SHA-256 values were computed from the files in this repository. The original checkpoints of the AGPL-3.0 models
(MosaicGone's detector and restorer, and Lada's restorer) are in `Releases/`, described in `Releases/README.md`. The
original checkpoint will be provided as a separate attachment to the same GitHub release as the APK. Other upstream
weights are not mirrored here: their download locations are given as the upstream projects publish them.

Scripts named below are in `model-development/training/` unless a path says otherwise. The training data is not
included (see `model-development/README.md`).

## 1. `mosaicgone_detector.onnx` — YOLOX-Nano mosaic detector (MosaicGone)

| | |
|---|---|
| SHA-256 | `0e31922e4cb65387cdffc042c9e25c5c873c1c579972bdb8a5cb9d6b7ac714bd` |
| Checkpoint | `Releases/mosaicgone_detector_det_v2_step10000.pt` (training run `det_v2`, step 10000, EMA weights), SHA-256 `7c5e26206c135251021733a077ac32c28faa2e7165f46d97d680df159594a298` |
| Architecture | YOLOX-Nano (Megvii) plus a mask head, re-implemented in `dmtrain/yolox.py` |
| Starting weights | Official YOLOX-Nano `yolox_nano.pth` (Apache-2.0), https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth; the copy used had SHA-256 `cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b` |
| Training | `train_detector.py`. Run `det_v1` (40000 steps from the YOLOX-Nano weights), then `det_v2`, a fine-tune of `det_v1/last.pt` with `det_v1` as reference (`--iters 14000 --lr 3e-4 --warmup 300 --batch 32`); best.pt = step 10000. Exact commands: comments in `run_all.ps1`. |
| Teacher | Lada mosaic detection model v4 accurate (`lada_mosaic_detection_model_v4_accurate.pt`, AGPL-3.0, Lada Authors); its SHA-256 `c244d7e4…801bc1f` matches Lada's `model_weights/checksums_sha256.txt` |
| Training data | synthetic mosaics and blur on REDS, YouTube-VOS 2019 and super-resolution benchmark stills (`dmtrain/data.py`, `dmtrain/synth.py`) |
| Export | `export_mosaicgone.py`: input `(1,3,416,416)` BGR 0..255 letterboxed, output `(1,1,104,104)` sigmoid mask, ONNX opset 17 |
| License | AGPL-3.0 (MosaicGone); starting weights Apache-2.0 |

## 2. `mosaicgone_restorer.onnx`, `mosaicgone_restorer_t4.onnx` — slim BasicVSR++ restorer (MosaicGone)

| | |
|---|---|
| SHA-256 | 16-frame: `111283df2591c3139da6ee28391df76040136ceb14c8ba6d8ef70c6d1650dbc5`; 4-frame (`_t4`): `c2afcfe21ed0bfb4c68166d73b5638817eec9c4d31f2723c2ccce4a5f21c9acf` |
| Checkpoint | `Releases/mosaicgone_restorer_res_s2_step15000.pt` (training run `res_s2`, step 15000, 32 channels, 5 blocks), SHA-256 `2bb1cf9814621ec8141d136706edf274e8290fcf0dff0901800dd26a2f409d41` |
| Architecture | BasicVSR++ (Chan et al.) at 32 channels / 5 blocks, re-implemented in `dmtrain/basicvsrpp.py` |
| Starting weights | SPyNet part only: OpenMMLab `spynet_20210409-c6c1bd09.pth` (Apache-2.0), https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth; the copy used had SHA-256 `c6c1bd09b52d05ba17f3e701f549d6faf5e314aabce8ae462c1c171a8d6c4914`. The rest started untrained. |
| Training | Stage 1, `train_restorer.py`: run `res_s1`, 80000 steps, 16 frames, batch 2; best.pt = step 78000. Stage 2 (GAN and perceptual losses), `train_restorer_s2.py`: a chain of pilot runs, each starting from the previous one (`s2_pilot_lines` → `s2_pilot_animbranch_ap` → `s2_pilot_colour_faces`), then `res_s2` from `s2_pilot_colour_faces/pilot_step3000.pt`. Per the comments in `run_all.ps1`, `res_s2` was resumed at step 14750 with `--tone 5 --out_gate_real` added. The shipped checkpoint is step 15000, the run's best.pt; the later resume with `--edge` was not used. Commands and options for every run are in `run_all.ps1`. |
| Teacher | Lada restoration model generic v1.2 (AGPL-3.0, Lada Authors), see model 3 |
| Training-only helpers | VGG19 `vgg19-dcbb9e9d.pth` (torchvision), Danbooru ResNet50 `resnet50-13306192.pth` (M. Baas), Real-ESRGAN's U-Net discriminator (`dmtrain/discriminator.py`) |
| Training data | REDS, YouTube-VOS 2019, super-resolution benchmark stills and animation stills (`dmtrain/data.py`) |
| Export | `export_mosaicgone.py`: input `(1,T,3,256,256)` RGB 0..1, output the same shape, clamped to 0..1; T = 16 and T = 4; `grid_sample` form of the deformable convolution; opset 17 |
| License | AGPL-3.0 (MosaicGone); SPyNet starting weights Apache-2.0 |

## 3. `lada_restorer.onnx`, `lada_restorer_t4.onnx` — Lada restoration model generic v1.2 (unchanged weights)

| | |
|---|---|
| SHA-256 | 16-frame: `473f4045d174b275e544df036c2b8676d8079d1ba8518837438c04bce8de7588`; 4-frame: `30ff274ca34c617287253e464ed1a94f38a8ae65032a2ae2918d9f2752da36a3` |
| Origin | `lada_mosaic_restoration_model_generic_v1.2.pth` from Lada, https://github.com/ladaapp/lada |
| Source checkpoint | `Releases/lada_mosaic_restoration_model_generic_v1.2.pth`, SHA-256 `d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee`, the same as Lada's `model_weights/checksums_sha256.txt`. The original checkpoint will be provided as a separate attachment to the same GitHub release as the APK. |
| Architecture | BasicVSR++, 64 channels / 15 blocks; loaded into `dmtrain/basicvsrpp.py` (`load_lada_teacher`) |
| Changes | Only the EMA weights (`generator_ema.*`) are used, as Lada's app does. Converted to ONNX with the deformable convolution as `grid_sample`, output clamped to 0..1, the same inputs and outputs as model 2. Weights are not retrained. |
| Export | `export_lada.py` (path arguments added to this copy; see its header) |
| License | AGPL-3.0, Copyright Lada Authors (upstream `model_weights/lada_mosaic_restoration_model_generic_v1.2.pth.license`) |

## 4. Real-ESRGAN sharpening models (optional "Sharpen the cleaned area")

| File | SHA-256 | Made from |
|---|---|---|
| `realesr-animevideov3.onnx` | `c27ba6ed703c67738a8778c978a6fc9b5523811984398b9ecfd7edc35d86d3a9` | `realesr-animevideov3.pth` |
| `realesr-general-x4v3_dn000.onnx` | `f5d8ab1e9ff122fdcef120013d6349fc363d943179fcf64d6ff1c50435f060a8` | denoise strength 0: the weights of `realesr-general-wdn-x4v3.pth` |
| `realesr-general-x4v3_dn050.onnx` | `0e8272b01d8e065e9ebe5195a291f5f17c1e98ee7269e3b954c6585f8ea0c1c4` | 0.5 × `realesr-general-x4v3.pth` + 0.5 × `realesr-general-wdn-x4v3.pth` (Real-ESRGAN's own `dni` interpolation) |
| `realesr-general-x4v3_dn100.onnx` | `e859a54006946df52e810ea79c6ba71ef522ed1b6f8b04ae1646d66edddce293` | denoise strength 1: the weights of `realesr-general-x4v3.pth` |

- Upstream: https://github.com/xinntao/Real-ESRGAN, weights from its v0.2.5.0 release
  (the URLs in Real-ESRGAN's `inference_realesrgan.py`):

  | Weights | SHA-256 | URL |
  |---|---|---|
  | `realesr-animevideov3.pth` | `b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d` | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth |
  | `realesr-general-x4v3.pth` | `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292` | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth |
  | `realesr-general-wdn-x4v3.pth` | `1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca` | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth |

  The weights inside the four ONNX files are identical to these files, compared tensor by tensor. The `_dn050` file
  was compared with the 0.5 / 0.5 mix.
- Architecture: SRVGGNetCompact, x4. Weights are not retrained; the denoise levels are fixed interpolations.
- Export: `tools/export_realesrgan.py` (also writes the reference images the tests compare against). Input and output
  are RGB 0..1 with dynamic size.
- License: BSD 3-Clause, Copyright (c) 2021, Xintao Wang (`third_party_licenses/real-esrgan/LICENSE`).
