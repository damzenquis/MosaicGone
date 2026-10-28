# Release checkpoints

The original checkpoints behind the AGPL-3.0 models in the APK. Each will also be attached to the GitHub release that
carries the APK.

The Real-ESRGAN, YOLOX and SPyNet weights are not here: their licenses (BSD 3-Clause, Apache-2.0) do not require them
to be shipped with the app. Where to get them is in `../MODELS.md` and `../model-development/README.md`.

## `lada_mosaic_restoration_model_generic_v1.2.pth`

| | |
|---|---|
| SHA-256 | `d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee` (the same as Lada's `model_weights/checksums_sha256.txt`) |
| Upstream | Lada, https://github.com/ladaapp/lada |
| License | AGPL-3.0, Copyright Lada Authors: `lada_mosaic_restoration_model_generic_v1.2.pth.license` (upstream file, unchanged) and `licenses/AGPL-3.0.txt` |
| Converted to ONNX | Yes: `lada_restorer.onnx` (16 frames) and `lada_restorer_t4.onnx` (4 frames), the app's second restorer. The EMA weights were converted unchanged. |
| Also used for | the teacher model when training the MosaicGone restorer |
| Model definition | `../model-development/training/dmtrain/basicvsrpp.py` (`BasicVSRPlusPlus(64, 15)`, `load_lada_teacher`) |
| ONNX export | `../model-development/training/export_lada.py` (uses `export_mosaicgone.py` in the same folder); details in `../MODELS.md`, model 3 |

## `mosaicgone_detector_det_v2_step10000.pt`

| | |
|---|---|
| SHA-256 | `7c5e26206c135251021733a077ac32c28faa2e7165f46d97d680df159594a298` |
| Upstream | MosaicGone (this project). Training started from YOLOX-Nano `yolox_nano.pth`, https://github.com/Megvii-BaseDetection/YOLOX (Copyright (c) 2021-2022 Megvii Inc.). |
| License | AGPL-3.0 (`../LICENSE`); the YOLOX-Nano starting weights are Apache-2.0 (`licenses/Apache-2.0.txt`) |
| Converted to ONNX | Yes: `mosaicgone_detector.onnx` |
| Model definition | `../model-development/training/dmtrain/yolox.py` |
| Training configuration | `../model-development/training/train_detector.py`, and the run commands in `run_all.ps1` (runs `det_v1`, then `det_v2`) |
| ONNX export | `../model-development/training/export_mosaicgone.py --det` |

## `mosaicgone_restorer_res_s2_step15000.pt`

| | |
|---|---|
| SHA-256 | `2bb1cf9814621ec8141d136706edf274e8290fcf0dff0901800dd26a2f409d41` |
| Upstream | MosaicGone (this project). Its motion estimator started from OpenMMLab SPyNet `spynet_20210409-c6c1bd09.pth`, https://github.com/open-mmlab/mmagic (Copyright (c) OpenMMLab). |
| License | AGPL-3.0 (`../LICENSE`); the SPyNet starting weights are Apache-2.0 (`licenses/Apache-2.0.txt`) |
| Converted to ONNX | Yes: `mosaicgone_restorer.onnx` (16 frames) and `mosaicgone_restorer_t4.onnx` (4 frames) |
| Model definition | `../model-development/training/dmtrain/basicvsrpp.py` (`BasicVSRPlusPlus(32, 5)`) |
| Training configuration | `../model-development/training/train_restorer.py` (stage 1), `train_restorer_s2.py` (stage 2), and the run commands in `run_all.ps1` |
| ONNX export | `../model-development/training/export_mosaicgone.py --res` |
