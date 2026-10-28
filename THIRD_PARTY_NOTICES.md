# Third-party notices

MosaicGone is licensed under the GNU Affero General Public License v3.0 (`LICENSE`). It includes, or was built from,
the third-party work listed below. Each keeps its own license; the full texts are in `third_party_licenses/`.
This file is also included in the app (Credits row → License → Third-party notices).

## Included in the app (APK)

### Lada: restoration model "generic v1.2"
- Shipped as: `lada_restorer.onnx`, `lada_restorer_t4.onnx` (the second, optional restorer)
- Source: https://github.com/ladaapp/lada — weights `lada_mosaic_restoration_model_generic_v1.2.pth`
- Copyright: Lada Authors
- License: AGPL-3.0 (upstream `model_weights/lada_mosaic_restoration_model_generic_v1.2.pth.license`:
  `SPDX-License-Identifier: AGPL-3.0`)
- Changes: converted to ONNX (EMA weights, deformable convolution expressed with `grid_sample`, output clamped
  to 0..1); see `MODELS.md`.
- License texts: `third_party_licenses/lada/` (Lada's `LICENSE.md` and `LICENSES/` folder, unchanged)

### Real-ESRGAN: models and enhancement code
- Shipped as: `realesr-animevideov3.onnx`, `realesr-general-x4v3_dn000.onnx`, `_dn050.onnx`, `_dn100.onnx`
- Code: `core/src/main/kotlin/com/mosaicgone/core/Enhance.kt` is a Kotlin port of `realesrgan.utils.RealESRGANer.enhance`,
  and `model-development/training/dmtrain/discriminator.py` contains Real-ESRGAN's `UNetDiscriminatorSN` (training
  only); both files carry the notice below.
- Source: https://github.com/xinntao/Real-ESRGAN
- License: BSD 3-Clause

```
Copyright (c) 2021, Xintao Wang
All rights reserved.
```
Full text with conditions and disclaimer: `third_party_licenses/real-esrgan/LICENSE`

### DeepMosaics: mosaic finding and pasting code
- Code: `core/src/main/kotlin/com/mosaicgone/core/ImageOps.kt` (ports of `util/image_processing.py`) and
  `Filt.kt` (port of `util/filt.py`); the app's pipeline began as a port of DeepMosaics 0.5.1
- Source: https://github.com/HypoX64/DeepMosaics
- License: GPL-3.0 (`third_party_licenses/deepmosaics/LICENSE`). Combined with AGPL-3.0 code as section 13 of both
  licenses permits.

### YOLOX
- Used as: architecture of the MosaicGone detector (`mosaicgone_detector.onnx`), re-implemented for this project, and
  its training started from the official `yolox_nano.pth` weights (see `MODELS.md`)
- Source: https://github.com/Megvii-BaseDetection/YOLOX
- License: Apache-2.0 (`third_party_licenses/apache-2.0/LICENSE-2.0.txt`)
- Copyright (c) 2021-2022 Megvii Inc. All rights reserved.

### MMagic (OpenMMLab): SPyNet weights
- Used as: the starting weights of the motion-estimation part (SPyNet) of the MosaicGone restorer
  (`mosaicgone_restorer.onnx`, `mosaicgone_restorer_t4.onnx`), `spynet_20210409-c6c1bd09.pth`; changed by training
- Source: https://github.com/open-mmlab/mmagic
- License: Apache-2.0 (`third_party_licenses/apache-2.0/LICENSE-2.0.txt`)
- Copyright (c) OpenMMLab. All rights reserved.

### BasicVSR++
- Used as: the design of the MosaicGone restorer (Chan et al., CVPR 2022), re-implemented for this project with
  parameter names matching MMagic's `BasicVSRPlusPlusNet`
- Source: https://github.com/ckkelvinchan/BasicVSR_PlusPlus — License: Apache-2.0

### ONNX Runtime 1.20.0
- Used as: the AI engine (`com.microsoft.onnxruntime:onnxruntime-android:1.20.0`)
- License: MIT, Copyright (c) Microsoft Corporation (`third_party_licenses/onnxruntime/LICENSE`)
- ONNX Runtime's own third-party notices: `third_party_licenses/onnxruntime/ThirdPartyNotices.txt`, unchanged from
  https://github.com/microsoft/onnxruntime/blob/v1.20.0/ThirdPartyNotices.txt (the Android package contains no notices
  file). The LICENSE is from the same tag.

### Java Native Access (JNA) 5.14.0
- Used as: calls `sched_setaffinity` to keep the AI on the fast CPU cores
- License: dual LGPL-2.1-or-later or Apache-2.0; used here under Apache-2.0
  (`third_party_licenses/apache-2.0/LICENSE-2.0.txt`). JNA's own license statement, from its package:
  `third_party_licenses/jna/LICENSE`. Source: https://github.com/java-native-access/jna

### Android and Kotlin libraries
All under Apache-2.0 as declared in each library's Maven POM (`third_party_licenses/apache-2.0/LICENSE-2.0.txt`):
AndroidX (AppCompat 1.7.0, Activity 1.9.3, Lifecycle 2.8.7 and the AndroidX libraries they pull in), Material
Components for Android 1.12.0, Kotlin standard library 2.0.21 and kotlinx-coroutines 1.9.0. Also included is Guava's
`com.google.guava:listenablefuture:1.0` (pulled in by AndroidX), whose POM declares no license itself (the Guava project
is Apache-2.0). The full list (69 artifacts) comes from
`gradlew :app:dependencies --configuration releaseRuntimeClasspath`. None of these 69 packages contains an Apache
`NOTICE` file (checked inside every `.aar` / `.jar`, including their `classes.jar`), so there is no NOTICE text to
pass on under Apache-2.0 section 4(d).

## Used only to develop the models (not in the app)

These were used by the training scripts in `model-development/`; none of them is shipped in the APK.

| Item | Use | License |
|---|---|---|
| Lada mosaic detection model v4 accurate (`lada_mosaic_detection_model_v4_accurate.pt`) | teacher for the detector | AGPL-3.0, Lada Authors (upstream `.license` file) |
| Lada restoration model generic v1.2 | teacher for the restorer | AGPL-3.0, Lada Authors |
| Ultralytics 8.3.203 (Python package) | runs the Lada detection teacher | AGPL-3.0 |
| Real-ESRGAN `UNetDiscriminatorSN` | copied into `model-development/training/dmtrain/discriminator.py` | BSD 3-Clause, (c) 2021 Xintao Wang |
| torchvision VGG19 ImageNet weights (`vgg19-dcbb9e9d.pth`) | perceptual loss | torchvision: BSD 3-Clause |
| Danbooru pretrained ResNet50 by Matthew Baas (`resnet50-13306192.pth`, https://github.com/RF5/danbooru-pretrained) | anime perceptual loss | MIT-style, requires crediting M Baas |
| APISR (https://github.com/Kiteretsu77/APISR) | ideas for the anime losses, re-implemented (no code copied, per the file headers) | GPL-3.0 |
