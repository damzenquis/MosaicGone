# Test fixtures

What the `core` tests read (`gradlew :core:test`). Paths in the tests are relative to the `core` module, where Gradle
runs them (`PipelineTest.data` and the paths after it). Test output goes to `core/build/`, never here.

## In Git

| Path | Used by | Origin |
|---|---|---|
| `gt/shapes/gt/`, `gt/shapes/mosaic/` | `PipelineTest.mosaicFrame()`, `QualityTest` | a synthetic clip, 21 frames of 384 x 214: random shapes, gradients and stripes, panning, with a 10 px block mosaic over an ellipse. Drawn by `tools/make_test_clips.py` from a fixed seed; no third-party content |
| `images/shapes.gif` | `GifTest` (GIF decode / encode / clean) | the same clip as an animated GIF, clean frames on the left and mosaicked on the right (768 x 214, 21 frames, 2 fps), by the same script |
| `reference/esrgan_input_*.png` | `PipelineTest` (Real-ESRGAN checks), `UnitTest` (colour conversion, on `esrgan_input_ADE_val_00000114.png`) | `0014.jpg`, `0030.jpg` and `ADE_val_00000114.jpg` from Real-ESRGAN's `inputs/` folder, as read by `tools/export_realesrgan.py` |
| `reference/esrgan_*_x4*.png`, `esrgan_general100_0030_x2.png` | `PipelineTest` (Real-ESRGAN checks) | the same images through Real-ESRGAN's own Python `RealESRGANer` (fp32, CPU), written by `tools/export_realesrgan.py` |

`python -I tools/make_test_clips.py` (numpy, opencv-python, pillow) remakes the `shapes` files exactly.

## Optional clips (not in Git)

`QualityTest` also scores the clips `real`, `onepiece`, `small`, `mid` and `tiny` when they are present in `gt/`
(each `gt/<clip>/gt/NNN.png` + `gt/<clip>/mosaic/NNN.png`). They are ignored by Git and not part of this repository.
Without them every test still runs except `mosaicGoneParallelMatchesSerialAndKeepsOutside` (reported as skipped), and
the clip loops in `QualityTest` check `shapes` only, printing `skip <clip>` for the others.

## Where the third-party images came from

| Fixture | Upstream file | SHA-256 of the upstream file | Upstream repository and license | Notes |
|---|---|---|---|---|
| `reference/esrgan_input_0014.png` | `inputs/0014.jpg` | `444d2cc3bd9304b9a54e676e10081c8cdd2f3f2665e8c742760c97da481523a2` | https://github.com/xinntao/Real-ESRGAN, BSD 3-Clause | re-saved as PNG by `tools/export_realesrgan.py` |
| `reference/esrgan_input_0030.png` | `inputs/0030.jpg` | `2f49c8cf6249940408fff736f394a4f8a228a6488bcc2fe9f92bdfb90b54a504` | Real-ESRGAN, BSD 3-Clause | re-saved as PNG |
| `reference/esrgan_input_ADE_val_00000114.png` | `inputs/ADE_val_00000114.jpg` | `ebf8662783a1a5f5d97c893f7257e46f59738a5f3848c7a1d05cd16d5670dc6b` | Real-ESRGAN, BSD 3-Clause | re-saved as PNG; the name follows the ADE20K dataset's file naming, but upstream does not say so |
| `reference/esrgan_*_x4*.png`, `esrgan_general100_0030_x2.png` | made from the three inputs above | — | as the inputs | outputs of Real-ESRGAN's own Python code (`tools/export_realesrgan.py`) |

Real-ESRGAN's license covers how these files are distributed there; the repository does not state where the three
photos themselves came from.
