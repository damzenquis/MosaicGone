# Model development

Everything used to make MosaicGone's own models (`mosaicgone_detector.onnx`, `mosaicgone_restorer[_t4].onnx`), and
to convert Lada's restorer for the app. `MODELS.md` (one folder up) says which file came from what.

Nothing in this folder is needed to build or run the app.

## Contents

| Path | What it is |
|---|---|
| `../Releases/mosaicgone_detector_det_v2_step10000.pt` | the shipped detector's PyTorch checkpoint (EMA weights, step 10000, plus its validation metrics) |
| `../Releases/mosaicgone_restorer_res_s2_step15000.pt` | the shipped restorer's PyTorch checkpoint (32 channels, 5 blocks, step 15000, plus metrics) |
| `../Releases/lada_mosaic_restoration_model_generic_v1.2.pth` | Lada's original restoration checkpoint (source of `lada_restorer*.onnx`) |
| `training/dmtrain/yolox.py` | YOLOX-Nano + mask head (architecture; loads the official `yolox_nano.pth`) |
| `training/dmtrain/basicvsrpp.py` | BasicVSR++ (architecture; slim students and the Lada teacher; ONNX-friendly deformable conv) |
| `training/dmtrain/discriminator.py` | stage-2 GAN discriminator (copied from Real-ESRGAN, BSD 3-Clause) and perceptual losses |
| `training/dmtrain/synth.py` | synthetic mosaic / blur obscuring and video damage |
| `training/dmtrain/data.py` | builds training samples from the datasets (expects an index file, see below) |
| `training/train_detector.py` | detector training / distillation (teacher: Lada mosaic detection v4 accurate) |
| `training/train_restorer.py` | restorer stage 1 (distillation from the Lada restoration teacher) |
| `training/train_restorer_s2.py` | restorer stage 2 (GAN + perceptual fine-tune) |
| `training/run_all.ps1` | the exact command line of every run, in order, with notes on why each run was made or stopped |
| `training/export_mosaicgone.py` | exports the two checkpoints to the ONNX files the app ships, and checks them against PyTorch |
| `training/export_lada.py` | converts Lada's restoration weights to the app's ONNX format |
| `training/verify_*.py`, `preview_samples.py` | checks made before / after training (architecture vs official weights, teacher outputs, samples) |
| `../tools/export_realesrgan.py` | converts the Real-ESRGAN weights and writes the test reference images |

These are copies of the scripts as they were run for v0.2. Since then their paths have changed so that they run from
this repository (`export_lada.py` takes its paths as arguments; the other scripts find everything from their own
location, below), and some comments, one class (`synth.Obscurer`) and one option (`--line_mask` of
`train_restorer_s2.py`) were renamed. What the code computes is unchanged. They have not been re-run from this folder.

### Paths
Relative to `training/`, none of these in Git:

| Path | What goes there |
|---|---|
| `models/` | the upstream weights in the table below (except Lada's restoration checkpoint, read from `../Releases/`) |
| `models/helpers/` | `spynet_20210409-c6c1bd09.pth`, `vgg19-dcbb9e9d.pth`, `danbooru_resnet50-13306192.pth` |
| `data/TRAINING DATA/` | the datasets listed under "Training data"; `data/index.json` is written on the first run |
| `runs/` | training output (one folder per run, as in `run_all.ps1`) |
| `.venv/` | the training Python environment (`run_all.ps1` runs `.venv\Scripts\python.exe`) |

The ground-truth clips are read from `core/testdata/gt` (see `core/testdata/README.md`); the scripts that score on the
optional clips (`onepiece`, `real`) need them there.

## Upstream weights used

| File | Used for | Where | SHA-256 of the copy used |
|---|---|---|---|
| `yolox_nano.pth` | detector starting weights | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth (Apache-2.0) | `cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b` |
| `spynet_20210409-c6c1bd09.pth` | restorer motion-estimator starting weights | https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth (Apache-2.0) | `c6c1bd09b52d05ba17f3e701f549d6faf5e314aabce8ae462c1c171a8d6c4914` |
| `lada_mosaic_restoration_model_generic_v1.2.pth` | restorer teacher; source of `lada_restorer*.onnx` | included: `../Releases/` (Lada, https://github.com/ladaapp/lada, AGPL-3.0) | `d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee` (matches Lada's) |
| `lada_mosaic_detection_model_v4_accurate.pt` | detector teacher | Lada, as above (AGPL-3.0) | `c244d7e49d8f88e264b8dc15f91fb21f5908ad8fb6f300b7bc88462d0801bc1f` (matches Lada's) |
| `vgg19-dcbb9e9d.pth` | stage-2 perceptual loss | torchvision's ImageNet VGG19 weights | `dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44` |
| `resnet50-13306192.pth` | stage-2 anime perceptual loss | Danbooru pretrained ResNet50 by M. Baas, https://github.com/RF5/danbooru-pretrained (credit M Baas) | `13306192d93e6f4346faf557b93f7e6577327cdf7bf59c6e95df04164a9fa2ca` |

## Training data (not included)

The models were trained on samples made on the fly from these sources (see `dmtrain/data.py`):

- REDS `train_sharp` (CC BY 4.0, per `data.py`)
- YouTube-VOS 2019 train
- super-resolution benchmark stills (BSD, DIV2K, Manga109, Urban100 and others)
- animation stills (styles "manga", "classic" and "tv")

The datasets, and `training/data/index.json` (the list of the files used), are not part of this repository.

## Software used

- **Training:** Python 3.12.7 with torch 2.11.0+cu128, torchvision 0.26.0+cu128, numpy 2.5.2, opencv-python
  4.14.0.94, pillow 12.3.0, scipy 1.18.1, and ultralytics 8.3.203 (AGPL-3.0, only to run Lada's detection teacher).
  An NVIDIA GPU was used.
- **ONNX export:** the export environment has torch 2.5.1+cpu, onnx 1.23.2, onnxruntime 1.31.0 and numpy 2.5.2.

These are the versions installed in those environments now. No separate record was kept of the exact versions used
for each run.
