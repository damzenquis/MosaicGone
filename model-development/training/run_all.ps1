# Trains the detector, then the restorer (stage 1), one after the other on this PC's GPU.
# Output: training/runs/det_v1 and training/runs/res_s1 (log.csv, val.csv, best.pt, last.pt, val_*.jpg, stdout.txt).
# Safe to stop (close the window / end python.exe) and restart: both scripts resume from last.pt.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:YOLO_OFFLINE = "1"
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force runs\det_v1, runs\det_v2, runs\det_v3, runs\res_s1, runs\s2_pilot_real, runs\s2_pilot_anime, runs\s2_pilot_nogan, runs\s2_pilot_noganime, runs\s2_pilot_perc02, runs\s2_pilot_perc05, runs\s2_pilot_lines, runs\s2_pilot_lines_gated, runs\s2_pilot_animbranch, runs\s2_pilot_animbranch_ap, runs\s2_pilot_colour_faces, runs\res_s2 | Out-Null
# det_v1 finished (40000 steps); kept for reference
# & $py train_detector.py --run runs/det_v1 --iters 40000 --batch 32 --workers 8 --resume *>> runs\det_v1\stdout.txt
# det_v2: fine-tune of det_v1 on data without "flat area = obscured" labels (false alarms on clean video), plus tiny
# mosaics; det_v1 is the frozen reference for the keep loss and the floors (runs/det_v2_try1 = an aborted first try)
# det_v2 finished (14000 steps, best.pt = step 10000); kept for reference
# & $py train_detector.py --run runs/det_v2--init runs/det_v1/last.pt --ref runs/det_v1/last.pt --iters 14000 --lr 3e-4 --warmup 300 --batch 32 --workers 8 --resume *>> runs\det_v2\stdout.txt
# res_s1 finished (80000 steps, best.pt = step 78000); kept for reference
# & $py train_restorer.py --run runs/res_s1 --iters 80000 --t 16 --batch 2 --workers 8 --resume *>> runs\res_s1\stdout.txt
# det_v3: det_v2 + animation stills (TRAINING DATA/Animation), mostly to stop false alarms on halftone / screentone /
# flat fills (det_v2 flags 5 of 61 held-out clean stills; floor: at most 1); det_v2 is the reference for keep loss + floors
# det_v3 finished (10000 steps): anime false alarms 5 -> 3 of 61, misses the floor of 1; det_v2 stays the detector
# & $py train_detector.py --run runs/det_v3 --init runs/det_v2/best.pt --ref runs/det_v2/best.pt --anime 0.3 --iters 10000 --lr 3e-4 --warmup 300 --batch 32 --workers 8 --resume *>> runs\det_v3\stdout.txt
# restorer stage 2 pilots A + B finished (3000 steps each): both made onepiece worse (psnr, vgg, seam, flicker)
# & $py train_restorer_s2.py --run runs/s2_pilot_real --anime 0 --iters 3000 --resume *>> runs\s2_pilot_real\stdout.txt
# & $py train_restorer_s2.py --run runs/s2_pilot_anime --anime 0.35 --iters 3000 --resume *>> runs\s2_pilot_anime\stdout.txt
# pilot C (no GAN) finished: just as bad on onepiece -> not the GAN; the perceptual loss (weight 0.2) swamps the gt loss
# & $py train_restorer_s2.py --run runs/s2_pilot_nogan --anime 0.35 --gan 0 --iters 3000 --resume *>> runs\s2_pilot_nogan\stdout.txt
# pilot D (GAN on real footage only) stopped at ~step 250: C already answered its question
# & $py train_restorer_s2.py --run runs/s2_pilot_noganime --anime 0.35 --gan_anime 0 --iters 3000 --resume *>> runs\s2_pilot_noganime\stdout.txt
# pilots E + F finished: perc 0.02 (E) is the better one; onepiece still slips (missing line art)
# & $py train_restorer_s2.py --run runs/s2_pilot_perc02 --anime 0.35 --perc 0.02 --gan_anime 0 --iters 3000 --resume *>> runs\s2_pilot_perc02\stdout.txt
# & $py train_restorer_s2.py --run runs/s2_pilot_perc05 --anime 0.35 --perc 0.05 --gan_anime 0 --iters 3000 --resume *>> runs\s2_pilot_perc05\stdout.txt
# pilots G + H, on the fixed still-image clips (obscuring + crop bug): G = line art x3 on animation, H = G + teacher loss on
# animation only where the teacher is closer to the truth; pilot E (re-checked at start) is a floor too (--ref)
# G + H finished: G (s2_pilot_lines) is the go-to model (better than E everywhere but onepiece vgg); H broke real_seam
# & $py train_restorer_s2.py --run runs/s2_pilot_lines --anime 0.35 --perc 0.02 --gan_anime 0 --line_w 3 --ref runs/s2_pilot_perc02/last.pt --iters 3000 --resume *>> runs\s2_pilot_lines\stdout.txt
# & $py train_restorer_s2.py --run runs/s2_pilot_lines_gated --anime 0.35 --perc 0.02 --gan_anime 0 --line_w 3 --out_gated 0.25 --ref runs/s2_pilot_perc02/last.pt --iters 3000 --resume *>> runs\s2_pilot_lines_gated\stdout.txt
# animation branch pilot: G + APISR-style animation clips (codec damage, mild line-sharpened targets, New_Dataset tv frames
# at half the animation clips, tv val set); starts from G; floors = stage 1 + the record of G / H / E; --keep pulls
# towards G wherever G is closer to the truth (no drift below G, on real footage or animation)
# stopped at step 350 (user's call): replaced by the combined pilot below, which adds the anime perceptual loss
# & $py train_restorer_s2.py --run runs/s2_pilot_animbranch --init runs/s2_pilot_lines/last.pt --ref runs/s2_pilot_lines/last.pt runs/s2_pilot_lines_gated/last.pt runs/s2_pilot_perc02/last.pt --keep 1 --anime 0.35 --perc 0.02 --gan_anime 0 --line_w 3 --line_sharp 0.5 --tv 0.5 --d_warmup 200 --iters 3000 --resume *>> runs\s2_pilot_animbranch\stdout.txt
# combined pilot: the animation branch above + the anime perceptual loss (Danbooru ResNet50, half of the perceptual loss on
# animation frames; real frames VGG only)
# & $py train_restorer_s2.py --run runs/s2_pilot_animbranch_ap --init runs/s2_pilot_lines/last.pt --ref runs/s2_pilot_lines/last.pt runs/s2_pilot_lines_gated/last.pt runs/s2_pilot_perc02/last.pt --keep 1 --anime 0.35 --perc 0.02 --gan_anime 0 --line_w 3 --line_sharp 0.5 --tv 0.5 --anime_perc 0.5 --d_warmup 200 --iters 3000 --resume *>> runs\s2_pilot_animbranch_ap\stdout.txt
# combined pilot finished: better than G / H / E on real footage, tv, manga, classic; onepiece flat, manga / classic tint;
# its step 3000 saved by the user as candidate_step3000.pt = the go-to model
# colour + faces pilot: from the candidate, + colour loss and half the animation obscured regions on dense line art (faces / hair);
# onepiece reported only (does not block best.pt); records = G / H / E / candidate, --keep towards the candidate
# colour + faces pilot finished: beats the candidate on real / tv / manga / classic; manga colour 0.019 short of E's record
# (no best.pt); its step 3000 copied to pilot_step3000.pt = the start of res_s2
# & $py train_restorer_s2.py --run runs/s2_pilot_colour_faces --init runs/s2_pilot_animbranch_ap/candidate_step3000.pt --ref runs/s2_pilot_animbranch_ap/candidate_step3000.pt runs/s2_pilot_lines/last.pt runs/s2_pilot_lines_gated/last.pt runs/s2_pilot_perc02/last.pt --keep 1 --anime 0.35 --perc 0.02 --gan_anime 0 --line_w 3 --line_sharp 0.5 --tv 0.5 --anime_perc 0.5 --chroma 1 --line_mask 0.5 --info_sets onepiece --d_warmup 500 --iters 3000 --resume *>> runs\s2_pilot_colour_faces\stdout.txt
# res_s2 (full stage 2): the colour + faces settings, 40000 steps, from that pilot's step 3000; records = pilot / candidate /
# G / H / E, --keep towards the pilot
# resumed at step 14750 (user's call) with --tone 5 (the "colour" drift was the obscured area turning darker: manga
# -1.7 levels, not tint) and --out_gate_real (the teacher is the weaker model on real footage: follow it only where closer)
# resumed at step 19500 (user's call): --tone 5 -> 2 (fixed manga / classic but overshot brighter: onepiece +0.84 levels,
# real colour creeping), + --edge 0.05 (animation: edge strength near the true lines no weaker than the truth's, against
# blurred faces), + opapp = the onepiece clip at the app's 1.5x crop (report only)
& $py train_restorer_s2.py --run runs/res_s2 --init runs/s2_pilot_colour_faces/pilot_step3000.pt --ref runs/s2_pilot_colour_faces/pilot_step3000.pt runs/s2_pilot_animbranch_ap/candidate_step3000.pt runs/s2_pilot_lines/last.pt runs/s2_pilot_lines_gated/last.pt runs/s2_pilot_perc02/last.pt --keep 1 --anime 0.35 --perc 0.02 --gan_anime 0 --line_w 3 --line_sharp 0.5 --tv 0.5 --anime_perc 0.5 --chroma 1 --tone 2 --edge 0.05 --out_gate_real --line_mask 0.5 --info_sets onepiece opapp --d_warmup 500 --iters 40000 --resume *>> runs\res_s2\stdout.txt
$trainOk = $LASTEXITCODE -eq 0
# only when training really finished (a stopped / killed run exits non-zero and must not look done)
if ($trainOk) {
    "ALL TRAINING DONE $(Get-Date)" | Out-File -Append runs\done.txt
    # started as a one-time scheduled task: remove it now that training is done
    Unregister-ScheduledTask -TaskName "MosaicGoneTraining" -Confirm:$false -ErrorAction SilentlyContinue
} else {
    "STOPPED (exit $LASTEXITCODE) $(Get-Date)" | Out-File -Append runs\stopped.txt
}
