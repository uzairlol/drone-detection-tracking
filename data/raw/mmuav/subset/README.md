# data/raw/mmuav/subset — PLACEHOLDER EMPTY (this is the one you want)

Drop zone for the **MM-UAV RGB subset**. Tri-modal MOT benchmark, `drone` only.

Use this directory, not `../full`. See `../full/README.md` for why.

## Fill it with

```powershell
anti-uav download --dataset mmuav --max-sequences 150 --modalities rgb
```

## The honest situation

MM-UAV is the hardest and most valuable data in this project, and it is also the
most awkward to obtain:

- **Baidu Pan is the only distribution channel so far.** The Google Drive mirror
  is listed as "coming soon" on the project page. Share link and extraction code
  are in `configs/datasets/registry.yaml`.
- **It needs a cookie plus an interactive captcha.** There is no way to automate
  that. `anti-uav download` will report `needs_credentials` and print the link and
  code. Fetch with the official client, then re-run with `--skip-download` to
  build the frame index from what already landed.
- **~400 GB extracted.** That is why you subset. 150 RGB sequences is ~12 GB and
  is enough for every experiment in the matrix.

Full walkthrough, including the 400 GB -> 12 GB subsetting arithmetic:
`docs/DATASETS.md`.

## What you should see afterwards

Frames are already extracted by the publisher, so ingest is a copy and rename:

```
data/raw/mmuav/subset/.../<sequence>/
  rgb_frame/    *.jpg
  gt/           MOT ground truth (identity-preserving)
```

Then:

```powershell
anti-uav convert --dataset mmuav --variant subset
anti-uav stats   --dataset mmuav --variant subset
```

## Things that will bite you

- **12x5 px median target.** This is the single hardest fact in the project. At a
  640 px input a drone shrinks to roughly 6x3 px and is genuinely not detectable.
  Tiling is **mandatory** for any combo containing mmuav — the registry sets a
  `tiling_override` of 256 px tiles, upscaled 2.5x, which turns 12x5 into ~30x12.
  Do not "optimise" this away.
- **RGB frames are only 640x360.** That is why the tile is 256 px rather than
  640: a 640 px crop of a 640x360 frame would crop the whole frame and magnify
  nothing.
- **Event modality is ~1/3 of the payload and cannot train an RGB detector.** It is
  excluded by default via `exclude_globs`.
- **Strict trajectory annotation.** A UAV that leaves and returns keeps its
  original id. Most MOT metrics score that as an id switch, so numbers here are
  not directly comparable to MOTA on MOT17-style data.
- **Baidu throttles aggressively** and will interrupt multi-GB transfers. Budget
  several attempts; `scripts/fetch_wheel.py` exists because large resumable
  fetches are worth doing by hand.

## Why this dataset is in the matrix

Real multi-object MOT with identity preserved across leave-and-re-enter. That is
precisely what the global track manager's fragmentation repair has to cope with,
and it is the only dataset that makes `anti-uav track-eval` meaningful.

Dataset card: `configs/datasets/registry.yaml` -> `datasets.mmuav`