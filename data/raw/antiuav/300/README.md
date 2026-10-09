# data/raw/antiuav/300 — PLACEHOLDER EMPTY

Drop zone for **Anti-UAV300** (INEE / Zhao et al.), RGB **and** IR, `drone` only.
This is the publisher's recommended variant and the project's default.

## Fill it with

```powershell
anti-uav download --dataset antiuav --variant 300
```

Ships from Google Drive (id `1NPYaop35ocVTYWHOYQQHn8YHsM9jmLGr`, extraction
code `sagx`). No account needed. The archive extracts to an `extracted/` subtree.

## What you should see afterwards

One directory per sequence, each holding a video and a JSON annotation:

```
data/raw/antiuav/300/extracted/.../<sequence>/
  <sequence>.mp4
  <sequence>.json
```

The JSON shapes vary between challenge releases; the converter accepts the
documented variants and reports the one it found. Sequences with a video but no
JSON are skipped with a note rather than silently dropped.

Then:

```powershell
anti-uav convert  --dataset antiuav --variant 300
anti-uav stats    --dataset antiuav --variant 300
```

## Things that will bite you

- **~60 GB.** Budget the disk before you start; `anti-uav verify-env` checks free
  space against the declared size and refuses if short.
- **NO BIRD NEGATIVES.** Anti-UAV is drone-only. A model trained on this alone has
  never been told what a bird looks like, so its validation precision will look
  excellent and mean nothing. This is why `dvb` and `mavvid` are in the matrix.
- **RGB and IR are UNALIGNED** — the publisher says so explicitly. Do not assume
  pixel correspondence between modalities; the fusion modules treat them as
  independent sequences.
- Label spellings differ across releases: `target`, `uav`, `drone`, `0`, `1` all
  map to `drone`. The converter reports which spelling it saw.
- `ingest.stride: 3` — 30 fps sources become ~10 fps, ample for a fixed camera.

## Why this dataset is in the matrix

The only source of **official per-frame visibility flags**, so it is the natural
place to validate the rule layer's persistence and occlusion handling, and the
natural source of MOT-style ground truth for the local-tracker benchmark.

Dataset card: `configs/datasets/registry.yaml` -> `datasets.antiuav`
Acquisition notes: `docs/DATASETS.md`