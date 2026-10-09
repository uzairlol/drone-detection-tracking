# data/raw/dvb/full — PLACEHOLDER EMPTY

Drop zone for **Drone-vs-Bird (WOSDETC / AVSS 2021)**, RGB, `drone` + `bird`.

Nothing here is committed. This file is the placeholder: it tells you what is
supposed to land in this directory before you spend an evening finding out.

## Fill it with

```powershell
anti-uav download --dataset dvb
```

## What you should see afterwards

The Kaggle mirror `romsham/dronevsbird-foryolo` unzips to a `Data/` directory:

```
data/raw/dvb/full/
  Data/
    images_drones/    image_000001.jpg ...     (0 = drone)
    labels_drones/    image_000001.txt ...     (YOLO: cls xc yc w h)
    images_birds/                                (1 = bird)
    labels_birds/
```

If you see that shape, the download is good. Then:

```powershell
anti-uav convert  --dataset dvb
anti-uav stats    --dataset dvb
```

## Things that will bite you

- **~14 GB.** Not small.
- **Needs Kaggle credentials.** `kaggle login --accept-dictionary`, which writes
  `~/.kaggle/kaggle.json`. See `data/raw/_credentials/README.md`.
- **The mirror uses bare integers**, not names: `0` = drone, `1` = bird. The
  converter cross-checks the observed class ratio against the published
  statistics and **aborts** if it looks inverted. That abort is the protection
  against silently training "drone = bird"; do not work around it.
- **Native resolution is 3840x2160.** `ingest.max_dimension: 1920` halves the
  storage with no measurable recall cost at a 28 px target.
- Sources run at 25-30 fps and `ingest.stride: 2` is applied at convert time, so
  roughly half the frames are kept.

## Why this dataset is in the matrix

It is the **only** source of genuine bird negatives besides MAV-VID, and the only
one where a precision number means something. A model that never sees birds will
happily classify birds as drones, which is the exact failure mode the anti-UAV
system exists to prevent. If you have to cut scope, keep this one.

Dataset card: `configs/datasets/registry.yaml` -> `datasets.dvb`
Acquisition notes: `docs/DATASETS.md`