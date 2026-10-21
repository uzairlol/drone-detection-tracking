# data/raw/mavvid/full — PLACEHOLDER EMPTY

Drop zone for **MAV-VID** (Multirotor Aerial Vehicle VID, Cranfield), RGB,
`drone` + `bird`.

## Fill it with

```powershell
anti-uav download --dataset mavvid
```

The registry prefers the **Bitbucket** mirror
(`https://bitbucket.org/alejodosr/mav-vid-dataset`), because the original Kaggle
page is dead. This dataset needs **no credentials**.

The mirror is not a frame tree: it holds one `mav_vid_dataset.zip` (11.5 GB). The
git backend shallow-clones it, unpacks the zip into this directory, and removes
the clone (including `.git`) so the payload is not stored twice. Bitbucket's raw
and archive URLs return nothing for a file this size, so the clone is the only
working transport.

## What you should see afterwards

Already YOLO-annotated and pre-split by the publisher, so ingest is a copy and
rename rather than a video decode. Roughly:

```
data/raw/mavvid/full/
  train/
    images/   *.jpg
    labels/   *.txt
  val/
    images/
    labels/
```

Then:

```powershell
anti-uav convert  --dataset mavvid
anti-uav stats    --dataset mavvid
```

## Things that will bite you

- **~11.5 GB** as shipped (the registry used to say 6 GB, which cost people a
  failed free-space check). It is still the smallest download in the matrix.
- Label spelling varies between releases (`drone` / `Drone` / `0`). The converter
  normalises case and numeric ids, then asserts every observed label was mapped.
  It prints which spelling it actually saw — read that line.
- Captured from other drones, ground cameras **and handheld devices**. The
  handheld subset has ego-motion, so it behaves differently at test time than the
  fixed-camera sequences that dominate deployment.
- `ingest.stride: 1` — frames are already extracted, do not thin them.

## Why this dataset is in the matrix

Easiest one: 215x128 px targets, pre-annotated, ~40k frames. Good for a first
sanity training run and as the "easy" column of the comparison table — it
separates "the detector is wrong" from "the dataset is hard". If you want one run
to prove the whole pipeline works end to end before touching the hard data, make
it `yolo11n --combo mavvid`.

Dataset card: `configs/datasets/registry.yaml` -> `datasets.mavvid`
Acquisition notes: `docs/DATASETS.md`