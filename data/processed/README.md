# data/processed — one materialised YOLO dataset per experiment combo

Written by `anti-uav build`. This is the only layer ultralytics ever sees.

```powershell
anti-uav build --combo dvb
anti-uav build --combo all4
anti-uav sanity --combo all4     # gate. non-zero exit means do not train
```

## The seven combo directories

| directory | sources | tiling | why it is in the matrix |
| --- | --- | --- | --- |
| `dvb/` | dvb | 640 px | bird negatives, 28 px targets. The most informative single run. |
| `mavvid/` | mavvid | none | 171 px targets, pre-annotated. The sanity run. |
| `antiuav/` | antiuav | none | drone-only; second modality; visibility flags. |
| `mmuav/` | mmuav | 256 px | 12 px targets, MOT ground truth. |
| `dvb+mavvid/` | dvb, mavvid | per-dataset | two target scales in one model. |
| `dvb+mavvid+antiuav/` | + antiuav | per-dataset | first combo mixing RGB and thermal. |
| `all4/` | all four | per-dataset | candidate production model. |

The `+` in a slug is a literal directory name. Quote it in PowerShell.

## Order is not interchangeable

**convert -> splits -> build -> sanity.** `build` reads each frame record's
`split` field, which `splits` writes. Build before splitting and you get an empty
train set plus a `sanity` failure — not a subtle degradation, an outright one.

## What is in each directory

```
<combo>/
  data.yaml            the ultralytics dataset descriptor
  build_report.json    the verdict - read this, not just the return value
  images/train/...     the frames
  labels/train/...     the YOLO labels
  images/val/...
  labels/val/...
```

## build_report.json is the gate

`report.ok` must be `true` before you spend GPU hours. It records per-source frame
counts, the tiling actually applied, and anything skipped. If a source is missing
— say the Baidu transfer for mmuav never finished — the build **skips that source
with a warning** and the combo is quietly smaller than declared. The rest of the
matrix still runs, which is the intended behaviour, but it means a combo name is
not evidence that a dataset was in it. Check `skipped` before trusting a number.

## Tiling is per-dataset, not per-combo

A source is tiled when its `median_target_px` is at or below
`matrix.tiling_threshold_px` (40 px). So MM-UAV gets 256 px crops upscaled 2.5x
while MAV-VID's 171 px targets are left alone. Tiling the whole `dvb+mavvid`
combo would penalise MAV-VID for no reason; the cost is paid only where recall
lives.

## Disk

`build` copies images by default, so this tree is large. Use
`--no-copy-images` to symlink instead when disk is tight, or
`--max-frames-per-source` for a smoke configuration — though that last one
warns, correctly, that the resulting metrics will not reflect the full dataset.