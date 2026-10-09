# artifacts/tracks — tracker row dumps

Written by `anti-uav track-eval` (one summary per dataset/variant) and by
`anti-uav replay --dump` (one row file per sequence).

## Row format

`replay --dump <dir>` writes `<dataset>_<sequence>_<tracker>.txt` in the MOT
layout this project uses throughout:

```
frame,id,x1,y1,x2,y2,conf,class,visibility
```

1-based frame numbers, `class` always `1` and `visibility` always `1.0` for
predictions.

**This is xyxy, not the MOTChallenge `bb_left,bb_top,bb_width,bb_height`.** It
matches MM-UAV's own `gt.txt` and matches what `anti-uav convert` writes into
`data/interim/<dataset>/<variant>/mot/`, so our rows and our converted ground
truth are mutually readable by `load_mot_ground_truth`. A toolkit that assumes
the MOT17 column order will misread these rows — convert the 3rd–6th columns
first (`w = x2 - x1`, `h = y2 - y1`).

These files are also the thing to read when you want to know *why* a track broke
rather than just that it did: dump the same sequence with two different
`--tracker` values and diff the rows.