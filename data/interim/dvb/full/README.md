# data/interim/dvb/full — write with `anti-uav convert --dataset dvb`

Produced from `data/raw/dvb/full/`. See `../README.md` for the index format.

Expected: `index.jsonl`, `convert_report.json`, `frames/vid_images_*/rgb/*.jpg`,
`labels/vid_images_*/rgb/*.txt`.

## The check that matters here

`dvb` is the only dataset where precision is falsifiable, so the converter's
class-ratio cross-check is load-bearing rather than decorative. The Kaggle mirror
labels `0` = drone and `1` = bird; if the observed ratio looks inverted the
conversion **aborts**.

If you hit that abort, the mirror's convention changed or you mixed up
`images_drones` and `images_birds`. Do not edit the mapping to get past it —
check which directory you actually downloaded. `has_bird_negatives: true` in the
registry is the whole reason this dataset is in the matrix, and a build where
"bird" secretly means "drone" is worse than no build.

`anti-uav stats --dataset dvb` prints the bird-negative table; it should show a
meaningful bird count.