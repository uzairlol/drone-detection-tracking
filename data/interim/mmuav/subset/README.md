# data/interim/mmuav/subset — write with `anti-uav convert --dataset mmuav --variant subset`

Produced from `data/raw/mmuav/subset/`. See `../README.md` for the index format.

This is the variant the project defaults to (`default_variant: subset`), and the
one `anti-uav build --combo mmuav` reads.

Expected: `index.jsonl`, `convert_report.json`, `frames/<sequence>/rgb/*.jpg`,
`labels/<sequence>/rgb/*.txt`, **and identity-preserving MOT ground truth** in
the frame records.

## Track ids

`has_track_ids: true`. This is what makes `anti-uav track-eval` (MOTA, IDF1, HOTA,
ID switches) possible at all — it is the only dataset with multi-object identity
to score against. Anti-UAV has track ids too, but no birds; MM-UAV is the one
that makes the tracking benchmark real.

## Tiling is mandatory

A 12x5 px median target at a 640 px input becomes ~6x3 px and is not detectable.
The registry sets a `tiling_override` for this dataset (256 px tiles, 2.5x
magnification) and `configs/matrix.yaml` sets `force_tiling: true` on the `mmuav`
combo. If a build report says tiling was off for mmuav, stop — the metrics will
be meaningless and the failure will look like a bad model rather than a bad
config.

## Strict trajectories

A UAV that leaves frame and returns keeps its original id. Most MOT metrics score
that as an id switch, so these numbers are not directly comparable to MOTA on
MOT17-style data. The evaluator knows this; do not "fix" it by merging ids.