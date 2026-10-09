# data/interim/antiuav/300 — write with `anti-uav convert --dataset antiuav --variant 300`

Produced from `data/raw/antiuav/300/`. See `../README.md` for the index format.

Expected: `index.jsonl`, `convert_report.json`, `frames/<sequence>/rgb/*.jpg`,
`labels/<sequence>/rgb/*.txt`.

## Two modalities, two frame trees

Anti-UAV300 ships RGB **and** IR. The converter emits both, and the modality is
part of the normalised path so `group.pattern` can tell them apart. RGB and IR
are **unaligned** — the publisher states this explicitly — so do not assume pixel
correspondence between them; the fusion modules treat them as independent
sequences.

`ingest.stride: 3` turns 30 fps sources into ~10 fps.

## What to check

- `convert_report.json` reports which object spelling it saw. `target`, `uav`,
  `drone`, `0` and `1` all map to `drone` across different challenge releases.
- This dataset has **no bird annotations at all**
  (`has_bird_negatives: false`). `anti-uav stats` prints a class-coverage table
  that will show a bird count of zero. That is the dataset's nature, not a
  conversion failure — and it is why precision measured here is reported as
  unfalsifiable rather than as a number you can act on.
- Sequences with a video but no `.json` are skipped with a note. Read the notes.

## Visibility flags

This is the only source of official per-frame visibility flags, so the converted
index carries them. That is what makes it the natural dataset for validating the
rule layer's persistence and occlusion handling, and for MOT-style local-tracker
ground truth.

The `410` and `600` variants write to sibling directories and are IR only.