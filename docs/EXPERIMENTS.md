# Experiments

14 runs: two detector families across seven dataset combinations. This file
explains what the matrix is for, how to run it, and — the part that matters most —
how to read the results without fooling yourself.

---

## The matrix

Combos, in planner order (single-source first, cumulative last):

| slug | sources | what it answers |
| --- | --- | --- |
| `dvb` | Drone-vs-Bird | does bird suppression work at all? |
| `mavvid` | MAV-VID | does it transfer to a different capture domain? |
| `antiuav` | Anti-UAV | does it track, with occlusion? |
| `mmuav` | MM-UAV | does it detect 12 px targets when tiled? |
| `dvb+mavvid` | both bird datasets | is more bird data better? |
| `dvb+mavvid+antiuav` | + Anti-UAV | bird suppression plus occlusion? |
| `all4` | all four | is the combination worth deploying? |

Models: `yolo11n` (fast, small) and `rtdetr_x2` (NMS-free, stronger on small
targets). 7 x 2 = 14.

```powershell
anti-uav matrix list                                    # all 14, with resolved settings
anti-uav matrix run --models yolo11n --combos all --dry-run
.\scripts\train_matrix.ps1 -Profile blackwell            # run them, resumable
```

`--dry-run` costs nothing and prints the resolved hyperparameters, the output
path and the dataset summary. Use it before every expensive job.

---

## What to run first

You do not have time for all 14 on the first pass. Two runs answer most of it:

```powershell
anti-uav train --model yolo11n --combo dvb
anti-uav train --model yolo11n --combo all4
```

- **`dvb`** — the only combo that can falsify a false-positive claim. If precision
  here is poor, nothing else matters yet.
- **`all4`** — whether combining four sources beats the best single one.

Then `mmuav` alone, because tiling is a different regime and you want to know
whether it works before it contaminates every combo.

---

## Reading results

### The number that matters is per-dataset, not combined

```powershell
anti-uav evaluate --run artifacts/runs/yolo11n/all4 --cross
```

A single combined mAP over four datasets is close to meaningless, because the
datasets disagree about what a hard sample is. MM-UAV's 12 px targets dominate the
loss; Drone-vs-Bird's birds dominate the precision. Read the per-dataset table.

### Three of the four datasets cannot falsify precision

`antiuav` and `mmuav` contain **no birds**. Precision measured on them is
unfalsifiable: a detector that labels every bird as a drone would score perfectly.
The evaluator prints a note next to those numbers rather than leaving you to
remember.

Real precision evidence comes only from `dvb` and `mavvid` — **two** sources, not
one. `matrix.yaml` has a combo called `dvb+mavvid` for exactly that reason.

### Check what a run is allowed to claim, before you spend the GPU hours

```powershell
anti-uav matrix list        # per-source and per-combo capability tables
```

Every dataset and every combo carries an evidential contract, derived once in
`src/anti_uav/data/capabilities.py` from `configs/datasets/registry.yaml`:

```
combo                birds      identity  trackGT         tiled
dvb                  dvb        -         -               dvb
mavvid               mavvid     -         -               -
antiuav              -          -         antiuav         -
mmuav                -          mmuav     mmuav           mmuav
dvb+mavvid           dvb,mavvid -         -               dvb
dvb+mavvid+antiuav   dvb,mavvid -         antiuav         dvb
all4                 dvb,mavvid mmuav     antiuav,mmuav   dvb,mmuav
```

Note `all4`: it *can* falsify false positives, because `dvb` and `mavvid` are in
it. That says nothing about how either behaved on its own — which is why the
per-dataset table is still the thing to read.

`anti-uav track-eval` enforces the tracking half: it **refuses** a dataset with no
MOT ground truth and **warns** on single-target ground truth that IDF1 and ID
switches will be meaningless.

### Compare like with like

- Same image size. A combo with `mmuav` tiles at 256 px; the others train at
  640. That is not a like-for-like comparison of architectures, and the log says
  so per run.
- Same profile. `batch` is `-1` (auto) by default and resolves differently per GPU.
  Runs from different boxes are not comparable.
- Same tiling. `anti-uav matrix list` prints `tiling: on/off` per run.

### What "good" looks like

| result | what it means |
| --- | --- |
| high mAP on `dvb` and `mavvid` | the detector generalises across capture domains |
| high recall on `mmuav` | tiling is working; if it is not, check tile size before blaming the model |
| good IDF1, high ID switches | the detector is fine and the **tracker** is fragmenting — go look at `track-eval` |
| high mAP everywhere, poor `all4` | the sources are interfering; check the class balance `anti-uav stats` prints |
| `rtdetr_x2` beats `yolo11n` on `mmuav` | expected: RT-DETR is NMS-free and does better on small, densely packed targets. It is the more expensive answer. |

### The tracker is a separate experiment

Detection metrics say nothing about tracking. Score them separately:

```powershell
anti-uav track-eval --run artifacts/runs/yolo11n/all4 --dataset mmuav
anti-uav track-eval --run artifacts/runs/yolo11n/all4 --dataset mmuav --trackers sort,bytetrack,botsort
```

This needs MOT ground truth, which means MM-UAV. And MM-UAV's strict trajectory
annotation means its scores are **not** comparable to MOTA on MOT17-style data —
read them as absolute, not relative to published numbers.

To understand a specific failure rather than a score:

```powershell
anti-uav replay --run artifacts/runs/yolo11n/all4 --dataset mmuav --sequence 0007 --tracker botsort
anti-uav replay --run artifacts/runs/yolo11n/all4 --dataset mmuav --sequence 0007 --tracker bytetrack --dump out/
```

`replay` needs no ground truth. Watch the same clip with two trackers and the
break will be obvious.

---

## Recording a result

Runs write `artifacts/runs/<model>/<combo>/` containing `best.pt`, `last.pt`,
`results.csv`, `args.yaml` and a metadata file. That directory is the record —
copy it, do not regenerate it.

```powershell
anti-uav list-runs
anti-uav serve     # the UI reads runs/ and shows the matrix as a table
```

---

## Cost

Rough, from the per-run estimate `matrix list` prints (multiply by the frame count
`anti-uav build` reports; tiled combos are 4–6x that):

| where | throughput guide | 14 runs |
| --- | --- | --- |
| CPU | ~0.3 it/s at batch 4, AMP off | not viable |
| RTX 1070 (Pascal) | ~12 it/s at batch 8 | days, and FP16 is unusable |
| RTX 4090 | ~90 it/s at batch 96 | hours |
| RTX 5090 | ~150 it/s at batch 128 | hours |

The estimate is an order-of-magnitude guide, not a measurement. `matrix list`
prints it per run with the actual batch and AMP state so you can compare against
a known run.

If you only have one GPU-day, run `yolo11n` on `dvb` and `all4` first. They are
the two that change what you do next.

---

## Reproducibility

- Seeds are explicit and honoured. The split generator uses a stable hash, not
  Python's per-process-salted `hash()`, so **the same seed gives the same split in
  a different process** — there is a regression test that runs a subprocess to
  prove it.
- Splits are **by sequence**, so adjacent frames cannot straddle train and val.
  `anti-uav sanity` fails the build if one does.
- `configs/coverage/coverage_map.yaml` is generated by
  `scripts/gen_coverage_map.py`. Regenerate it rather than editing by hand.

```powershell
anti-uav sanity --combo all4      # before every training run
```

---

## A note on what 14 runs can and cannot tell you

This is four public datasets, two architectures, and one site model. It answers
"which of these combinations is worth deploying on this hardware", and it will
not answer "how well does this detect an unseen drone type at 40 m against a
cluttered sky". Four datasets is a design choice made under a time budget, not a
claim of coverage. Treat the ranking as a ranking; treat the absolute numbers as
specific to these four sources.