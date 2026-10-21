# Experiments

28 runs: two detector families across fourteen dataset combinations. This file
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
| `kaggle_cybersimar` | Kaggle (cybersimar08) | does it work on a community-curated set? |
| `github_maciullo` | GitHub (Maciullo) | bird rejection on a third bird-positive source? |
| `hf_seraphim` | HuggingFace (Seraphim) | does it work on a hub-hosted distribution? |
| `kaggle_banderastepan` | Kaggle (banderastepan) | does more of the same community data help? |
| `dvb+mavvid` | both original bird datasets | is more bird data better? |
| `birds3` | all three bird-negative sources | is three times the bird data better still? |
| `dvb+mavvid+antiuav` | + Anti-UAV | bird suppression plus occlusion? |
| `birds3+antiuav` | three bird sources + Anti-UAV | bird rejection plus a second modality? |
| `all4` | the original four | is the combination worth deploying? |
| `all8` | everything | does the extra data help, and where does it cost? |

Models: `yolo11n` (fast, small) and `rtdetr_x2` (NMS-free, stronger on small
targets). 14 x 2 = 28.

```powershell
anti-uav matrix list                                    # all 28, with resolved settings
anti-uav matrix run --models yolo11n --combos all --dry-run
.\scripts\train_matrix.ps1 -Profile blackwell            # run them, resumable
```

`--dry-run` costs nothing and prints the resolved hyperparameters, the output
path and the dataset summary. Use it before every expensive job.

> The two cumulative slugs are short on purpose: `birds3` is the three
> bird-negative sources and `birds3+antiuav` adds Anti-UAV to them. The
> capability table is width-budgeted to fit an 80-column console, and
> `dvb+mavvid+github_maciullo` does not fit in a column.

---

## What to run first

You do not have time for all 28 on the first pass. Three runs answer most of it:

```powershell
anti-uav train --model yolo11n --combo dvb
anti-uav train --model yolo11n --combo github_maciullo
anti-uav train --model yolo11n --combo all8
```

- **`dvb`** — the combo that can falsify a false-positive claim. If precision
  here is poor, nothing else matters yet.
- **`github_maciullo`** — the *third* bird-negative source, and the cheapest one
  to fetch (a public git clone, no credentials). It answers "does bird rejection
  hold on a distribution the model has never seen" without a 14 GB download.
- **`all8`** — whether combining eight sources beats the best single one.

Then `mmuav` alone, because tiling is a different regime and you want to know
whether it works before it contaminates every combo.

### Training one model on many datasets

Yes — and that is the point of the cumulative combos. `anti-uav train --combo
all8` builds a single YOLO11n on all eight sources, and `notebooks/
kaggle_combined.ipynb` is that run end to end on Kaggle.

The comparison to make is **not** "is the combined mAP higher". It is:

- is the combined model at least as good on the *bird-negative* sources
  (`dvb`, `mavvid`, `github_maciullo`), and
- is it better on the sources it has never seen?

The first half is the regression check and the second is the whole hypothesis. A
model trained on `dvb` alone will score badly on `github_maciullo`; a model
trained on both should not.

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
github_maciullo      github...  -         -               -
dvb+mavvid           dvb,mavvid -         -               dvb
birds3               dvb,mav,gh -         -               dvb
dvb+mavvid+antiuav   dvb,mavvid -         antiuav         dvb
birds3+antiuav       dvb,mav,gh -         antiuav         dvb
all4                 dvb,mavvid mmuav     antiuav,mmuav   dvb,mmuav
all8                 dvb,mav,gh mmuav     antiuav,mmuav   dvb,mmuav
```

Note `all8`: it *can* falsify false positives, because `dvb`, `mavvid` and
`github_maciullo` are in it. That says nothing about how any of them behaved on
its own — which is why the per-dataset table is still the thing to read.

The alias lists are abbreviated when the table would otherwise wrap: the report
is printed through a Rich console that hard-wraps at the terminal width, and a
wrapped capability table is worse than none. The per-source table above the
combo table spells every alias out in full.

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