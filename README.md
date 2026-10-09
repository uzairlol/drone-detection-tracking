# Drone Detection & Tracking

Research pipeline for the 100-camera anti-UAV system: dataset acquisition,
conversion and leakage-safe splitting, training and evaluating **YOLO11n** and
**RT-DETR-x2** across four public datasets, then local multi-object tracking,
cross-camera identity fusion, and the camera-coordination layer.

The system design this implements is in
[`docs/diagrams/system block diagram.md`](docs/diagrams/system%20block%20diagram.md)
(Mermaid source; `hr.svg` / `mermaid-diagram.png` / the PDF are exports).

That diagram is the **deployed** half. The **research** half — where the data
comes from, how a detector is trained, how the trackers are scored, and what is
exported — is in
[`docs/diagrams/ml pipeline diagram.md`](docs/diagrams/ml%20pipeline%20diagram.md).

> **Nothing in this repo trains or downloads by itself.** Every stage is a command
> you run. See [`docs/COMMANDS.md`](docs/COMMANDS.md) for the full list, or start
> with `anti-uav --help`.

---

## The one-paragraph version

Four public datasets, harmonised into a single two-class `drone` / `bird` label
space, split **by sequence** so no video's adjacent frames straddle the train/val
boundary, and used in seven combinations. Two detectors are trained on all seven —
fourteen runs — sharing one data pipeline, one evaluator and one tracker stack.
The hard part is not the detectors; it is that three of the four datasets are
**drone-only**, so precision measured on them cannot falsify a false-positive
claim, and that the smallest targets (MM-UAV, 12×5 px) are invisible at a 640 px
input without tiling. Both are handled explicitly rather than assumed away.

---

## Datasets

| alias | dataset | modality | frames | median target | bird negatives | acquisition |
| --- | --- | --- | --- | --- | --- | --- |
| `dvb` | Drone-vs-Bird (WOSDETC/AVSS) | RGB | 104,760 | 34×23 px | **yes** | Kaggle |
| `mavvid` | MAV-VID (Cranfield) | RGB | 40,232 | 215×128 px | **yes** | Bitbucket |
| `antiuav` | Anti-UAV300 (INEE) | RGB+IR | ~225,000 | 125×59 / 52×29 px | no | Google Drive |
| `mmuav` | MM-UAV (tri-modal MOT) | RGB (+IR) | 2.8 M (subset) | **12×5 px** | no | Baidu Pan |

`antiuav` and `mmuav` contain no bird annotations. Any precision number from a
combo built only from those is **unfalsifiable** — the tooling refuses to present
it as if it meant something. See [`docs/DATASETS.md`](docs/DATASETS.md), including
the MM-UAV acquisition walkthrough and the 400 GB → 12 GB subsetting arithmetic.

Every dataset slot ships as an empty directory with a `README.md` saying what
belongs in it, how big it is, whether it needs credentials, and the exact command
that fills it — see `data/raw/dvb/full/README.md` for the pattern. Credentials go
in `data/raw/_credentials/` and are gitignored; only `*.example` templates are
tracked.

---

## Quick start

```powershell
# 0. environment (reuses an existing conda env; see docs/ENVIRONMENT.md)
.\scripts\setup_env.ps1 -Profile pascal -UseLockFile
anti-uav verify-env
anti-uav fetch-weights

# 1. data. Always --dry-run first: two of these are 60 GB and 400 GB.
anti-uav download --dataset all --dry-run
anti-uav download --dataset dvb
anti-uav download --dataset mavvid
anti-uav download --dataset antiuav --variant 300
anti-uav download --dataset mmuav --max-sequences 150 --modalities rgb

# 2. preprocess - in this order, and the order matters
anti-uav convert --dataset all
anti-uav stats   --dataset all
anti-uav splits  --combo all4
anti-uav build   --combo all4
anti-uav sanity  --combo all4

# 3. train (14 runs: 2 models x 7 combos)
anti-uav matrix list
anti-uav matrix run --models yolo11n,rtdetr_x2 --combos all --dry-run
.\scripts\train_matrix.ps1 -Profile blackwell -DryRun

# 4. evaluate, track, export
anti-uav evaluate       --run artifacts/runs/yolo11n/all4 --cross
anti-uav track-eval --run artifacts/runs/yolo11n/all4 --dataset mmuav
anti-uav replay     --run artifacts/runs/yolo11n/all4 --dataset mmuav --sequence 0007
anti-uav predict    --run artifacts/runs/yolo11n/all4 --source clip.mp4 --save out/
anti-uav export     --run artifacts/runs/yolo11n/all4 --formats onnx,tensorrt --fp16

# 5. the interface
anti-uav serve      # http://127.0.0.1:8000
```

### Or all of it in one command

```powershell
.\scripts\run_pipeline.ps1 -DryRun      # print all 13 stages, touch nothing
.\scripts\run_pipeline.ps1 -Profile pascal
```

Stages run in order, each skipping itself when its output already exists, so it
is safe to re-run after an interruption. `verify-env` and `sanity` are hard gates:
a non-zero exit from either stops the run there rather than wasting GPU hours on a
box with the wrong torch build or a leaked split.

Full command reference: **[docs/COMMANDS.md](docs/COMMANDS.md)**.

---

## Layout

```
configs/          every tunable, in one place
  datasets/       registry: sources, licences, class maps, grouping
  train/          yolo11n.yaml, rtdetr_x2.yaml, per-GPU overrides
  matrix.yaml     the 14-run experiment matrix
  rules/          drone_rules.yaml  - section 4 of the diagram
  coverage/       coverage_map.yaml  - section 3, 100 cameras (generated)
src/anti_uav/
  config/         pydantic schemas; every YAML validated at load
  data/           download / convert / harmonise / split / build / verify
  detection/      one trainer for two families, export, evaluation
  tracking/
    local/        SORT, ByteTrack, BoT-SORT
    global_tracker/  cross-camera identity fusion
    coordination/ coverage, risk, handoff, recovery, scheduling, PTZ
    metrics.py    MOTA, IDF1, HOTA, Anti-UAV accuracy
    replayer.py   offline detect -> track -> score harness
  rules/          the threshold engine
  deploy/         DeepStream pipeline + probes, NvDCF config
  api/ + ui/      FastAPI and a no-build operator console
scripts/          setup_env, run_pipeline, train_matrix, gen_coverage_map
tests/            pytest
requirements-lock.txt   115 pinned packages, verified against this suite
data/             raw / interim / processed  (gitignored payloads; README stubs kept)
artifacts/        runs / exports / tracks  (gitignored; README stubs kept)
docs/             commands, datasets, experiments, environment, architecture
```

## Design notes worth knowing before you change anything

* **Splits are by sequence, never by frame.** Adjacent frames of a video are
  near-identical; a frame-level split leaks and inflates mAP by double digits.
  `anti-uav sanity` fails the build if a sequence appears in two splits.
* **Tiling is per-dataset, not per-combo.** A source is tiled when its
  `median_target_px` is at or below `matrix.tiling_threshold_px`, so MM-UAV gets a
  256 px crop upscaled 2.5× while MAV-VID's 171 px targets are left alone.
* **The detector is permissive and the rule layer is strict.** Inference runs at
  `conf_threshold: 0.25`; alerting happens at
  `rules.confidence.initiate: 0.60`. Raising the detector threshold to 0.60
  starves the tracker.
* **The profile system is not decoration.** CUDA 12.8 removed sm_61, so a default
  `pip install torch` cannot run on a GTX 1070 at all.
  `configs/train/overrides/<profile>.yaml` plus `anti-uav verify-env` make that a
  startup error instead of a training run wasted. `requirements-lock.txt`
  therefore omits torch deliberately: one file cannot pin five GPU tiers, so
  `-Profile` still owns it and `verify-env` still checks the pairing.
* **The checkpoints are prefetched, not discovered mid-run.** Ultralytics
  downloads a missing `base_weights` from inside the trainer, so on a blocked
  network the run dies after the dataset is built. `anti-uav fetch-weights` moves
  that failure to the point where it costs five seconds.
* **Rules report `skipped`, never `passing`.** A gate whose input is missing
  (no calibration, no global identity) says so, so a missing input is never
  mistaken for a satisfied rule.

## Documentation

| file | what is in it |
| --- | --- |
| [docs/COMMANDS.md](docs/COMMANDS.md) | every command, in run order, with what it refuses to do |
| [docs/DATASETS.md](docs/DATASETS.md) | the four datasets, how to fetch each, the MM-UAV 400 GB → 12 GB subsetting |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | the 14-run matrix, what to run first, how to read results without fooling yourself |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | GPU profiles, why CUDA 12.8 breaks a GTX 1070, extras, disk planning |
| [notebooks/kaggle_dvb.ipynb](notebooks/kaggle_dvb.ipynb) | end-to-end run on Kaggle T4x2: install, download, preprocess, train, evaluate |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how it fits together and why the non-obvious decisions were made |
| [docs/diagrams/](docs/diagrams/) | the system block diagram (Mermaid source plus exports) and the ML pipeline diagram |

## Tests

```powershell
python -m pytest                      # 295 tests, no GPU, no datasets, ~25 s
python -m ruff check src tests scripts
python -m mypy
```

Eight smoke scripts cover the end-to-end paths pytest cannot reach without a GPU
or a 100 GB download. They run on CPU in a few seconds:

```powershell
python scripts/smoke_data.py          # convert -> split -> build -> sanity, synthetic
python scripts/smoke_tracking.py      # trackers, metrics, appearance, on a scripted scenario
python scripts/smoke_coordination.py  # the whole section-3 stack on the real coverage map
python scripts/smoke_rules.py         # every rule gate, including that each one rejects
python scripts/smoke_api.py           # every endpoint, via FastAPI's TestClient
python scripts/smoke_cross_eval.py    # the per-dataset evaluation table
python scripts/smoke_commands.py      # predict and replay against a real checkpoint
```
