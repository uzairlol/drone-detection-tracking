# Commands

Every command the project exposes, in the order you actually run them. `anti-uav
--help` is authoritative; this file explains *when* to use each one and what it
will refuse to do.

```powershell
anti-uav --help
anti-uav <command> --help
```

Substitute your environment's interpreter if `anti-uav` is not on `PATH`:

```powershell
& "D:\envs\ml\python.exe" -m anti_uav.cli <command>
```

---

## 0. Environment

```powershell
.\scripts\setup_env.ps1                              # into the current conda env, CPU profile
.\scripts\setup_env.ps1 -Profile blackwell -UpgradeTorch   # on the RTX 5090 box
.\scripts\setup_env.ps1 -Profile pascal -UseLockFile -UpgradeTorch   # pinned set, GTX 1070 box
.\scripts\setup_env.ps1 -EnvName ml -DataRoot D:\drone-data
anti-uav verify-env
anti-uav fetch-weights
```

`verify-env` is the one command worth running before anything else. It checks
the interpreter, the torch build against the GPU's compute capability, the
checkpoints, the disk you are about to fill, and prints the exact remedy for
anything wrong. Read [ENVIRONMENT.md](ENVIRONMENT.md) for the profile table and
the lock file.

**Exit codes matter.** `verify-env`, `fetch-weights` and `sanity` return non-zero
when they find a blocking problem. Everything else returns 0 unless the command
itself failed, so in a script:

```powershell
anti-uav verify-env
if ($LASTEXITCODE -ne 0) { throw "fix the environment first" }
```

### Base weights

Each recipe in `configs/train/` names a starting checkpoint — `yolo11n.pt`,
`rtdetr-x2.pt`. Ultralytics downloads a missing one from *inside* the trainer,
so on a machine that cannot reach the release host the run dies after the dataset
is built and the run directory created. Prefetch instead:

```powershell
anti-uav fetch-weights --dry-run    # what is present, what is missing
anti-uav fetch-weights              # download the missing ones
anti-uav fetch-weights --models yolo11n   # only the edge candidate
```

`verify-env` reports the same facts as a `[WARN] base-weights` line — a warning,
not a failure, because data prep, evaluation, the API and the test suite all work
without checkpoints.

---

## 1. Data acquisition

Always `--dry-run` first. Two of these datasets are 60 GB and 400 GB.

```powershell
anti-uav download --dataset all --dry-run          # what would happen, sizes, credentials
anti-uav download --dataset dvb
anti-uav download --dataset mavvid
anti-uav download --dataset antiuav --variant 300
anti-uav download --dataset mmuav --max-sequences 150 --modalities rgb
```

| flag | effect |
| --- | --- |
| `--dataset` | `dvb`, `mavvid`, `antiuav`, `mmuav`, or `all` |
| `--variant` | `300` / `410` / `600` for Anti-UAV; ignored for the single-release datasets |
| `--source` | pick one distribution channel when a dataset has several |
| `--dry-run` | print plans and touch nothing |
| `--max-sequences`, `--modalities` | MM-UAV subsetting; see [DATASETS.md](DATASETS.md) |
| `--skip-download` | re-run the pipeline against files already fetched |

**What will not automate.** MM-UAV is Baidu-Pan-only and needs a cookie plus an
interactive captcha; the command reports `needs_credentials` and prints the share
link and extraction code instead of failing obscurely. Fetch it with the official
client, then re-run with `--skip-download`.

---

## 2. Preprocessing

```powershell
anti-uav convert --dataset all        # normalise into the interim frame index
anti-uav stats   --dataset all        # per-dataset stats + bird-negative table
anti-uav splits  --combo all4         # assign whole SEQUENCES to train/val
anti-uav build   --combo all4         # materialise a YOLO dataset
anti-uav sanity  --combo all4         # invariants; non-zero exit means do not train
anti-uav dedup   --dataset mmuav      # pHash near-duplicates across the split
```

The order matters and is not interchangeable: **convert → splits → build →
sanity**. `build` reads each record's split field, so building before splitting
produces an empty train set and a `sanity` failure.

| command | what it refuses to do |
| --- | --- |
| `convert` | stops on an unreadable source rather than silently skipping frames; reports unknown labels separately |
| `splits` | will not split by frame unless you pass `strategy=file`, which warns that the resulting metrics are upper bounds |
| `build` | refuses to run if a required source has no converted frames |
| `sanity` | fails if any sequence appears in two splits, if a val split has too few sequences, or if median target size is below `min_target_px` |

Build per combo, not per dataset — the combos are what you train on:

```powershell
anti-uav build --combo dvb
anti-uav build --combo dvb+mavvid
anti-uav build --combo all4
```

---

## 3. Training

```powershell
anti-uav matrix list
anti-uav matrix run --models yolo11n,rtdetr_x2 --combos all --dry-run
anti-uav matrix run --models yolo11n --combos all --profile blackwell
anti-uav matrix run --models yolo11n --combos dvb --profile turing   # Kaggle T4
.\scripts\train_matrix.ps1 -Profile blackwell          # 14 runs, resumable
```

Or one cell at a time. The flag is `--combo`, not `--dataset`:

```powershell
anti-uav train --model yolo11n --combo all4 --epochs 120
anti-uav train --model rtdetr_x2 --combo dvb --dry-run
```

`--dry-run` prints the resolved hyperparameters, the output path and the dataset
summary without allocating a GPU. Use it whenever you are about to start an
expensive run — it is the cheapest way to catch a wrong `--dataset`.

The matrix runs single-dataset combos first and cumulative ones last, so a
failure tells you which source caused it.

---

## 4. Evaluation

```powershell
anti-uav list-runs
anti-uav evaluate --run artifacts/runs/yolo11n/all4
anti-uav evaluate --run artifacts/runs/yolo11n/all4 --cross       # the table worth reading
anti-uav evaluate --run artifacts/runs/yolo11n/all4 --cross --cross-datasets all4
```

`--cross` scores the run against **each dataset's held-out split separately**,
which is the only way to see that a model trained on everything did not just
learn MM-UAV's 12 px targets. A single combined mAP hides that.

The evaluator adds a falsifiability note: for `antiuav` and `mmuav`, precision
cannot falsify a false-positive claim because neither contains bird annotations.
It is printed with the number rather than left for you to remember.

### Tracking and replay

```powershell
anti-uav track-eval --run artifacts/runs/yolo11n/all4 --dataset mmuav
anti-uav track-eval --run artifacts/runs/yolo11n/all4 --dataset mmuav --trackers sort,bytetrack,botsort
anti-uav replay     --run artifacts/runs/yolo11n/all4 --dataset mmuav --variant subset
anti-uav replay     --run artifacts/runs/yolo11n/all4 --dataset mmuav --sequence 0007 --tracker bytetrack
```

- `track-eval` **scores** trackers against MOT ground truth (MOTA, IDF1, HOTA,
  ID switches). Needs a dataset with track ids — that means MM-UAV. Detection is
  computed once per sequence and reused across trackers, so comparing three
  trackers is one inference pass, not three. `--reid <ckpt>` selects the
  appearance model explicitly (default: the newest
  `artifacts/runs/reid/*/best.pt`); `--no-reid` ablates the gate for the A/B row.
- `replay` needs **no** ground truth. Use it to see *why* a track breaks on a
  specific clip: switch `--tracker` and watch the same sequence again. `--dump`
  writes per-frame MOT rows (xyxy layout — see `artifacts/tracks/README.md`).

### Single predictions

```powershell
anti-uav predict --run artifacts/runs/yolo11n/all4 --source clip.mp4
anti-uav predict --run artifacts/runs/yolo11n/all4 --source frame.jpg --save out/ --tracker botsort
anti-uav predict --run artifacts/runs/yolo11n/all4 --source clip.mp4 --stride 2 --max-frames 300
```

`--tracker` uses this project's own trackers, not ultralytics' — see
[ARCHITECTURE.md](ARCHITECTURE.md) for why. `predict` prints the per-detection
table rather than a count, because 40 detections at conf 0.25 and 3 at 0.8 are
completely different situations.

---

## 5. Export and deploy

```powershell
anti-uav export --run artifacts/runs/yolo11n/all4 --formats onnx,tensorrt --fp16
anti-uav deploy render --output artifacts/deploy --rtsp-host 10.0.0.5 --engine ./model.engine
anti-uav deploy render --node node-00          # one node only, while iterating
```

`export` writes ONNX, a TensorRT engine and the matching `nvinfer` config. FP16 is
unsafe on Pascal and the profile system will tell you so.

`deploy render` writes one DeepStream pipeline per node plus the on-edge probe
source. It reports a warning per node when the engine file is missing, listing
the exact `trtexec` command to build it — it will not fail the render, because
the engine is usually built on the target device.

---

## 6. The interface

```powershell
anti-uav serve                      # http://127.0.0.1:8000
anti-uav serve --host 0.0.0.0 --port 9000
anti-uav serve --reload             # development
```

Serves the operator console and the JSON API from one process. Everything the UI
shows is available as `GET /api/...`, including `/api/rules/explain`, which
evaluates a hypothetical track against the live rule set — that is how you answer
"why was this not alerted?" without re-running anything.

The bundle is offline: Chart.js is vendored under `ui/static/vendor/`. There is no
CDN reference, because a site that cannot reach the internet is exactly the site
this runs on.

---

## 7. Inspecting configuration

```powershell
anti-uav version
anti-uav config show registry        # or: rules, coverage, matrix, or a config path
anti-uav config datasets
anti-uav rules show                  # the active thresholds, exactly as written
anti-uav rules validate              # cross-rule tensions
anti-uav rules explain               # run the engine over recorded tracks, gate by gate
```

`rules validate` reports *cross-rule* tensions — thresholds that are individually
valid but contradict each other. It is advisory: a listed tension is a decision
for you to make, not a broken config. `rules explain` needs recorded tracks, so
it is most useful against a run you have already produced.

---

## Pipeline script

`scripts/run_pipeline.ps1` runs the whole chain in order, resumable:

```powershell
.\scripts\run_pipeline.ps1 -DryRun                 # print every stage, touch nothing
.\scripts\run_pipeline.ps1 -Profile pascal         # all 13 stages
.\scripts\run_pipeline.ps1 -Combos dvb -Models yolo11n
.\scripts\run_pipeline.ps1 -From train             # data is ready, start here
.\scripts\run_pipeline.ps1 -Stage sanity           # one stage
.\scripts\run_pipeline.ps1 -Skip download          # already have the data
```

Stages, in order:

```
verify-env -> fetch-weights -> download -> convert -> stats
           -> splits -> build -> sanity
           -> train -> evaluate -> track-eval -> export -> deploy render
```

Two properties make it worth using rather than pasting the list:

- **`verify-env` and `sanity` are hard gates.** A non-zero exit from either stops
  the run there. A 14-run matrix on a box with the wrong torch build, or a
  training set with a sequence leaked across the train/val boundary, wastes days
  in ways that are only obvious in hindsight.
- **It is resumable.** Every stage skips itself when its output already exists —
  `data/raw/<dataset>/` is populated, `index.jsonl` exists, `splits.json` exists,
  `build_report.json` has `ok: true`, `weights/best.pt` exists. Re-run after an
  interruption and it picks up where it stopped.

`-Combos` defaults to `all4`, not `all`: building all seven combos copies every
frame seven times. Ask for `all` only when you mean to run the whole matrix and
have the disk for it.

`scripts/train_matrix.ps1` still exists for training alone. Use `run_pipeline.ps1`
when you are starting from nothing.

---

## Exit codes

| code | meaning |
| --- | --- |
| 0 | success |
| 1 | the command failed, or `verify-env` / `fetch-weights` / `sanity` found a blocking problem |
| 2 | bad command line (typo in a flag or dataset name) |

Every command prints the remedy with the error. If one of them does not tell you
what to do next, that is a bug worth reporting.