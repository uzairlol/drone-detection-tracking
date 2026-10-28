# Handover — Drone Detection and Tracking

Read this first. It records the state of the project at transfer: what works, what
does not, what is missing, and what to do next. The rest of the documentation
describes the intended system; this file describes what actually exists today.

---

## Honest summary

The **software is complete and verified**. The **experiment matrix is not**. One
of the twenty-eight planned training runs exists, and it is the right first run.
Twenty-seven runs, the real per-dataset evaluation table, and the tracking
benchmark are the work that remains.

Do not present the contents of `artifacts/tracks/` or the zero-valued rows in
`artifacts/runs/_eval/cross_yolo11n.json` as benchmark results. They are smoke
outputs from four-to-six-frame toy sequences. The one real result is described
below.

---

## Status at a glance

| area | state |
| --- | --- |
| Data download, conversion, splitting, building, sanity | implemented; DVB download and conversion exercised locally |
| Detection training and evaluation (YOLO11n, RT-DETR-x2) | implemented; one YOLO11n run complete |
| Local tracking (SORT, ByteTrack, BoT-SORT) and MOT metrics | implemented and unit-tested; **no real benchmark yet** |
| Cross-camera fusion, coverage, rules, API, UI | implemented and smoke-tested |
| Export and DeepStream deployment rendering | ONNX exported; **no TensorRT engine built or site-tested** |
| Tests | `347 passed` |
| Lint and typecheck | `ruff` clean; `mypy` clean across 77 source files |
| Smoke scripts | all 7 exit 0 |
| Documentation | complete and audited for handover |
| Raw datasets | **not included**; local `data/` is partial (~9 GB) |
| Trained artifacts | 36 MB; include `artifacts/` in the handover archive |

---

## Environment

The verified environment on the handover machine is a conda env at:

```
D:\envs\ml\python.exe
```

| package | version |
| --- | --- |
| Python | 3.11.17 |
| torch | 2.13.0+cu126 |
| torchvision | 0.28.0+cu126 |
| ultralytics | 8.4.171 |
| opencv | 5.0.0.93 |
| numpy | 2.4.6 |
| pytest | 9.1.1 |
| ruff | 0.16.10 |
| mypy | 2.4.0 |

The machine has an NVIDIA GTX 1070 (sm_61), so the `pascal` profile applies here.
On the RTX 4090 or RTX 5090 boxes, use `-Profile ada` or `-Profile blackwell`
respectively. Do **not** reuse this machine's torch wheel on those cards.

```powershell
# On the handover machine
.\scripts\setup_env.ps1 -EnvName ml -Profile pascal -UseLockFile
& "D:\envs\ml\python.exe" -m anti_uav.cli verify-env
& "D:\envs\ml\python.exe" -m anti_uav.cli fetch-weights
```

If `anti-uav` is already on `PATH`, the shorter `anti-uav ...` form is equivalent.
The global `python` on this machine is 3.13 and does **not** have `ultralytics`
or `cv2`; always use the `ml` interpreter explicitly.

Full profile and lock-file guidance: [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

---

## The one real result

`artifacts/runs/yolo11n/yolo11n__dvb/` is a complete 120-epoch YOLO11n run on the
Drone-vs-Bird combo, trained on Kaggle T4x2 (`turing` profile, batch 32, 640 px,
AMP, ~4 h 35 m).

| metric (best validation epoch) | value |
| --- | --- |
| mAP50-95 | **0.7687** |
| mAP50 | 0.9186 |
| precision | 0.9640 |
| recall | 0.8658 |

This is a genuine result, and it is also a narrow one:

- It is **DVB only**. It says nothing yet about MAV-VID, MM-UAV, or the new
  Kaggle/HuggingFace sources.
- DVB carries bird annotations, so its precision is meaningful. That is why this
  run was chosen first.
- It was trained on two T4s. A run on a 4090/5090 uses a different batch and is
  **not** a like-for-like comparison unless the batch is pinned.

The ONNX export of this run is at
`artifacts/exports/yolo11n__dvb/best_640.onnx`. There is no TensorRT engine; it
must be built on the target device with `trtexec` (the command is printed by
`anti-uav deploy render`).

---

## What is not finished

### 1. Twenty-seven of twenty-eight runs

`configs/matrix.yaml` plans 2 models × 14 combos. Only `yolo11n` / `dvb` exists.
The recommended order if time is limited:

```powershell
# 1. The three runs that answer the main questions
& "D:\envs\ml\python.exe" -m anti_uav.cli train --model yolo11n --combo dvb
& "D:\envs\ml\python.exe" -m anti_uav.cli train --model yolo11n --combo github_maciullo
& "D:\envs\ml\python.exe" -m anti_uav.cli train --model yolo11n --combo all8

# 2. Tiling regime, before it contaminates the cumulative combos
& "D:\envs\ml\python.exe" -m anti_uav.cli train --model yolo11n --combo mmuav

# 3. The accuracy reference, starting with the same two combos
& "D:\envs\ml\python.exe" -m anti_uav.cli train --model rtdetr_x2 --combo dvb
& "D:\envs\ml\python.exe" -m anti_uav.cli train --model rtdetr_x2 --combo all8
```

`dvb` is already done, but the command is listed for completeness. Always
`--dry-run` first. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for why these
runs, and for the full matrix.

### 2. A real cross-dataset table

`artifacts/runs/_eval/cross_yolo11n.json` is **partial and misleading**:

- `dvb` has a real 2,690-image evaluation and matches the run above.
- `mavvid` is empty because it has no held-out split yet.
- `antiuav` and `mmuav` show `0.0` because they were evaluated against six
  images each.

Regenerate it after the data is in place:

```powershell
& "D:\envs\ml\python.exe" -m anti_uav.cli splits --datasets mavvid
& "D:\envs\ml\python.exe" -m anti_uav.cli evaluate `
    --run artifacts/runs/yolo11n/yolo11n__dvb --cross
```

### 3. A real tracking benchmark

`artifacts/tracks/*.json` were produced on toy sequences (4–5 frames, 3 ground-
truth boxes per sequence). The negative MOTA and zero IDF1 are artefacts of that
input, not a tracker failure. Regenerate on real MM-UAV data:

```powershell
& "D:\envs\ml\python.exe" -m anti_uav.cli track-eval `
    --run artifacts/runs/yolo11n/yolo11n__dvb --dataset mmuav `
    --trackers sort,bytetrack,botsort
```

MM-UAV is Baidu-Pan-only. Follow the walkthrough in
[`docs/DATASETS.md`](docs/DATASETS.md) before expecting this to run.

### 4. Deployment

`anti-uav deploy render` produces the eight node pipelines, but the TensorRT
engine is deliberately not in the repo: it is built on the target Jetson/edge
device. Nothing here has been run against live RTSP or on a DeepStream node.

### 5. Repository housekeeping

These do not affect correctness, but they should be fixed before the repository
is published or transferred:

- **No `LICENSE` file**, although `pyproject.toml` declares MIT.
- `pyproject.toml` has `Homepage = "https://example.invalid/anti-uav"` and
  `authors = [{ name = "Anti-UAV internship" }]`.
- The git remote is personal (`github.com:uzairlol/drone-detection-tracking`),
  and the Kaggle notebooks hardcode the same URL in `REPO_URL`.
- `notebooks/kaggle_dvb.executed.ipynb` is a stale 19 MB output.
- Root `yolo11n.pt` and `weights/yolo26n.pt` are stray downloads.
- `data/processed/does-not-exist/` is an empty test artefact.
- The local `Drone/` folder (271 MB) is not part of the pipeline.

---

## What is in the repository, and what is not

| path | contents | in git? |
| --- | --- | --- |
| `src/anti_uav/` | the package: data, detection, tracking, rules, deploy, api, ui | yes |
| `configs/` | datasets registry, recipes, matrix, rules, coverage map | yes |
| `scripts/` | setup and pipeline scripts, smoke tests | yes |
| `tests/` | 347-test suite | yes |
| `docs/` | commands, datasets, experiments, environment, architecture, diagrams | yes |
| `notebooks/` | seven Kaggle notebooks (one per dataset plus a combined run) | yes, except outputs |
| `artifacts/` | one run, ONNX export, rendered deploy configs, partial evals | **no** (gitignored) |
| `data/` | local working data; only DVB is downloaded (~9 GB) | **no** (gitignored) |
| `Drone/` | unrelated local videos | **no** (gitignored) |

The handover archive is expected to include `artifacts/` (36 MB) so the trained
model and export travel with the code. It is expected to exclude `data/`,
`Drone/`, caches, the executed notebook, and loose `*.pt` downloads.

---

## Reproduce the verification

```powershell
$py = "D:\envs\ml\python.exe"

& $py -m pytest -q
& $py -m ruff check src tests scripts
& $py -m mypy

& $py scripts\smoke_data.py
& $py scripts\smoke_tracking.py
& $py scripts\smoke_coordination.py
& $py scripts\smoke_rules.py
& $py scripts\smoke_api.py
& $py scripts\smoke_cross_eval.py
& $py scripts\smoke_commands.py
```

Last verified: **2026-10-28**. Expected result: 347 tests pass, ruff reports no
issues, mypy reports no issues in 77 source files, every smoke script exits 0.

The training data is not needed for any of the above; the suite uses synthetic
fixtures and the smoke scripts use scripted scenarios.

---

## Caveats that change how results should be read

1. **Five of eight datasets contain no birds.** `antiuav`, `mmuav`,
   `kaggle_cybersimar`, `hf_seraphim`, and `kaggle_banderastepan` cannot falsify a
   false-positive claim. Only `dvb`, `mavvid`, and `github_maciullo` can. The
   evaluator prints this next to each row.
2. **MM-UAV is tiled; nothing else is.** A combo containing MM-UAV trains at a
   different effective resolution. Architecture comparisons across tiled and
   untiled combos are not like-for-like.
3. **Splits are by sequence.** Never introduce a frame-level split on this data;
   adjacent frames leak and inflate mAP by double digits.
4. **Profiles pin different batches.** Runs from different GPUs/profiles are not
   directly comparable unless the batch is pinned explicitly.
5. **MM-UAV MOTA is not MOT17 MOTA.** Its trajectory annotation is strict; read
   the number as absolute, not against published benchmarks.
6. **The absolute numbers apply to these eight sources only.** The matrix is a
   ranking exercise, not a claim about unseen drones at an arbitrary site.

---

## Ownership and credentials

No credentials are committed. Only `*.example` templates are tracked under
`data/raw/_credentials/`. Real Kaggle tokens and Baidu cookies belong in that
directory and are gitignored.

Before transfer, confirm:

- [ ] GitHub repository ownership or a new remote under the receiving account.
- [ ] Kaggle account access for `dvb`, `kaggle_cybersimar`, and
      `kaggle_banderastepan`.
- [ ] Access to the Baidu Pan share for MM-UAV (link and code are in
      `configs/datasets/registry.yaml`).
- [ ] Access to the RTX 4090 / RTX 5090 training boxes and the site hardware.
- [ ] A named owner for the dataset licences; two sources restrict
      redistribution.

| role | name | contact |
| --- | --- | --- |
| Previous owner | _fill in_ | _fill in_ |
| Receiving owner | _fill in_ | _fill in_ |
| GPU / site access | _fill in_ | _fill in_ |

---

## Recommended first week

| day | task |
| --- | --- |
| 1 | Run the verification block above. Confirm `verify-env` and `anti-uav version` point at `D:\envs\ml`. |
| 2 | Download DVB and MAV-VID; convert, split, build, and sanity-check `all8`. Read `build_report.json`, not just the exit code. |
| 3 | Run `yolo11n` on `github_maciullo` and `all8`; regenerate the cross-dataset table. |
| 4 | Fetch ~150 MM-UAV RGB sequences and run the real `track-eval` comparison. |
| 5 | Start `rtdetr_x2` on `dvb` and `all8`; write the first honest results table into `docs/EXPERIMENTS.md`. |
| 6–7 | Build the TensorRT engine on the target device, render deploy configs, and run one live RTSP stream through DeepStream. |

Do not start the full 28-run matrix until the tiling path (`mmuav`) and the
tracking path have produced one believable result each.
