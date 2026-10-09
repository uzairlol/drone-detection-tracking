# Architecture

How the pieces fit, and why the non-obvious decisions were made. Read
[`diagrams/system block diagram.md`](diagrams/system%20block%20diagram.md) for
the block-level picture of the deployed system, and
[`diagrams/ml pipeline diagram.md`](diagrams/ml%20pipeline%20diagram.md) for the
research pipeline that produces the weights it consumes; this file explains the
choices behind both.

---

## Shape

```
raw datasets
  -> convert          normalise to one interim frame index, one label space
  -> splits           whole SEQUENCES to train/val
  -> build            materialise a YOLO dataset, tile per source
  -> sanity           prove the invariants before spending GPU hours
        |
        v
   trainer  ->  Detector  ->  local tracker  ->  global fusion  ->  rules
        |                         |                   |              |
        v                         v                   v              v
     eval, export          coverage + handoff    geofence, risk   alert / PTZ
     DeepStream            + scheduling          + recovery
```

Data flows one way and every stage is a command. Nothing runs implicitly; nothing
trains or downloads by itself.

---

## Decisions worth defending

### The detector is permissive, the rule layer is strict

Inference runs at `conf_threshold: 0.25`. Alerting happens at
`rules.confidence.initiate: 0.60`.

Raising the detector threshold to 0.60 starves the tracker. The tracker's job is
to turn a noisy stream of detections into a persistent identity, and it cannot do
that if the low-confidence detections that carry the motion evidence are already
discarded. The rule layer exists precisely so that precision can be bought *after*
tracking, using persistence and kinematics and cross-camera agreement, rather
than by throwing away the evidence early.

Two different stages, two different jobs: the detector maximises recall, the rules
maximise precision.

### Rules report `skipped`, never `passing`

Only `FAIL` blocks an alert. A gate whose input is unavailable — no ground plane,
no horizon — returns `SKIPPED` with a reason naming the setting to configure.

The consequence is worth stating plainly: **an uncalibrated site runs three of the
five gates** (confidence, persistence, cross-camera) and reports the other two as
skipped. That is a deliberate choice. Blocking all alerts during bring-up would
mean a silent site, and the reasons list makes the gap visible on every
evaluation. Calibrate, and all five run. Both halves are tested.

### Cross-camera confirmation is not optional

`cross_camera.min_cameras: 2`, and a local track with no global identity is a
**failure**, not a skip. One view cannot distinguish a drone from a bird with
enough confidence to alert, and letting a lone per-camera detection raise an alert
would reintroduce exactly the false positives the system exists to prevent.

This is why the global tracker and the coverage map are not optional extras: they
are load-bearing for correctness.

### Splits are by sequence, and `sanity` enforces it

All four datasets are video-derived. Adjacent frames are near-identical, so a
frame-level split leaks and inflates mAP by double digits. `sanity` fails the
build if any sequence appears in two splits.

### Tiling is per-source, not per-combo

A source is tiled when its `median_target_px` is at or below
`matrix.tiling_threshold_px` (40 px). So MM-UAV tiles at 256 px — six tiles per
640x360 frame, ~2.5x upscale — while MAV-VID's 171 px targets are left alone.

Tiling everything would multiply the cost of every run by 4-6x for no benefit on
the datasets that do not need it.

### Stable hashing, not `hash()`

`rng_for(seed, *stream)` feeds a **stable** hash into the RNG. Python salts string
hashing per process, so using `hash()` gave every run a different train/val split
— which makes the seed a lie and the experiment matrix irreproducible. There is a
regression test that spawns a subprocess to prove the property holds.

### This project owns its trackers

ultralytics 8.4 ships no `track` task: `persist` and `tracker` are not in the
predict config, and `model.predict(..., persist=True)` raises
`SyntaxError: 'persist' is not a valid YOLO argument`.

So video inference uses `predict_image` per frame and feeds this project's own
SORT / ByteTrack / BoT-SORT. That has a second benefit: the live path and the
`track-eval` metrics harness share **one** association implementation, so a
number from the harness describes what the pipeline does.

### RT-DETR feeds the same tracker

RT-DETR is NMS-free — its decoder emits a fixed, already-deduplicated set of
queries. The boxes arrive through the same `.boxes` attribute as YOLO's, so the
conversion path is identical for both families. That is what makes one tracker
stack legitimate for both models.

### The profile system is not decoration

CUDA 12.8 removed sm_61, so a default `pip install torch` **cannot run on a GTX
1070 at all**. Blackwell (sm_120) needs CUDA 12.8+. `configs/train/overrides/`
plus `verify-env` turn both into a startup error rather than a training run
wasted.

---

## Layer by layer

### `config/`

Every tunable lives in YAML and is validated by pydantic on load. A bad threshold
is a load-time error, not a surprise at epoch 40. Environment variables
(`ANTI_UAV_*`) override individual fields without editing files.

### `data/`

`download` → `convert` → `splits` → `build` → `sanity`.

Converters are per-dataset because the sources are genuinely different: mp4 +
JSON, YOLO txt, MOT gt. They all emit the same `FrameRecord` index, and everything
downstream reads only that.

The converters **report rather than guess**. Unknown labels are counted, class
mappings are cross-checked against published statistics, and a mismatch aborts.

### `data/capabilities.py` — what a run is allowed to claim

Four public datasets do not support the same conclusions. Only `dvb` and `mavvid`
carry birds, so only they can turn precision into evidence about false positives.
Only `mmuav` has multi-object identity, so only it makes IDF1 and ID-switch
counts meaningful — `antiuav` has MOT ground truth but is single-target, which
satisfies a naive "has track ids" check while making identity metrics vacuous.

Those facts were already in the registry and were already honoured in four
separate places. That works, but adding a fifth dataset then means remembering
four places, and forgetting one is a silent hole rather than a failure. So the
derivation lives here once, and `matrix list`, `evaluate --cross`, the `track-eval`
gate, `resolve_tiling` and `/api/capabilities` all read from it.

The pipeline does not fork per dataset. One combo, one build, one trainer, and
weights per combo — forking would delete the cumulative combos and with them the
production model, and would make "do the sources interfere?" unanswerable.

### `detection/`

One `Trainer` for two model families — they share a data pipeline, an evaluator
and an export path, and the only thing that differs is the kwargs passed to
ultralytics. `matrix.py` plans the 14 runs; `predictor.py` loads one;
`evaluate.py` scores it, including per-dataset cross-evaluation.

`augment.py` holds the two augmentations ultralytics does not provide — cutout
and IR-grayscale — applied to the batch tensor from an `on_train_batch_start`
callback rather than deleted, because the intent behind both is real: cutout is
what teaches the backbone not to fire on a bird, and IR-grayscale is what stops it
keying on palette as a modality shortcut when RGB and thermal are mixed in one
training set. Routing them through albumentations would have meant depending on
an optional extra this project does not install.

### `tracking/`

- `local/` — SORT, ByteTrack, BoT-SORT. Association, state, and the metrics
  harness (MOTA, IDF1, HOTA, Anti-UAV accuracy). All three accept an optional
  `global_motion` homography on `update()`, which removes a PTZ pan or a mast
  sway from the association *and* from the velocity measurement — correcting only
  the first leaves the camera's motion being learned as the target's.
- `global_tracker/` — cross-camera identity fusion on a metric-space Kalman
  filter.
- `coordination/` — the section-3 stack: coverage geometry, risk scoring, handoff,
  recovery, scheduling, PTZ.
- `appearance/` + `reid_model.py` + `reid_train.py` — a small ReID embedding for
  the appearance gate. Deliberately cheap: it runs per detection inside a
  DeepStream pipeline already batch-inferring 13 streams.
- `replayer.py` — offline detect → track → score. Detection is computed **once per
  sequence** and replayed per tracker, so a three-tracker comparison is one
  inference pass rather than three and any difference between rows is the
  tracker's rather than the detector's.

### `rules/`

Five gates in order: confidence (with hysteresis), persistence, kinematics,
spatial, cross-camera. `explain()` returns every gate's verdict and the value that
produced it, which is exposed at `POST /api/rules/explain` so you can ask "why was
this not alerted?" against the live rule set.

### `deploy/`

Renders one DeepStream pipeline per node from `coverage_map.yaml`, plus the
on-edge probe source and an NvDCF config. It warns rather than fails when the
engine file is missing, and prints the `trtexec` command to build it — because
the engine is usually built on the target device, not here.

### `api/` + `ui/`

FastAPI plus a no-build console. Chart.js is vendored: this runs on a site that
may not reach the internet, so there is no CDN reference anywhere.

---

## Coverage, and one honest wrinkle

The 100-camera map is generated, not typed, from a site model:
820 x 640 m, 8 nodes, 80 fixed + 20 PTZ, perimeter and interior placement.

Four of the five protected zones are covered by fixed cameras. The fifth,
`pz-approach-west`, is covered **only by a reachable PTZ pose** — the fence-mounted
cameras are tilted ~8 degrees up to watch the sky, which means they cannot see
the ground at their own feet.

That is not a modelling bug, it is the actual geometry. `coverage_report` reports
the distinction explicitly (`fixed_covered_zones` vs `ptz_only_zones`) rather than
collapsing it into a boolean, so the scheduler knows a PTZ must be pre-positioned
there.

---

## Testing

```powershell
python -m pytest              # 295 tests, no GPU, no datasets, ~25 s
.\scripts\setup_env.ps1 -Extras dev
```

The suite is organised around the invariants rather than around coverage:
no sequence straddles a split, every protected zone is watched, each rule gate
rejects the case it exists to catch, seeds are reproducible across processes, and
the trained-model paths work on a real random-initialised checkpoint.

Seven smoke scripts cover the end-to-end paths that pytest would need a GPU or a
100 GB download to reach:

```powershell
python scripts/smoke_data.py          # convert -> split -> build -> sanity, synthetic
python scripts/smoke_tracking.py      # trackers, metrics, appearance, on a scripted scenario
python scripts/smoke_coordination.py  # the whole section-3 stack on the real coverage map
python scripts/smoke_rules.py         # every rule gate, including that each one rejects
python scripts/smoke_api.py           # every endpoint, with FastAPI's TestClient
python scripts/smoke_cross_eval.py    # the per-dataset evaluation table
python scripts/smoke_commands.py      # predict and replay against a real checkpoint
```