# Datasets

Eight datasets, harmonised into one two-class label space. This file covers what
each one actually contains, how to get it, and — more importantly — what each one
can and cannot be used to prove.

---

## The one-paragraph version

Five of the eight datasets contain **no birds at all**. They are drone-only
detection and tracking datasets. A model trained on Anti-UAV alone will report
excellent precision, and that number will mean nothing, because nothing in the
training set ever resembled a bird. Only Drone-vs-Bird, MAV-VID, and GitHub
Maciullo can falsify a false-positive claim. This is why the matrix trains on all
fourteen dataset combinations rather than "the best dataset": `dvb` is the one
that tests whether the detector rejects what it should.

The other structural fact is target size. Median targets run from 28 px
(Drone-vs-Bird) down to **12 px** (MM-UAV). At a 640 px input a 12 px drone
becomes roughly 6x3 px, which is below what any of these detectors reliably sees.
Tiling is therefore mandatory for MM-UAV and is applied per-source rather than
per-combo.

---

## At a glance

| alias | dataset | modality | frames | seqs | median target | birds | MOT GT | identity | ~size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dvb` | Drone-vs-Bird | RGB | 104,760 | 77 | 28 px | **yes** | no | no | 14 GB |
| `mavvid` | MAV-VID | RGB | 40,232 | 64 | 171 px | **yes** | no | no | 6 GB |
| `antiuav` | Anti-UAV300 | RGB+IR | ~225,000 | 300 | 92 px | no | yes | no (1 target) | 60 GB |
| `mmuav` | MM-UAV | RGB+IR+event | 2.8 M | 1,321 | **12 px** | no | yes | **yes** | 380 GB |
| `kaggle_cybersimar` | Kaggle (cybersimar08) | RGB | 8,500 | 50 | 64 px | no | no | no | 2.5 GB |
| `github_maciullo` | GitHub (Maciullo) | RGB | 12,000 | 30 | 80 px | **yes** | no | no | 3 GB |
| `hf_seraphim` | HuggingFace (Seraphim) | RGB | 15,000 | 40 | 72 px | no | no | no | 4 GB |
| `kaggle_banderastepan` | Kaggle (banderastepan) | RGB | 20,000 | 60 | 56 px | no | no | no | 5 GB |

The `birds` and `identity` columns decide which conclusions each source can
support, and they are different questions:

- **birds** — can precision here falsify a false-positive claim? Only `dvb`,
  `mavvid`, and `github_maciullo` answer yes. A detector that labelled every bird
  as a drone scores perfectly on the other five, because they have no bird to
  misclassify.
- **identity** — can IDF1 and ID-switch counts mean anything? Only `mmuav`
  answers yes. `antiuav` has MOT ground truth but is **single-target**: every box
  is id 1, so "has track ids" is true while every identity trivially matches
  every other. MOTA and HOTA are still meaningful there; IDF1 and IDSW are not.

`src/anti_uav/data/capabilities.py` derives this table from
`configs/datasets/registry.yaml` and every surface reads from it —
`anti-uav matrix list`, `evaluate --cross`, the `track-eval` gate, and
`GET /api/capabilities`.

Total if you take everything: ~475 GB extracted. A working subset — Anti-UAV300,
Drone-vs-Bird, MAV-VID, GitHub Maciullo, ~150 MM-UAV sequences, and the three
small Kaggle/HuggingFace drone-only sets — is roughly **105 GB**. Check your
volume first; `anti-uav verify-env` does it for you.

> **Status at handover:** raw data is not committed. The local `data/` tree
> contains a partial DVB download (~9 GB) and its converted frames; treat it as a
> working copy, not the full eight-dataset corpus.

---

## Before you download anything

Every dataset slot already exists as an empty directory with a `README.md`
explaining what belongs in it, how big it is, whether it needs credentials, and
the exact command that fills it. They are the fastest way to answer "what belongs
here" — read the file rather than reconstructing it.

```
data/raw/<dataset>/<variant>/README.md          what the download produces
data/raw/_credentials/README.md                which datasets need auth, and how
data/interim/<dataset>/<variant>/README.md     what convert writes, and how to read it
data/processed/README.md                        the fourteen combo directories
data/manifests/README.md                        observed vs published figures
```

Credentials are never committed. `data/raw/_credentials/` holds only
`*.example` templates; drop a real `kaggle.json` or `.baidu_cookies.json` there
and it stays gitignored.

---

## `dvb` — Drone-vs-Bird Detection Challenge

The only dataset here whose *primary purpose* is drone-versus-bird. That makes it
the most important one for a deployed system, and the least sufficient one: it has
no tracking annotations at all.

- **Contains birds** — a large fraction of frames are labelled birds.
- **Tiny targets**: 28 px median in a 4K frame. `ingest.max_dimension` downsamples
  to 1920 px, which halves storage with no measurable recall cost at that size.
- **No track ids**, so it cannot be used for `track-eval`. It trains and scores
  detection only.
- The Kaggle mirror renames classes to bare integers; the converter maps
  `0 -> drone`, `1 -> bird` and cross-checks the class ratio against the published
  statistics. A mismatch **aborts** the conversion rather than silently producing
  a wrong dataset.

```powershell
anti-uav download --dataset dvb --dry-run
anti-uav download --dataset dvb          # needs Kaggle credentials
```

Kaggle requires an account and accepted rules. Run `kaggle login
--accept-dictionary` first, which writes `~/.kaggle/kaggle.json` outside the repo;
`kaggle` picks it up automatically. `--dry-run` reports whether it resolved.

---

## `mavvid` — MAV-VID

Multirotor detection *and* tracking from aerial platforms. Bird negatives, and the
largest targets in the project, which makes it the easiest to learn from.

- **Contains birds.**
- **171 px median targets** — comfortably detectable at 640 px, and the reason it
  is *not* tiled.
- Videos come from other drones, ground cameras and handheld devices. The
  handheld subset has ego-motion, so it behaves differently at test time than the
  fixed-camera sequences that dominate deployment. It is grouped by sequence for
  splitting, so it cannot leak across the train/val boundary, but it will make the
  validation score noisier.
- The original Kaggle mirror may be dead; a Bitbucket mirror is listed first for
  that reason.

```powershell
anti-uav download --dataset mavvid --dry-run
anti-uav download --dataset mavvid
```

---

## `antiuav` — Anti-UAV

Single-target drone tracking with visibility flags, which makes it the natural
source for tracking metrics that account for occlusion.

> **No bird negatives.** A model trained on Anti-UAV alone has never been told what
> a bird looks like. Its validation precision will look excellent and mean
> nothing. This is exactly why `dvb` and `mavvid` are in the matrix, and why
> `anti-uav stats` prints a class-coverage table.

> **RGB and IR are unaligned.** The publisher states this explicitly. Do not assume
> pixel correspondence between modalities; the converters treat them as
> independent sequences.

Variants:

| variant | content | notes |
| --- | --- | --- |
| `300` | RGB + IR | **default.** The only variant with both modalities, matching the fleet's mixed visible/thermal coverage |
| `410` | IR only | via Baidu Pan; needs credentials |
| `600` | IR only | via ModelScope; no auth, but IR only |

```powershell
anti-uav download --dataset antiuav --variant 300
```

Google Drive, file id `1NPYaop35ocVTYWHOYQQHn8YHsM9jmLGr`, extraction code
`sagx`. The code is recorded in `configs/datasets/registry.yaml` and printed by
`--dry-run`, so you never have to hunt for it.

---

## `mmuav` — MM-UAV

Tri-modal multi-object tracking benchmark: 1,321 sequences, ~2.8 M frames. The
only dataset here with genuine multi-object MOT ground truth, so it is what
`track-eval` scores against.

> **12x5 px median target.** At a 640 px input a drone shrinks to roughly 6x3 px
> and is simply not detectable. Tiling is mandatory for any combo containing
> `mmuav`.

### Acquisition notes

MM-UAV is distributed through **Baidu Pan only**; the project's Google Drive
mirror is listed as "coming soon" on the publisher's page. Baidu has no anonymous
download API, so this cannot be automated. The command reports
`needs_credentials` and prints the share link rather than failing obscurely.

1. Open `https://pan.baidu.com/s/1xYzvams9X972rGBm-RgY_g?pwd=mmmm`
   (extraction code `mmmm`).
2. Sign in. Solve the captcha.
3. Save your cookies to `data/raw/_credentials/.baidu_cookies.json`.
   `data/raw/_credentials/.baidu_cookies.json.example` is the exact shape.
4. Re-run `anti-uav download --dataset mmuav --skip-download` to verify the tree.

If you skip this, `anti-uav build --combo mmuav` **skips that source with a
warning** and the rest of the matrix still runs. It will not pretend mmuav was
there — check `skipped` in `data/processed/mmuav/build_report.json` before
trusting a combo name as evidence of dataset membership.

Baidu throttles aggressively and will interrupt multi-GB transfers. Budget
several attempts.

### The 400 GB problem, and the ~12 GB answer

The full extraction is ~380 GB across RGB, IR and event. For a detector training
pipeline you want:

```
train/0001-0150/{rgb_frame,gt_rgb}/
```

RGB frames and RGB ground truth for the first ~150 sequences — roughly **12 GB**
instead of 380. That is the `subset` variant, and it is the default.

```powershell
anti-uav download --dataset mmuav --max-sequences 150 --modalities rgb --dry-run
anti-uav download --dataset mmuav --max-sequences 150 --modalities rgb
```

> **The Event modality (~1/3 of the download) cannot train an RGB detector.** It is
> excluded by default in the source's `exclude_globs`.

> **Strict trajectory annotation.** A UAV that leaves and returns keeps its
> original id. Most MOT metrics penalise this as an id switch, so MM-UAV scores
> are **not** directly comparable to MOTA on MOT17-style data. Read the numbers,
> do not compare them across benchmarks.

---

## `kaggle_cybersimar` — Kaggle Drone Detection (cybersimar08)

Small curated drone-only dataset from Kaggle.

- **No bird negatives** — drone-only, cannot falsify false-positive claims.
- **Median target 64 px** — detectable at 640 px without tiling.
- ~8,500 frames, ~50 sequences, ~2.5 GB.
- YOLO-format annotations on Kaggle.

```powershell
anti-uav download --dataset kaggle_cybersimar --dry-run
anti-uav download --dataset kaggle_cybersimar          # needs Kaggle credentials
```

---

## `github_maciullo` — DroneDetectionDataset (Maciullo)

GitHub-hosted dataset with both drone and bird annotations.

- **Contains birds** — can falsify false-positive claims (third source with birds).
- **Median target 80 px** — comfortably detectable at 640 px.
- ~12,000 frames, ~30 sequences, ~3 GB.
- Public GitHub repo, no authentication needed.

```powershell
anti-uav download --dataset github_maciullo --dry-run
anti-uav download --dataset github_maciullo
```

---

## `hf_seraphim` — Seraphim Drone Detection (HuggingFace)

HuggingFace-hosted drone detection dataset.

- **No bird negatives** — drone-only, cannot falsify false-positive claims.
- **Median target 72 px** — detectable at 640 px without tiling.
- ~15,000 frames, ~40 sequences, ~4 GB.
- Available via HuggingFace datasets library, no auth for public datasets.

```powershell
anti-uav download --dataset hf_seraphim --dry-run
anti-uav download --dataset hf_seraphim
```

---

## `kaggle_banderastepan` — Kaggle Drone Detection (banderastepan)

Larger Kaggle drone detection dataset.

- **No bird negatives** — drone-only, cannot falsify false-positive claims.
- **Median target 56 px** — detectable at 640 px without tiling.
- ~20,000 frames, ~60 sequences, ~5 GB.
- Dataset includes a `dataset` subfolder.

```powershell
anti-uav download --dataset kaggle_banderastepan --dry-run
anti-uav download --dataset kaggle_banderastepan          # needs Kaggle credentials
```

---

## Two things that will silently corrupt your results

### 1. Frame-level splits leak

Every dataset here is video-derived, so adjacent frames are near-identical. Split
by frame and the same target's near-duplicates land on both sides; mAP inflates by
double digits and means nothing.

The project splits **by sequence**, always, and `anti-uav sanity` fails the build
if any sequence appears in two splits. A frame-level split exists for genuinely
frame-independent data and warns loudly when used:

```powershell
anti-uav splits --combo all8 --strategy sequence   # the default, and the only correct one here
```

### 2. Near-duplicates across datasets

Datasets overlap: the same site, the same event, sometimes the same frames
re-extracted at a different stride. `anti-uav dedup` uses pHash to find them.

```powershell
anti-uav dedup --dataset mmuav
```

---

## Harmonisation

All eight are normalised into one interim frame index under
`data/interim/<dataset>/<variant>/` with `frames/` and `labels/`, and then into the
unified label space:

```
0 = drone
1 = bird
```

The ordering is fixed and asserted by the test suite. Class order is not
cosmetic: it is baked into every trained checkpoint and every exported engine.

Converters report what they saw rather than silently guessing:

- `antiuav`: `target` / `0` both mean drone. Different challenge releases label
  the same object differently; the converter records which spelling it saw.
- `dvb`: bare integers mapped as above, with a class-ratio cross-check.
- Unknown labels are counted and reported, never dropped in silence.

---

## Disk planning

| what | where | size |
| --- | --- | --- |
| raw downloads | `data/raw/` | up to ~475 GB (recommended subset: ~105 GB) |
| interim frame index | `data/interim/` | ~0.9x raw for frames, labels are small |
| processed YOLO datasets | `data/processed/` | 1x interim per combo, tiled combos are 4–6x |
| runs, exports, tracks | `artifacts/` | plan 20 GB per model across its 14 runs |

Tiled combos are the ones to watch: MM-UAV at 256 px tiles produces six tiles per
frame, so a combo containing it is several times larger than the same combo
without.

```powershell
anti-uav verify-env      # reports free space on the data volume
```

---

## References

- Drone-vs-Bird: WOSDETC/AVSS Drone-vs-Bird Detection Challenge
  (`wosdetc/challenge`)
- MAV-VID: Cranfield University, Multirotor Aerial Vehicle VID
- Anti-UAV: INEE, Anti-UAV410 / Anti-UAV600 (publisher states redistribution is
  restricted)
- MM-UAV: Tri-Modal Multi-UAV Tracking Benchmark (research use, confirm with the
  authors)
- Kaggle cybersimar08: Drone Detection (`cybersimar08/drone-detection`)
- GitHub Maciullo: DroneDetectionDataset (`Maciullo/DroneDetectionDataset`)
- HuggingFace Seraphim: Drone Detection (`lgrzybowski/seraphim-drone-detection-dataset`)
- Kaggle banderastepan: Drone Detection (`banderastepan/drone-detection`)

Licences differ and two of them restrict redistribution. Check before sharing any
of this data, and before putting it on a machine that leaves the site.