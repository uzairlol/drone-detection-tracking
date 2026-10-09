# Environment

The project runs on three kinds of machine: a CPU-only dev box, a GTX 1070 that
cannot run current torch at all, and two supervisor boxes with an RTX 4090 and an
RTX 5090. One repository handles all of them; the profile system is what makes that
possible.

**This project does not create virtual environments.** It is meant to live in an
environment you already maintain, because a second environment per machine is
where dependency drift comes from. `scripts/setup_env.ps1` installs into an
existing one and tells you which.

---

## Quick start

```powershell
.\scripts\setup_env.ps1 -EnvName ml
anti-uav verify-env
anti-uav fetch-weights
```

`verify-env` is the command to trust. It checks the interpreter, compares the
installed torch build against the live device's compute capability, checks disk
space, checks that the pretrained checkpoints are on disk, and prints the exact
remedy for anything wrong. It returns non-zero when something is actually
blocking.

`fetch-weights` is separate because a missing checkpoint is a five-minute
download, not a broken environment — see [Base weights](#base-weights) below.

To reproduce the pinned, known-good set rather than re-resolving:

```powershell
.\scripts\setup_env.ps1 -EnvName ml -Profile pascal -UseLockFile -UpgradeTorch
```

---

## Locked dependency set

`requirements-lock.txt` is a verified-good snapshot: 115 pinned packages taken
from an environment that passes the test suite, `ruff`, `mypy` and
`verify-env`. Use it to provision a new machine, or to undo drift.

```powershell
.\scripts\setup_env.ps1 -EnvName ml -UseLockFile
```

**torch and torchvision are deliberately absent from it.** One file cannot pin
five GPU tiers, and a wrong torch build is the single most expensive mistake in
this project — it installs cleanly, imports cleanly, reports CUDA as available,
and then fails hours into training. So `-Profile` still governs torch, and the
two work together rather than conflicting:

| profile | torch index |
| --- | --- |
| `pascal` | `https://download.pytorch.org/whl/cu126` |
| `ampere` | `https://download.pytorch.org/whl/cu124` |
| `ada` | `https://download.pytorch.org/whl/cu128` |
| `blackwell` | `https://download.pytorch.org/whl/cu128` |
| `cpu` | default PyPI wheel |

If you have a pre-downloaded wheel — this repo's Pascal box uses
`D:\wheels\torch-<version>+cu126-cp311-cp311-win_amd64.whl`, which is how a cu126
build survives on a card whose driver predates that runtime — install it with
`pip install <path-to-wheel>`, or point the script at a wheel directory:

```powershell
.\scripts\setup_env.ps1 -UseLockFile -WheelDir D:\wheels
```

`--find-links` is used without `--no-index`, so a local wheel wins where present
and the index still covers anything not cached. A hard `--no-index` would turn one
missing wheel into a failed install on a machine that had fine connectivity.

---

## Base weights

Each recipe in `configs/train/` names a starting checkpoint in `base_weights`:
`yolo11n.pt` for the edge candidate, `rtdetr-x2.pt` for the accuracy reference.

```powershell
anti-uav fetch-weights              # download whatever is missing
anti-uav fetch-weights --dry-run    # report only, touch nothing
```

`verify-env` also reports these as a `[WARN] base-weights` line — a warning, not
a failure, because a machine with no checkpoints is still perfectly capable of
data prep, evaluation, the API and the whole test suite.

The reason this is a command rather than something to leave to the trainer:
ultralytics downloads a missing checkpoint from *inside* `train_one`, so on a
machine that cannot reach the release host the run dies after the dataset is
built, the split is validated and the run directory is created. Fetching first
turns a five-minute problem into a five-second one.

---

## GPU profiles

| profile | compute capability | hardware | torch index URL | notes |
| --- | --- | --- | --- | --- |
| `pascal` | sm_61 | GTX 1070 | `.../whl/cu126` | **CUDA 12.8 removed sm_61.** Needs an older wheel line or the card cannot run at all. Also the only tier with **no fp16 tensor cores** |
| `volta` | sm_70 | V100 | `.../whl/cu126` | fp16 tensor cores present. **CUDA 13.x removed sm_70** |
| `turing` | sm_75 | T4 (Kaggle, Colab) | `.../whl/cu126` | fp16 tensor cores present. 16 GB, so batch is an absolute 32 |
| `ampere` | sm_80 / sm_86 | RTX 3090 | `.../whl/cu124` | |
| `ada` | sm_89 | RTX 4090 | `.../whl/cu128` | |
| `blackwell` | sm_120 | RTX 5090 | `.../whl/cu128` | **Needs CUDA 12.8+.** Older wheels have no sm_120 kernels |
| `cpu` | — | any | — | Data prep, API, UI and the whole test suite |

`pascal` used to absorb `volta` and `turing` as well, on the reasoning that they
were "the last line with fp16 tensor cores". That was wrong twice over: sm_61 has
no fp16 path at all, and sm_70/sm_75 have a genuinely fast one. Folding them in
meant a Kaggle T4 trained with AMP off and batch 8 — a large throughput loss on
the one GPU tier most people outside the lab actually have access to.

The profile defaults to `auto` in `configs/app.yaml`, which reads the live torch
build and the device rather than anything written down. Override when you need to:

```powershell
anti-uav verify-env --profile blackwell
$env:ANTI_UAV_PROFILE = "pascal"      # or set it for your user account
```

### Installing the right torch

```powershell
# RTX 5090
.\scripts\setup_env.ps1 -EnvName ml -Profile blackwell -UpgradeTorch

# or by hand
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

`setup_env.ps1` deliberately does **not** touch torch unless you pass
`-UpgradeTorch`. Replacing a working torch is a ~2.5 GB download and should not
happen because someone forgot to read the table.

The consequence of getting it wrong is silent-ish: `torch 2.13.0+cpu` on a machine
with a GPU, or a CUDA build that reports `no kernel image is available for
execution on the device`. `verify-env` catches both.

---

## Extras

Defined in `pyproject.toml` and selected by `setup_env.ps1 -Extras`:

| extra | pulls in | needed for |
| --- | --- | --- |
| `download` | gdown, kaggle, modelscope | dataset acquisition |
| `train` | ultralytics | training and inference |
| `export` | onnx, onnxruntime, TensorRT tooling | ONNX/TensorRT export |
| `dev` | pytest, ruff, mypy | the test suite |
| `all` | everything above | |

```powershell
.\scripts\setup_env.ps1 -Extras train,dev
.\scripts\setup_env.ps1 -Extras all          # the default
```

There is no `reid` extra: the appearance model is plain torch + torchvision, which
are already hard dependencies.

---

## Where things live

```powershell
anti-uav version
```

```
project    D:\Drone Detection and Tracking     # configs, src, scripts
data       <project>\data                       # raw, interim, processed, manifests
runs       <project>\artifacts\runs             # one dir per (model, combo)
processed  <project>\data\processed
```

Override the data root when it must live on another volume — with 460 GB of
datasets involved, this comes up:

```powershell
.\scripts\setup_env.ps1 -DataRoot D:\drone-data     # sets ANTI_UAV_DATA_DIR
$env:ANTI_UAV_DATA_DIR = "D:\drone-data"           # or just for this session
```

pip's cache and temp are pinned off the system drive by the same script, which
matters when a download extra pulls ~2.5 GB of wheels and C: has 12 GB free.

---

## Known-good environment

Recorded automatically from the environment this repository is developed and
verified in — Windows, Python 3.11, CUDA 12.6, on a GTX 1070 (sm_61):

```
python      3.11.17
torch       2.13.0+cu126      <- local wheel, D:\wheels\...+cu126...whl
torchvision 0.28.0+cu126
numpy       2.4.6
pydantic    2.13.5
opencv      5.0.0.93
ultralytics 8.4.171
lapx        0.10.0            (module name: lap)
onnx        1.23.1
onnxruntime 1.30.0
gdown       6.4.1
kaggle      2.2.4
modelscope  1.40.1
pandas      3.0.6
scipy       1.17.1
pytest      9.1.1
ruff        0.16.10
mypy        2.4.0
```

The full 115-package set, machine-readable and installable, is
[`requirements-lock.txt`](../requirements-lock.txt). Swap the torch pair for
your profile's CUDA build and leave the rest alone.

A second `ml` environment exists on some machines under both
`%USERPROFILE%\.conda\envs\ml` and `D:\envs\ml`. They are not interchangeable —
different torch builds, different numpy. The scripts resolve the **active** conda
env first (`$env:CONDA_PREFIX`), and `run_pipeline.ps1 -Python` overrides it
explicitly when in doubt. Check which one you got with `anti-uav version`.

### ultralytics 8.4 specifics worth knowing

- There is **no `track` task**. `persist` and `tracker` are not in the predict
  config, so `model.predict(..., persist=True)` raises
  `SyntaxError: 'persist' is not a valid YOLO argument`. This project therefore
  uses its **own** trackers (`anti_uav.tracking.local`) on top of `predict_image`,
  which is also why `track-eval` and live video inference share one association
  implementation.
- `select_device` rejects the literal string `"auto"`; it auto-selects only on an
  empty string. `configs/app.yaml` says `device: auto` for readability and a
  schema validator normalises it, because pinning a device string in a shared
  config is how a box ends up trying to train on a GPU that is not there.
- `half=True` is deprecated in favour of `quantize` and prints a warning on every
  call. Harmless, but noisy.

---

## Reinstalling

```powershell
.\scripts\setup_env.ps1 -EnvName ml
```

Safe to re-run. It upgrades pip, installs the project editable, and re-verifies.
It will not touch torch or any other existing package unless `-UpgradeTorch` is
passed.

If the editable install fails, the cause is almost always `README.md` missing —
`pyproject.toml` declares it as the long description, and hatchling refuses to
build without it.

---

## Checks

```powershell
anti-uav verify-env          # the real check: torch vs GPU, disk, packages
anti-uav verify-env --json   # machine-readable, for a provisioning script
python -c "import anti_uav; print(anti_uav.__version__)"
```

What `verify-env` reports and what to do:

| line | meaning |
| --- | --- |
| `[ok] profile` | the installed torch build matches this GPU tier |
| `[ok] arch-kernels` | the build actually carries kernels for this card's `sm_XX` |
| `[FAIL] arch-kernels` | imports fine, has CUDA, and cannot run — the message prints the exact `--index-url` to use |
| `[WARN] base-weights` | a recipe's checkpoint is missing; run `anti-uav fetch-weights` before training |
| `[WARN] optional absent` | an extra is not installed; the feature needing it will not work, everything else will |
| `[FAIL] disk` | not enough room for the dataset you are about to fetch; the message names the dataset and the subset that fits |
| `[FAIL] torch` | the build cannot run on this GPU; the message prints the exact `--index-url` to use |