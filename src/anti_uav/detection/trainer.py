"""Unified detection trainer.

One code path for two families. ``YOLO11n`` and ``RT-DETR-x2`` differ in their
architecture but share every part of the pipeline that matters here - the same
converted dataset, the same sequence-level split, the same evaluator, the same
exporter, the same tracker downstream. Only the recipe differs, and it lives in
``configs/train/``.

The one genuinely different code path is *loading* the model:

* ``YOLO(...)``   - a single class wrapping both CNN and DETR architectures.
* ``RTDETR(...)`` - ultralytics keeps a separate class for RT-DETR because it
  overrides ``predict``/``train`` to handle the NMS-free query decoder. Passing
  an ``rtdetr-x2.pt`` to ``YOLO`` works but routes through the generic path and
  loses the RT-DETR-specific training loop details.

Getting that wrong is easy and produces confusing errors, so it is handled once,
here, and asserted at startup.

Hardware profiles
-----------------
``configs/train/overrides/<profile>.yaml`` adjusts batch, image size and AMP for
the target GPU. The Pascal entry matters most: CUDA 12.8 removed sm_61 kernels,
so the GTX 1070 needs an older torch wheel, and fp16 on Pascal is emulated and
slower than fp32. ``--profile auto`` reads the live device.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.loader import (
    detect_profile,
    load_profile_override,
    load_recipe,
    load_settings,
)
from ..config.schema import GpuProfile, ProfileOverride, RunPlan, TrainRecipe
from ..utils.io import read_json, write_json
from ..utils.logging import get_logger
from ..utils.paths import combo_root, relative_to_root, subdir
from ..utils.seed import seed_everything
from . import augment

log = get_logger(__name__)

#: Family -> the ultralytics class that must load it. Verified at startup so a
#: mistake surfaces before a 40-minute run rather than inside it.
_LOADERS: dict[str, str] = {
    "yolo11n": "YOLO",
    "yolo11s": "YOLO",
    "rtdetr_x2": "RTDETR",
    "rtdetr_l": "RTDETR",
}

#: Where ultralytics writes its run directories. Redirected under artifacts/ so
#: nothing lands in the CWD or in the user's home.
RUNS_DIR = "runs"


@dataclass(slots=True)
class TrainingResult:
    run_name: str
    output_dir: str
    best_weights: str | None
    last_weights: str | None
    results_csv: str | None
    epochs_requested: int
    epochs_completed: int
    duration_s: float
    profile: str
    model: str
    combo: str
    imgsz: int
    batch: int
    amp: bool
    best_metrics: dict[str, float] = field(default_factory=dict)
    #: The results.csv column this run is judged on, resolved from the recipe's
    #: ``metric:``, and its best value. ``None`` until a results.csv exists.
    primary_metric: str | None = None
    primary_metric_value: float | None = None
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.best_weights is not None


# --------------------------------------------------------------------------- #
# environment inspection
# --------------------------------------------------------------------------- #


def torch_environment() -> dict[str, Any]:
    """Facts about torch that the log lines and the UI both want."""
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
    except ImportError:
        info["torch"] = None
        return info

    info["torch"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda
    info["cudnn"] = torch.backends.cudnn.version()
    info["arch_list"] = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "capability": f"{props.major}.{props.minor}",
                    "vram_gb": round(props.total_memory / 1024**3, 1),
                }
            )
        info["devices"] = devices
    return info


def _loader_for(family: str):
    """The ultralytics class that must be used for a model family."""
    expected = _LOADERS.get(family, "YOLO")
    try:
        import ultralytics
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ultralytics is not installed. Install it with:\n"
            "    pip install 'anti-uav[train]'"
        ) from exc

    cls = getattr(ultralytics, expected, None)
    if cls is None:
        raise RuntimeError(
            f"ultralytics {ultralytics.__version__} has no {expected} class. "
            f"Expected it for model family {family!r}. Upgrade ultralytics."
        )
    return cls


def resolve_batch(recipe: TrainRecipe, override: ProfileOverride, *, device: str) -> int:
    """Final batch size, after the profile override.

    Precedence, highest first:

    1. ``override.force_batch`` — an absolute value. This is what the constrained
       profiles actually use, because an absolute number is the only thing that
       can be reasoned about on a card nobody has measured yet.
    2. ``recipe.batch * override.batch_scale`` — only meaningful when the recipe
       pins a batch. Both shipped recipes use ``-1`` (let ultralytics auto-fit),
       and auto-fit cannot be scaled from outside: ultralytics picks the value
       from free VRAM after it reads the card.
    3. ``-1`` — auto-fit.
    """
    if override.force_batch:
        return override.force_batch
    if recipe.batch > 0:
        return max(1, round(recipe.batch * override.batch_scale))
    if override.batch_scale != 1.0:
        # Not fatal: -1 auto-fit already adapts to the card. But the operator set
        # a knob that will not do anything, and silence would read as "honoured".
        log.warning(
            "batch_scale has no effect because the recipe leaves batch at -1 "
            "(ultralytics auto-fit). Set force_batch instead.",
            extra={"profile": override.profile, "batch_scale": override.batch_scale},
        )
    return -1


def resolve_imgsz(recipe: TrainRecipe, override: ProfileOverride, combo_imgsz: int | None) -> int:
    if combo_imgsz:
        return combo_imgsz
    if override.force_imgsz:
        return override.force_imgsz
    scaled = round(recipe.imgsz * override.imgsz_scale)
    return max(64, scaled)


def plan_run(
    family: str,
    combo_slug: str,
    *,
    profile: GpuProfile | str | None = None,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    tiling: bool | None = None,
    tile_size: int | None = None,
) -> RunPlan:
    """Resolve a training plan without running anything.

    ``matrix run --dry-run`` uses this to print the exact command for every run
    in the matrix, so the printed plan and the executed plan cannot diverge.
    """
    recipe = load_recipe(family)
    active = profile or detect_profile()
    override = load_profile_override(active)

    final_imgsz = imgsz or resolve_imgsz(recipe, override, combo_imgsz=None)
    final_batch = batch if batch is not None else resolve_batch(recipe, override, device="")
    final_epochs = epochs or recipe.epochs
    final_amp = recipe.amp if override.force_amp is None else override.force_amp
    final_tiling = recipe.tiling.enabled if tiling is None else tiling
    final_tile = tile_size or recipe.tiling.tile_size

    root = combo_root(combo_slug)
    return RunPlan(
        run_name=f"{family}__{combo_slug}",
        model=recipe.family,
        combo=combo_slug,
        datasets=[combo_slug],
        profile=active if isinstance(active, GpuProfile) else GpuProfile(str(active)),
        imgsz=final_imgsz,
        batch=final_batch,
        epochs=final_epochs,
        amp=final_amp,
        tiling=final_tiling,
        tile_size=final_tile,
        output_dir=str(subdir("runs") / family / combo_slug),
        data_yaml=str(root / "data.yaml"),
        estimate_note=_estimate(family, final_epochs, final_imgsz, final_batch, final_amp, active),
    )


def _estimate(
    family: str, epochs: int, imgsz: int, batch: int, amp: bool, profile: GpuProfile | str
) -> str:
    """A deliberately rough wall-clock note. Never present it as a promise."""
    profile_value = profile.value if isinstance(profile, GpuProfile) else str(profile)
    table = {
        ("yolo11n", "blackwell"): "~4-8 it/s at batch 128",
        ("yolo11n", "ada"): "~3-5 it/s at batch 96",
        ("yolo11n", "ampere"): "~3-5 it/s at batch 96",
        ("yolo11n", "turing"): "~10-14 it/s at batch 32 (per GPU; T4 is a small card)",
        ("yolo11n", "volta"): "~8-12 it/s at batch 32",
        ("yolo11n", "pascal"): "~4-6 it/s at batch 8 (AMP forced off; sm_61)",
        ("yolo11n", "cpu"): "~0.3 it/s at batch 4",
        ("rtdetr_x2", "blackwell"): "~1.2-2 s/it at batch 20",
        ("rtdetr_x2", "ada"): "~1.5-2.5 s/it at batch 24",
        ("rtdetr_x2", "ampere"): "~1.5-2.5 s/it at batch 24",
        ("rtdetr_x2", "turing"): "~0.6-1.0 s/it at batch 32 - not a good T4 workload",
        ("rtdetr_x2", "volta"): "~0.8-1.4 s/it at batch 32",
        ("rtdetr_x2", "pascal"): "~2-4 s/it at batch 8 (AMP forced off; sm_61)",
        ("rtdetr_x2", "cpu"): "~8-15 s/it - not worth running, use a GPU box",
    }
    rate = table.get((family, profile_value), "rate unknown for this profile")
    flag = "" if amp else ", AMP OFF"
    size_note = f"at imgsz {imgsz}" if imgsz != 640 else ""
    return (
        f"{profile_value}: {rate} {size_note} -> {epochs} epochs{flag}. "
        f"Multiply by the frame count printed by `anti-uav build`; tiled combos "
        f"(tiling on) are roughly 4-6x that. This is an order-of-magnitude guide, "
        f"not a measurement."
    )


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def train(
    family: str,
    combo_slug: str,
    *,
    profile: GpuProfile | str | None = None,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    amp: bool | None = None,
    device: str | None = None,
    workers: int | None = None,
    seed: int | None = None,
    resume: str | Path | None = None,
    project: str | Path | None = None,
    name: str | None = None,
    val: bool = True,
    plots: bool = True,
    exist_ok: bool = True,
    dry_run: bool = False,
    on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainingResult:
    """Train one (model family, dataset combo) pair.

    Returns a :class:`TrainingResult` describing where everything landed, whether
    it worked, and what to do next. Never raises for an expected failure - a
    training run that dies at epoch 40 of 100 still has a ``last.pt`` worth
    keeping, and the caller should be able to see that.
    """
    recipe = load_recipe(family)
    active_profile = profile or detect_profile()
    override = load_profile_override(active_profile)
    settings = load_settings()

    final_device = device or _resolve_device(settings.device, active_profile)
    final_epochs = epochs or recipe.epochs
    final_imgsz = imgsz or resolve_imgsz(recipe, override, None)
    final_batch = batch if batch is not None else resolve_batch(recipe, override, device=final_device)
    final_amp = (recipe.amp if override.force_amp is None else override.force_amp) if amp is None else amp
    final_workers = workers if workers is not None else (override.workers or recipe.workers)
    final_seed = seed if seed is not None else recipe.seed

    data_yaml = combo_root(combo_slug) / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"no dataset for combo {combo_slug!r}: {data_yaml} is missing.\n"
            f"Run: anti-uav build --combo {combo_slug}"
        )

    run_name = name or f"{family}__{combo_slug}"
    project_dir = Path(project) if project else subdir("runs") / family
    run_dir = project_dir / run_name

    result = TrainingResult(
        run_name=run_name,
        output_dir=str(run_dir),
        best_weights=None,
        last_weights=None,
        results_csv=None,
        epochs_requested=final_epochs,
        epochs_completed=0,
        duration_s=0.0,
        profile=active_profile.value if isinstance(active_profile, GpuProfile) else str(active_profile),
        model=family,
        combo=combo_slug,
        imgsz=final_imgsz,
        batch=final_batch,
        amp=final_amp,
    )

    env = torch_environment()
    log.info(
        "training configuration",
        extra={
            "run": run_name,
            "model": family,
            "combo": combo_slug,
            "epochs": final_epochs,
            "imgsz": final_imgsz,
            "batch": final_batch,
            "amp": final_amp,
            "device": final_device,
            "profile": result.profile,
            "torch": env.get("torch"),
            "torch_cuda": env.get("torch_cuda"),
        },
    )
    _preflight(family, final_device, final_amp, result)

    if dry_run:
        result.warnings.append("dry run - nothing was trained")
        return result

    seed_everything(final_seed)
    loader = _loader_for(family)

    # Ultralytics reads these from the environment; setting them keeps the run
    # reproducible and keeps its scratch files inside the repo.
    os.environ.setdefault("YOLO_VERBOSE", "true")

    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": final_epochs,
        "imgsz": final_imgsz,
        "batch": final_batch,
        "device": final_device,
        "workers": final_workers,
        "seed": final_seed,
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": exist_ok,
        "val": val,
        "plots": plots,
        "verbose": True,
    }

    if resume:
        kwargs["resume"] = str(resume)
    else:
        kwargs.update(
            {
                "optimizer": recipe.optimizer.name,
                "lr0": recipe.optimizer.lr0,
                "lrf": recipe.optimizer.lrf,
                "momentum": recipe.optimizer.momentum,
                "weight_decay": recipe.optimizer.weight_decay,
                "warmup_epochs": recipe.optimizer.warmup_epochs,
                "warmup_momentum": recipe.optimizer.warmup_momentum,
                "warmup_bias_lr": recipe.optimizer.warmup_bias_lr,
                "cos_lr": recipe.scheduler == "cos-linear",
                "patience": recipe.patience,
                "freeze": recipe.freeze or None,
                "hsv_h": recipe.augmentation.hsv_h,
                "hsv_s": recipe.augmentation.hsv_s,
                "hsv_v": recipe.augmentation.hsv_v,
                "degrees": recipe.augmentation.degrees,
                "translate": recipe.augmentation.translate,
                "scale": recipe.augmentation.scale,
                "shear": recipe.augmentation.shear,
                "perspective": recipe.augmentation.perspective,
                "flipud": recipe.augmentation.flipud,
                "fliplr": recipe.augmentation.fliplr,
                "mosaic": recipe.augmentation.mosaic,
                "mixup": recipe.augmentation.mixup,
                "copy_paste": recipe.augmentation.copy_paste,
                "close_mosaic": recipe.augmentation.close_mosaic or None,
                "box": recipe.loss.box,
                "cls": recipe.loss.cls,
                "dfl": recipe.loss.dfl,
                "amp": final_amp,
                "save_period": recipe.save_period,
                # The tracker associates against deduplicated boxes. YOLO applies
                # NMS by default; this makes the intent explicit rather than
                # inherited, and it is the switch RT-DETR deliberately lacks.
                "nms": bool(recipe.extra.get("nms", True)),
            }
        )

    try:
        model = loader(recipe.base_weights)
    except Exception as exc:
        result.error = f"could not load base weights {recipe.base_weights!r}: {exc}"
        log.error("base weight load failed", extra={"weights": recipe.base_weights})
        return result

    started = time.monotonic()
    if not resume:
        # ultralytics has no cutout and no IR-grayscale knob, so these two come
        # from a project-owned callback. Skipped on resume: a resumed run is
        # already past the epoch it would start applying them from.
        augment.register(model, recipe, final_seed)
    try:
        model.train(**kwargs)
    except KeyboardInterrupt:
        result.error = "interrupted by the user"
        result.duration_s = time.monotonic() - started
        _collect_outputs(result, run_dir, recipe.metric)
        result.warnings.append(
            "Interrupted. last.pt is kept; resume with: "
            f"anti-uav train --model {family} --combo {combo_slug} --resume {run_dir / 'weights' / 'last.pt'}"
        )
        return result
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.duration_s = time.monotonic() - started
        _collect_outputs(result, run_dir, recipe.metric)
        result.warnings.append(
            "Training raised. Check the traceback above; last.pt (if written) is still resumable."
        )
        log.exception("training failed", extra={"run": run_name})
        return result

    result.duration_s = time.monotonic() - started
    _collect_outputs(result, run_dir, recipe.metric)
    _write_run_metadata(result, run_dir, env, recipe, override)
    return result


def _resolve_device(requested: str, profile: GpuProfile | str) -> str:
    """``auto`` means "use the GPU when the profile has one"."""
    if requested and requested != "auto":
        return requested
    profile_value = profile.value if isinstance(profile, GpuProfile) else str(profile)
    if profile_value == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return "cpu"


def _device_indices(device: str) -> list[int]:
    """Parse an ultralytics ``device`` string into CUDA ordinals.

    Handles ``0``, ``0,1``, ``cuda:0,1`` and ``cpu`` (empty list). Mirrors
    ultralytics' own rule: a comma-separated list of more than one entry means
    DDP.
    """
    if device == "cpu":
        return []
    spec = device.split(":", 1)[-1]
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if part.isdigit():
            indices.append(int(part))
    return indices or [0]


def _preflight(family: str, device: str, amp: bool, result: TrainingResult) -> None:
    """Catch the setup mistakes that would otherwise waste hours."""
    if device == "cpu" and family == "rtdetr_x2":
        result.warnings.append(
            "RT-DETR-x2 on CPU is not practical (roughly 8-15 s per iteration). This run "
            "will take weeks. Use a CUDA profile: anti-uav train --profile blackwell ..."
        )

    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available() or device == "cpu":
        return

    # ultralytics accepts "0", "0,1", "cuda:0,1" and treats any comma-separated
    # list of length > 1 as a request for DDP. Parse the list rather than
    # assuming index 0, so a run that asked for two GPUs and found one is caught
    # here instead of half an hour into a spawned worker.
    indices = _device_indices(device)
    present = torch.cuda.device_count()
    missing = [i for i in indices if i >= present]
    if missing:
        result.warnings.append(
            f"device={device!r} asks for GPU(s) {missing} but only {present} CUDA "
            f"device(s) are present. Training will fail; pass --device 0 to use one."
        )
        return

    index = indices[0]
    props = torch.cuda.get_device_properties(index)
    capability = props.major * 10 + props.minor

    if len(indices) > 1:
        mismatched = {
            i: f"sm_{torch.cuda.get_device_properties(i).major}"
            f"{torch.cuda.get_device_properties(i).minor}"
            for i in indices[1:]
            if f"sm_{torch.cuda.get_device_properties(i).major}"
            f"{torch.cuda.get_device_properties(i).minor}" != f"sm_{props.major}{props.minor}"
        }
        if mismatched:
            result.warnings.append(
                f"device={device!r} spans heterogeneous GPUs: rank 0 is sm_{props.major}"
                f"{props.minor} but {mismatched}. DDP requires identical architectures - "
                "NCCL will hang or the run will silently train on one card. Use one GPU."
            )

    if capability < 70 and amp:
        result.warnings.append(
            f"GPU is sm_{props.major}{props.minor} (Pascal, sm_61) but amp=True. There are "
            "no fp16 tensor cores on this architecture, so fp16 is emulated or demoted to "
            "fp32 - slower than fp32 and prone to silent mAP loss from grad-scaler "
            "instability. The pascal profile forces amp=False for this reason; pass "
            "--profile pascal."
        )

    arch_list = torch.cuda.get_arch_list()
    if arch_list and f"sm_{props.major}{props.minor}" not in arch_list:
        result.warnings.append(
            f"This torch build ({torch.__version__}, CUDA {torch.version.cuda}) does not "
            f"include sm_{props.major}{props.minor} kernels for {props.name}. Training will "
            f"fail with 'no kernel image is available'. "
            + (
                "CUDA 12.8 removed Maxwell/Pascal (sm_50-sm_62) support and CUDA 13.x "
                "removed Volta too - install the last build line carrying kernels for this "
                "card (see docs/ENVIRONMENT.md)."
                if capability < 70
                else "Upgrade torch or install a CUDA build that still ships kernels for "
                "this card (see docs/ENVIRONMENT.md)."
            )
        )

    free, total = torch.cuda.mem_get_info(index)
    log.info(
        "gpu memory",
        extra={
            "device": props.name,
            "total_gb": round(total / 1024**3, 1),
            "free_gb": round(free / 1024**3, 1),
            "batch": result.batch,
        },
    )


def _collect_outputs(result: TrainingResult, run_dir: Path, primary_metric: str = "") -> None:
    weights = run_dir / "weights"
    for key, name in (("best_weights", "best.pt"), ("last_weights", "last.pt")):
        candidate = weights / name
        if candidate.is_file():
            setattr(result, key, str(candidate))
    csv = run_dir / "results.csv"
    if csv.is_file():
        result.results_csv = str(csv)
        result.best_metrics = _best_from_csv(csv)
        result.epochs_completed = _epochs_from_csv(csv)
        if primary_metric:
            resolved = resolve_primary_metric(result.best_metrics, primary_metric)
            if resolved is not None:
                result.primary_metric, result.primary_metric_value = resolved
            else:
                log.warning(
                    "primary metric not found in results.csv",
                    extra={"metric": primary_metric, "run": result.run_name},
                )


def _best_from_csv(csv: Path) -> dict[str, float]:
    """Best value per metric column, taken from the row ultralytics saved."""
    import csv as csv_module

    try:
        with csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv_module.DictReader(handle))
    except OSError:
        return {}
    if not rows:
        return {}

    best: dict[str, float] = {}
    for column in rows[0]:
        if column == "epoch":
            continue
        values = []
        for row in rows:
            try:
                values.append(float(row[column]))
            except (TypeError, ValueError, KeyError):
                continue
        if not values:
            continue
        # Loss is minimised, everything else is maximised.
        best[column] = min(values) if "loss" in column else max(values)
    return best


#: Short metric name -> the results.csv column that carries it.
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "map50-95": ("metrics/map50-95(b)", "map50-95", "map_50_95", "map"),
    "map50": ("metrics/map50(b)", "map50"),
    "precision": ("metrics/precision(b)", "precision"),
    "recall": ("metrics/recall(b)", "recall"),
}


def resolve_primary_metric(metrics: dict[str, float], metric: str) -> tuple[str, float] | None:
    """Find the ``(column, best value)`` a recipe's ``metric:`` refers to.

    Ultralytics early-stops on its own internal ``fitness`` blend and exposes no
    hook to change that, so ``metric`` cannot control early stopping — the
    ``patience`` field does. What ``metric`` does control is which number this
    run is *judged* on, and recording the resolved column plus its best value
    keeps that decision in the artefact instead of in someone's memory.
    """
    wanted = _METRIC_ALIASES.get(metric.strip().lower().replace("_", "-"), (metric,))
    lowered = {k.lower(): k for k in metrics}
    for candidate in wanted:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()], metrics[lowered[candidate.lower()]]
    for candidate in wanted:
        for key_lower, key in lowered.items():
            if candidate.lower() in key_lower:
                return key, metrics[key]
    return None


def _epochs_from_csv(csv: Path) -> int:
    import csv as csv_module

    try:
        with csv.open(encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv_module.DictReader(handle))
    except OSError:
        return 0


def _write_run_metadata(
    result: TrainingResult,
    run_dir: Path,
    env: dict[str, Any],
    recipe: TrainRecipe,
    override: ProfileOverride,
) -> None:
    """Persist everything needed to reproduce or explain this run months later."""
    payload = {
        "run_name": result.run_name,
        "model": result.model,
        "combo": result.combo,
        "profile": result.profile,
        "epochs_requested": result.epochs_requested,
        "epochs_completed": result.epochs_completed,
        "imgsz": result.imgsz,
        "batch": result.batch,
        "amp": result.amp,
        "duration_s": round(result.duration_s, 1),
        "best_metrics": result.best_metrics,
        "primary_metric": result.primary_metric,
        "primary_metric_value": result.primary_metric_value,
        "best_weights": result.best_weights,
        "last_weights": result.last_weights,
        "relative_output": relative_to_root(run_dir),
        "recipe": json.loads(recipe.model_dump_json()),
        "profile_override": json.loads(override.model_dump_json()),
        "environment": env,
        "warnings": result.warnings,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(run_dir / "run_metadata.json", payload)


def describe_plan(plan: RunPlan) -> str:
    lines = [
        f"run     : {plan.run_name}",
        f"model   : {plan.model.value}  (profile {plan.profile.value})",
        f"data    : {relative_to_root(Path(plan.data_yaml))}",
        f"out     : {relative_to_root(Path(plan.output_dir))}",
        f"epochs  : {plan.epochs}",
        f"imgsz   : {plan.imgsz}",
        f"batch   : {plan.batch}",
        f"amp     : {plan.amp}",
        f"tiling  : {'on, tile ' + str(plan.tile_size) if plan.tiling else 'off'}",
        f"estimate: {plan.estimate_note}",
    ]
    return "\n".join(lines)


def list_runs() -> list[dict[str, Any]]:
    """Every run that has a ``run_metadata.json`` or ``results.csv``.

    Backs the UI's dashboard and ``anti-uav matrix status``.
    """
    from ..utils.io import iter_files

    out: list[dict[str, Any]] = []
    for meta in iter_files(subdir("runs"), ["run_metadata.json"]):
        try:
            data = read_json(meta)
        except (OSError, ValueError):
            continue
        run_dir = meta.parent
        data["_dir"] = relative_to_root(run_dir)
        data["_has_weights"] = (run_dir / "weights" / "best.pt").is_file()
        data["_has_results"] = (run_dir / "results.csv").is_file()
        out.append(data)
    out.sort(key=lambda r: str(r.get("trained_at", "")), reverse=True)
    return out
