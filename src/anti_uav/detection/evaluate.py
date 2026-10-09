"""Detection evaluation.

Two distinct things are measured here, and conflating them is the classic way to
end up with a misleading results table.

**Within-combo validation.** The model is scored on the val split built from the
same datasets it trained on. Answers "how well does it fit this data". Useful for
debugging, weak evidence for deployment.

**Cross-dataset evaluation.** The model is scored on *each source dataset*
separately, including ones it never saw. Answers "does this generalise to new
footage, a new sensor, a new site" - which is the only question that matters when
the thing being shipped runs on 100 unseen cameras.

The cross-dataset numbers are reported with a caveat column, because two of the
sources make a headline number unfalsifiable on their own:

* ``antiuav`` and ``mmuav`` are drone-only, so precision measured on them says
  nothing about false positives;
* ``dvb`` and ``mavvid`` carry bird negatives, so bird recall is measurable there
  and nowhere else.

Small-object performance is reported via ``AP_S``-style buckets. At a 12-28 px
target, aggregate mAP hides the fact that most of the failures are on the small
class, which is the class that matters for a 4K camera watching a 30 m sky.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.loader import load_registry, load_settings
from ..config.schema import DatasetSpec
from ..utils.io import write_json
from ..utils.logging import get_logger
from ..utils.paths import subdir
from .predictor import Detector, resolve_weights

log = get_logger(__name__)

#: Normalised sqrt(area) cut-offs for the small/medium/large buckets.
SCALE_THRESHOLDS = {"small": 0.33, "medium": 0.66}


@dataclass(slots=True)
class EvalResult:
    run: str
    scope: str  # "val" or a dataset alias
    metrics: dict[str, float] = field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    per_class_and_scale: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    images: int = 0
    boxes_gt: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def map50(self) -> float:
        return float(self.metrics.get("map50", 0.0))

    @property
    def map(self) -> float:
        return float(self.metrics.get("map50-95", 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "scope": self.scope,
            "metrics": self.metrics,
            "per_class": self.per_class,
            "per_class_and_scale": self.per_class_and_scale,
            "images": self.images,
            "boxes_gt": self.boxes_gt,
            "notes": self.notes,
        }


def evaluate_run(
    run_or_path: str | Path,
    *,
    split: str = "val",
    data_yaml: str | Path | None = None,
    conf: float | None = None,
    imgsz: int | None = None,
    device: str | None = None,
    batch: int | None = None,
    plots: bool = True,
    save_json: bool = True,
) -> EvalResult:
    """Evaluate one checkpoint on one split.

    Delegates the metric computation to ultralytics so the numbers are directly
    comparable with any other ultralytics run. The per-scale breakdown is
    computed separately because ``results_dict`` does not expose it portably.
    """
    settings = load_settings()
    weights = resolve_weights(run_or_path)
    data_path = Path(data_yaml) if data_yaml else _guess_data_yaml(run_or_path)

    result = EvalResult(run=str(run_or_path), scope=split)
    if not data_path.is_file():
        result.notes.append(
            f"No data.yaml found for this run (looked for {data_path}). "
            f"Pass --data explicitly."
        )
        return result

    try:
        import ultralytics
    except ImportError as exc:
        result.notes.append(f"ultralytics not installed: {exc}")
        return result

    name = str(weights).lower()
    loader = ultralytics.RTDETR if "rtdetr" in name else ultralytics.YOLO
    model = loader(str(weights))

    metrics = model.val(
        data=str(data_path),
        split=split,
        imgsz=imgsz or settings.imgsz,
        conf=conf if conf is not None else settings.conf_threshold,
        device=device or settings.device,
        batch=batch if batch is not None else (16 if settings.device != "cpu" else 4),
        plots=plots,
        verbose=False,
        project=str(subdir("runs") / "_eval"),
        name=weights.parent.parent.name if weights.parent.name == "weights" else weights.stem,
        exist_ok=True,
    )

    result.metrics = _extract(metrics)
    result.per_class, result.per_class_and_scale = _per_class(metrics)
    result.images = _count(metrics, "nt_per_image", "nt")
    result.boxes_gt = _count(metrics, "nt_per_class", "n_gt", "nt")

    if data_path.is_file():
        result.notes.extend(_data_yaml_notes(data_path))

    if save_json:
        target = weights.parent.parent / f"eval_{split}.json"
        write_json(target, result.to_dict())
        log.info("eval written", extra={"file": str(target)})

    return result


def _guess_data_yaml(run_or_path: str | Path) -> Path:
    """Find the data.yaml this run was trained on.

    Prefers the run's own recorded metadata, because several combos can share a
    model name and guessing from the directory name gets it wrong.
    """
    weights = resolve_weights(run_or_path)
    run_dir = weights.parent.parent if weights.parent.name == "weights" else weights.parent

    meta = run_dir / "run_metadata.json"
    if meta.is_file():
        from ..utils.io import read_json

        try:
            data = read_json(meta)
            combo = str(data.get("combo", ""))
            if combo:
                candidate = subdir("processed") / combo / "data.yaml"
                if candidate.is_file():
                    return candidate
        except (OSError, ValueError, KeyError):
            pass

    stem = run_dir.name.split("__")[-1]
    candidate = subdir("processed") / stem / "data.yaml"
    if candidate.is_file():
        return candidate
    return run_dir / "data.yaml"


def _data_yaml_notes(data_yaml: Path) -> list[str]:
    """Turn the data.yaml's provenance into caveats about the numbers."""
    notes: list[str] = []
    try:
        text = data_yaml.read_text(encoding="utf-8")
    except OSError:
        return notes

    sources = []
    for line in text.splitlines():
        if line.startswith("# sources:"):
            sources = [s.strip() for s in line.split(":", 1)[1].split(",") if s.strip()]

    if not sources:
        return notes

    registry = load_registry()
    bird_sources = [
        name
        for name in sources
        if name in registry.datasets and registry.datasets[name].has_bird_negatives
    ]
    if not bird_sources:
        notes.append(
            "This combo contains NO source with bird annotations. Precision measured "
            "here is unfalsifiable - report bird recall from the dvb/mavvid rows of the "
            "cross-dataset table instead."
        )
    else:
        notes.append(f"Bird negatives available from: {', '.join(bird_sources)}.")

    drone_only = [n for n in sources if n in registry.datasets and n not in bird_sources]
    if drone_only:
        notes.append(
            f"Drone-only sources in this combo: {', '.join(drone_only)}. These dominate "
            f"the frame count and carry no false-positive signal."
        )
    return notes


def _extract(metrics: Any) -> dict[str, float]:
    """Pull the headline numbers out of an ultralytics ``DetMetrics``."""
    out: dict[str, float] = {}
    candidates = {
        "precision": ("box.mp", "metrics/precision(B)"),
        "recall": ("box.mr", "metrics/recall(B)"),
        "map50": ("box.map50", "metrics/mAP50(B)"),
        "map50-95": ("box.map", "metrics/mAP50-95(B)"),
    }
    for key, attributes in candidates.items():
        for attribute in attributes:
            value = getattr(metrics, attribute, None)
            if value is not None:
                out[key] = _as_float(value)
                break

    for key, attribute in (("fitness", "fitness"), ("fitness_full", "fitness_full")):
        value = getattr(metrics, attribute, None)
        if value is not None:
            out[key] = _as_float(value)
    return out


def _per_class(metrics: Any) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    """Per-class and per-class-per-scale metrics.

    ultralytics exposes AP_S/AP_M/AP_L on the metrics object but the naming has
    shifted between versions, so every known spelling is probed and anything
    missing is simply absent from the result rather than reported as zero.
    """
    names = getattr(metrics, "names", None) or {}
    per_class: dict[str, dict[str, float]] = {}
    per_scale: dict[str, dict[str, dict[str, float]]] = {}

    for index, name in names.items():
        if index >= 1000:  # the names dict pads to 1000 classes in some versions
            continue
        entry: dict[str, float] = {}
        scale_entry: dict[str, dict[str, float]] = {}

        precision = _index(getattr(metrics, "box", None), "p", index)
        recall = _index(getattr(metrics, "box", None), "r", index)
        map50 = _index(getattr(metrics, "box", None), "map50", index)
        map95 = _index(getattr(metrics, "box", None), "map", index)

        if precision is not None:
            entry["precision"] = precision
        if recall is not None:
            entry["recall"] = recall
        if map50 is not None:
            entry["map50"] = map50
        if map95 is not None:
            entry["map50-95"] = map95

        for scale, suffixes in (
            ("small", ("s",)),
            ("medium", ("m",)),
            ("large", ("l",)),
        ):
            values: dict[str, float] = {}
            for suffix in suffixes:
                for label, base in (("map50", f"map50{suffix}"), ("map50-95", f"map{suffix}")):
                    value = _index(getattr(metrics, "box", None), base, index)
                    if value is not None:
                        values[label] = value
            if values:
                scale_entry[scale] = values

        if entry:
            per_class[str(name)] = entry
        if scale_entry:
            per_scale[str(name)] = scale_entry

    return per_class, per_scale


def _index(container: Any, attribute: str, index: int) -> float | None:
    values = getattr(container, attribute, None)
    if values is None:
        return None
    try:
        value = values[index]
    except (IndexError, TypeError):
        return None
    number = _as_float(value)
    return None if number != number else number  # drop NaN


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _count(metrics: Any, *names: str) -> int:
    """Ground-truth image / instance counts, across ultralytics versions.

    Current ultralytics reports these as per-class numpy histograms
    (``nt_per_image``, ``nt_per_class``) and has no scalar ``nt``/``n_gt``
    attribute at all. Reading only the old scalar names made ``getattr`` fall
    through to 0, so every evaluation printed ``images: 0  gt boxes: 0`` next to
    a perfectly good mAP - a report that looks broken even when the model is
    fine. Several names are tried in order so one implementation covers both the
    histogram and the older scalar layout.
    """
    for name in names:
        value = getattr(metrics, name, None)
        if value is None:
            continue
        if hasattr(value, "sum"):
            try:
                return int(value.sum())
            except (TypeError, ValueError):
                continue
        total = _as_int(value)
        if total:
            return total
    return 0


# --------------------------------------------------------------------------- #
# cross-dataset evaluation
# --------------------------------------------------------------------------- #


def evaluate_cross_dataset(
    run_or_path: str | Path,
    *,
    datasets: Sequence[str] | None = None,
    imgsz: int | None = None,
    conf: float | None = None,
    device: str | None = None,
    save: bool = True,
) -> dict[str, EvalResult]:
    """Score one checkpoint on every source dataset, including unseen ones.

    Each source is evaluated as its own one-class problem in the unified
    ``{drone, bird}`` space, using only its own val frames. A model trained
    without a given dataset is being asked a question it has not been taught the
    answer to, which is exactly the point.
    """
    registry = load_registry()
    targets = list(datasets) if datasets else list(registry.aliases)
    out: dict[str, EvalResult] = {}

    for alias in targets:
        spec: DatasetSpec | None = registry.datasets.get(alias)
        if spec is None:
            log.warning("unknown dataset in cross-eval", extra={"dataset": alias})
            continue

        data_yaml = _cross_eval_yaml(alias, spec)
        if data_yaml is None:
            result = EvalResult(run=str(run_or_path), scope=alias)
            result.notes.append(
                f"{alias}: no held-out split available. Cross-dataset evaluation needs "
                f"this dataset converted and split. Run: anti-uav splits --datasets {alias}"
            )
            out[alias] = result
            continue

        log.info("cross-dataset eval", extra={"dataset": alias})
        result = evaluate_run(
            run_or_path,
            split="val",
            data_yaml=data_yaml,
            imgsz=imgsz,
            conf=conf,
            device=device,
            plots=False,
            save_json=False,
        )
        result.notes.append(_falsifiability_note(alias, spec))
        out[alias] = result

    if save:
        write_json(
            subdir("runs") / "_eval" / f"cross_{_slug(run_or_path)}.json",
            {alias: result.to_dict() for alias, result in out.items()},
        )
    return out


def _cross_eval_yaml(alias: str, spec: DatasetSpec) -> Path | None:
    """Build (once) a data.yaml for a single source's val split.

    Written into ``data/processed/_cross/<alias>/`` rather than reusing the combo
    builds, because a combo's val set mixes sources and mixing them would hide
    exactly the per-dataset differences this table exists to show.
    """
    from ..data.frameindex import load_index

    variant = spec.default_variant or "full"
    records = [r for r in load_index(alias, variant) if r.split == "val"]
    if not records:
        return None

    from ..config.schema import ComboSpec
    from ..data.build import _write_data_yaml

    root = subdir("processed") / "_cross" / alias
    root.mkdir(parents=True, exist_ok=True)

    # Symlink the val images rather than copying: a full copy would duplicate
    # tens of GB for no reason.
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)

    link_images(records, root)

    # `_write_data_yaml` renders the combo slug and source list into the header
    # comment, so it needs a ComboSpec. A one-source combo is the honest
    # description of what this directory contains.
    combo = ComboSpec(
        slug=f"cross-{alias}",
        datasets=[alias],
        description=f"held-out {alias} val split, for cross-dataset evaluation",
    )

    return _write_data_yaml(
        root,
        combo,
        spec.classes.unified_labels,
        {alias: variant},
    )


def link_images(records: Sequence[Any], root: Path) -> int:
    """Hard-link a dataset's frames into a cross-eval split tree."""
    from ..data.frameindex import interim_root

    linked = 0
    for record in records:
        if record.split not in {"train", "val"}:
            continue
        source = interim_root(record.dataset, record.variant) / record.image
        target = root / "images" / record.split / Path(record.image).name
        if target.exists() or not source.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            import os

            os.link(source, target)
        except OSError:
            import shutil

            shutil.copy2(source, target)
        linked += 1
    return linked


def _falsifiability_note(alias: str, spec: DatasetSpec) -> str:
    """Read off the capability contract rather than restating the rule.

    The wording lives in one place (``data/capabilities.py``) so a fifth dataset
    cannot quietly acquire an unstated exemption.
    """
    from ..data.capabilities import source_capabilities_for

    caps = source_capabilities_for(alias)
    if caps.can_falsify_precision:
        return (
            f"{alias}: carries bird annotations, so precision and bird recall are both "
            f"measurable here. This is the row to trust for false positives."
        )
    return (
        f"{alias}: DRONE-ONLY. Precision on this dataset is not evidence about false "
        f"positives, because no birds appear in the ground truth. Bird recall must be read "
        f"from the dvb/mavvid row instead."
    )


def format_cross_table(results: dict[str, EvalResult]) -> str:
    """The comparison table for the write-up.

    The ``falsifiable?`` column is not decoration - it is the difference between a
    number you can act on and one that merely looks good.
    """
    registry = load_registry()
    header = (
        f"{'dataset':<10}{'map50':>8}{'map50-95':>10}{'prec':>8}{'recall':>8}"
        f"{'drone AP_S':>12}{'bird AP':>10}  {'falsifiable?'}"
    )
    lines = [header, "-" * len(header)]

    for alias in sorted(results):
        result = results[alias]
        spec = registry.datasets.get(alias)
        bird_falsifiable = bool(spec and spec.has_bird_negatives)
        drone_small = _lookup(result, "drone", "small", "map50-95")
        bird = _lookup(result, "bird", None, "map50")

        lines.append(
            f"{alias:<10}"
            f"{result.map50:>8.4f}"
            f"{result.map:>10.4f}"
            f"{result.metrics.get('precision', 0.0):>8.4f}"
            f"{result.metrics.get('recall', 0.0):>8.4f}"
            f"{drone_small:>12.4f}"
            f"{bird:>10.4f}"
            f"  {'yes' if bird_falsifiable else 'NO (drone-only)'}"
        )

    lines.append("")
    lines.append("map50-95 is the metric to compare on. mAP50 at a 12-28 px target is")
    lines.append("generous: a loose box on the right patch scores well.")
    lines.append("")
    lines.append("Rows marked NO cannot falsify a precision claim. Judge false positives")
    lines.append("only on the dvb and mavvid rows.")
    return "\n".join(lines)


def _lookup(result: EvalResult, class_name: str, scale: str | None, metric: str) -> float:
    if scale:
        return float(
            result.per_class_and_scale.get(class_name, {}).get(scale, {}).get(metric, 0.0)
        )
    return float(result.per_class.get(class_name, {}).get(metric, 0.0))


def format_result(result: EvalResult) -> str:
    lines = [
        f"run   : {result.run}",
        f"scope : {result.scope}",
        f"images: {result.images}   gt boxes: {result.boxes_gt}",
        "",
        f"  precision {result.metrics.get('precision', 0.0):.4f}",
        f"  recall    {result.metrics.get('recall', 0.0):.4f}",
        f"  mAP50     {result.map50:.4f}",
        f"  mAP50-95  {result.map:.4f}",
    ]
    if result.per_class:
        lines.append("")
        lines.append(f"{'class':<10}{'prec':>9}{'recall':>9}{'mAP50':>9}{'mAP50-95':>10}{'AP_S':>9}")
        for name, values in sorted(result.per_class.items()):
            small = result.per_class_and_scale.get(name, {}).get("small", {})
            lines.append(
                f"{name:<10}"
                f"{values.get('precision', 0.0):>9.4f}"
                f"{values.get('recall', 0.0):>9.4f}"
                f"{values.get('map50', 0.0):>9.4f}"
                f"{values.get('map50-95', 0.0):>10.4f}"
                f"{small.get('map50-95', 0.0):>9.4f}"
            )
    if result.notes:
        lines.append("")
        lines.append("notes:")
        lines.extend(f"  - {n}" for n in result.notes)
    return "\n".join(lines)


def _slug(run_or_path: str | Path) -> str:
    return Path(str(run_or_path)).parent.name or Path(str(run_or_path)).stem


def load_detector_for_manifest(run: str | Path) -> Detector:
    """Open a detector from a run name, for the API and the replayer."""
    return Detector(run)
