"""Combo builder: turn converted datasets into a trainable YOLO dataset.

Given one or more converted datasets, this produces the directory + ``data.yaml``
that ``anti-uav train`` consumes, and it is where **tiling** is applied.

Tiling, and why it is per-dataset rather than per-combo
-------------------------------------------------------
The project's smallest targets are MM-UAV's 12x5 px drones and Drone-vs-Bird's
34x23 px ones. Letterboxed into a 640x640 input, a 12 px drone becomes roughly
6x3 px - below what any of these detectors reliably sees. Cropping a small window
and letting the model upscale it fixes that: MM-UAV's 640x360 RGB frames tiled at
256 px and upscaled to 640 give a 2.5x magnification, taking the target to ~30x12 px.

Tiling is decided **per source dataset** from its ``median_target_px`` against
``matrix.tiling_threshold_px``, not per combo. Tiling a whole ``dvb+mavvid`` combo
would push MAV-VID's 171 px targets through 4-6x magnification each for no reason
while dvb's 28 px targets genuinely need it. Per-dataset keeps the cost where it
buys recall.

Layout produced::

    data/processed/<combo>/
        images/train/...   images/val/...
        labels/train/...   labels/val/...
        data.yaml
        build_report.json

Tiles are written as flat images under their sequence, so the sequence id in the
path still groups them for any later re-split.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config.schema import ComboSpec, DatasetSpec, ExperimentMatrix, TilingSpec
from ..utils.imaging import read_image, tile_boxes, tile_grid, write_image
from ..utils.io import require_free_space, write_json, write_text
from ..utils.logging import get_logger
from ..utils.paths import combo_root
from .frameindex import FrameRecord, load_index, load_report
from .harmonize import read_yolo_label, write_yolo_label

log = get_logger(__name__)

#: Hard ceiling on the area blow-up tiling may cause. At overlap 0.25 a 4x6 grid
#: over a 3840x2160 frame is ~11x the pixel area; past this the gain stops being
#: worth the GPU hours and something is configured wrong.
MAX_AREA_EXPANSION = 12.0


@dataclass(slots=True)
class SourceBuild:
    dataset: str
    variant: str
    tiling: bool
    tile_size: int
    frames_in: int = 0
    frames_out: int = 0
    boxes_out: int = 0
    train_frames: int = 0
    val_frames: int = 0
    skipped: bool = False
    reason: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BuildReport:
    combo: str
    datasets: list[str]
    sources: list[SourceBuild] = field(default_factory=list)
    total_train_frames: int = 0
    total_val_frames: int = 0
    total_boxes: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    output_dir: str = ""
    data_yaml: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    ok: bool = False

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)


def resolve_tiling(
    spec: DatasetSpec,
    combo: ComboSpec,
    matrix: ExperimentMatrix,
) -> TilingSpec:
    """Decide tiling for one source inside one combo.

    Precedence: the dataset's own ``tiling_override`` > the combo's
    ``force_tiling`` > the automatic median-size rule.
    """
    if spec.tiling_override is not None:
        return spec.tiling_override
    if combo.force_tiling is not None:
        return TilingSpec(enabled=combo.force_tiling)
    if spec.median_target_px is not None:
        return TilingSpec(enabled=spec.median_target_px <= matrix.tiling_threshold_px)
    return TilingSpec(enabled=False)


def build(
    combo: ComboSpec,
    specs: dict[str, DatasetSpec],
    matrix: ExperimentMatrix,
    *,
    variants: dict[str, str] | None = None,
    unified_labels: Sequence[str] = ("drone", "bird"),
    clean: bool = True,
    copy_images: bool = True,
    max_frames_per_source: int | None = None,
) -> BuildReport:
    """Materialise one combo. Returns the report; ``report.ok`` gates training."""
    root = combo_root(combo.slug)
    report = BuildReport(combo=combo.slug, datasets=list(combo.datasets), output_dir=str(root))
    variants = variants or {}
    tiling_specs: dict[str, TilingSpec] = {}

    if clean and root.exists():
        log.info("cleaning previous build", extra={"combo": combo.slug, "dir": str(root)})
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    total_expected = 0
    for alias in combo.datasets:
        spec = specs.get(alias)
        if spec is None:
            report.errors.append(f"unknown dataset {alias!r} in combo {combo.slug}")
            continue
        tiling = resolve_tiling(spec, combo, matrix)
        tiling_specs[alias] = tiling
        variant = variants.get(alias) or spec.default_variant or "full"
        records = load_index(alias, variant)

        if not records:
            source = SourceBuild(
                dataset=alias,
                variant=variant,
                tiling=tiling.enabled,
                tile_size=tiling.tile_size,
                skipped=True,
                reason=(
                    f"data/interim/{alias}/{variant} is empty or missing. The download "
                    f"probably has not run (or, for MM-UAV, the Baidu transfer has not "
                    f"finished). This source is SKIPPED, so the combo is smaller than "
                    f"declared - check build_report.json before training."
                ),
            )
            report.sources.append(source)
            report.warn(source.reason)
            log.warning("source missing", extra={"combo": combo.slug, "dataset": alias})
            continue

        if max_frames_per_source:
            records = records[:max_frames_per_source]
            report.warn(
                f"{alias}: capped at {max_frames_per_source:,} frames by "
                f"--max-frames-per-source. This is a smoke-test configuration; the "
                f"resulting metrics will not reflect the full dataset."
            )

        total_expected += len(records)
        _emit_source(
            records=records,
            alias=alias,
            variant=variant,
            tiling=tiling,
            root=root,
            report=report,
            copy_images=copy_images,
            unified_labels=unified_labels,
            source_report=report.sources,
        )

    if total_expected == 0:
        report.errors.append(
            f"{combo.slug}: no source produced any frame. Nothing to build."
        )
        return report

    if report.total_train_frames == 0 or report.total_val_frames == 0:
        report.errors.append(
            f"{combo.slug}: built {report.total_train_frames:,} train and "
            f"{report.total_val_frames:,} val frames. One of the splits is empty - run "
            f"`anti-uav splits --combo {combo.slug}` before building."
        )
        return report

    _write_data_yaml(root, combo, unified_labels, variants)
    report.data_yaml = str(root / "data.yaml")
    # `ok` must be settled BEFORE the report is serialised. build_report.json is
    # what `anti-uav sanity` and the trainer read to decide whether this combo is
    # safe to train on, so a report written first leaves every successful build
    # claiming `"ok": false` on disk while the returned object says otherwise.
    report.ok = not report.errors
    write_json(root / "build_report.json", _report_dict(report, tiling_specs))
    return report


def _emit_source(
    *,
    records: list[FrameRecord],
    alias: str,
    variant: str,
    tiling: TilingSpec,
    root: Path,
    report: BuildReport,
    copy_images: bool,
    unified_labels: Sequence[str],
    source_report: list[SourceBuild],
) -> None:
    from .frameindex import interim_root

    interim = interim_root(alias, variant)
    source = SourceBuild(
        dataset=alias,
        variant=variant,
        tiling=tiling.enabled,
        tile_size=tiling.tile_size,
        frames_in=len(records),
    )

    train_images = root / "images" / "train"
    val_images = root / "images" / "val"
    train_labels = root / "labels" / "train"
    val_labels = root / "labels" / "val"

    expansion_seen: list[float] = []

    for index, record in enumerate(records):
        if record.split not in {"train", "val"}:
            continue

        image_path = interim / record.image
        label_path = interim / record.label
        if not image_path.is_file():
            continue

        boxes = read_yolo_label(
            label_path,
            width=record.width,
            height=record.height,
            strict=False,
        )

        if tiling.enabled:
            emitted = _emit_tiles(
                image=read_image(image_path),
                boxes=boxes,
                record=record,
                tiling=tiling,
                images_dir=train_images if record.split == "train" else val_images,
                labels_dir=train_labels if record.split == "train" else val_labels,
                expansion_seen=expansion_seen,
            )
        else:
            _emit_single(
                image_path=image_path,
                boxes=boxes,
                record=record,
                images_dir=train_images if record.split == "train" else val_images,
                labels_dir=train_labels if record.split == "train" else val_labels,
                copy_images=copy_images,
            )
            emitted = 1

        source.frames_out += emitted
        source.boxes_out += len(boxes) * (emitted if not tiling.enabled else 1)
        if record.split == "train":
            source.train_frames += emitted
            report.total_train_frames += emitted
        else:
            source.val_frames += emitted
            report.total_val_frames += emitted

        for box in boxes:
            class_id = box[0]
            name = unified_labels[class_id] if class_id < len(unified_labels) else f"_{class_id}"
            report.class_counts[name] = report.class_counts.get(name, 0) + 1

        if (index + 1) % 5_000 == 0:
            log.info(
                "build progress",
                extra={
                    "combo": report.combo,
                    "dataset": alias,
                    "done": index + 1,
                    "total": len(records),
                    "frames": report.total_train_frames + report.total_val_frames,
                },
            )

    report.total_boxes += source.boxes_out
    if expansion_seen and max(expansion_seen) > MAX_AREA_EXPANSION:
        source.warnings.append(
            f"Tiling expanded this source by {max(expansion_seen):.1f}x in pixel area "
            f"(limit {MAX_AREA_EXPANSION:.0f}x). Lower overlap or raise tile_size if "
            f"training time becomes a problem."
        )
    source_report.append(source)

    log.info(
        "source built",
        extra={
            "dataset": alias,
            "tiling": tiling.enabled,
            "tile_size": tiling.tile_size,
            "frames_in": source.frames_in,
            "frames_out": source.frames_out,
            "train": source.train_frames,
            "val": source.val_frames,
        },
    )


def _emit_single(
    *,
    image_path: Path,
    boxes: list[tuple[int, float, float, float, float]],
    record: FrameRecord,
    images_dir: Path,
    labels_dir: Path,
    copy_images: bool,
) -> None:
    """One image, unchanged, at original resolution."""
    # Namespace by source so two datasets with colliding sequence ids cannot
    # overwrite each other's frames.
    stem = f"{record.dataset}__{record.group_key.replace('/', '_')}__{Path(record.image).stem}"
    target_image = images_dir / f"{stem}.jpg"
    target_label = labels_dir / f"{stem}.txt"

    if copy_images:
        write_image(target_image, read_image(image_path))
    else:
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
    write_yolo_label(target_label, boxes)


def _emit_tiles(
    *,
    image: np.ndarray,
    boxes: list[tuple[int, float, float, float, float]],
    record: FrameRecord,
    tiling: TilingSpec,
    images_dir: Path,
    labels_dir: Path,
    expansion_seen: list[float],
) -> int:
    """Slice a frame into tiles, remapping each box into every tile it overlaps."""
    height, width = image.shape[:2]
    tiles = tile_grid(width, height, tile_size=tiling.tile_size, overlap=tiling.overlap)
    if not tiles:
        return 0

    box_array = (
        np.asarray(
            [
                [
                    (cx - bw / 2.0) * width,
                    (cy - bh / 2.0) * height,
                    (cx + bw / 2.0) * width,
                    (cy + bh / 2.0) * height,
                ]
                for _cls, cx, cy, bw, bh in boxes
            ],
            dtype=float,
        )
        if boxes
        else np.empty((0, 4), dtype=float)
    )
    classes = [box[0] for box in boxes]

    if tiles:
        covered = sum(t.width * t.height for t in tiles)
        expansion_seen.append(covered / max(width * height, 1))

    emitted = 0
    base = f"{record.dataset}__{record.group_key.replace('/', '_')}__{Path(record.image).stem}"

    for tile in tiles:
        # Tiles with no target are still emitted, as negatives. With 25% overlap a
        # frame becomes 4-6 windows and most contain sky and clutter rather than
        # the drone, which is exactly the material that teaches the detector not to
        # fire. Dropping them would bias the training set towards positives and
        # leave the rule layer's confidence.initiate threshold uncalibratable.
        tile_boxes_xyxy = tile_boxes(box_array, tile, min_visibility=tiling.min_visibility)

        crop = image[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
        if crop.size == 0:
            continue

        tile_name = f"{base}__{tile.name}"
        write_image(images_dir / f"{tile_name}.jpg", crop)

        yolo_boxes: list[tuple[int, float, float, float, float]] = []
        for index in range(tile_boxes_xyxy.shape[0]):
            x1, y1, x2, y2 = tile_boxes_xyxy[index]
            box = (x1, y1, x2 - x1, y2 - y1)
            cx = (x1 + (x2 - x1) / 2.0) / tile.width
            cy = (y1 + (y2 - y1) / 2.0) / tile.height
            bw = (x2 - x1) / tile.width
            bh = (y2 - y1) / tile.height
            yolo_boxes.append((classes[index], cx, cy, bw, bh))

        write_yolo_label(labels_dir / f"{tile_name}.txt", yolo_boxes)
        emitted += 1

    return emitted


def _write_data_yaml(
    root: Path,
    combo: ComboSpec,
    unified_labels: Sequence[str],
    variants: dict[str, str],
) -> Path:
    """Render the ultralytics ``data.yaml``.

    Paths are written as ``.`` relative so the whole processed tree can be moved
    or the drive letter changed without editing the YAML. ``names`` is written
    as a dict because ultralytics resolves the ordering from the keys, and an
    explicit mapping makes the drone=0 / bird=1 contract visible in the file.
    """
    lines = [
        "# GENERATED by anti-uav build. Edit configs/matrix.yaml and rebuild.",
        f"# combo: {combo.slug}",
        f"# sources: {', '.join(combo.datasets)}",
        "",
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(unified_labels))
    lines.extend(
        [
            "",
            f"# variants used: {variants or 'defaults'}",
            f"# tiling threshold: median_target_px <= matrix.tiling_threshold_px",
        ]
    )
    return write_text(root / "data.yaml", "\n".join(lines) + "\n")


def _report_dict(report: BuildReport, tiling: dict[str, TilingSpec]) -> dict:
    return {
        "combo": report.combo,
        "datasets": report.datasets,
        "output_dir": report.output_dir,
        "data_yaml": report.data_yaml,
        "total_train_frames": report.total_train_frames,
        "total_val_frames": report.total_val_frames,
        "total_boxes": report.total_boxes,
        "class_counts": report.class_counts,
        "tiling_per_source": {
            alias: {"enabled": spec.enabled, "tile_size": spec.tile_size, "overlap": spec.overlap}
            for alias, spec in tiling.items()
        },
        "sources": [
            {
                "dataset": s.dataset,
                "variant": s.variant,
                "tiling": s.tiling,
                "tile_size": s.tile_size,
                "frames_in": s.frames_in,
                "frames_out": s.frames_out,
                "boxes": s.boxes_out,
                "train": s.train_frames,
                "val": s.val_frames,
                "skipped": s.skipped,
                "reason": s.reason,
                "warnings": s.warnings,
            }
            for s in report.sources
        ],
        "warnings": report.warnings,
        "errors": report.errors,
        "ok": report.ok,
    }


def check_disk_budget(records: Sequence[FrameRecord], tiling: TilingSpec) -> str:
    """Warn before the build fills the disk. Returns a human-readable estimate."""
    if not records:
        return ""
    average_bytes = 90_000  # ~90 kB for a 640x360 JPEG from these datasets
    multiplier = 1.0
    if tiling.enabled:
        multiplier = 1.0 + tiling.overlap**-1 if tiling.overlap else 2.0
    estimate = len(records) * average_bytes * max(multiplier, 1.0)
    return (
        f"Estimated build size: {estimate / (1024**3):.1f} GB "
        f"(tiling={'on' if tiling.enabled else 'off'}, {len(records):,} frames)"
    )


def require_room(root: Path, estimate_bytes: float) -> None:
    require_free_space(root, int(estimate_bytes * 1.5), context="combo build")


def build_report_for(combo_slug: str) -> dict | None:
    """Read back a previous build report."""
    from ..utils.io import read_json

    path = combo_root(combo_slug) / "build_report.json"
    return read_json(path) if path.is_file() else None


def conversion_problems(alias: str, variant: str) -> list[str]:
    """Surface converter warnings/errors when building, so they are not buried."""
    report = load_report(alias, variant)
    if report is None:
        return [f"{alias}/{variant}: no convert_report.json - has `anti-uav convert` run?"]
    return [*report.errors, *report.warnings]