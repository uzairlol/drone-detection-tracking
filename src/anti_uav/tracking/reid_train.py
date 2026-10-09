"""Train the ReID model used by the appearance gate.

Identity-classification training on detection crops: every crop from the same
source sequence gets the same label, so the network is pushed to make two views
of one target similar before the head is even read. That is a stronger and much
simpler signal than the contrastive alternatives, and it suits the budget here.

Which dataset
-------------
**MM-UAV**, and this is not incidental. Appearance learning needs *many identities*,
and Anti-UAV - the one dataset with clean ground truth and a natural flight
geometry - is single-target, so it has exactly one identity and a classification
head trained on it learns nothing. MM-UAV is multi-object MOT with identity
preserved across leave-and-re-enter, so every sequence contributes several
identities and the model actually has something to separate.

The trade-off is uncomfortable but correct: the ReID model trains on the hardest
targets in the project (12 px) and then has to generalise to easier ones.

Leakage
-------
Split by **sequence**, never by frame. Two adjacent crops of the same target
landing in train and val would report a near-perfect mAP for a model that had
memorised one drone. This reuses the sequence-level splitter for exactly the reason
the detection pipeline does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config.loader import load_registry
from ..utils.io import write_json
from ..utils.logging import get_logger
from ..utils.paths import subdir
from ..utils.seed import seed_everything
from .appearance import CROP_CONTEXT, CROP_SIZE, crop_box
from .reid_model import ARCHITECTURES, ReIdNet, summarise_model

log = get_logger(__name__)


@dataclass(slots=True)
class CropSample:
    """One labelled crop."""

    image_path: Path
    box: tuple[float, float, float, float]
    label: int
    sequence_id: str
    dataset: str
    frame_index: int = 0


@dataclass(slots=True)
class ReIdConfig:
    arch: str = "color"
    embedding_dim: int = 512
    input_size: int = CROP_SIZE
    crop_context: float = CROP_CONTEXT
    epochs: int = 30
    batch: int = 64
    lr: float = 0.001
    weight_decay: float = 0.0005
    label_smoothing: float = 0.1
    max_crops_per_sequence: int = 400
    #: Skip crops smaller than this many pixels. A 6 px crop teaches nothing.
    min_crop_px: float = 12.0
    val_fraction: float = 0.15
    seed: int = 0
    workers: int = 0
    amp: bool = True
    device: str = "auto"
    notes: str = ""


@dataclass(slots=True)
class ReIdResult:
    run_dir: str
    best_checkpoint: str | None
    epochs: int
    classes: int
    best_val_accuracy: float
    duration_s: float
    samples: int
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.best_checkpoint is not None


def collect_crops(
    dataset: str,
    variant: str,
    config: ReIdConfig,
    *,
    modality: str = "rgb",
) -> list[CropSample]:
    """Gather labelled crops from a converted dataset.

    The identity label is the **sequence id**, which is only correct when the
    sequence contains exactly one target (Anti-UAV's case). For a multi-object
    dataset the MOT track id is used instead, giving one class per target identity
    - which is what makes MM-UAV the right source. If neither is available the
    sequence id is used and a warning is emitted, because silently training on
    labels that do not correspond to identities produces a model that scores well
    and re-identifies nothing.
    """
    from ..data.frameindex import interim_root, load_index
    from ..data.harmonize import read_yolo_label

    registry = load_registry()
    spec = registry.datasets.get(dataset)
    if spec is None:
        raise KeyError(f"unknown dataset {dataset!r}")

    root = interim_root(dataset, variant)
    records = [
        r for r in load_index(dataset, variant)
        if r.modality == modality and r.box_count > 0
    ]
    if not records:
        return []

    labels: dict[str, int] = {}
    samples: list[CropSample] = []
    per_sequence: dict[str, int] = {}
    multi_object = spec.has_track_ids and len({r.track_id for r in records if r.track_id}) > 1

    for record in records:
        key = record.track_id if (multi_object and record.track_id) else record.sequence_id
        key_str = f"id{key}"
        if key_str not in labels:
            labels[key_str] = len(labels)

        if per_sequence.get(record.sequence_id, 0) >= config.max_crops_per_sequence:
            continue

        label_path = root / record.label
        if not label_path.is_file() or not record.width or not record.height:
            continue

        boxes = read_yolo_label(
            label_path,
            width=record.width,
            height=record.height,
            strict=False,
        )
        drone_index = spec.classes.unified_labels.index("drone")
        image_path = root / record.image
        if not image_path.is_file():
            continue

        for class_id, cx, cy, bw, bh in boxes:
            if class_id != drone_index:
                continue
            x1 = (cx - bw / 2.0) * record.width
            y1 = (cy - bh / 2.0) * record.height
            x2 = (cx + bw / 2.0) * record.width
            y2 = (cy + bh / 2.0) * record.height
            if (x2 - x1) < config.min_crop_px or (y2 - y1) < config.min_crop_px:
                continue

            samples.append(
                CropSample(
                    image_path=image_path,
                    box=(x1, y1, x2, y2),
                    label=labels[key_str],
                    sequence_id=record.sequence_id,
                    dataset=dataset,
                    frame_index=record.frame_index,
                )
            )
            per_sequence[record.sequence_id] = per_sequence.get(record.sequence_id, 0) + 1

    if not multi_object:
        log.warning(
            "Using sequence id as the identity label. This is only a real identity "
            "for a single-target dataset (Anti-UAV). For multi-object footage train on "
            "MM-UAV, where the MOT track id gives one class per target.",
            extra={"dataset": dataset, "classes": len(labels)},
        )

    log.info(
        "crops collected",
        extra={
            "dataset": dataset,
            "variant": variant,
            "crops": len(samples),
            "identities": len(labels),
            "multi_object": multi_object,
        },
    )
    return samples


def split_by_sequence(samples: Sequence[CropSample], val_fraction: float, seed: int) -> tuple[list[CropSample], list[CropSample]]:
    """Split crops by sequence. Never by frame - see the module docstring."""
    from ..utils.seed import rng_for

    sequences = sorted({s.sequence_id for s in samples})
    if not sequences:
        return [], []
    generator = rng_for(seed, "reid_split")
    shuffled = [sequences[i] for i in generator.permutation(len(sequences))]

    n_val = round(len(shuffled) * val_fraction)
    n_val = max(1, min(n_val, len(shuffled) - 1)) if len(shuffled) > 1 else 0
    val_sequences = set(shuffled[:n_val])

    train = [s for s in samples if s.sequence_id not in val_sequences]
    val = [s for s in samples if s.sequence_id in val_sequences]
    log.info(
        "reid split",
        extra={
            "train": len(train),
            "val": len(val),
            "train_sequences": len(shuffled) - n_val,
            "val_sequences": n_val,
        },
    )
    return train, val


if TYPE_CHECKING:  # torch is imported lazily inside train()
    from torch.utils.data import Dataset


class _CropDataset(Dataset[tuple[Any, int]]):
    """Minimal Dataset that decodes crops on the fly.

    Deliberately not using torchvision's ImageFolder machinery: the labels here are
    per-crop identities derived from annotations, not directory names, and
    reconstructing that mapping would add more code than it saves.
    """

    def __init__(self, samples: Sequence[CropSample], config: ReIdConfig) -> None:
        self.samples = list(samples)
        self.config = config

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        from ..utils.imaging import read_image

        sample = self.samples[index]
        image = read_image(sample.image_path)
        crop = crop_box(
            image,
            sample.box,
            context=self.config.crop_context,
            out_size=self.config.input_size,
        )
        return crop, sample.label


def train_reid(
    dataset: str,
    variant: str,
    config: ReIdConfig,
    *,
    modality: str = "rgb",
    run_name: str | None = None,
) -> ReIdResult:
    """Train a ReID model. Runs on CPU; the GPU makes it ~50x faster."""
    import time

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    result = ReIdResult(
        run_dir="", best_checkpoint=None, epochs=0, classes=0,
        best_val_accuracy=0.0, duration_s=0.0, samples=0,
    )
    name = run_name or f"{dataset}_{variant}_{config.arch}"
    run_dir = subdir("runs") / "reid" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    result.run_dir = str(run_dir)

    if config.arch not in ARCHITECTURES:
        result.error = f"unknown arch {config.arch!r}; available: {ARCHITECTURES}"
        return result

    seed_everything(config.seed)
    device = _device(config.device)

    samples = collect_crops(dataset, variant, config, modality=modality)
    result.samples = len(samples)
    if not samples:
        result.error = (
            f"no usable crops from {dataset}/{variant}. Everything was filtered out - "
            f"check that the dataset was converted and that min_crop_px "
            f"({config.min_crop_px} px) is not above the median target size."
        )
        return result

    num_classes = len({s.label for s in samples})
    result.classes = num_classes
    if num_classes < 2:
        result.error = (
            f"only {num_classes} identity in the crops. An identity-classification head "
            f"needs at least 2. Use a multi-object dataset (mmuav) rather than a "
            f"single-target one."
        )
        return result

    train_samples, val_samples = split_by_sequence(samples, config.val_fraction, config.seed)
    if not val_samples:
        result.warnings.append(
            "No validation sequences after splitting; accuracy is reported on the "
            "training set and is not meaningful. Add sequences."
        )
        val_samples = train_samples[: max(1, len(train_samples) // 10)]

    loader_kwargs: dict[str, Any] = {
        "batch_size": config.batch,
        "num_workers": config.workers,
        "shuffle": True,
        "drop_last": len(train_samples) > config.batch,
    }
    if config.workers > 0:  # Windows needs the spawn-safe guard
        loader_kwargs["persistent_workers"] = True

    train_loader: DataLoader[tuple[Any, int]] = DataLoader(
        _CropDataset(train_samples, config), **loader_kwargs
    )
    eval_loader: DataLoader[tuple[Any, int]] = DataLoader(
        _CropDataset(val_samples, config),
        batch_size=config.batch,
        num_workers=config.workers,
        shuffle=False,
    )

    model = ReIdNet(
        arch=config.arch,
        num_classes=num_classes,
        embedding_dim=config.embedding_dim,
        input_size=config.input_size,
    ).to(device)

    log.info("reid model", extra={"summary": summarise_model(model), "device": str(device)})
    if num_classes > 512:
        result.warnings.append(
            f"{num_classes} identities with a {config.embedding_dim}-d embedding is a "
            f"hard problem for a model this small. Expect a mediocre appearance gate; "
            f"increase embedding_dim or reduce the identity count."
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    started = time.monotonic()
    best_accuracy = 0.0
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        seen = 0
        for crops, labels in train_loader:
            crops = crops.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, _features = model(crops, mode="classify")
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * labels.size(0)
            seen += labels.size(0)

        accuracy = _evaluate(model, eval_loader, device)
        scheduler.step()
        history.append({"epoch": epoch + 1, "loss": running / max(seen, 1), "val_acc": accuracy})
        result.epochs = epoch + 1

        log.info(
            "reid epoch",
            extra={
                "epoch": epoch + 1,
                "of": config.epochs,
                "loss": round(running / max(seen, 1), 4),
                "val_acc": round(accuracy, 4),
            },
        )

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "arch": config.arch,
                        "embedding_dim": config.embedding_dim,
                        "input_size": config.input_size,
                        "num_classes": num_classes,
                    },
                    "epoch": epoch + 1,
                    "val_accuracy": accuracy,
                },
                str(run_dir / "best.pt"),
            )

    result.duration_s = time.monotonic() - started
    result.best_val_accuracy = best_accuracy
    result.best_checkpoint = str(run_dir / "best.pt") if (run_dir / "best.pt").is_file() else None

    write_json(
        run_dir / "reid_report.json",
        {
            "dataset": dataset,
            "variant": variant,
            "classes": num_classes,
            "samples": len(samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "epochs": config.epochs,
            "best_val_accuracy": round(best_accuracy, 5),
            "duration_s": round(result.duration_s, 1),
            "model": summarise_model(model),
            "config": config.__dict__,
            "history": history,
            "warnings": result.warnings,
            "notes": (
                "Identity-classification training. val accuracy measures how well crops "
                "of the same identity are grouped - it is NOT the tracking metric. The "
                "value that matters is HOTA/IDF1 from `anti-uav track-eval`, which uses "
                "this model through the BoT-SORT appearance gate."
            ),
        },
    )
    return result


def _evaluate(model: ReIdNet, loader: Any, device: Any) -> float:
    import torch

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for crops, labels in loader:
            crops = crops.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, _features = model(crops, mode="classify")
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.numel()
    return correct / max(total, 1)


def _device(requested: str) -> Any:
    import torch

    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def format_result(result: ReIdResult) -> str:
    lines = [
        f"run dir      : {result.run_dir}",
        f"samples      : {result.samples}",
        f"identities   : {result.classes}",
        f"epochs       : {result.epochs}",
        f"best val acc : {result.best_val_accuracy:.4f}",
        f"duration     : {result.duration_s / 60:.1f} min",
    ]
    if result.best_checkpoint:
        lines.append(f"checkpoint   : {result.best_checkpoint}")
        lines.append("")
        lines.append("Use it with:")
        lines.append(
            f"    anti-uav track-eval --dataset <ds> --reid {result.best_checkpoint}"
        )
        lines.append(
            "    (omit --reid to pick up the newest artifacts/runs/reid/*/best.pt automatically;"
        )
        lines.append("     add --no-reid to ablate the appearance gate and compare the two rows)")
    if result.warnings:
        lines.append("")
        lines.append("warnings:")
        lines.extend(f"  ! {w}" for w in result.warnings)
    if result.error:
        lines.append("")
        lines.append(f"ERROR: {result.error}")
    return "\n".join(lines)


def default_config() -> ReIdConfig:
    return ReIdConfig()


def available_datasets() -> list[str]:
    registry = load_registry()
    return [
        alias for alias, spec in registry.datasets.items() if spec.has_track_ids or spec.has_bird_negatives
    ]
