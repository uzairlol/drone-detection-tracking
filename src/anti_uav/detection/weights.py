"""Pretrained base weights: locate them before a run needs them.

Every training recipe names a starting checkpoint in ``base_weights`` -
``yolo11n.pt`` for the edge candidate, ``rtdetr-x2.pt`` for the accuracy
reference. Ultralytics will fetch a missing one on demand, which is convenient
right up until it is not: the fetch happens *inside* the trainer, so on a
machine that cannot reach the release host the run dies after the dataset has
been built, the split validated and the run directory created. You find out at
the one moment you can least afford it.

So the check is hoisted out of the trainer into its own command. ``fetch-weights``
reports what is present, and downloads what is missing, before a single GPU hour
is committed. ``anti-uav verify-env`` reports the same facts as a non-blocking
warning, because a missing checkpoint is a five-minute problem - but one that is
much cheaper to discover at 09:00 than at 09:00 the day after a deadline.

Resolution order for a bare filename, which is all a recipe ever stores:

1. the current working directory (this is where ultralytics drops a manual
   download, and where ``yolo11n.pt`` sits in this repo)
2. the project root
3. ultralytics' own weights directory, if the package is importable

An absolute or explicitly relative path in ``base_weights`` is honoured as-is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config.loader import load_matrix, load_recipe
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class WeightTarget:
    """One recipe's starting checkpoint and whether we can find it."""

    family: str
    declared: str
    path: Path | None

    @property
    def name(self) -> str:
        return Path(self.declared).name

    @property
    def present(self) -> bool:
        return self.path is not None and self.path.is_file()

    @property
    def size_mb(self) -> float:
        # Bind to a local: mypy cannot narrow `self.path` through the `present`
        # property, and a property that re-checks the filesystem is not free.
        path = self.path
        if path is None or not path.is_file():
            return 0.0
        return path.stat().st_size / 1024**2

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "declared": self.declared,
            "present": self.present,
            "path": str(self.path) if self.path else None,
            "size_mb": round(self.size_mb, 1),
        }


def _search_dirs() -> list[Path]:
    from ..utils.paths import project_root

    dirs = [Path.cwd(), project_root()]
    try:
        from ultralytics.utils import WEIGHTS_DIR

        dirs.append(Path(WEIGHTS_DIR))
    except Exception:  # pragma: no cover - ultralytics absent or moved the constant
        log.debug("ultralytics weights dir unavailable", exc_info=True)
    return dirs


def locate(declared: str) -> Path | None:
    """Resolve a ``base_weights`` entry to a file, or ``None`` if absent."""
    candidate = Path(declared).expanduser()
    if candidate.is_absolute() or declared.startswith(("./", "..")):
        return candidate if candidate.is_file() else None

    name = candidate.name
    for directory in _search_dirs():
        path = directory / name
        if path.is_file():
            return path
    return None


def targets(models: Sequence[str] | None = None) -> list[WeightTarget]:
    """Every recipe's base weights, in matrix order, de-duplicated.

    ``models`` filters by family; ``None`` means every family the matrix
    declares. An unknown family raises, because silently reporting fewer targets
    than the matrix would run is exactly the kind of quiet gap this module
    exists to close.
    """
    matrix = load_matrix()
    declared = list(matrix.models)
    if models:
        asked = [m.strip() for m in models if m.strip()]
        unknown = [m for m in asked if m not in declared]
        if unknown:
            raise KeyError(
                f"unknown model(s) {unknown}; the matrix declares {[str(d) for d in declared]}"
            )
        # Filter `declared` rather than reassigning to `asked`: that keeps the
        # matrix's ordering (so a report lists families in the order the matrix
        # trains them) and keeps the element type, which is a str enum.
        selected = [family for family in declared if str(family) in asked]
    else:
        selected = declared

    found: dict[str, WeightTarget] = {}
    for family in selected:
        declared_weights = load_recipe(family).base_weights
        # Two families may legitimately share one checkpoint; report it once.
        found.setdefault(
            declared_weights,
            WeightTarget(str(family), declared_weights, locate(declared_weights)),
        )
    return list(found.values())


def missing(models: Sequence[str] | None = None) -> list[WeightTarget]:
    return [t for t in targets(models) if not t.present]


def fetch(
    models: Sequence[str] | None = None,
    *,
    dry_run: bool = False,
) -> tuple[list[WeightTarget], list[str]]:
    """Ensure every declared base weight is on disk.

    Returns ``(targets, errors)``. The returned targets are re-located after the
    attempt, so ``present`` reflects reality rather than intent. Downloading is
    delegated to ultralytics, which is the component that knows the URLs; this
    function owns the *sequencing* (before training, not during it) and the
    reporting.
    """
    resolved = targets(models)
    wanted = [t for t in resolved if not t.present]
    if not wanted:
        return resolved, []
    if dry_run:
        return resolved, []

    from ultralytics import YOLO

    errors: list[str] = []
    for target in wanted:
        # A relative declaration is what ultralytics expects; an absolute one
        # would make it treat the path as a model name and fail confusingly.
        try:
            YOLO(target.declared)
        except Exception as exc:
            errors.append(f"{target.family}: {target.declared}: {type(exc).__name__}: {exc}")
            log.error(
                "base weight fetch failed",
                extra={"family": target.family, "weights": target.declared},
            )
            continue
        log.info("fetched base weights", extra={"family": target.family})

    # Re-locate: the object above is stale the moment a download succeeds.
    return targets(models), errors
