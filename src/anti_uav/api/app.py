"""FastAPI application.

A local operator console and a look at the training state. Deliberately a *reader*
of what the pipeline already wrote - the runs directory, the frame indexes, the
config files - rather than a second source of truth that has to be kept in sync
with them. Everything here resolves from disk at request time, so the UI cannot
drift from reality and needs no database migration when a run changes shape.

No authentication: this binds to 127.0.0.1 by default and is an internal tool on
an offline edge network. That assumption is stated rather than hidden - see
``configs/app.yaml``'s ``api_host``.
"""

from __future__ import annotations

import base64
import contextlib
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import __version__
from ..utils.logging import get_logger, setup_logging
from . import schemas

log = get_logger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "ui" / "templates"
_STATIC_DIR = Path(__file__).resolve().parent.parent / "ui" / "static"

app = FastAPI(
    title="anti-uav",
    version=__version__,
    description=(
        "Drone detection & tracking console. Reads the pipeline's own output: runs, "
        "frame indexes and config files."
    ),
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

_templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #


@app.on_event("startup")
def _startup() -> None:
    setup_logging()
    log.info("api starting", extra={"version": __version__})


# --------------------------------------------------------------------------- #
# meta
# --------------------------------------------------------------------------- #


@app.get("/api/health", response_model=schemas.HealthResponse, tags=["meta"])
def health() -> schemas.HealthResponse:
    """Environment summary and what is actually present on disk."""
    from ..config.loader import detect_profile, load_registry, load_settings
    from ..detection.trainer import list_runs, torch_environment

    # load_settings() validates configs/app.yaml and the GPU profile override on
    # every health call: a broken config should surface here, not on first use.
    load_settings()
    env = torch_environment()
    registry = load_registry()

    present: list[str] = []
    warnings: list[str] = []
    for alias in registry.aliases:
        spec = registry.get(alias)
        variant = spec.default_variant or "full"
        from ..data.frameindex import index_path

        if index_path(alias, variant).is_file():
            present.append(alias)
        else:
            warnings.append(f"{alias}: not converted yet")

    devices = env.get("devices") or []
    return schemas.HealthResponse(
        status="ok" if present else "degraded",
        version=__version__,
        profile=detect_profile().value,
        cuda_available=bool(env.get("cuda_available")),
        gpu=devices[0]["name"] if devices else None,
        datasets_present=present,
        runs=len(list_runs()),
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #


@app.get("/api/datasets", response_model=list[schemas.DatasetSummary], tags=["data"])
def datasets() -> list[schemas.DatasetSummary]:
    """The registry, annotated with what has actually been converted."""
    from ..config.loader import load_registry
    from ..data.capabilities import all_source_capabilities
    from ..data.frameindex import index_path, load_index

    registry = load_registry()
    capabilities = all_source_capabilities()
    out: list[schemas.DatasetSummary] = []

    for alias in registry.aliases:
        spec = registry.get(alias)
        variant = spec.default_variant or "full"
        converted = index_path(alias, variant).is_file()
        frames = len(load_index(alias, variant)) if converted else 0
        caps = capabilities[alias]

        out.append(
            schemas.DatasetSummary(
                alias=alias,
                display_name=spec.display_name,
                modalities=[m.value for m in spec.modalities],
                median_target_px=spec.median_target_px,
                has_bird_negatives=spec.has_bird_negatives,
                approx_frames=spec.approx_frames,
                has_track_ids=spec.has_track_ids,
                has_multi_object_identity=spec.has_multi_object_identity,
                has_visibility_flags=spec.has_visibility_flags,
                official_metric=spec.official_metric,
                converted=converted,
                frames=frames,
                notes=spec.notes,
                caveats=spec.caveats,
                can_falsify_precision=caps.can_falsify_precision,
                can_score_identity=caps.can_score_identity,
                can_score_tracking=caps.can_score_tracking,
                can_score_visibility=caps.can_score_visibility,
                requires_tiling=caps.requires_tiling,
                tile_size=caps.tile_size,
                sources=[
                    schemas.SourceSummary(
                        label=s.label,
                        kind=s.kind.value,
                        url=s.url,
                        approx_size_gb=s.approx_size_gb,
                        needs_credentials=s.needs_credentials,
                        notes=s.notes,
                    )
                    for s in spec.sources
                ],
            )
        )
    return out


@app.get("/api/capabilities", tags=["data"])
def capabilities() -> dict[str, object]:
    """What each dataset and combo is allowed to prove.

    The evidential contract, served so the operator console cannot present a
    precision number from a drone-only source as if it were evidence about birds.
    """
    from ..config.loader import load_matrix
    from ..data.capabilities import (
        CLAIM_IDENTITY,
        CLAIM_PRECISION,
        CLAIM_TRACKING,
        CLAIM_VISIBILITY,
        all_source_capabilities,
        combo_capabilities,
    )

    sources = {
        alias: {
            "median_target_px": caps.median_target_px,
            "requires_tiling": caps.requires_tiling,
            "tile_size": caps.tile_size,
            "can_falsify_precision": caps.can_falsify_precision,
            "can_score_identity": caps.can_score_identity,
            "can_score_tracking": caps.can_score_tracking,
            "can_score_visibility": caps.can_score_visibility,
            "claims": caps.claims(),
            "caveats": list(caps.caveats),
        }
        for alias, caps in all_source_capabilities().items()
    }

    combos: dict[str, object] = {}
    for spec in load_matrix().combos:
        caps = combo_capabilities(spec.slug, tuple(spec.datasets))
        combos[spec.slug] = {
            "datasets": list(spec.datasets),
            "claims": caps.claims(),
            "supported_by": {k: list(v) for k, v in caps.supports.items()},
            "tiled_sources": list(caps.tiled_sources),
            "comparable_within_combo_only": caps.is_comparable_across_combos,
            "warnings": caps.warnings(),
        }

    return {
        "claims": {
            "precision": CLAIM_PRECISION,
            "identity": CLAIM_IDENTITY,
            "visibility": CLAIM_VISIBILITY,
            "tracking": CLAIM_TRACKING,
        },
        "sources": sources,
        "combos": combos,
    }


@app.get("/api/datasets/{alias}/stats", response_model=schemas.DatasetStatsResponse, tags=["data"])
def dataset_stats(alias: str) -> schemas.DatasetStatsResponse:
    """Statistics for one converted dataset."""
    from ..config.loader import load_registry
    from ..data.stats import for_dataset

    registry = load_registry()
    spec = registry.datasets.get(alias)
    if spec is None:
        raise HTTPException(404, f"unknown dataset {alias!r}")

    variant = spec.default_variant or "full"
    stats = for_dataset(alias, variant, unified_labels=spec.classes.unified_labels)
    if stats.frames == 0:
        raise HTTPException(
            404,
            f"{alias} is not converted yet. Run: anti-uav convert --dataset {alias}",
        )
    return schemas.DatasetStatsResponse(**stats.to_dict())


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #


@app.get("/api/runs", response_model=list[schemas.RunSummary], tags=["runs"])
def runs() -> list[schemas.RunSummary]:
    """Every training run found on disk."""
    from ..detection.trainer import list_runs as _list

    out: list[schemas.RunSummary] = []
    for run in _list():
        out.append(
            schemas.RunSummary(
                run_name=str(run.get("run_name", "?")),
                model=run.get("model"),
                combo=run.get("combo"),
                profile=run.get("profile"),
                epochs_requested=run.get("epochs_requested"),
                epochs_completed=int(run.get("epochs_completed", 0) or 0),
                imgsz=run.get("imgsz"),
                batch=run.get("batch"),
                amp=run.get("amp"),
                duration_s=run.get("duration_s"),
                best_metrics=run.get("best_metrics") or {},
                best_weights=run.get("best_weights"),
                has_weights=bool(run.get("_has_weights")),
                has_results=bool(run.get("_has_results")),
                dir=str(run.get("_dir", "")),
                trained_at=run.get("trained_at"),
                warnings=run.get("warnings") or [],
            )
        )
    return out


@app.get("/api/runs/{run_name}", response_model=schemas.RunDetail, tags=["runs"])
def run_detail(run_name: str) -> schemas.RunDetail:
    """One run, with its full training curve."""
    from ..detection.trainer import list_runs as _list

    match = next((r for r in _list() if r.get("run_name") == run_name), None)
    if match is None:
        raise HTTPException(404, f"no run named {run_name!r}")

    curve: list[schemas.TrainingCurvePoint] = []
    best_epoch: int | None = None
    csv_path = Path(match.get("results_csv") or "")
    if not csv_path.is_absolute():
        csv_path = _resolve_run_dir(match) / "results.csv"

    if csv_path.is_file():
        curve, best_epoch = _read_curve(csv_path)

    return schemas.RunDetail(
        run_name=str(match.get("run_name", run_name)),
        model=match.get("model"),
        combo=match.get("combo"),
        profile=match.get("profile"),
        epochs_requested=match.get("epochs_requested"),
        epochs_completed=int(match.get("epochs_completed", 0) or 0),
        imgsz=match.get("imgsz"),
        batch=match.get("batch"),
        amp=match.get("amp"),
        duration_s=match.get("duration_s"),
        best_metrics=match.get("best_metrics") or {},
        best_weights=match.get("best_weights"),
        has_weights=bool(match.get("_has_weights")),
        has_results=bool(match.get("_has_results")),
        dir=str(match.get("_dir", "")),
        trained_at=match.get("trained_at"),
        warnings=match.get("warnings") or [],
        environment=match.get("environment") or {},
        recipe=match.get("recipe") or {},
        profile_override=match.get("profile_override") or {},
        results_csv=str(csv_path) if csv_path.is_file() else None,
        curve=curve,
        best_epoch=best_epoch,
    )


@app.get("/api/runs/{run_name}/weights", tags=["runs"])
def run_weights(run_name: str) -> FileResponse:
    """Download ``best.pt``."""
    from ..detection.trainer import list_runs as _list

    match = next((r for r in _list() if r.get("run_name") == run_name), None)
    if match is None or not match.get("best_weights"):
        raise HTTPException(404, f"no best.pt for run {run_name!r}")
    path = Path(str(match["best_weights"]))
    if not path.is_file():
        raise HTTPException(404, f"best_weights recorded but missing: {path}")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@app.get("/api/matrix", response_model=schemas.MatrixResponse, tags=["runs"])
def matrix() -> schemas.MatrixResponse:
    """The run x combo grid, for the comparison table."""
    from ..config.loader import load_matrix
    from ..detection.trainer import list_runs as _list

    spec = load_matrix()
    runs_found = _list()
    lookup = {(r.get("model"), r.get("combo")): r for r in runs_found}

    cells: list[schemas.MatrixCell] = []
    for combo in spec.enabled_combos():
        for model in (m.value for m in spec.models):
            run = lookup.get((model, combo.slug))
            metrics = (run or {}).get("best_metrics") or {}
            cells.append(
                schemas.MatrixCell(
                    run_name=f"{model}__{combo.slug}",
                    model=model,
                    combo=combo.slug,
                    epochs_completed=int((run or {}).get("epochs_completed", 0) or 0),
                    map50=_as_float(metrics.get("metrics/mAP50(B)")),
                    map50_95=_as_float(metrics.get("metrics/mAP50-95(B)")),
                    precision=_as_float(metrics.get("metrics/precision(B)")),
                    recall=_as_float(metrics.get("metrics/recall(B)")),
                    has_weights=bool((run or {}).get("_has_weights")),
                )
            )

    return schemas.MatrixResponse(
        cells=cells,
        models=[m.value for m in spec.models],
        combos=[c.slug for c in spec.enabled_combos()],
        note=(
            "mAP is only comparable within a combo. Cross-dataset numbers come from "
            "/api/runs/{run}/cross, and a combo with no bird-negative source has an "
            "unfalsifiable precision."
        ),
    )


@app.get("/api/runs/{run_name}/cross", response_model=dict[str, schemas.EvalResultModel], tags=["runs"])
def cross_eval(run_name: str) -> dict[str, schemas.EvalResultModel]:
    """Cross-dataset evaluation, with the per-dataset caveat attached."""
    from ..detection.evaluate import evaluate_cross_dataset

    results = evaluate_cross_dataset(run_name, save=False)
    return {alias: schemas.EvalResultModel(**r.to_dict()) for alias, r in results.items()}


def _resolve_run_dir(match: dict[str, Any]) -> Path:
    from ..utils.paths import project_root

    return project_root() / str(match.get("_dir", ""))


def _read_curve(csv_path: Path) -> tuple[list[schemas.TrainingCurvePoint], int | None]:
    import csv as csv_module

    try:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv_module.DictReader(handle))
    except OSError:
        return ([], None)

    points: list[schemas.TrainingCurvePoint] = []
    best_epoch: int | None = None
    best_value = -1.0

    for row in rows:
        metrics: dict[str, float] = {}
        epoch = 0
        for key, value in row.items():
            if key == "epoch":
                with contextlib.suppress(TypeError, ValueError):
                    epoch = int(float(value))
                continue
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                continue
        points.append(schemas.TrainingCurvePoint(epoch=epoch, metrics=metrics))

        candidate = metrics.get("metrics/mAP50-95(B)")
        if candidate is not None and candidate > best_value:
            best_value = candidate
            best_epoch = epoch

    return (points, best_epoch)


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #


@app.post("/api/predict", response_model=schemas.PredictResponse, tags=["inference"])
def predict(request: schemas.PredictRequest) -> schemas.PredictResponse:
    """Run a checkpoint over one image and return detections, tracks and an overlay."""
    import cv2

    from ..detection.predictor import Detector, overlay
    from ..tracking.local import build_tracker
    from ..tracking.types import TrackObservation

    notes: list[str] = []
    if not request.image_path:
        raise HTTPException(
            400,
            "image_path is required. Serve an existing file; the UI lists the val split.",
        )

    path = Path(request.image_path)
    if not path.is_file():
        raise HTTPException(404, f"no such image: {path}")

    try:
        detector = Detector(
            request.run,
            conf=request.conf,
            iou=request.iou,
            imgsz=request.imgsz,
            device=request.device,
        )
        detector.load()
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc

    tracker_name = request.tracker or detector.settings.default_tracker.value
    try:
        tracker = build_tracker(tracker_name)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, f"could not decode image: {path}")

    started = time.perf_counter()
    try:
        frame = detector.predict_image(image)
    finally:
        detector.close()
    elapsed = (time.perf_counter() - started) * 1000.0

    observations = [
        TrackObservation(
            frame_index=0,
            timestamp_s=0.0,
            box=d.box,
            confidence=d.confidence,
            class_id=d.class_id,
            class_name=d.class_name,
            image_height_px=frame.height,
        )
        for d in frame.detections
    ]
    tracks = tracker.update(observations, frame_index=0, timestamp_s=0.0)

    keep = sorted(tracks.tracks, key=lambda t: -t.confidence)[: request.max_tracks]
    for track in keep:
        for detection in frame.detections:
            if detection.box == track.box:
                detection.track_id = track.track_id

    if not tracks.tracks:
        notes.append("no tracks: detections below the tracker threshold, or the target is too small")

    overlay_b64: str | None = None
    if request.return_overlay:
        canvas = overlay(image, frame)
        success, buffer = cv2.imencode(".png", canvas)
        if success:
            overlay_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")

    return schemas.PredictResponse(
        detections=[schemas.DetectionModel(**d.to_dict()) for d in frame.detections],
        tracks=[
            schemas.TrackModel(
                id=t.track_id,
                state=t.state.value,
                box=[round(v, 2) for v in t.box],
                hits=t.hits,
                confidence=round(t.confidence, 4),
            )
            for t in keep
        ],
        width=frame.width,
        height=frame.height,
        elapsed_ms=round(elapsed, 2),
        source=str(path),
        overlay_png_b64=overlay_b64,
        notes=notes,
    )


@app.get("/api/samples", tags=["inference"])
def samples(
    limit: int = Query(default=24, ge=1, le=200),
    split: str = Query(default="val"),
    combo: str = Query(default="dvb"),
) -> dict[str, Any]:
    """Images from a built combo, for the inference playground.

    Served from the processed dataset rather than a fixture directory, so the
    playground shows exactly what the model was trained on - including the tiles,
    which is where the tiling decision becomes visible.
    """
    from ..utils.paths import subdir

    root = subdir("processed") / combo
    listing = root / f"{split}.txt"
    if not listing.is_file():
        return {
            "images": [],
            "note": (
                f"no {split}.txt in {root}. Run: anti-uav splits --combo {combo} "
                f"then anti-uav build --combo {combo}"
            ),
        }

    images: list[dict[str, str]] = []
    with listing.open(encoding="utf-8") as handle:
        for line in handle:
            entry = line.strip()
            if not entry:
                continue
            p = Path(entry)
            if p.is_file():
                with contextlib.suppress(ValueError):
                    p = p.resolve().relative_to(root.resolve())
                images.append({"path": entry, "name": p.name})
            if len(images) >= limit:
                break

    return {"images": images, "note": ""}


# --------------------------------------------------------------------------- #
# rules and coverage
# --------------------------------------------------------------------------- #


@app.get("/api/rules", response_model=schemas.RulesResponse, tags=["rules"])
def rules() -> schemas.RulesResponse:
    """The active rule set plus any cross-rule problems."""
    from ..config.loader import config_path, load_rules
    from ..rules import validate_rules

    path = config_path("rules", "drone_rules.yaml")
    resolved = load_rules()
    return schemas.RulesResponse(
        path=str(path),
        rules=_yaml_to_json(path.read_text(encoding="utf-8")),
        problems=validate_rules(resolved),
    )


@app.post("/api/rules/explain", response_model=schemas.RuleExplainResponse, tags=["rules"])
def rules_explain(request: schemas.RuleExplainRequest) -> schemas.RuleExplainResponse:
    """Evaluate one hypothetical track and show every gate's verdict.

    Exposing this over HTTP is what makes the thresholds debuggable: you can ask
    "why was this not alerted" against the *real* rule set instead of guessing.
    """
    from ..config.loader import load_rules
    from ..rules import RuleEngine
    from ..tracking.types import Track, TrackState

    config = _rule_config()
    engine = RuleEngine(config)

    if request.sightings:
        for global_id, entries in request.sightings.items():
            for entry in entries or []:
                if isinstance(entry, dict):
                    config.record_sighting(
                        int(global_id),
                        str(entry.get("camera_id", "")),
                        str(entry.get("node", "")),
                        float(entry.get("t", 0.0)),
                        tuple(entry.get("ground_xy", (0.0, 0.0))),  # type: ignore[arg-type]
                    )

    track = Track(
        track_id=0,
        class_id=request.class_id,
        class_name="drone" if request.class_id == 0 else "bird",
        camera_id=request.camera_id,
        state=TrackState.CONFIRMED,
        confirmed=True,
        box=tuple(request.box),  # type: ignore[arg-type]
        confidence=request.confidence,
        confidence_ema=request.confidence,
        hits=request.hits,
        misses=request.misses,
        first_timestamp_s=0.0,
        last_timestamp_s=request.duration_s,
        global_id=request.global_id,
        image_height_px=request.image_height_px,
        ground_xy=tuple(request.ground_xy) if request.ground_xy else None,  # type: ignore[arg-type]
        ground_z=request.ground_z,
        velocity_m_s=tuple(request.velocity_m_s or (0.0, 0.0)),  # type: ignore[arg-type]
    )
    del load_rules

    evaluation = engine.evaluate(track, timestamp_s=request.duration_s)
    return schemas.RuleExplainResponse(
        alerted=evaluation.alerted,
        severity=evaluation.severity.value,
        first_failure=evaluation.first_failure,
        gates=[
            schemas.RuleGateResult(
                name=r.name, outcome=r.outcome.value, detail=r.detail, measured=r.measured
            )
            for r in evaluation.results
        ],
        reasons=evaluation.reasons,
    )


def _rule_config():
    from ..rules import build_engine

    return build_engine().config


def _yaml_to_json(text: str) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(text) or {}


@app.get("/api/coverage", response_model=schemas.CoverageResponse, tags=["rules"])
def coverage() -> schemas.CoverageResponse:
    """The 100-camera fleet and its protected zones."""
    from ..tracking.coordination import load_from_config

    model = load_from_config()
    summary = model.coverage_report()
    return schemas.CoverageResponse(
        summary=schemas.CoverageSummary(
            frame=summary["frame"],
            cameras=summary["cameras"],
            fixed=summary["fixed"],
            ptz=summary["ptz"],
            nodes=summary["nodes"],
            cameras_per_node=summary["cameras_per_node"],
            overlap_zones=summary["overlap_zones"],
            protected_zones=summary["protected_zones"],
            uncovered_zones=summary["uncovered_zones"],
            all_zones_covered=summary["all_zones_covered"],
            fixed_covered_zones=summary.get("fixed_covered_zones", []),
            ptz_only_zones=summary.get("ptz_only_zones", []),
        ),
        cameras=[
            schemas.CameraInfo(**view.to_dict())
            for view in sorted(model.views.values(), key=lambda v: v.camera_id)
        ],
        protected_zones=[
            {
                "id": z.id,
                "polygon": z.polygon,
                "priority": z.priority,
                "height_m": z.height_m,
                "description": z.description,
            }
            for z in model.protected_zones()
        ],
        geofences=[
            {"id": g.id, "action": g.action, "polygon": g.polygon, "height_m": g.height_m}
            for g in model.config.geofences
        ],
    )


# --------------------------------------------------------------------------- #
# tracking metrics
# --------------------------------------------------------------------------- #


@app.get("/api/tracking", response_model=schemas.TrackerReportResponse, tags=["runs"])
def tracking(dataset: str = "mmuav", variant: str | None = None) -> schemas.TrackerReportResponse:
    """Stored tracker comparison results, if a ``track-eval`` has been run."""
    from ..config.loader import load_registry
    from ..utils.io import read_json
    from ..utils.paths import subdir

    registry = load_registry()
    spec = registry.datasets.get(dataset)
    resolved = variant or (spec.default_variant if spec else "full") or "full"
    path = subdir("tracks") / f"eval_{dataset}_{resolved}.json"

    if not path.is_file():
        return schemas.TrackerReportResponse(
            dataset=dataset,
            variant=resolved,
            scores=[],
            note=(
                f"no stored results at {path}. Run: anti-uav track-eval --run <run> "
                f"--dataset {dataset}"
            ),
        )

    payload = read_json(path)
    scores = [
        schemas.TrackerScore(
            tracker=name,
            mota=entry.get("mota", 0.0),
            idf1=entry.get("idf1", 0.0),
            hota=entry.get("hota", 0.0),
            precision=entry.get("precision", 0.0),
            recall=entry.get("recall", 0.0),
            id_switches=int(entry.get("id_switches", 0)),
            fragmentations=int(entry.get("fragmentations", 0)),
            sequences=int(entry.get("sequences", 0)),
        )
        for name, entry in payload.items()
    ]
    scores.sort(key=lambda s: -s.hota)
    return schemas.TrackerReportResponse(
        dataset=dataset,
        variant=resolved,
        scores=scores,
        note="HOTA is the column to rank on: MOTA and IDF1 disagree about which tracker is better.",
    )


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request) -> HTMLResponse:
    """The operator console."""
    return _templates.TemplateResponse(request, "index.html", {"version": __version__})


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    return {"status": "ok"}
