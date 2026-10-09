"""Pydantic response/request models for the API.

Kept separate from the routers so the wire format is visible in one place, and so
adding a field is a deliberate act rather than something that happens wherever a
dict is built.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --------------------------------------------------------------------------- #
# health / meta
# --------------------------------------------------------------------------- #


class HealthResponse(_Model):
    status: Literal["ok", "degraded"]
    version: str
    profile: str
    cuda_available: bool
    gpu: str | None = None
    datasets_present: list[str] = Field(default_factory=list)
    runs: int = 0
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(_Model):
    detail: str
    hint: str | None = None


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #


class DatasetSummary(_Model):
    alias: str
    display_name: str
    modalities: list[str]
    median_target_px: float | None = None
    has_bird_negatives: bool = Field(
        description="False means precision measured on this dataset cannot falsify "
        "a false-positive claim."
    )
    approx_frames: int = 0
    has_track_ids: bool = False
    #: More than one identity per sequence, so IDF1 / ID switches mean something.
    #: Distinct from `has_track_ids`: a single-target dataset labels every box
    #: id=1, which satisfies a naive check while making identity metrics vacuous.
    has_multi_object_identity: bool = False
    has_visibility_flags: bool = False
    official_metric: str | None = None
    converted: bool = False
    frames: int = 0
    notes: str = ""
    caveats: list[str] = Field(default_factory=list)
    sources: list[SourceSummary] = Field(default_factory=list)
    #: The evidential contract, derived once in `data/capabilities.py`, so the UI
    #: cannot disagree with what the CLI prints.
    can_falsify_precision: bool = False
    can_score_identity: bool = False
    can_score_tracking: bool = False
    can_score_visibility: bool = False
    requires_tiling: bool = False
    tile_size: int | None = None


class SourceSummary(_Model):
    label: str
    kind: str
    url: str
    approx_size_gb: float
    needs_credentials: bool
    notes: str = ""


class DatasetStatsResponse(_Model):
    dataset: str
    variant: str
    frames: int = 0
    boxes: int = 0
    sequences: int = 0
    empty_frames: int = 0
    class_counts: dict[str, int] = Field(default_factory=dict)
    bird_sequences: int = 0
    provides_bird_negatives: bool = False
    size_percentiles: dict[str, float] = Field(default_factory=dict)
    size_histogram: dict[str, int] = Field(default_factory=dict)
    frame_sizes: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #


class RunSummary(_Model):
    run_name: str
    model: str | None = None
    combo: str | None = None
    profile: str | None = None
    epochs_requested: int | None = None
    epochs_completed: int = 0
    imgsz: int | None = None
    batch: int | None = None
    amp: bool | None = None
    duration_s: float | None = None
    best_metrics: dict[str, Any] = Field(default_factory=dict)
    best_weights: str | None = None
    has_weights: bool = False
    has_results: bool = False
    dir: str
    trained_at: str | None = None
    warnings: list[str] = Field(default_factory=list)


class TrainingCurvePoint(_Model):
    epoch: int
    metrics: dict[str, float] = Field(default_factory=dict)


class RunDetail(RunSummary):
    environment: dict[str, Any] = Field(default_factory=dict)
    recipe: dict[str, Any] = Field(default_factory=dict)
    profile_override: dict[str, Any] = Field(default_factory=dict)
    results_csv: str | None = None
    curve: list[TrainingCurvePoint] = Field(default_factory=list)
    best_epoch: int | None = None


class EvalResultModel(_Model):
    run: str
    scope: str
    metrics: dict[str, float] = Field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict)
    per_class_and_scale: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    images: int = 0
    boxes_gt: int = 0
    notes: list[str] = Field(default_factory=list)


class MatrixCell(_Model):
    run_name: str
    model: str
    combo: str
    epochs_completed: int = 0
    map50: float | None = None
    map50_95: float | None = None
    precision: float | None = None
    recall: float | None = None
    has_weights: bool = False


class MatrixResponse(_Model):
    cells: list[MatrixCell]
    models: list[str]
    combos: list[str]
    note: str = ""


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #


class DetectionModel(_Model):
    box: list[float] = Field(description="[x1, y1, x2, y2] in source pixels")
    confidence: float
    class_id: int
    class_name: str = ""
    track_id: int | None = None


class TrackModel(_Model):
    id: int
    state: str
    box: list[float]
    hits: int
    confidence: float


class PredictRequest(_Model):
    run: str = Field(description="Run name, run dir, or a .pt path.")
    image_path: str | None = None
    tracker: str | None = Field(default=None, description="sort | bytetrack | botsort")
    conf: float | None = Field(default=None, ge=0.0, le=1.0)
    iou: float | None = Field(default=None, ge=0.0, le=1.0)
    imgsz: int | None = Field(default=None, ge=64, le=4096)
    device: str | None = None
    max_tracks: int = Field(default=50, ge=1, le=500)
    return_overlay: bool = True


class PredictResponse(_Model):
    detections: list[DetectionModel]
    tracks: list[TrackModel]
    width: int
    height: int
    elapsed_ms: float
    source: str = ""
    overlay_png_b64: str | None = None
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# rules / coverage
# --------------------------------------------------------------------------- #


class RulesResponse(_Model):
    path: str
    rules: dict[str, Any]
    problems: list[str] = Field(default_factory=list)


class RuleExplainRequest(_Model):
    box: list[float]
    confidence: float
    hits: int = 0
    misses: int = 0
    duration_s: float = 0.0
    camera_id: str = ""
    image_height_px: int = 0
    global_id: int | None = None
    ground_xy: list[float] | None = None
    ground_z: float | None = None
    velocity_m_s: list[float] | None = None
    class_id: int = 0
    sightings: dict[str, Any] | None = None


class RuleGateResult(_Model):
    name: str
    outcome: str
    detail: str = ""
    measured: dict[str, float] = Field(default_factory=dict)


class RuleExplainResponse(_Model):
    alerted: bool
    severity: str
    first_failure: str | None = None
    gates: list[RuleGateResult]
    reasons: list[str] = Field(default_factory=list)


class CoverageSummary(_Model):
    frame: str
    cameras: int
    fixed: int
    ptz: int
    nodes: int
    cameras_per_node: dict[str, int] = Field(default_factory=dict)
    overlap_zones: int = 0
    protected_zones: int = 0
    uncovered_zones: list[str] = Field(default_factory=list)
    all_zones_covered: bool = False
    fixed_covered_zones: list[str] = Field(default_factory=list)
    ptz_only_zones: list[str] = Field(default_factory=list)


class CameraInfo(_Model):
    id: str
    node: str
    role: str
    model: str = ""
    position_m: list[float]
    yaw_deg: float
    pitch_deg: float
    zoom: float = 1.0
    hfov_deg: float = 0.0
    vfov_deg: float = 0.0
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)


class CoverageResponse(_Model):
    summary: CoverageSummary
    cameras: list[CameraInfo]
    protected_zones: list[dict[str, Any]] = Field(default_factory=list)
    geofences: list[dict[str, Any]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# tracking
# --------------------------------------------------------------------------- #


class TrackerScore(_Model):
    tracker: str
    mota: float
    idf1: float
    hota: float
    precision: float
    recall: float
    id_switches: int
    fragmentations: int
    sequences: int


class TrackerReportResponse(_Model):
    dataset: str
    variant: str
    scores: list[TrackerScore]
    note: str = ""


DatasetSummary.model_rebuild()
