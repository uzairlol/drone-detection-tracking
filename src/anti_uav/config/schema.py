"""Typed configuration schemas.

Every YAML file under ``configs/`` is validated through one of these models at
load time. The alternative - passing raw dicts into the trainer - means a typo
in ``configs/train/yolo11n.yaml`` surfaces 40 minutes into a training run instead
of at load time, so validation is deliberately strict:

* unknown keys are rejected (``extra="forbid"``);
* units are explicit in the field name (``_px``, ``_s``, ``_deg``, ``_m_s``);
* ranges that would break training (non-positive batch, NaN learning rate) are
  rejected rather than clamped.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# enums
# --------------------------------------------------------------------------- #


class Modality(StrEnum):
    RGB = "rgb"
    IR = "ir"
    EVENT = "event"


class SourceKind(StrEnum):
    """How a download backend should fetch the artefact."""

    GDRIVE = "gdrive"
    KAGGLE = "kaggle"
    BITBUCKET = "bitbucket"
    MODELSCOPE = "modelscope"
    BAIDU = "baidu"
    HF = "hf"
    HTTP = "http"
    MANUAL = "manual"


class ModelFamily(StrEnum):
    YOLO11N = "yolo11n"
    YOLO11S = "yolo11s"
    RTDETR_X2 = "rtdetr_x2"
    RTDETR_L = "rtdetr_l"


class GpuProfile(StrEnum):
    """CUDA/PyTorch build compatibility tiers.

    See docs/ENVIRONMENT.md. ``PASCAL`` exists because CUDA 12.8 and 13.x removed
    sm_61 kernels, so a GTX 1070 needs the last cu126-era wheel line.

    ``TURING`` (sm_75) and ``VOLTA`` (sm_70) are separate tiers because they are
    the oldest cards with *working* fp16 tensor cores. Folding them into PASCAL
    - as this enum used to - silently forces AMP off on a Kaggle T4, halving
    throughput for a card whose fp16 path is its main selling point.
    """

    #: Defer to :func:`anti_uav.config.loader.detect_profile`, which reads the
    #: live torch build and the device's compute capability. The default in
    #: ``configs/app.yaml``, and the right answer nearly always.
    AUTO = "auto"
    PASCAL = "pascal"  # sm_61  - GTX 1070, needs cu126 or older
    TURING = "turing"  # sm_75  - T4, Kaggle/Colab. fp16 tensor cores ARE present
    VOLTA = "volta"  # sm_70  - V100. fp16 tensor cores, no bf16
    AMPERE = "ampere"  # sm_80/86 - RTX 3090
    ADA = "ada"  # sm_89 - RTX 4090
    BLACKWELL = "blackwell"  # sm_120 - RTX 5090, needs cu128+
    CPU = "cpu"


class TrackerName(StrEnum):
    SORT = "sort"
    BYTETRACK = "bytetrack"
    BOTSORT = "botsort"
    NVDCF = "nvdcf"  # DeepStream-only; config artifact, not runnable on Windows


class SplitStrategy(StrEnum):
    """How train/val assignment is decided. See data/splits.py for why the
    default is sequence-level."""

    SEQUENCE = "sequence"
    #: Honour the publisher's own split via a regex over the sequence id. The
    #: only way a validation number is comparable to their published one.
    PATTERN = "pattern"
    GROUP = "group"
    SOURCE = "source"  # whole datasets held out (cross-dataset eval)
    FILE = "file"  # frame-level; only for genuinely static image datasets


# --------------------------------------------------------------------------- #
# dataset specs
# --------------------------------------------------------------------------- #


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=False)


class DownloadSource(_Strict):
    """One acquisition route for a dataset artefact."""

    kind: SourceKind
    url: str = ""
    #: Archive password for encrypted sources (Anti-UAV300, MM-UAV).
    password: str | None = None
    #: Human label shown in ``--dry-run`` output.
    label: str = ""
    #: Approximate download size, for the disk-budget warning.
    approx_size_gb: float = 0.0
    #: SHA-256 of the primary archive, when the publisher provides one.
    sha256: str | None = None
    #: Members to pull from a multi-part archive, e.g. ``["rgb_frame", "gt_rgb"]``.
    include_members: list[str] = Field(default_factory=list)
    #: Directories to skip inside an extracted tree.
    exclude_globs: list[str] = Field(default_factory=list)
    #: Requires the operator to solve a captcha / paste a cookie first.
    needs_credentials: bool = False
    notes: str = ""


class ClassMapping(_Strict):
    """Maps a source dataset's own label names onto the unified label space.

    The unified space is fixed by the system block diagram: ``drone``, ``bird``.
    Source datasets disagree wildly - MAV-VID uses ``drone``/``bird``, MM-UAV uses
    ``UAV``, Anti-UAV uses ``target`` - and silently dropping unknown labels
    would quietly turn a bird dataset into a drone-only one. Unmapped names are
    an error, not a warning.
    """

    #: ``{source_label: unified_label}``. Values must be in :attr:`unified_labels`.
    mapping: dict[str, str]
    #: The unified vocabulary, in training index order.
    unified_labels: list[str] = Field(default_factory=lambda: ["drone", "bird"])
    #: Source labels that are valid targets but intentionally dropped.
    ignore_labels: list[str] = Field(default_factory=list)

    @field_validator("unified_labels")
    @classmethod
    def _labels_must_be_nonempty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("unified_labels cannot be empty")
        return value

    @model_validator(mode="after")
    def _mapping_targets_must_exist(self) -> ClassMapping:
        unknown = sorted({v for v in self.mapping.values() if v not in self.unified_labels})
        if unknown:
            raise ValueError(
                f"class mapping targets not in unified_labels {self.unified_labels}: {unknown}"
            )
        overlap = set(self.mapping) & set(self.ignore_labels)
        if overlap:
            raise ValueError(f"labels both mapped and ignored: {sorted(overlap)}")
        return self

    def to_index(self, source_label: str) -> int | None:
        """Unified class index for a source label, or ``None`` to ignore."""
        if source_label in self.ignore_labels:
            return None
        unified = self.mapping.get(source_label)
        if unified is None:
            return None
        return self.unified_labels.index(unified)

    @property
    def provides_bird_negatives(self) -> bool:
        """Whether this source can teach the model to reject birds.

        Anti-UAV and MM-UAV are drone-only, so a model trained on them alone has
        no signal for the false positives the whole system exists to suppress.
        Surfaced by ``anti-uav stats`` on every run.
        """
        return any(v == "bird" for v in self.mapping.values())


class IngestSpec(_Strict):
    """Video -> frame extraction parameters."""

    #: Temporal subsampling. 1 keeps every frame; 2-4 is typical.
    stride: int = Field(default=1, ge=1)
    max_frames_per_sequence: int | None = Field(default=None, ge=1)
    start_frame: int = Field(default=0, ge=0)
    image_format: Literal["jpg", "png"] = "jpg"
    jpeg_quality: int = Field(default=92, ge=50, le=100)
    #: Long edge after resize. ``None`` keeps native resolution.
    max_dimension: int | None = Field(default=None, ge=64)
    #: Drop frames whose focus score (Laplacian variance) is below this.
    min_focus_score: float | None = Field(default=None, ge=0.0)
    #: Delete source videos once frames are extracted. Saves a lot of space.
    delete_source_video: bool = False


class TilingSpec(_Strict):
    """Sliding-window tiling for tiny targets.

    Tiling works because the model input size is fixed: cropping a 256 px window
    out of MM-UAV's 640x360 RGB frames and upscaling it to ``imgsz``=640 gives a
    ~2.5x magnification, turning a 12x5 px drone into ~30x12 px. Without it the
    drone lands at 12x5 px inside a 640x640 letterbox and is not detectable.

    Enabled automatically for any dataset whose ``median_target_px`` falls below
    ``matrix.tiling_threshold_px``, so cost is only paid where it buys recall.
    """

    enabled: bool = False
    tile_size: int = Field(default=640, ge=128, le=4096)
    overlap: float = Field(default=0.25, ge=0.0, lt=1.0)
    #: Boxes whose visible area in a tile falls below this are discarded.
    min_visibility: float = Field(default=0.35, ge=0.0, le=1.0)
    #: Invert window order so adjacent frames are not all in the same tile
    #: column, which would let tile-level split leakage reappear.
    random_offset: bool = True


class SequenceGroupSpec(_Strict):
    """How to group frames into the unit that must not be split."""

    #: Regex with a capture group; frames matching the same group stay together.
    pattern: str
    #: Which capture group forms the group key.
    group: int = 1
    #: Optional extra regex applied first, e.g. to isolate the RGB vs IR member
    #: of one physical sequence.
    split_pattern: str | None = None


class DatasetSpec(_Strict):
    """Everything the pipeline needs to know about one source dataset."""

    #: Short key used on the CLI: ``--dataset dvb``.
    alias: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str
    #: Publication / landing page.
    homepage: str = ""
    paper: str = ""
    license: str = "unknown"
    citation: str = ""

    modalities: list[Modality] = Field(min_length=1)
    #: Subset identifier for datasets with several releases (Anti-UAV300/410/600).
    variants: list[str] = Field(default_factory=list)
    default_variant: str | None = None

    sources: list[DownloadSource] = Field(min_length=1)
    ingest: IngestSpec = Field(default_factory=IngestSpec)
    classes: ClassMapping

    #: Per-dataset tiling override. ``None`` means "derive from
    #: median_target_px vs matrix.tiling_threshold_px", which is right for
    #: everything except a dataset whose frames are already small: MM-UAV's RGB
    #: frames are 640x360, so a 640 px tile gives zero magnification and the tile
    #: size has to drop to 256 to actually enlarge the target.
    tiling_override: TilingSpec | None = None

    #: Frame grouping regex - the backbone of leakage-safe splitting.
    group: SequenceGroupSpec
    #: Frames per sequence on average, used for the progress estimate.
    approx_sequences: int = Field(default=0, ge=0)
    approx_frames: int = Field(default=0, ge=0)
    #: Median target edge length in px, from the dataset paper. Drives tiling.
    median_target_px: float | None = Field(default=None, gt=0)
    #: Contains real bird annotations.
    has_bird_negatives: bool = False
    #: MOT-style track identity available.
    has_track_ids: bool = False
    #: More than one identity per sequence, so identity metrics (IDF1, ID
    #: switches) are actually measurable. Distinct from `has_track_ids`: a
    #: single-target dataset can still label every box id=1, which satisfies a
    #: naive "has identity" check while making IDF1 meaningless — every identity
    #: trivially matches every other. Anti-UAV is the case in point: one target
    #: per sequence, identity labels present, and its own benchmark metric is
    #: IoU times visibility rather than anything identity-based.
    has_multi_object_identity: bool = False
    #: Per-frame visibility flag (target absent / occluded).
    has_visibility_flags: bool = False
    #: Sequence-level metric the dataset's benchmark uses (Anti-UAV).
    official_metric: str | None = None

    notes: str = ""
    caveats: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _variant_must_exist(self) -> DatasetSpec:
        if self.variants and self.default_variant and self.default_variant not in self.variants:
            raise ValueError(
                f"default_variant {self.default_variant!r} not in variants {self.variants}"
            )
        return self

    @model_validator(mode="after")
    def _bird_flag_matches_mapping(self) -> DatasetSpec:
        """``has_bird_negatives`` must agree with the class mapping.

        Keeping them in sync matters: the flag drives the "trained without bird
        negatives" warning on every eval table, so a mismatch would either hide a
        real limitation or cry wolf.
        """
        if self.has_bird_negatives and not self.classes.provides_bird_negatives:
            raise ValueError(
                f"{self.alias}: has_bird_negatives=true but the class mapping "
                f"{self.classes.mapping} never maps anything to 'bird'"
            )
        return self


class DatasetRegistry(_Strict):
    """The parsed ``configs/datasets/registry.yaml``."""

    schema_version: int = 1
    #: Unified label space every dataset is harmonised into.
    unified_labels: list[str] = Field(default_factory=lambda: ["drone", "bird"])
    datasets: dict[str, DatasetSpec]

    @field_validator("datasets")
    @classmethod
    def _keys_match_aliases(cls, value: dict[str, DatasetSpec]) -> dict[str, DatasetSpec]:
        for key, spec in value.items():
            if spec.alias != key:
                raise ValueError(f"registry key {key!r} != spec alias {spec.alias!r}")
        return value

    def get(self, alias: str) -> DatasetSpec:
        try:
            return self.datasets[alias]
        except KeyError as exc:
            raise KeyError(
                f"unknown dataset {alias!r}; known: {sorted(self.datasets)}"
            ) from exc

    @property
    def aliases(self) -> list[str]:
        return sorted(self.datasets)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


class OptimizerSpec(_Strict):
    name: Literal["auto", "SGD", "AdamW", "Adam", "RMSProp"] = "auto"
    lr0: float = Field(default=0.01, gt=0.0, le=1.0)
    lrf: float = Field(default=0.01, gt=0.0, le=1.0)
    momentum: float = Field(default=0.937, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0005, ge=0.0, le=1.0)
    warmup_epochs: float = Field(default=3.0, ge=0.0)
    warmup_momentum: float = Field(default=0.8, ge=0.0, lt=1.0)
    warmup_bias_lr: float = Field(default=0.1, gt=0.0, le=1.0)


class AugmentationSpec(_Strict):
    """Mosaic-style augmentation, with a per-family switch.

    RT-DETR's hybrid encoder was not designed around mosaics; ultralytics'
    own RT-DETR recipe keeps mosaic off for fine-tuning. Both defaults below
    match that guidance.
    """

    hsv_h: float = Field(default=0.015, ge=0.0, le=1.0)
    hsv_s: float = Field(default=0.7, ge=0.0, le=1.0)
    hsv_v: float = Field(default=0.4, ge=0.0, le=1.0)
    degrees: float = Field(default=0.0, ge=0.0, le=180.0)
    translate: float = Field(default=0.1, ge=0.0, le=1.0)
    scale: float = Field(default=0.5, ge=0.0)
    shear: float = Field(default=0.0, ge=-45.0, le=45.0)
    perspective: float = Field(default=0.0, ge=0.0, le=1.0)
    flipud: float = Field(default=0.0, ge=0.0, le=1.0)
    fliplr: float = Field(default=0.5, ge=0.0, le=1.0)
    mosaic: float = Field(default=1.0, ge=0.0, le=1.0)
    mixup: float = Field(default=0.0, ge=0.0, le=1.0)
    cutout: int = Field(default=0, ge=0, le=50)
    copy_paste: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Close mosaic for this many final epochs to settle on single-scale inputs.
    close_mosaic: int = Field(default=10, ge=0)
    #: HSV jitter strength specifically for the IR half of a mixed dataset.
    ir_grayscale_probability: float = Field(default=0.0, ge=0.0, le=1.0)


class LossSpec(_Strict):
    box: float = Field(default=7.5, gt=0.0)
    cls: float = Field(default=0.5, ge=0.0)
    dfl: float = Field(default=1.5, ge=0.0)


class TrainRecipe(_Strict):
    """A full training configuration for one (model family) pair.

    Model-agnostic where possible; ``family_overrides`` carries the deltas that
    genuinely differ between a CNN detector and RT-DETR so the two share one
    file shape and one code path.
    """

    name: str
    family: ModelFamily
    #: Pretrained checkpoint name or path. Downloaded by ultralytics if a bare name.
    base_weights: str
    epochs: int = Field(default=100, ge=1, le=5000)
    imgsz: int = Field(default=640, ge=64, le=4096)
    #: ``-1`` = auto-detect (60% of VRAM), ``0.7`` = 70% fraction.
    batch: float = Field(default=-1.0, ge=-1.0)
    batch_note: str = ""
    optimizer: OptimizerSpec = Field(default_factory=OptimizerSpec)
    augmentation: AugmentationSpec = Field(default_factory=AugmentationSpec)
    loss: LossSpec = Field(default_factory=LossSpec)
    tiling: TilingSpec = Field(default_factory=TilingSpec)

    patience: int = Field(default=50, ge=0)
    #: cos-linear is what RT-DETR was trained with.
    scheduler: Literal["cos-linear", "linear", "cosine", "one-cycle"] = "cos-linear"
    #: Half precision. Pascal has no fp16 tensor-core path, so this is forced
    #: off for the GTX 1070 profile regardless of what the recipe says.
    amp: bool = True
    seed: int = 0
    workers: int = Field(default=4, ge=0, le=32)
    #: Freeze the backbone for this many epochs (optional warm start).
    freeze: int = Field(default=0, ge=0)
    #: The metric this run is judged on, and the headline written to
    #: ``run_metadata.json``. Plain mAP50 is generous at a 12-28 px target, so
    #: mAP50-95 is the default. Note this does NOT control early stopping:
    #: ultralytics always early-stops on its internal ``fitness`` blend and
    #: exposes no hook to change that, which is what ``patience`` bounds.
    metric: str = "map50-95"
    save_period: int = Field(default=-1, ge=-1)

    extra: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class ProfileOverride(_Strict):
    """Per-GPU deltas applied on top of a recipe.

    ``configs/train/overrides/pascal.yaml`` is the important one: it forces AMP
    off and drops batch size, because fp16 master weights on sm_61 are both slow
    and numerically fragile. sm_61 is the *only* tier in this project without fp16
    tensor cores, so this is a Pascal-specific correction rather than a general
    one - do not copy it into the turing or volta overrides.
    """

    profile: GpuProfile
    #: Minimum torch CUDA version that still ships kernels for this profile.
    min_torch_cuda: str = ""
    #: Compute capability this profile targets, for the verify step.
    target_capability: str = ""
    batch_scale: float = Field(default=1.0, gt=0.0, le=1.0)
    imgsz_scale: float = Field(default=1.0, gt=0.0, le=1.0)
    force_amp: bool | None = None
    #: Absolute batch for every model on this GPU tier.
    force_batch: int | None = Field(default=None, ge=1)
    #: Per-model absolute batch, overriding :attr:`force_batch`.
    #:
    #: One number per profile is not enough, because the two families in the
    #: matrix do not have the same memory profile by anything like the same
    #: factor. RT-DETR-x2's hybrid encoder keeps far more activations live than
    #: YOLO11n's, and memory scales with the denoising query count on top of
    #: that - so the batch that fits the CNN OOMs the transformer on the same
    #: card. Without this, every operator has to rediscover the right number per
    #: model and `matrix run` plans a run that dies at allocation.
    force_batch_by_model: dict[str, int] = Field(default_factory=dict)
    force_imgsz: int | None = Field(default=None, ge=64)
    workers: int | None = Field(default=None, ge=0, le=32)
    notes: str = ""


class ComboSpec(_Strict):
    """One dataset combination in the experiment matrix."""

    #: Slug used in run names, e.g. ``dvb+mavvid+antiuav``.
    slug: str
    datasets: list[str] = Field(min_length=1)
    description: str = ""
    #: Per-combo tiling decision. Defaults come from the dataset specs.
    force_tiling: bool | None = None
    force_imgsz: int | None = Field(default=None, ge=64)
    #: Skip this combo - e.g. MM-UAV when its Baidu download failed.
    enabled: bool = True
    notes: str = ""


class ExperimentMatrix(_Strict):
    """``configs/matrix.yaml``: every run we intend to produce."""

    schema_version: int = 1
    models: list[ModelFamily]
    combos: list[ComboSpec]
    #: A dataset is tiled when its ``median_target_px`` is at or below this.
    #: 40 px is where a target stops being reliably detectable after letterboxing
    #: to ``default_imgsz``.
    tiling_threshold_px: float = Field(default=40.0, gt=0.0)
    #: Input size for untiled sources. Tiled sources produce ``tile_size`` crops
    #: which are then upscaled to this.
    default_imgsz: int = Field(default=640, ge=64, le=4096)
    #: Cross-dataset evaluation: train on each combo, test on each dataset.
    cross_dataset_eval: bool = True
    #: Datasets used as the cross-eval test sets.
    cross_eval_datasets: list[str] = Field(default_factory=list)
    seed: int = 0
    notes: str = ""

    @field_validator("combos")
    @classmethod
    def _slugs_unique(cls, value: list[ComboSpec]) -> list[ComboSpec]:
        slugs = [c.slug for c in value]
        if len(slugs) != len(set(slugs)):
            dupes = sorted({s for s in slugs if slugs.count(s) > 1})
            raise ValueError(f"duplicate combo slugs: {dupes}")
        return value

    @property
    def n_runs(self) -> int:
        return len(self.models) * sum(1 for c in self.combos if c.enabled)

    def enabled_combos(self) -> list[ComboSpec]:
        return [c for c in self.combos if c.enabled]


class RunPlan(_Strict):
    """A single resolved training run - the unit the matrix runner executes."""

    run_name: str
    model: ModelFamily
    combo: str
    datasets: list[str]
    profile: GpuProfile
    imgsz: int
    batch: int
    epochs: int
    amp: bool
    tiling: bool
    tile_size: int
    output_dir: str
    data_yaml: str
    overrides: list[str] = Field(default_factory=list)
    estimate_note: str = ""


# --------------------------------------------------------------------------- #
# rule layer  (system block diagram section 4)
# --------------------------------------------------------------------------- #


class ConfidenceRule(_Strict):
    initiate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.60
    maintain: Annotated[float, Field(ge=0.0, le=1.0)] = 0.35

    @model_validator(mode="after")
    def _maintain_below_initiate(self) -> ConfidenceRule:
        if self.maintain >= self.initiate:
            raise ValueError(
                f"maintain ({self.maintain}) must be below initiate ({self.initiate})"
            )
        return self


class PersistenceRule(_Strict):
    min_hits: int = Field(default=12, ge=1)
    min_duration_s: float = Field(default=1.0, ge=0.0)
    max_gap_frames: int = Field(default=5, ge=0)
    #: Minimum track age before a drone alert may fire.
    alert_min_hits: int = Field(default=12, ge=1)


class KinematicsRule(_Strict):
    min_speed_m_s: float = Field(default=1.5, ge=0.0)
    max_speed_m_s: float = Field(default=35.0, gt=0.0)
    max_turn_rate_deg_s: float = Field(default=90.0, gt=0.0)
    #: Hover = vertical displacement below this for hover_duration_s.
    hover_threshold_m: float = Field(default=0.5, ge=0.0)
    hover_duration_s: float = Field(default=2.0, gt=0.0)

    @model_validator(mode="after")
    def _speed_range_ordered(self) -> KinematicsRule:
        if self.min_speed_m_s >= self.max_speed_m_s:
            raise ValueError(
                f"min_speed_m_s ({self.min_speed_m_s}) >= max_speed_m_s ({self.max_speed_m_s})"
            )
        return self


class SpatialRule(_Strict):
    require_above_horizon: bool = True
    #: Per-camera horizon line in normalised image coords; empty = use metadata.
    horizon_y: float | None = Field(default=None, ge=0.0, le=1.0)
    geofence_breach_suppresses: bool = True
    #: Ground-plane height for the speed estimate, metres.
    ground_plane_z_m: float | None = Field(default=None, ge=0.0)


class CrossCameraRule(_Strict):
    min_cameras: int = Field(default=2, ge=1)
    max_window_s: float = Field(default=3.0, gt=0.0)
    #: Require the agreeing views to be from different worker nodes, not just
    #: different cameras on the same node.
    require_distinct_nodes: bool = False
    #: Max disagreement between two views of the same target, metres.
    max_position_disagreement_m: float = Field(default=25.0, ge=0.0)


class TrackingRule(_Strict):
    """Mirrors the DeepStream ``nvtracker`` + global-track settings."""

    local_tracker: TrackerName = TrackerName.BOTSORT
    max_target_age_frames: int = Field(default=4, ge=1)
    min_trackable_height_px: float = Field(default=15.0, gt=0.0)
    handoff_trigger_lead_s: float = Field(default=2.0, ge=0.0)
    handoff_verify_frames: int = Field(default=3, ge=1)
    handoff_verify_window_s: float = Field(default=0.5, gt=0.0)
    recovery_budget_s: float = Field(default=1.5, gt=0.0)
    reid_similarity_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    #: Association gate combining motion and appearance cost.
    max_association_cost: float = Field(default=0.75, ge=0.0, le=1.0)
    #: Process noise scale for the constant-velocity Kalman filter.
    process_noise_std_m_s: float = Field(default=0.35, gt=0.0)
    measurement_noise_std_m: float = Field(default=1.0, gt=0.0)
    #: Target is declared lost when the 3-sigma ellipse grows past this radius.
    max_uncertainty_radius_m: float = Field(default=18.0, gt=0.0)


class DroneRules(_Strict):
    """``configs/rules/drone_rules.yaml`` - one file, centrally tunable."""

    schema_version: int = 1
    confidence: ConfidenceRule = Field(default_factory=ConfidenceRule)
    persistence: PersistenceRule = Field(default_factory=PersistenceRule)
    kinematics: KinematicsRule = Field(default_factory=KinematicsRule)
    spatial: SpatialRule = Field(default_factory=SpatialRule)
    cross_camera: CrossCameraRule = Field(default_factory=CrossCameraRule)
    tracking: TrackingRule = Field(default_factory=TrackingRule)
    #: Severities for the alerting stage.
    severity_order: list[str] = Field(default_factory=lambda: ["info", "warn", "critical"])
    notes: str = ""


# --------------------------------------------------------------------------- #
# coverage map  (system block diagram section 3)
# --------------------------------------------------------------------------- #


class CameraIntrinsics(_Strict):
    fx_px: float = Field(gt=0.0)
    fy_px: float = Field(gt=0.0)
    cx_px: float | None = None
    cy_px: float | None = None
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    #: Brown-Conrady ``k1, k2, p1, p2, k3``.
    distortion: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    #: Derive fx from FOV when the calibrator has not run yet.
    hfov_deg: float | None = Field(default=None, gt=0.0, lt=180.0)


class CameraMount(_Strict):
    #: Site coordinates in the shared geospatial frame: east, north, up, metres.
    position_m: tuple[float, float, float]
    #: Mount orientation. 0 = north, increasing clockwise.
    yaw_deg: float = 0.0
    pitch_deg: float = -5.0
    roll_deg: float = 0.0


class PTZCapabilities(_Strict):
    enabled: bool = False
    pan_min_deg: float = Field(default=-170.0)
    pan_max_deg: float = Field(default=170.0)
    tilt_min_deg: float = Field(default=-90.0)
    tilt_max_deg: float = Field(default=90.0)
    zoom_min: float = Field(default=1.0, gt=0.0)
    zoom_max: float = Field(default=30.0, gt=1.0)
    max_pan_speed_deg_s: float = Field(default=30.0, gt=0.0)
    max_tilt_speed_deg_s: float = Field(default=20.0, gt=0.0)
    #: Time for the PTZ to physically settle after a move command.
    settle_time_s: float = Field(default=1.2, ge=0.0)
    #: Observed overshoot as a fraction of commanded delta.
    overshoot_ratio: float = Field(default=0.08, ge=0.0, le=1.0)


class CameraSpec(_Strict):
    """One entry of the 100-camera fleet."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    node: str = Field(default="node-00", description="Worker node that owns this camera")
    role: Literal["fixed", "ptz"] = "fixed"
    model: str = ""
    mount: CameraMount
    intrinsics: CameraIntrinsics
    ptz: PTZCapabilities = Field(default_factory=PTZCapabilities)
    #: Normalised horizon line (0 = top, 1 = bottom) for the spatial rule.
    horizon_y: float | None = Field(default=None, ge=0.0, le=1.0)
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)


class ProtectedZone(_Strict):
    """A region that must always keep at least one camera watching."""

    id: str
    #: Polygon in the geospatial frame as flat ``[east, north, ...]`` pairs.
    polygon: list[float] = Field(min_length=6)
    priority: int = Field(default=1, ge=1, le=10)
    #: Zone centre height used for FOV intersection tests.
    height_m: float = Field(default=10.0, ge=0.0)
    description: str = ""


class Geofence(_Strict):
    """Region in which a drone alert is escalated / suppressed."""

    id: str
    polygon: list[float] = Field(min_length=6)
    action: Literal["alert", "escalate", "suppress"] = "alert"
    height_m: float = Field(default=0.0, ge=0.0)
    description: str = ""


class CoverageMap(_Strict):
    """``configs/coverage/coverage_map.yaml``."""

    schema_version: int = 1
    #: Name of the shared geospatial frame, e.g. ``site-local-ENU-metres``.
    frame: str = "site-local-ENU"
    #: Vertical drop used when rasterising a fixed camera's FOV to a polygon.
    default_ground_plane_z_m: float = Field(default=0.0, ge=-1000.0)
    #: Horizontal extent covered by a boundary camera, used for the coverage
    #: heatmap when no explicit polygon is supplied.
    nominal_range_m: float = Field(default=300.0, gt=0.0)
    cameras: list[CameraSpec]
    protected_zones: list[ProtectedZone] = Field(default_factory=list)
    geofences: list[Geofence] = Field(default_factory=list)
    #: Cached overlap polygons between cameras; empty means compute on demand.
    overlap_zones: list[dict[str, Any]] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _camera_ids_unique(self) -> CoverageMap:
        ids = [c.id for c in self.cameras]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate camera ids: {dupes}")
        return self

    @field_validator("cameras")
    @classmethod
    def _ptz_role_needs_capabilities(
        cls, value: list[CameraSpec]
    ) -> list[CameraSpec]:
        bad = [c.id for c in value if c.role == "ptz" and not c.ptz.enabled]
        if bad:
            raise ValueError(f"cameras with role='ptz' must set ptz.enabled: {bad}")
        return value

    def camera(self, camera_id: str) -> CameraSpec:
        for cam in self.cameras:
            if cam.id == camera_id:
                return cam
        raise KeyError(f"unknown camera {camera_id!r}")


# --------------------------------------------------------------------------- #
# application settings
# --------------------------------------------------------------------------- #


class AppSettings(_Strict):
    """Runtime settings, resolved from ``configs/app.yaml`` + env overrides."""

    profile: GpuProfile = GpuProfile.CPU
    #: Ultralytics device string. ``auto`` means "let ultralytics pick", which
    #: it spells as the empty string - passing the literal ``"auto"`` through
    #: makes ``select_device`` raise on every train and predict call. The
    #: validator below translates it so config authors can write what they mean.
    device: str = "auto"
    seed: int = 0
    imgsz: int = Field(default=640, ge=64, le=4096)
    conf_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25
    iou_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.70
    max_detections: int = Field(default=100, ge=1, le=1000)
    #: Prediction letterbox size for the ONNX/TensorRT engines.
    half: bool = False
    log_level: str = "INFO"
    log_json: bool = False
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    #: Registry database for run/metric bookkeeping.
    registry_db: str = "artifacts/registry.sqlite3"
    #: Inference settings used by the tracking replayer.
    default_tracker: TrackerName = TrackerName.BOTSORT
    tensorrt_workspace_gb: float = Field(default=4.0, gt=0.0)
    notes: str = ""

    @field_validator("device", mode="before")
    @classmethod
    def _translate_auto_device(cls, value: Any) -> Any:
        """Map ``auto`` onto ultralytics' spelling of auto-select.

        ``select_device`` raises ``Invalid CUDA 'device=auto'``; it auto-selects
        only on an empty string. Normalising here means no caller - trainer,
        predictor, exporter - has to remember the distinction.
        """
        if isinstance(value, str) and value.strip().lower() in {"auto", "default"}:
            return ""
        return value

    @model_validator(mode="after")
    def _cpu_forces_half_off(self) -> AppSettings:
        if self.profile is GpuProfile.CPU and self.half:
            # Not fatal - TensorRT builds still need the flag - but the UI warns.
            object.__setattr__(self, "notes", (self.notes + " half=True on a CPU profile.").strip())
        return self
