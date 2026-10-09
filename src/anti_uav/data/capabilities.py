"""What each source dataset is allowed to prove.

Four public datasets do not support the same conclusions, and the differences are
not cosmetic:

===========  ==================  =============  ==================
source       falsify FP claims?  identity?      tiling
===========  ==================  =============  ==================
``dvb``      **yes**             no             640 px
``mavvid``   **yes**             no             none
``antiuav``  no                  no (1 target)  none
``mmuav``    no                  **yes**        256 px
===========  ==================  =============  ==================

Only ``dvb`` and ``mavvid`` contain birds, so only they can turn a precision
number into evidence about false positives — a detector that labelled every bird
as a drone would score perfectly on ``antiuav`` or ``mmuav``. Only ``mmuav`` has
multi-object identity, so only it makes IDF1 and ID-switch counts mean anything;
``antiuav`` labels every box id=1, which satisfies a naive "has track ids" check
while making identity metrics vacuous.

Those facts were already in ``configs/datasets/registry.yaml`` and were already
honoured in four separate places — a ``data.yaml`` provenance note, a
falsifiability string on each ``evaluate --cross`` row, a warning in
``track-eval``, and a ``median_target_px`` comparison in ``build``. That works but
it has the shape of an accident waiting to happen: adding a fifth dataset means
remembering four places, and forgetting one is a silent hole rather than a
failure.

This module is the single derivation, and every one of those surfaces reads from
it. The pipeline itself does not fork: one combo, one build, one trainer, and
weights per combo under ``artifacts/runs/<family>/<family>__<combo>/``. What is
explicit is *which claims a given combo's numbers can carry* — see
:class:`ComboCapabilities`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config.schema import DatasetSpec


#: Human-readable name for each claim, used as the dict key and in tables.
CLAIM_PRECISION = "precision (false positives)"
CLAIM_IDENTITY = "identity (IDF1, ID switches)"
CLAIM_VISIBILITY = "visibility-aware accuracy"
CLAIM_TRACKING = "detection-side tracking metrics (MOTA, HOTA)"


@dataclass(frozen=True, slots=True)
class SourceCapabilities:
    """One source dataset's evidential capabilities.

    Every field is derived from ``configs/datasets/registry.yaml``; none of it is
    computed from the data on disk, so this is answerable before anything is
    downloaded.
    """

    alias: str
    display_name: str
    #: Real bird annotations exist, so precision is falsifiable here.
    can_falsify_precision: bool
    #: More than one identity per sequence, so IDF1 / ID switches mean something.
    can_score_identity: bool
    #: MOT identity present at all. Weaker than `can_score_identity`: a
    #: single-target dataset labels every box id=1.
    has_track_ids: bool
    #: Per-frame visibility flags, so absence can be scored rather than guessed.
    can_score_visibility: bool
    #: This source is tiled at build time.
    requires_tiling: bool
    tile_size: int | None
    median_target_px: float | None
    #: The dataset's own benchmark metric, where it publishes one.
    official_metric: str | None
    #: The dataset declares its own ``tiling_override``, so its tiling decision is
    #: explicit rather than threshold-derived. A combo can still force tiling on
    #: or off, in which case :attr:`ComboCapabilities.tiling` is authoritative.
    tiling_is_forced: bool = False
    caveats: tuple[str, ...] = ()

    @property
    def can_score_tracking(self) -> bool:
        """MOTA / HOTA need identity-labelled ground truth to be meaningful at all."""
        return self.has_track_ids

    def claims(self) -> dict[str, bool]:
        """The claims this source can carry, keyed by :data:`CLAIM_*`."""
        return {
            CLAIM_PRECISION: self.can_falsify_precision,
            CLAIM_IDENTITY: self.can_score_identity,
            CLAIM_VISIBILITY: self.can_score_visibility,
            CLAIM_TRACKING: self.can_score_tracking,
        }


@dataclass(frozen=True, slots=True)
class ComboCapabilities:
    """What a combo's numbers are allowed to prove.

    A combo's capability is the **union** of its sources'. Training on
    ``all4`` can falsify a false-positive claim because ``dvb`` and ``mavvid`` are
    in it, even though ``mmuav`` and ``antiuav`` contribute nothing to that
    question — which is exactly why the per-dataset table still matters.
    """

    slug: str
    sources: tuple[SourceCapabilities, ...]
    #: Sources whose presence is what makes each claim available.
    supports: dict[str, tuple[str, ...]]
    #: Resolved per-source tiling for THIS combo, alias -> enabled.
    tiling: dict[str, bool] = field(default_factory=dict)

    # -- construction ----------------------------------------------------- #

    @classmethod
    def from_specs(
        cls,
        slug: str,
        specs: dict[str, DatasetSpec],
        *,
        aliases: tuple[str, ...],
        tiling_threshold_px: float,
        force_tiling: bool | None = None,
    ) -> ComboCapabilities:
        sources = tuple(
            source_capabilities(specs[alias], tiling_threshold_px=tiling_threshold_px)
            for alias in aliases
            if alias in specs
        )
        supports: dict[str, tuple[str, ...]] = {}
        for claim in (CLAIM_PRECISION, CLAIM_IDENTITY, CLAIM_VISIBILITY, CLAIM_TRACKING):
            supports[claim] = tuple(s.alias for s in sources if s.claims()[claim])

        # Mirror build.resolve_tiling's precedence exactly, so the report cannot
        # claim a source is untiled when build tiles it (or the reverse).
        from ..config.schema import TilingSpec

        tiling: dict[str, bool] = {}
        for alias in aliases:
            spec = specs.get(alias)
            if spec is None:
                continue
            if spec.tiling_override is not None:
                tiling[alias] = spec.tiling_override.enabled
            elif force_tiling is not None:
                tiling[alias] = bool(force_tiling)
            elif spec.median_target_px is not None:
                tiling[alias] = spec.median_target_px <= tiling_threshold_px
            else:
                tiling[alias] = TilingSpec().enabled

        return cls(slug=slug, sources=sources, supports=supports, tiling=tiling)

    # -- claims ----------------------------------------------------------- #

    def can_claim(self, claim: str) -> bool:
        return bool(self.supports.get(claim))

    @property
    def precision_sources(self) -> tuple[str, ...]:
        return self.supports.get(CLAIM_PRECISION, ())

    @property
    def identity_sources(self) -> tuple[str, ...]:
        return self.supports.get(CLAIM_IDENTITY, ())

    @property
    def tracking_sources(self) -> tuple[str, ...]:
        return self.supports.get(CLAIM_TRACKING, ())

    @property
    def tiled_sources(self) -> tuple[str, ...]:
        """Sources this combo tiles. Combo-aware: ``force_tiling`` wins over the
        dataset default, exactly as :func:`anti_uav.data.build.resolve_tiling` does."""
        return tuple(alias for alias, enabled in self.tiling.items() if enabled)

    @property
    def requires_tiling(self) -> bool:
        return bool(self.tiled_sources)

    @property
    def is_comparable_across_combos(self) -> bool:
        """False when the sources tile differently.

        A combo containing ``mmuav`` trains at 256 px tiles upscaled to 640 while
        the others train untiled at 640. The rows are not a like-for-like
        architecture comparison, and ``docs/EXPERIMENTS.md`` says so — this makes
        it queryable rather than something to remember.
        """
        return len({s.requires_tiling for s in self.sources}) <= 1

    def claims(self) -> dict[str, bool]:
        return {claim: bool(names) for claim, names in self.supports.items()}

    def sources_for(self, claim: str) -> tuple[str, ...]:
        return self.supports.get(claim, ())

    # -- reporting -------------------------------------------------------- #

    def warnings(self) -> list[str]:
        """Short headlines for claims this combo cannot support.

        Deliberately one clause each. The *why* is already in the per-source table and
        in :meth:`explain`; a warning that has to be wrapped is a warning that stops
        being read halfway down a terminal.
        """
        out: list[str] = []
        if not self.can_claim(CLAIM_PRECISION):
            others = self._bird_sources_available()
            out.append(
                "no bird negatives, so precision is not evidence about false positives"
                + (f" (use {', '.join(others)})" if others else "")
            )
        if not self.can_claim(CLAIM_IDENTITY):
            if self.can_claim(CLAIM_TRACKING):
                out.append(
                    "single-target ground truth, so IDF1 and ID switches are meaningless"
                )
            else:
                out.append("no track ids, so MOTA and HOTA cannot be computed")
        return out

    def _bird_sources_available(self) -> list[str]:
        """Which registered sources could supply the bird evidence this combo lacks."""
        try:
            from ..config.loader import load_registry

            registry = load_registry()
        except Exception:  # pragma: no cover - registry always loads in practice
            return ["dvb"]
        return [
            a
            for a in registry.aliases
            if registry.get(a).has_bird_negatives and a not in {s.alias for s in self.sources}
        ] or ["dvb"]

    def summary_row(self) -> str:
        """One line for a results table."""
        def mark(claim: str) -> str:
            names = self.supports.get(claim, ())
            return f"{','.join(names)}" if names else "-"

        return (
            f"{self.slug or '-':<20} "
            f"FP:{mark(CLAIM_PRECISION):<14} "
            f"ID:{mark(CLAIM_IDENTITY):<8} "
            f"track:{mark(CLAIM_TRACKING):<10} "
            f"tiled:{','.join(self.tiled_sources) or '-'}"
        )

    def explain(self) -> str:
        """Multi-line explanation, for ``--help``-style output and the docs."""
        lines = [f"capabilities of combo '{self.slug}':", ""]
        header = f"  {'source':<10} {'median px':>9} {'tiled':>6} {'FP ok':>6} {'ID ok':>6} {'track':>6}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for s in self.sources:
            lines.append(
                f"  {s.alias:<10} {str(int(s.median_target_px or 0)):>9} "
                f"{(str(s.tile_size) if s.requires_tiling else '-'):>6} "
                f"{('yes' if s.can_falsify_precision else 'no'):>6} "
                f"{('yes' if s.can_score_identity else 'no'):>6} "
                f"{('yes' if s.can_score_tracking else 'no'):>6}"
            )
        lines.append("")
        for claim in (CLAIM_PRECISION, CLAIM_IDENTITY, CLAIM_VISIBILITY, CLAIM_TRACKING):
            names = self.supports.get(claim, ())
            verdict = f"supported by {', '.join(names)}" if names else "NOT SUPPORTED"
            lines.append(f"  {claim:<44} {verdict}")
        for warning in self.warnings():
            lines.append("")
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# derivation
# --------------------------------------------------------------------------- #


def source_capabilities(
    spec: DatasetSpec,
    *,
    tiling_threshold_px: float,
) -> SourceCapabilities:
    """Derive one source's capabilities from its registry spec.

    Tiling is resolved the same way :func:`anti_uav.data.build.resolve_tiling`
    does, so the capability report cannot disagree with what build actually did.
    """
    from ..config.schema import TilingSpec

    override = spec.tiling_override
    if override is not None:
        requires_tiling = override.enabled
        tile_size: int | None = override.tile_size if override.enabled else None
    else:
        median = spec.median_target_px
        requires_tiling = median is not None and median <= tiling_threshold_px
        tile_size = TilingSpec().tile_size if requires_tiling else None

    return SourceCapabilities(
        alias=spec.alias,
        display_name=spec.display_name,
        can_falsify_precision=spec.has_bird_negatives,
        can_score_identity=spec.has_multi_object_identity,
        has_track_ids=spec.has_track_ids,
        can_score_visibility=spec.has_visibility_flags,
        requires_tiling=requires_tiling,
        tile_size=tile_size,
        median_target_px=spec.median_target_px,
        tiling_is_forced=override is not None,
        official_metric=spec.official_metric,
        caveats=tuple(spec.caveats),
    )


def _matrix_threshold() -> float:
    from ..config.loader import load_matrix

    return load_matrix().tiling_threshold_px


def source_capabilities_for(alias: str) -> SourceCapabilities:
    """Capabilities of one dataset, by alias."""
    from ..config.loader import load_registry

    registry = load_registry()
    spec = registry.get(alias)
    if spec is None:
        raise KeyError(f"unknown dataset {alias!r}; known: {', '.join(registry.aliases)}")
    return source_capabilities(spec, tiling_threshold_px=_matrix_threshold())


def all_source_capabilities() -> dict[str, SourceCapabilities]:
    """Capabilities of every registered dataset, keyed by alias."""
    from ..config.loader import load_registry

    registry = load_registry()
    threshold = _matrix_threshold()
    return {
        alias: source_capabilities(registry.get(alias), tiling_threshold_px=threshold)
        for alias in registry.aliases
    }


def combo_capabilities(
    slug: str,
    aliases: tuple[str, ...] | list[str],
    *,
    force_tiling: bool | None = None,
) -> ComboCapabilities:
    """Capabilities of a combo, from its source aliases.

    ``force_tiling`` is taken from ``configs/matrix.yaml`` when the slug names a
    combo there, because tiling is decided per combo x source, not per source
    alone. Pass it explicitly to model a combo that is not in the matrix.
    """
    from ..config.loader import load_matrix, load_registry

    registry = load_registry()
    force = force_tiling
    if force is None:
        for spec in load_matrix().combos:
            if spec.slug == slug:
                force = spec.force_tiling
                break
    return ComboCapabilities.from_specs(
        slug,
        registry.datasets,
        aliases=tuple(aliases),
        tiling_threshold_px=_matrix_threshold(),
        force_tiling=force,
    )


def capabilities_for_run(run_metadata: dict[str, Any]) -> ComboCapabilities | None:
    """Capabilities implied by a ``run_metadata.json`` payload.

    Used by the API and the UI so a stored run can say what its own numbers are
    allowed to prove, rather than leaving the reader to remember.
    """
    combo = run_metadata.get("combo")
    if not isinstance(combo, str) or not combo:
        return None
    from ..config.loader import load_matrix

    for spec in load_matrix().combos:
        if spec.slug == combo:
            return combo_capabilities(combo, tuple(spec.datasets))
    return None