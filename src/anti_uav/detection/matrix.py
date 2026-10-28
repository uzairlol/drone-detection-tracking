"""Experiment matrix runner.

Turns ``configs/matrix.yaml`` into concrete, executable runs. The important
property is that ``--dry-run`` prints the *same* plan that a real invocation
would execute, because both go through :func:`anti_uav.detection.trainer.plan_run`.
There is no second, drifting code path that could show you one thing and run
another.

Run ordering is deliberate: single-dataset runs first, cumulative combos last.
If a source is broken - a failed download, a class mapping that dropped
everything - you find out on the cheap run rather than after the expensive one.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.loader import detect_profile, load_matrix, load_registry
from ..config.schema import ComboSpec, ExperimentMatrix, GpuProfile, RunPlan
from ..utils.logging import get_logger
from ..utils.paths import subdir
from .trainer import TrainingResult, describe_plan, plan_run, train

log = get_logger(__name__)


@dataclass(slots=True)
class MatrixPlan:
    plans: list[RunPlan] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (run, reason)

    @property
    def n_runs(self) -> int:
        return len(self.plans)


def _order_key(plan: RunPlan, combo_order: dict[str, int]) -> tuple[int, str, int]:
    """Single-dataset combos first, then cumulative ones, alphabetical within.

    The cheap, diagnostic runs go first so a broken source is caught before the
    multi-day ones start.
    """
    return (
        combo_order.get(plan.combo, 999),
        plan.combo,
        0 if plan.model.value.startswith("yolo") else 1,
    )


def build_matrix_plan(
    matrix: ExperimentMatrix,
    *,
    models: Sequence[str] | None = None,
    combos: Sequence[str] | None = None,
    profile: GpuProfile | str | None = None,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    require_dataset: bool = True,
) -> MatrixPlan:
    """Resolve every requested run into a :class:`RunPlan`.

    ``require_dataset=True`` skips combos whose ``data.yaml`` has not been built
    yet, with the reason recorded - so an interrupted download degrades the
    matrix instead of aborting it.
    """
    # `all` is a documented shorthand for "every enabled one". It has to be
    # expanded here, at the single point both `matrix list` and `matrix run` go
    # through: otherwise the literal string "all" is treated as a combo slug,
    # matches nothing, and the plan comes back empty - while the command still
    # exits 0, so a mistyped-or-shorthand run looks like it succeeded.
    wanted_models = (
        [m.value for m in matrix.models]
        if not models or "all" in models
        else list(models)
    )
    wanted_combos = (
        {c.slug for c in matrix.enabled_combos()}
        if not combos or "all" in combos
        else set(combos)
    )

    result = MatrixPlan()
    combo_order = {c.slug: i for i, c in enumerate(matrix.enabled_combos())}

    for combo in matrix.enabled_combos():
        if combo.slug not in wanted_combos:
            continue
        data_yaml = subdir("processed") / combo.slug / "data.yaml"

        if require_dataset and not data_yaml.is_file():
            result.skipped.append(
                (
                    f"{combo.slug}",
                    f"not built: {data_yaml} is missing. Run `anti-uav build --combo {combo.slug}`",
                )
            )
            continue

        for model in wanted_models:
            if model not in {m.value for m in matrix.models}:
                result.skipped.append(
                    (f"{model}__{combo.slug}", f"model {model!r} is not in matrix.models")
                )
                continue

            plan = plan_run(
                model,
                combo.slug,
                profile=profile,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
            )
            plan.combo = combo.slug
            plan.datasets = list(combo.datasets)
            plan.tiling = combo.force_tiling if combo.force_tiling is not None else False
            result.plans.append(plan)

    result.plans.sort(key=lambda p: _order_key(p, combo_order))
    return result


def format_plan(matrix_plan: MatrixPlan) -> str:
    lines: list[str] = []
    if matrix_plan.n_runs:
        lines.append(f"{matrix_plan.n_runs} run(s):")
        lines.append("")
        for index, plan in enumerate(matrix_plan.plans, start=1):
            lines.append(f"[{index:>2}/{matrix_plan.n_runs}] {plan.run_name}")
            for line in describe_plan(plan).splitlines():
                lines.append(f"     {line}")
            lines.append("")
    else:
        lines.append("No runs to execute.")

    if matrix_plan.skipped:
        lines.append("SKIPPED")
        for name, reason in matrix_plan.skipped:
            lines.append(f"  - {name}: {reason}")
    return "\n".join(lines).rstrip()


def execute(
    matrix_plan: MatrixPlan,
    *,
    dry_run: bool = False,
    stop_on_failure: bool = False,
    resume: bool = False,
    device: str | None = None,
) -> Iterator[TrainingResult]:
    """Run the matrix in order, yielding each result as it completes.

    Failures do not stop the sweep unless ``stop_on_failure``. A single-dataset
    run failing usually means that source is broken, and the cumulative combos
    that include it will fail too - but you want to see the whole picture, and a
    cross-dataset combo might still succeed.
    """
    for plan in matrix_plan.plans:
        resume_path: str | Path | None = None
        if resume:
            weights = Path(plan.output_dir) / "weights" / "last.pt"
            resume_path = weights if weights.is_file() else None
            if resume_path is None:
                log.warning(
                    "resume requested but no last.pt found; starting fresh",
                    extra={"run": plan.run_name, "expected": str(weights)},
                )

        log.info("starting run", extra={"run": plan.run_name, "estimate": plan.estimate_note})
        result = train(
            plan.model.value,
            plan.combo,
            profile=plan.profile,
            epochs=plan.epochs,
            imgsz=plan.imgsz,
            batch=plan.batch,
            amp=plan.amp,
            device=device,
            resume=resume_path,
            dry_run=dry_run,
        )
        yield result

        if result.error and stop_on_failure:
            log.error(
                "stopping matrix because a run failed",
                extra={"run": plan.run_name, "error": result.error},
            )
            return


def combo_for(matrix: ExperimentMatrix, slug: str) -> ComboSpec | None:
    for combo in matrix.combos:
        if combo.slug == slug:
            return combo
    return None


def summary_table(results: Sequence[TrainingResult]) -> str:
    """One line per run - what ``scripts/train_matrix.ps1`` prints at the end."""
    if not results:
        return "no runs executed"
    header = (
        f"{'run':<34}{'status':<9}{'epochs':>8}{'mAP50':>9}{'mAP50-95':>11}"
        f"{'prec':>8}{'recall':>8}{'minutes':>9}{'FP claim':>10}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        metrics = result.best_metrics
        status = "ok" if result.ok else ("dry-run" if result.warnings and not result.error else "FAILED")
        lines.append(
            f"{result.run_name:<34}{status:<9}"
            f"{result.epochs_completed:>8}"
            f"{_m(metrics, 'metrics/mAP50(B)'):>9.4f}"
            f"{_m(metrics, 'metrics/mAP50-95(B)'):>11.4f}"
            f"{_m(metrics, 'metrics/precision(B)'):>8.4f}"
            f"{_m(metrics, 'metrics/recall(B)'):>8.4f}"
            f"{result.duration_s / 60.0:>9.1f}"
            f"{_fp_claim(result.combo):>10}"
        )
    lines.append("")
    lines.append("NOTE: compare rows only within the same combo. mAP across different")
    lines.append("      datasets is not comparable - use `anti-uav evaluate --cross`.")
    lines.append("")
    lines.append("FP claim: which sources in the combo let its precision number count as")
    lines.append("      evidence about false positives. '-' means the precision column on that")
    lines.append("      row proves nothing about birds, and the bird-free sources are why.")
    return "\n".join(lines)


def _fp_claim(combo: str) -> str:
    """Which sources make a combo's precision falsifiable, for the summary table."""
    from ..data.capabilities import CLAIM_PRECISION, combo_capabilities

    if not combo:
        return "-"
    for spec in load_matrix().combos:
        if spec.slug != combo:
            continue
        names = combo_capabilities(combo, tuple(spec.datasets)).sources_for(CLAIM_PRECISION)
        return ",".join(names) if names else "-"
    return "-"


def capability_report(aliases: tuple[str, ...] | None = None) -> str:
    """Per-source and per-combo capability tables, for ``anti-uav matrix list``.

    The point is that a reader finds out which numbers are allowed to prove what
    *before* spending GPU hours, rather than working it out from a results table
    afterwards. Lines are kept short and warnings are wrapped, because this is
    printed through a Rich console that hard-wraps at the terminal width and a
    wrapped capability table is worse than none.
    """
    import textwrap

    from ..data.capabilities import (
        combo_capabilities,
        source_capabilities_for,
    )

    registry = load_registry()
    wanted = tuple(aliases) if aliases else tuple(registry.aliases)

    lines: list[str] = ["", "WHAT EACH SOURCE CAN PROVE", "=" * 60]
    source_rows: list[tuple[str, ...]] = []
    for alias in wanted:
        c = source_capabilities_for(alias)
        source_rows.append(
            (
                c.alias,
                f"{int(c.median_target_px or 0)}px",
                str(c.tile_size) if c.requires_tiling else "-",
                "yes" if c.can_falsify_precision else "NO",
                "yes" if c.can_score_identity else "NO",
                "yes" if c.can_score_tracking else "NO",
            )
        )
    lines.extend(_render_table(("source", "median", "tiled", "birds", "identity", "trackGT"), source_rows))
    lines.append("")
    for text in (
        "birds    precision here can falsify a false-positive claim",
        "identity IDF1 / ID switches meaningful (needs >1 id per sequence)",
        "trackGT  MOT ground truth, so MOTA / HOTA can be computed at all",
    ):
        lines.append(f"  {text}")

    lines.extend(["", "PER-COMBO CLAIMS", "=" * 60])
    lines.extend(_combo_claims_table())
    lines.append("")
    lines.extend(
        textwrap.wrap(
            "A combo inherits a claim from its sources: all8 can falsify false "
            "positives because dvb and mavvid are in it. That says nothing about "
            "how either behaved alone - read the per-dataset table for that.",
            width=76,
            initial_indent="  ",
            subsequent_indent="  ",
        )
    )

    problems: list[tuple[str, str]] = [
        (spec.slug, w)
        for spec in load_matrix().combos
        for w in combo_capabilities(spec.slug, tuple(spec.datasets)).warnings()
    ]
    if problems:
        lines.extend(["", "COMBOS THAT CANNOT SUPPORT EVERY CLAIM", "=" * 60])
        for slug, warning in problems:
            lines.extend(
                textwrap.wrap(
                    f"{slug}: {warning}",
                    width=74,
                    initial_indent="  ! ",
                    subsequent_indent="      ",
                )
            )
    return "\n".join(lines)


#: Hard budget for a table line. The report is printed through a Rich console
#: that hard-wraps at the terminal width, and a wrapped capability table is
#: worse than none - so a row that will not fit is abbreviated, not truncated
#: mid-word.
_LINE_BUDGET = 78


def _render_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> list[str]:
    """Render a table whose column widths come from the content.

    Fixed-width columns stop working the moment an alias gets longer than its
    slot: the row silently shifts and the columns stop lining up. Sizing from
    the content keeps the table readable, and the caller is responsible for
    keeping the result inside the console width.
    """
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    separator = "  "
    out = [
        separator.join(h.ljust(w) for h, w in zip(headers, widths, strict=True)).rstrip()
    ]
    out.append("-" * len(out[0]))
    for row in rows:
        out.append(
            separator.join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()
        )
    return out


def _combo_claims_table() -> list[str]:
    """The per-combo claims table, sized to fit the console width.

    Column widths come from the content rather than being fixed, because the
    alias lists grew: three bird-negative sources means ``dvb,mavvid,
    github_maciullo`` in one cell, which a hard-coded 11-column slot cannot
    hold. If even the adaptive table will not fit, the alias lists are
    abbreviated - the per-source table above already names every source in
    full, so the abbreviation loses no information a reader can act on.
    """
    from ..data.capabilities import (
        CLAIM_IDENTITY,
        CLAIM_PRECISION,
        CLAIM_TRACKING,
        combo_capabilities,
    )

    headers = ("combo", "birds", "identity", "trackGT", "tiled")
    rows: list[tuple[str, ...]] = []
    for spec in load_matrix().combos:
        caps = combo_capabilities(spec.slug, tuple(spec.datasets))
        rows.append(
            (
                spec.slug,
                _names(caps, CLAIM_PRECISION),
                _names(caps, CLAIM_IDENTITY),
                _names(caps, CLAIM_TRACKING),
                ",".join(caps.tiled_sources) or "-",
            )
        )

    for abbreviate in (False, True):
        rendered = [
            tuple(_abbreviate_aliases(cell) for cell in row) if abbreviate else row
            for row in rows
        ]
        widths = [
            max(len(headers[i]), *(len(row[i]) for row in rendered))
            for i in range(len(headers))
        ]
        if sum(widths) + 2 * (len(widths) - 1) <= _LINE_BUDGET:
            break

    return _render_table(headers, rendered)


def _abbreviate_aliases(cell: str) -> str:
    """``dvb,mavvid,github_maciullo`` -> ``dvb,mav,ghb``.

    Only used as a fallback when the full table will not fit the console
    width. Every alias is also spelled out in full in the per-source table
    above, so this shortens the reading without hiding a source.
    """
    if "," not in cell:
        return cell
    parts = cell.split(",")
    if all(len(p) <= 4 for p in parts):
        return cell
    return ",".join(p[:3] for p in parts)


def _names(caps: Any, claim: str) -> str:
    names = caps.sources_for(claim)
    return ",".join(names) if names else "-"


def _m(metrics: dict[str, float], key: str) -> float:
    return float(metrics.get(key, 0.0))


def default_plan(**kwargs) -> MatrixPlan:
    """Convenience used by the CLI: load the matrix and plan it in one call."""
    return build_matrix_plan(load_matrix(), **kwargs)


def active_profile_note() -> str:
    profile = detect_profile()
    return f"auto-detected GPU profile: {profile.value}"
