"""The ``anti-uav`` command line.

One entry point, subcommands per pipeline stage, in the order you actually run
them::

    verify-env  ->  download  ->  convert  ->  splits  ->  build  ->  sanity
                                                             |
                                                        train / eval
                                                             |
                                                    track-eval / export / serve

Design notes:

* Nothing here trains or downloads unless asked. ``--dry-run`` exists on the
  expensive commands and prints exactly what would happen, because the two largest
  downloads are 60 GB and 400 GB.
* Errors are messages, not tracebacks, wherever the cause is knowable. A missing
  dataset prints the command that fixes it.
* Every stage that writes to disk reports where it wrote, so the next command can
  be copy-pasted.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from . import __version__

app = typer.Typer(
    name="anti-uav",
    help=(
        "Drone detection & tracking pipeline.\n"
        "Data:  download -> convert -> splits -> build -> sanity\n"
        "Model: train -> eval -> export\n"
        "Track: track-eval -> replay\n"
        "Ops:   serve, rules, deploy, verify-env"
    ),
    no_args_is_help=True,
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

config_app = typer.Typer(help="Inspect and validate configuration.", no_args_is_help=True)
matrix_app = typer.Typer(help="The experiment matrix.", no_args_is_help=True)
rules_app = typer.Typer(help="The drone rule / threshold layer.", no_args_is_help=True)
deploy_app = typer.Typer(help="Render DeepStream deployment artifacts.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(matrix_app, name="matrix")
app.add_typer(rules_app, name="rules")
app.add_typer(deploy_app, name="deploy")

console = Console()


def _fail(message: str, code: int = 1) -> None:
    console.print(f"[bold red]error:[/] {message}")
    raise typer.Exit(code)


def _ok(message: str) -> None:
    console.print(f"[green]{message}[/]")


def _warn(message: str) -> None:
    console.print(f"[yellow]![/] {message}")


def _emit_json(payload: Any) -> None:
    console.print_json(json.dumps(payload, default=str))


def _comma_list(value: str | None) -> list[str]:
    """``--combos a,b`` or ``--combos all``."""
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #


@app.command()
def version() -> None:
    """Print the version and the resolved project paths."""
    from .utils.paths import data_root, project_root, subdir

    console.print(f"anti-uav {__version__}")
    console.print(f"python     {sys.version.split()[0]}")
    for key in ("project", "data", "runs", "processed"):
        try:
            path = project_root() if key == "project" else (
                data_root() if key == "data" else subdir(key)
            )
            console.print(f"{key:<10} {path}")
        except Exception as exc:
            _warn(f"{key}: {exc}")


@app.command("verify-env")
def verify_env(
    profile: Annotated[
        str | None, typer.Option(help="Force a GPU profile instead of auto-detecting.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Check the interpreter, torch build and GPU architecture.

    Run this before any training. The failure it exists to catch is silent: a torch
    wheel with no kernels for your GPU installs cleanly and fails hours later.
    """
    from .verify import run as run_verify_env

    code = run_verify_env(profile=profile, console=None if as_json else console)
    if as_json:
        from .verify import verify

        _emit_json(verify(profile).to_dict())
    raise typer.Exit(code)


@app.command("fetch-weights")
def fetch_weights(
    models: Annotated[
        str | None,
        typer.Option(help="Comma list of families. Default: every family in the matrix."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what is missing; download nothing.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Make sure every recipe's pretrained checkpoint is on disk.

    ``configs/train/*.yaml`` each name a ``base_weights`` file. Ultralytics
    downloads a missing one from inside the trainer, so without this command a
    blocked download surfaces only after the dataset is built and the run has
    started. Run it once per machine, before training.
    """
    from .detection.weights import fetch as _fetch

    selected = _comma_list(models) or None
    try:
        found, errors = _fetch(selected, dry_run=dry_run)
    except KeyError as exc:
        _fail(str(exc).strip("'"))
        return

    if as_json:
        _emit_json(
            {
                "dry_run": dry_run,
                "weights": [t.to_dict() for t in found],
                "errors": errors,
                "ok": not errors and all(t.present for t in found),
            }
        )
    else:
        table = Table(title="base weights" + ("  (dry run)" if dry_run else ""))
        table.add_column("family")
        table.add_column("file")
        table.add_column("state", justify="right")
        table.add_column("size", justify="right")
        table.add_column("path")
        for target in found:
            table.add_row(
                target.family,
                target.name,
                "[green]present[/]" if target.present else "[red]MISSING[/]",
                f"{target.size_mb:.0f} MB" if target.present else "-",
                str(target.path) if target.path else "-",
            )
        console.print(table)

    absent = [t for t in found if not t.present]
    for message in errors:
        _warn(message)

    if dry_run and absent:
        console.print()
        _warn(f"{len(absent)} checkpoint(s) would be downloaded.")
        return

    if errors:
        raise typer.Exit(1)
    if absent:
        _fail(
            f"{', '.join(t.name for t in absent)} still missing after the fetch. "
            f"Training will not start. If this machine has no route to the "
            f"ultralytics release host, copy the file in by hand from a machine that does."
        )
    _ok(f"all {len(found)} checkpoint(s) present")


@app.command("list-runs")
def list_runs() -> None:
    """Every training run under ``artifacts/runs``."""
    from .detection.trainer import list_runs as _list

    runs = _list()
    if not runs:
        _warn("no runs found. Train one: anti-uav train --model yolo11n --combo dvb")
        return

    table = Table(title=f"{len(runs)} run(s)", show_lines=False)
    for column in ("run", "model", "combo", "epochs", "mAP50", "mAP50-95", "best", "trained"):
        table.add_column(column, justify="right" if column != "run" else "left")

    for run in runs:
        metrics = run.get("best_metrics", {}) or {}
        table.add_row(
            str(run.get("run_name", "?")),
            str(run.get("model", "?")),
            str(run.get("combo", "?")),
            str(run.get("epochs_completed", 0)),
            f"{metrics.get('metrics/mAP50(B)', 0.0):.4f}",
            f"{metrics.get('metrics/mAP50-95(B)', 0.0):.4f}",
            "yes" if run.get("_has_weights") else "NO",
            str(run.get("trained_at", ""))[:19],
        )
    console.print(table)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


@app.command()
def download(
    dataset: Annotated[str, typer.Option(help="Alias (dvb, mavvid, antiuav, mmuav) or 'all'.")],
    variant: Annotated[str | None, typer.Option(help="e.g. 300 for Anti-UAV.")] = None,
    source: Annotated[
        str | None, typer.Option(help="Select one source by its label from the registry.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the plan and sizes without downloading.")
    ] = False,
    modalities: Annotated[
        str | None, typer.Option(help="Comma list for MM-UAV, e.g. rgb or rgb,ir.")
    ] = None,
    max_sequences: Annotated[
        int | None, typer.Option(help="MM-UAV subset size. 150 is ~12 GB of 400 GB.")
    ] = None,
) -> None:
    """Fetch a dataset. Always start with ``--dry-run``."""
    from .config.loader import load_dataset, load_registry
    from .config.schema import Modality
    from .data.download import download as _download

    wanted = _comma_list(modalities)
    chosen = [m.strip().lower() for m in wanted] or None
    modality_enum = [Modality(m) for m in chosen] if chosen else None

    aliases = list(load_registry().aliases) if dataset == "all" else [dataset]

    for alias in aliases:
        try:
            spec = load_dataset(alias)
        except KeyError as exc:
            _fail(str(exc))
            return

        for result in _download(
            spec,
            variant=variant,
            source_label=source,
            dry_run=dry_run,
            max_sequences=max_sequences,
            modalities=modality_enum,
        ):
            status = {
                "completed": "[green]completed[/]",
                "skipped": "[yellow]skipped[/]",
                "needs_credentials": "[magenta]needs credentials[/]",
                "failed": "[red]failed[/]",
            }.get(result.status, result.status)

            console.print(f"\n[bold]{alias}[/] - {result.source}  {status}")
            plan = result.extras.get("plan")
            if plan:
                console.print(str(plan), markup=False)
            if result.error:
                console.print(result.error, markup=False)
            if result.sha256:
                console.print(f"sha256: {result.sha256}")


@app.command()
def convert(
    dataset: Annotated[str, typer.Option(help="Alias, or 'all' for every converted dataset.")],
    variant: Annotated[str | None, typer.Option(help="Defaults to the registry's default.")] = None,
    raw: Annotated[
        str | None, typer.Option(help="Raw root. Defaults to data/raw/<alias>/<variant>.")
    ] = None,
    modalities: Annotated[str | None, typer.Option(help="MM-UAV only, e.g. rgb.")] = None,
    max_sequences: Annotated[int | None, typer.Option(help="MM-UAV subset size.")] = None,
    lenient: Annotated[
        bool, typer.Option("--lenient", help="Drop unmapped labels instead of failing.")
    ] = False,
) -> None:
    """Normalise a downloaded dataset into the interim frame index."""
    from .config.loader import load_dataset, load_registry
    from .data.convert import ConversionAborted, convert_dataset
    from .utils.paths import dataset_root

    aliases = list(load_registry().aliases) if dataset == "all" else [dataset]
    failures = 0

    for alias in aliases:
        spec = load_dataset(alias)
        chosen_variant = variant or spec.default_variant or "full"
        root = Path(raw) if raw else dataset_root(alias) / chosen_variant

        if not root.exists():
            _warn(f"{alias}: {root} does not exist - nothing to convert")
            _warn(f"    anti-uav download --dataset {alias} --dry-run")
            continue

        kwargs: dict[str, Any] = {}
        if alias == "mmuav":
            wanted = _comma_list(modalities)
            kwargs["modalities"] = wanted or ["rgb"]
            if max_sequences:
                kwargs["max_sequences"] = max_sequences

        console.print(f"\n[bold]{alias}[/] <- {root}")
        try:
            report = convert_dataset(
                spec, root, variant=chosen_variant, strict_labels=not lenient, **kwargs
            )
        except (ConversionAborted, FileNotFoundError) as exc:
            _fail(f"{alias}: {exc}")
            failures += 1
            continue
        except Exception as exc:
            _fail(f"{alias}: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        table = Table(show_header=False, box=None)
        for label, value in (
            ("frames", f"{report.frames_written:,}"),
            ("boxes", f"{report.boxes_written:,}"),
            ("sequences", f"{report.sequences:,}"),
            ("dropped boxes", f"{report.boxes_dropped:,}"),
            ("classes", json.dumps(report.class_histogram)),
        ):
            table.add_row(label, value)
        console.print(table)

        for warning in report.warnings:
            _warn(warning)
        _ok(f"index: data/interim/{alias}/{report.variant}/index.jsonl")

    if failures:
        raise typer.Exit(1)


@app.command()
def stats(
    dataset: Annotated[str, typer.Option(help="Alias or 'all'.")],
    variant: Annotated[str | None, typer.Option] = None,
) -> None:
    """Per-dataset statistics, including the bird-negative coverage table."""
    from .config.loader import load_dataset
    from .config.loader import load_registry as _registry
    from .data.stats import combo_summary, for_dataset, format_stats

    aliases = (
        _comma_list(dataset) if dataset != "all" else list(_registry().aliases)
    )

    collected = {}
    for alias in aliases:
        spec = load_dataset(alias)
        resolved = variant or spec.default_variant or "full"
        computed = for_dataset(alias, resolved, unified_labels=spec.classes.unified_labels)
        if computed.frames == 0:
            _warn(f"{alias}: no index. Run: anti-uav convert --dataset {alias}")
            continue
        collected[alias] = computed
        console.print(f"\n[bold]{alias}[/]")
        console.print(format_stats(computed), markup=False)

    if len(collected) > 1:
        console.print("\n[bold]combined[/]")
        console.print(
            combo_summary(collected, unified_labels=("drone", "bird")), markup=False
        )


@app.command()
def splits(
    combo: Annotated[str | None, typer.Option(help="Combo slug, e.g. dvb or all4.")] = None,
    datasets: Annotated[str | None, typer.Option(help="Override the combo's dataset list.")] = None,
    variant: Annotated[str | None, typer.Option(help="Dataset variant override.")] = None,
    val_fraction: Annotated[float, typer.Option(help="Target val share of SEQUENCES.")] = 0.15,
    test_fraction: Annotated[float, typer.Option(help="Test share, for cross-dataset eval.")] = 0.0,
    group_by: Annotated[
        str, typer.Option(help="sequence (default), group, source or file.")
    ] = "sequence",
    held_out: Annotated[str | None, typer.Option(help="Datasets to hold out as 'test'.")] = None,
    seed: Annotated[int, typer.Option] = 0,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Do not write.")] = False,
) -> None:
    """Assign whole sequences to train/val. Never individual frames.

    This is the step that decides whether your numbers mean anything. Adjacent
    frames of a video are near-identical, so a frame-level split leaks and inflates
    mAP by double digits. Read the warnings it prints.
    """
    from .config.loader import load_matrix, load_registry
    from .config.schema import SplitStrategy
    from .data.frameindex import load_index
    from .data.splits import format_report, persist, split_records

    if combo:
        matrix = load_matrix()
        entry = next((c for c in matrix.combos if c.slug == combo), None)
        if entry is None:
            _fail(f"unknown combo {combo!r}; known: {[c.slug for c in matrix.combos]}")
            return
        dataset_list = entry.datasets
        slug = entry.slug
    else:
        dataset_list = _comma_list(datasets)
        slug = "adhoc"
        if not dataset_list:
            _fail("pass --combo or --datasets")
            return

    registry = load_registry()
    records = []
    for alias in dataset_list:
        spec = registry.datasets.get(alias)
        resolved = variant or (spec.default_variant if spec else None) or "full"
        loaded = load_index(alias, resolved)
        if not loaded:
            _warn(f"{alias}: index is empty or missing. Run: anti-uav convert --dataset {alias}")
        records.extend(loaded)

    if not records:
        _fail("no frames found. Convert the datasets first.")
        return

    try:
        strategy = SplitStrategy(group_by)
    except ValueError:
        _fail(f"unknown strategy {group_by!r}; use sequence, group, source or file")
        return

    assigned, report = split_records(
        records,
        strategy=strategy,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        held_out_sources=_comma_list(held_out),
        combo=slug,
    )

    console.print(format_report(report), markup=False)

    if dry_run:
        _warn("dry run: nothing written")
        return

    persist(assigned, report, slug)
    _ok(f"wrote splits for {len(assigned):,} frames")
    _warn("now run: anti-uav build --combo " + slug)


@app.command()
def build(
    combo: Annotated[str, typer.Option(help="Combo slug, e.g. dvb, dvb+mavvid, all4, or 'all'.")],
    variants: Annotated[str | None, typer.Option(help="alias=variant pairs.")] = None,
    no_clean: Annotated[bool, typer.Option("--no-clean", help="Do not wipe the old build.")] = False,
    max_frames: Annotated[
        int | None, typer.Option(help="Cap frames per source, for a smoke build.")
    ] = None,
) -> None:
    """Materialise a YOLO dataset, applying per-source tiling."""
    from .config.loader import load_matrix, load_registry
    from .data.build import build as _build

    matrix = load_matrix()
    registry = load_registry()
    variant_map = dict(
        pair.split("=", 1) for pair in _comma_list(variants) if "=" in pair
    )

    combos = (
        matrix.enabled_combos() if combo == "all" else
        [c for c in matrix.combos if c.slug == combo]
    )
    if not combos:
        _fail(f"unknown combo {combo!r}; known: {[c.slug for c in matrix.combos]}")
        return

    specs = {alias: registry.datasets[alias] for alias in registry.datasets}
    failed = 0

    for entry in combos:
        console.print(f"\n[bold]{entry.slug}[/]  ({', '.join(entry.datasets)})")
        report = _build(
            entry,
            specs,
            matrix,
            variants=variant_map,
            unified_labels=matrix.unified_labels if hasattr(matrix, "unified_labels") else ("drone", "bird"),
            clean=not no_clean,
            max_frames_per_source=max_frames,
        )

        table = Table(show_header=True, box=None)
        for column in ("dataset", "tiling", "tile", "frames in", "frames out", "train", "val"):
            table.add_column(column, justify="right" if column != "dataset" else "left")
        for source in report.sources:
            table.add_row(
                source.dataset + (" (skipped)" if source.skipped else ""),
                "on" if source.tiling else "off",
                str(source.tile_size) if source.tiling else "-",
                f"{source.frames_in:,}",
                f"{source.frames_out:,}",
                f"{source.train_frames:,}",
                f"{source.val_frames:,}",
            )
        console.print(table)
        console.print(
            f"train {report.total_train_frames:,}   val {report.total_val_frames:,}   "
            f"boxes {report.total_boxes:,}"
        )

        for warning in report.warnings:
            _warn(warning)
        for source in report.sources:
            for warning in source.warnings:
                _warn(f"{source.dataset}: {warning}")

        if report.ok:
            _ok(f"data.yaml: {report.data_yaml}")
        else:
            for error in report.errors:
                _fail(error)
            failed += 1

    if failed:
        raise typer.Exit(1)


@app.command()
def sanity(
    combo: Annotated[str, typer.Option(help="Combo slug, or 'all'.")],
    min_target_px: Annotated[float, typer.Option(help="Warn below this target size.")] = 8.0,
) -> None:
    """Verify the pipeline invariants. Non-zero exit means do not train."""
    from .config.loader import load_matrix, load_registry
    from .data.frameindex import load_index
    from .data.sanity import format_report, run_all

    matrix = load_matrix()
    registry = load_registry()

    combos = (
        matrix.enabled_combos() if combo == "all" else
        [c for c in matrix.combos if c.slug == combo]
    )
    if not combos:
        _fail(f"unknown combo {combo!r}; known: {[c.slug for c in matrix.combos]}")
        return

    failed = False
    for entry in combos:
        records: list = []
        for alias in entry.datasets:
            spec = registry.datasets.get(alias)
            resolved = spec.default_variant if spec else "full"
            records.extend(load_index(alias, resolved or "full"))

        report = run_all(
            records,
            combo=entry.slug,
            expect_datasets=entry.datasets,
            min_target_px=min_target_px,
        )
        console.print(f"\n[bold]{entry.slug}[/]")
        console.print(format_report(report), markup=False)
        failed = failed or not report.ok

    raise typer.Exit(1 if failed else 0)


@app.command("dedup")
def dedup(
    combo: Annotated[str, typer.Option(help="Combo slug, or 'all'.")] = "all",
    threshold: Annotated[int, typer.Option(help="pHash Hamming distance.")] = 4,
    limit: Annotated[int | None, typer.Option(help="Cap frames hashed.")] = 200_000,
) -> None:
    """Find near-duplicate frames, especially ones straddling the split."""
    from .config.loader import load_matrix, load_registry
    from .data.dedup import find_duplicates, format_report, iter_dataset_records, save_report

    matrix = load_matrix()
    registry = load_registry()
    combos = (
        matrix.enabled_combos() if combo == "all" else
        [c for c in matrix.combos if c.slug == combo]
    )

    for entry in combos:
        variants = {
            alias: (registry.datasets[alias].default_variant or "full")
            for alias in entry.datasets
            if alias in registry.datasets
        }
        records = iter_dataset_records(entry.datasets, variants)
        if not records:
            _warn(f"{entry.slug}: no frames")
            continue

        console.print(f"\n[bold]{entry.slug}[/]  ({len(records):,} frames)")
        clusters, report = find_duplicates(
            records, threshold=threshold, limit=limit
        )
        console.print(format_report(report, clusters), markup=False)
        save_report(report, f"dedup_{entry.slug}")


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


@app.command()
def train(
    model: Annotated[str, typer.Option(help="yolo11n or rtdetr_x2.")],
    combo: Annotated[str, typer.Option(help="Combo slug from configs/matrix.yaml.")],
    profile: Annotated[str | None, typer.Option(help="pascal/ampere/ada/blackwell/cpu.")] = None,
    epochs: Annotated[int | None, typer.Option] = None,
    imgsz: Annotated[int | None, typer.Option] = None,
    batch: Annotated[int | None, typer.Option] = None,
    device: Annotated[str | None, typer.Option(help="auto, cpu, 0, 0,1")] = None,
    workers: Annotated[int | None, typer.Option] = None,
    seed: Annotated[int | None, typer.Option] = None,
    resume: Annotated[str | None, typer.Option(help="Path to last.pt.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the plan only.")] = False,
) -> None:
    """Train one (model, dataset combo) pair."""
    from .detection.trainer import describe_plan, plan_run
    from .detection.trainer import train as _train

    try:
        plan = plan_run(model, combo, profile=profile, epochs=epochs, imgsz=imgsz, batch=batch)
    except (FileNotFoundError, KeyError) as exc:
        _fail(str(exc))
        return

    console.print(describe_plan(plan), markup=False)
    console.print()

    if dry_run:
        _warn("dry run: nothing trained")
        return

    result = _train(
        model,
        combo,
        profile=profile,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        seed=seed,
        resume=resume,
    )

    if result.ok:
        _ok(f"finished in {result.duration_s / 60:.1f} min")
        for key, value in result.best_metrics.items():
            if key.startswith("metrics/"):
                console.print(f"  {key:<26} {value:.4f}")
        console.print(f"\nbest: {result.best_weights}")
        console.print(f"next: anti-uav eval --run {result.output_dir}")
        console.print(f"      anti-uav export --run {result.output_dir} --formats onnx,tensorrt")
    else:
        _fail(result.error or "training failed")
        for warning in result.warnings:
            _warn(warning)
        raise typer.Exit(1)


@app.command()
def evaluate(
    run: Annotated[str, typer.Option(help="Run name, run directory, or a .pt file.")],
    data: Annotated[str | None, typer.Option(help="data.yaml override.")] = None,
    split: Annotated[str, typer.Option] = "val",
    imgsz: Annotated[int | None, typer.Option] = None,
    device: Annotated[str | None, typer.Option] = None,
    cross: Annotated[
        bool, typer.Option("--cross", help="Evaluate on every source dataset, including unseen ones.")
    ] = False,
    datasets: Annotated[str | None, typer.Option(help="Restrict --cross to these datasets.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Score a run. ``--cross`` is the table worth reading."""
    from .detection.evaluate import (
        evaluate_cross_dataset,
        evaluate_run,
        format_cross_table,
        format_result,
    )

    if cross:
        results = evaluate_cross_dataset(
            run, datasets=_comma_list(datasets) or None, imgsz=imgsz, device=device
        )
        if as_json:
            _emit_json({k: v.to_dict() for k, v in results.items()})
        else:
            console.print(format_cross_table(results), markup=False)
        return

    try:
        result = evaluate_run(run, split=split, data_yaml=data, imgsz=imgsz, device=device)
    except FileNotFoundError as exc:
        _fail(str(exc))
        return

    if as_json:
        _emit_json(result.to_dict())
    else:
        console.print(format_result(result), markup=False)

    if not result.metrics:
        raise typer.Exit(1)


@app.command("track-eval")
def track_eval(
    run: Annotated[str, typer.Option(help="Detector run to replay with.")],
    dataset: Annotated[str, typer.Option(help="Dataset to replay.")],
    variant: Annotated[str | None, typer.Option] = None,
    trackers: Annotated[str | None, typer.Option(help="Comma list; default sort,bytetrack,botsort.")] = None,
    sequences: Annotated[int, typer.Option(help="How many sequences to replay.")] = 20,
    conf_floor: Annotated[float, typer.Option(help="Detections below this are not fed to the tracker.")] = 0.05,
    imgsz: Annotated[int | None, typer.Option] = None,
    device: Annotated[str | None, typer.Option] = None,
    no_reid: Annotated[bool, typer.Option("--no-reid", help="Ablate the appearance gate.")] = False,
    reid: Annotated[
        Path | None,
        typer.Option(
            "--reid",
            help=(
                "ReID checkpoint for the appearance gate. Defaults to the newest "
                "artifacts/runs/reid/*/best.pt; pass a path to compare two of them."
            ),
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Score trackers against a dataset's MOT ground truth.

    Detections are computed once per sequence and reused across trackers, so any
    difference between rows is the tracker's, not the detector's noise.
    """
    from .config.loader import load_registry
    from .detection.predictor import Detector
    from .tracking.replayer import evaluate_trackers_on_dataset, format_reports

    spec = load_registry().datasets.get(dataset)
    if spec is None:
        _fail(f"unknown dataset {dataset!r}")
        return
    resolved = variant or spec.default_variant or "full"

    names = _comma_list(trackers) or ["sort", "bytetrack", "botsort"]

    if reid is not None and not reid.is_file():
        _fail(
            f"ReID checkpoint {str(reid)!r} does not exist. Train one with "
            f"anti_uav.tracking.reid_train.train_reid, or drop the flag to use the default."
        )
        return

    # Capability gate. Refusing beats a confident table of numbers that cannot
    # mean what the column headings say.
    from .data.capabilities import (
        source_capabilities_for,
    )

    caps = source_capabilities_for(dataset)
    if not caps.can_score_tracking:
        usable = [a for a in ("dvb", "mavvid", "antiuav", "mmuav")
                  if source_capabilities_for(a).can_score_tracking]
        _fail(
            f"{dataset} carries no MOT track ids, so tracking metrics cannot be "
            f"computed on it at all. Use one of: {', '.join(usable)}."
        )
        return
    if not caps.can_score_identity and not no_reid:
        _warn(
            f"{dataset} is single-target ground truth: MOTA and HOTA are meaningful, but "
            f"IDF1 and ID-switch counts are not, because every identity looks like the "
            f"same one. Judge the tracker on HOTA, or run --dataset mmuav."
        )

    try:
        detector = Detector(run, imgsz=imgsz, device=device)
        detector.load()
    except (FileNotFoundError, RuntimeError) as exc:
        _fail(str(exc))
        return

    common: dict[str, Any] = {}
    if no_reid:
        common["with_reid"] = False

    try:
        reports = evaluate_trackers_on_dataset(
            detector,
            dataset,
            resolved,
            limit=sequences,
            trackers=names,  # type: ignore[arg-type]
            conf_floor=conf_floor,
            reid_checkpoint=reid,
            **common,
        )
    finally:
        detector.close()

    if as_json:
        _emit_json({k: v.to_dict() for k, v in reports.items()})
    else:
        console.print(format_reports(reports), markup=False)


@app.command()
def predict(
    run: Annotated[str, typer.Option(help="Run directory or weights path to load.")],
    source: Annotated[str, typer.Option(help="Image file, or video for a stream.")],
    tracker: Annotated[str | None, typer.Option(help="sort,bytetrack,botsort or nvdcf.")] = None,
    conf: Annotated[float | None, typer.Option] = None,
    imgsz: Annotated[int | None, typer.Option] = None,
    device: Annotated[str | None, typer.Option(help="cpu, 0, 0,1")] = None,
    stride: Annotated[int, typer.Option(help="Video only: process every Nth frame.")] = 1,
    max_frames: Annotated[int | None, typer.Option(help="Video only: stop after N.")] = None,
    save: Annotated[Path | None, typer.Option(help="Write annotated frames here.")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Run one trained model over an image or a video, and show what it found.

    This is the command that answers "is the thing I just trained actually
    detecting anything?", so it prints the per-detection table rather than just
    a count - a run that returns 40 detections at conf 0.25 is very different
    from one that returns 3 at 0.8, and the count alone hides that.
    """
    from .detection.predictor import Detector, TrackerName
    from .utils.imaging import IMAGE_SUFFIXES

    path = Path(source)
    if not path.exists():
        _fail(f"no such file or directory: {source}")
    if path.is_dir():
        _fail(f"{source} is a directory. Point --source at one image or one video.")

    tracker_name: TrackerName | None = None
    if tracker:
        try:
            tracker_name = TrackerName(tracker.lower())
        except ValueError:
            known = ", ".join(t.value for t in TrackerName)
            _fail(f"unknown tracker {tracker!r}; available: {known}")

    detector = Detector(run, conf=conf, imgsz=imgsz, device=device)
    is_image = path.suffix.lower() in IMAGE_SUFFIXES
    saved: list[str] = []

    from .detection.predictor import overlay
    from .utils.imaging import read_image, write_image

    if save is not None:
        save.mkdir(parents=True, exist_ok=True)

    # Overlays are drawn as each frame is decoded rather than afterwards:
    # re-decoding the video to recover the pixels would double the work and
    # decode at a different stride than the one that was inferred on.
    if is_image:
        frame_iter: Iterable[tuple[int, Any]] = [(0, read_image(path))]
        stream = iter([detector.predict_image(image) for _, image in frame_iter])
    else:
        from .utils.imaging import frames_from_video

        frame_iter = frames_from_video(path, stride=stride, max_frames=max_frames)
        # One generator, consumed once. Re-creating the stream per frame would
        # re-decode the video from the start on every iteration - quadratic, and
        # it would silently produce the wrong frames under a stride.
        stream = detector.predict_stream(
            path, tracker=tracker_name, stride=stride, max_frames=max_frames
        )

    results = []
    try:
        for (_frame_index, image), result in zip(frame_iter, stream, strict=False):
            results.append(result)
            if save is not None:
                out = save / (
                    f"{path.stem}.jpg"
                    if is_image
                    else f"{path.stem}_{result.frame_index:06d}.jpg"
                )
                write_image(out, overlay(image, result))
                saved.append(str(out))
    finally:
        detector.close()

    if not results:
        _fail(f"no frames decoded from {source}")

    shown = results[:20]
    total_detections = sum(len(r.detections) for r in results)
    entries: list[dict[str, Any]] = [
        {
            "frame_index": r.frame_index,
            "detections": len(r.detections),
            "tracks": len(r.tracks),
            "objects": [
                {
                    "box": [round(v, 1) for v in d.box],
                    "confidence": round(d.confidence, 3),
                    "class_name": d.class_name,
                    "track_id": d.track_id,
                }
                for d in r.detections
            ],
        }
        for r in shown
    ]

    if as_json:
        _emit_json(
            {
                "source": str(path),
                "frames": len(results),
                "detections": total_detections,
                "saved": saved,
                "results": entries,
            }
        )
        return

    console.print(
        f"[bold]{path.name}[/]  {len(results)} frame(s), {total_detections} detection(s)"
    )
    table = Table(title=f"{path.name} - first {len(shown)} frame(s)")
    table.add_column("frame")
    table.add_column("class")
    table.add_column("conf", justify="right")
    table.add_column("box (x1,y1,x2,y2)")
    table.add_column("track", justify="right")
    for entry in entries:
        if not entry["objects"]:
            table.add_row(str(entry["frame_index"]), "-", "-", "(none)", "-")
        for obj in entry["objects"]:
            table.add_row(
                str(entry["frame_index"]),
                obj["class_name"],
                f"{obj['confidence']:.3f}",
                ", ".join(str(v) for v in obj["box"]),
                str(obj["track_id"]) if obj["track_id"] is not None else "-",
            )
    console.print(table)
    if saved:
        _ok(f"wrote {len(saved)} annotated frame(s) to {save}")


@app.command()
def replay(
    run: Annotated[str, typer.Option(help="Run directory or weights path to load.")],
    dataset: Annotated[str, typer.Option(help="Dataset alias, e.g. mmuav.")],
    sequence: Annotated[
        str | None, typer.Option(help="Sequence id. Omit to replay every sequence.")
    ] = None,
    variant: Annotated[str | None, typer.Option(help="Dataset variant override.")] = None,
    tracker: Annotated[str, typer.Option(help="sort,bytetrack,botsort.")] = "botsort",
    modality: Annotated[str, typer.Option(help="rgb or ir.")] = "rgb",
    conf_floor: Annotated[float, typer.Option(
        help="Detections below this are not fed to the tracker."
    )] = 0.05,
    max_frames: Annotated[int | None, typer.Option] = None,
    no_embeddings: Annotated[bool, typer.Option(
        "--no-embeddings", help="Skip the ReID appearance feature."
    )] = False,
    reid: Annotated[
        Path | None,
        typer.Option(
            "--reid",
            help=(
                "ReID checkpoint for the appearance gate. Defaults to the newest "
                "artifacts/runs/reid/*/best.pt."
            ),
        ),
    ] = None,
    dump: Annotated[Path | None, typer.Option(help="Write per-frame MOT rows here.")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Detect -> track one sequence, and print the trajectory.

    `track-eval` scores trackers against ground truth; `replay` does not need
    any. That makes it the way to see *why* a track breaks on a specific clip -
    switch trackers with --tracker and watch the same sequence again.
    """
    from .detection.predictor import Detector, TrackerName
    from .tracking.replayer import (
        available_sequences,
        default_variant,
        dump_predictions,
        replay_sequence,
    )

    try:
        tracker_name = TrackerName(tracker.lower())
    except ValueError:
        known = ", ".join(t.value for t in TrackerName)
        _fail(f"unknown tracker {tracker!r}; available: {known}")

    chosen_variant = variant or default_variant(dataset)
    sequences = [sequence] if sequence else list(available_sequences(dataset, chosen_variant))
    if not sequences:
        _fail(
            f"{dataset}/{chosen_variant} has no replayable sequences. Run: "
            f"anti-uav convert --dataset {dataset}"
        )

    if dump is not None:
        dump.mkdir(parents=True, exist_ok=True)

    detector = Detector(run)
    results = []
    try:
        for seq in sequences:
            result = replay_sequence(
                detector,
                dataset,
                chosen_variant,
                seq,
                modality=modality,
                tracker_name=tracker_name,
                conf_floor=conf_floor,
                max_frames=max_frames,
                with_embeddings=not no_embeddings,
                reid_checkpoint=reid,
                dump_per_frame=dump is not None,
            )
            if dump is not None:
                path = dump / f"{dataset}_{seq}_{tracker_name.value}.txt"
                result.dump_path = str(dump_predictions(result, path))
            results.append(result)
    finally:
        detector.close()

    payload: list[dict[str, Any]] = [
        {
            "sequence": r.sequence,
            "tracker": r.tracker,
            "frames": r.frames,
            "detections": r.detections,
            "tracks_created": r.tracks_created,
            "tracks_confirmed": r.tracks_confirmed,
            "metrics": (
                {
                    "mota": round(r.metrics.mota, 4),
                    "idf1": round(r.metrics.idf1, 4),
                    "hota": round(r.metrics.hota, 4),
                    "id_switches": r.metrics.id_switches,
                    "fragmentations": r.metrics.fragmentations,
                    "track_length_avg": round(r.metrics.track_length_avg, 1),
                    "anti_uav_accuracy": (
                        round(r.metrics.anti_uav_accuracy, 4)
                        if r.metrics.anti_uav_accuracy is not None
                        else None
                    ),
                }
                if r.metrics is not None
                else None
            ),
            "warnings": list(r.warnings),
            "error": r.error,
            **({"dump_path": r.dump_path} if r.dump_path else {}),
        }
        for r in results
    ]

    if as_json:
        _emit_json(payload)
        return

    for entry in payload:
        if entry["error"]:
            _fail(f"{entry['sequence']}: {entry['error']}")
        console.print(
            f"[bold]{entry['sequence']}[/]  tracker={entry['tracker']}  "
            f"frames={entry['frames']}  detections={entry['detections']}  "
            f"tracks={entry['tracks_created']} "
            f"({entry['tracks_confirmed']} confirmed)"
        )
        metrics = entry["metrics"]
        if metrics is not None:
            console.print(
                f"  MOTA={metrics['mota']:.4f}  IDF1={metrics['idf1']:.4f}  "
                f"HOTA={metrics['hota']:.4f}  IDSW={metrics['id_switches']}  "
                f"frag={metrics['fragmentations']}"
            )
        for warning in entry["warnings"]:
            _warn(warning)
        if entry.get("dump_path"):
            _ok(f"MOT rows -> {entry['dump_path']}")


@app.command()
def export(
    run: Annotated[str, typer.Option(help="Run to export from.")],
    formats: Annotated[str, typer.Option(help="onnx, tensorrt")] = "onnx,tensorrt",
    imgsz: Annotated[int, typer.Option] = 640,
    half: Annotated[bool, typer.Option("--fp16", help="FP16 engine (Jetson only).")] = False,
    simplify: Annotated[bool, typer.Option("--simplify")] = False,
    int8: Annotated[bool, typer.Option("--int8", help="Needs --calib-data.")] = False,
    calib: Annotated[str | None, typer.Option(help="Calibration image dir for INT8.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Export ONNX and/or TensorRT, and render the matching nvinfer config."""
    from .detection.exporter import export as _export
    from .detection.exporter import write_deepstream_config

    result = _export(
        run,
        formats=_comma_list(formats),
        imgsz=imgsz,
        half=half,
        simplify=simplify,
        int8=int8,
        calib_data=calib,
        dry_run=dry_run,
    )
    console.print(result.describe(), markup=False)

    if result.ok:
        config = write_deepstream_config(
            run, imgsz=imgsz, half=half, class_names=("drone", "bird")
        )
        _ok(f"nvinfer config: {config}")
        _warn("build the engine with trtexec on the Jetson; TensorRT is usually absent here")

    if result.errors:
        raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# sub-apps
# --------------------------------------------------------------------------- #


@matrix_app.command("list")
def matrix_list(
    models: Annotated[str | None, typer.Option(help="Comma list.")] = None,
    combos: Annotated[str | None, typer.Option(help="Comma list.")] = None,
    profile: Annotated[str | None, typer.Option] = None,
    epochs: Annotated[int | None, typer.Option] = None,
    require_dataset: Annotated[bool, typer.Option("--no-require-dataset")] = False,
) -> None:
    """Show every planned run, with the resolved settings."""
    from .detection.matrix import build_matrix_plan, capability_report, format_plan

    plan = build_matrix_plan(
        _matrix(),
        models=_comma_list(models) or None,
        combos=_comma_list(combos) or None,
        profile=profile,
        epochs=epochs,
        require_dataset=require_dataset,
    )
    console.print(format_plan(plan), markup=False)
    console.print(capability_report(), markup=False)


@matrix_app.command("run")
def matrix_run(
    models: Annotated[str | None, typer.Option(help="Comma list.")] = None,
    combos: Annotated[str | None, typer.Option(help="Comma list.")] = None,
    profile: Annotated[str | None, typer.Option] = None,
    epochs: Annotated[int | None, typer.Option] = None,
    imgsz: Annotated[int | None, typer.Option] = None,
    batch: Annotated[int | None, typer.Option] = None,
    device: Annotated[str | None, typer.Option] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    stop_on_failure: Annotated[bool, typer.Option("--stop-on-failure")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print commands, train nothing.")] = False,
) -> None:
    """Execute the matrix in order: single-dataset runs first, cumulative last.

    Single-dataset first is deliberate - if a source is broken you find out on the
    cheap run rather than after the multi-day cumulative ones.
    """
    from .detection.matrix import build_matrix_plan, execute, format_plan, summary_table

    plan = build_matrix_plan(
        _matrix(),
        models=_comma_list(models) or None,
        combos=_comma_list(combos) or None,
        profile=profile,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
    )
    console.print(format_plan(plan), markup=False)

    if dry_run:
        _warn("dry run: nothing trained")
        return

    if not plan.plans:
        _fail("no runs to execute")
        return

    results = list(
        execute(
            plan,
            dry_run=False,
            stop_on_failure=stop_on_failure,
            resume=resume,
            device=device,
        )
    )
    console.print()
    console.print(summary_table(results), markup=False)


@rules_app.command("show")
def rules_show() -> None:
    """Print the active thresholds, exactly as ``drone_rules.yaml`` has them."""
    path = _config_file("rules", "drone_rules.yaml")
    console.print(path.read_text(encoding="utf-8"), markup=False)


@rules_app.command("validate")
def rules_validate() -> None:
    """Check the rule set for cross-rule inconsistencies."""
    from .rules import validate_rules

    problems = validate_rules()
    if not problems:
        _ok("no cross-rule problems found")
        return
    for problem in problems:
        _warn(problem)
    raise typer.Exit(1)


@rules_app.command("explain")
def rules_explain(
    track: Annotated[str, typer.Option(help="A recorded session JSONL, or '-' to explain the schema.")],
) -> None:
    """Run the rule engine over recorded tracks and explain every verdict."""
    from .rules import build_engine, format_evaluation
    from .utils.io import read_jsonl

    engine = build_engine()
    if track == "-":
        console.print(_RULES_HELP, markup=False)
        return

    path = Path(track)
    if not path.is_file():
        _fail(f"no such file: {path}")
        raise typer.Exit(1)

    count = 0
    for record in read_jsonl(path):
        subject = _record_to_track(record)
        if subject is None:
            continue
        evaluation = engine.evaluate(subject, timestamp_s=subject.last_timestamp_s)
        console.print(format_evaluation(evaluation), markup=False)
        count += 1
        if count >= 20:
            break

    if not count:
        _warn("no usable records. Expected jsonl with box/confidence/hits per line.")
        raise typer.Exit(1)


@deploy_app.command("render")
def deploy_render(
    output: Annotated[str | None, typer.Option(help="Output dir.")] = None,
    node: Annotated[str | None, typer.Option(help="Render one node only.")] = None,
    rtsp_host: Annotated[str, typer.Option(help="Per-camera host in the stream URI.")] = "192.168.1.10",
    engine: Annotated[str, typer.Option(help="Path to the exported engine.")] = "./model.engine",
) -> None:
    """Render the per-node DeepStream pipelines and the on-edge probes."""
    from .config.loader import load_coverage_map, load_rules
    from .deploy import deploy_dir, render, render_all, summary, write_deepstream_probe_source

    coverage = load_coverage_map()
    rules = load_rules()
    target = Path(output) if output else deploy_dir()

    if node:
        pipelines = [
            render(coverage, rules, node, rtsp_host=rtsp_host, engine_path=engine)
        ]
        pipelines[0].write(target / f"pipeline_{node}.txt")
    else:
        pipelines = render_all(
            coverage, rules, target, rtsp_host=rtsp_host, engine_path=engine
        )

    write_deepstream_probe_source(target / "deepstream_probe.py")

    console.print(summary(pipelines), markup=False)
    _ok(f"wrote {len(pipelines)} pipeline(s) + deepstream_probe.py to {target}")
    _warn("engine paths are placeholders; re-render after anti-uav export")


@config_app.command("show")
def config_show(
    what: Annotated[str, typer.Argument(help="registry, rules, coverage, matrix or a config path.")],
) -> None:
    """Print a config file."""
    path = _config_file(what)
    if not path.is_file():
        _fail(f"no such config: {path}")
        raise typer.Exit(1)
    console.print(path.read_text(encoding="utf-8"), markup=False)


@config_app.command("datasets")
def config_datasets() -> None:
    """The dataset registry as a table."""
    from .config.loader import load_registry

    registry = load_registry()
    table = Table(title="datasets")
    for column in ("alias", "name", "modality", "median px", "birds", "seqs", "frames", "sources"):
        table.add_column(column)

    for alias in registry.aliases:
        spec = registry.get(alias)
        table.add_row(
            alias,
            spec.display_name,
            "/".join(m.value for m in spec.modalities),
            f"{spec.median_target_px:.0f}" if spec.median_target_px else "-",
            "yes" if spec.has_bird_negatives else "NO",
            f"{spec.approx_sequences:,}",
            f"{spec.approx_frames:,}",
            str(len(spec.sources)),
        )
    console.print(table)
    console.print(
        "\n[yellow]bird negatives[/] matter: Anti-UAV and MM-UAV contain none, so precision "
        "measured on them\ncannot falsify a false-positive claim."
    )


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #


@app.command()
def serve(
    host: Annotated[str | None, typer.Option] = None,
    port: Annotated[int | None, typer.Option] = None,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    """Serve the operator UI and API."""
    import uvicorn

    from .config.loader import load_settings

    settings = load_settings()
    bind_host = host or settings.api_host
    bind_port = port or settings.api_port

    _ok(f"http://{bind_host}:{bind_port}")
    uvicorn.run(
        "anti_uav.api.app:app", host=bind_host, port=bind_port, reload=reload, log_level="info"
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _matrix():
    from .config.loader import load_matrix

    return load_matrix()


def _config_file(*parts: str) -> Path:
    from .config.loader import config_path

    return config_path(*parts)


def _record_to_track(record: dict[str, Any]):
    from .tracking.types import Track, TrackState

    box = record.get("box")
    if not box or len(box) < 4:
        return None
    return Track(
        track_id=int(record.get("track_id", 0)),
        class_id=int(record.get("class_id", 0)),
        class_name=str(record.get("class_name", "drone")),
        camera_id=str(record.get("camera_id", "")),
        node=str(record.get("node", "")),
        state=TrackState.CONFIRMED,
        confirmed=True,
        box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
        confidence=float(record.get("confidence", 0.0)),
        confidence_ema=float(record.get("confidence", 0.0)),
        hits=int(record.get("hits", 0)),
        misses=int(record.get("misses", 0)),
        global_id=record.get("global_id"),
        image_height_px=int(record.get("image_height_px", 0)),
        ground_xy=tuple(record["ground_xy"]) if record.get("ground_xy") else None,
        ground_z=record.get("ground_z"),
        velocity_m_s=tuple(record.get("velocity_m_s", (0.0, 0.0))),
    )


_RULES_HELP = """The rule layer consumes, per track per frame:

  box        [x1, y1, x2, y2] in pixels            (required)
  confidence 0..1                                    (required)
  hits, misses, duration_s                          (persistence)
  camera_id, image_height_px                        (spatial horizon test)
  global_id, ground_xy, velocity_m_s, ground_z      (cross-camera, kinematics)
  history / history_timestamps / history_heights    (turn rate, hover)

Anything optional is reported as a *skipped* gate rather than a passing one, so a
missing calibration never reads as a satisfied rule.
"""


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/]")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
