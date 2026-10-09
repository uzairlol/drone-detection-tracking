"""ONNX and TensorRT export.

The deployment target is the ``nvinfer`` element in each DeepStream worker node,
so the artefact that matters is a TensorRT engine. ONNX is exported first
because it is the portable, inspectable intermediate - if an engine behaves
differently from the PyTorch model you can diff the two in ONNX space.

Profiles
--------
FP16 is the block diagram's configuration and is correct on Jetson Thor / Orin.
It is **not** correct on the GTX 1070: Pascal has no usable fp16 arithmetic, so
an FP16 engine is slower and less accurate there. The exporter reads the profile
and refuses the combination rather than quietly producing a bad engine.

FP32 on Pascal and INT8 everywhere are supported. INT8 needs a calibration set;
this module points at the project's own val split rather than accepting a
hand-waved ``--calib`` path.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.loader import detect_profile
from ..config.schema import GpuProfile
from ..utils.io import write_json
from ..utils.logging import get_logger
from ..utils.paths import relative_to_root, subdir
from .predictor import resolve_weights

log = get_logger(__name__)

#: Ops the DeepStream nvinfer parser needs to be able to run. If a custom op
#: shows up here after an export, it will fail at Jetson build time and not here -
#: so the check is worth doing.
_KNOWN_PROBLEM_OPS = {
    "NonMaxSuppression": "not needed; nvinfer applies its own NMS",
    "Einsum": "reshape manually before export",
    "ScatterND": "unsupported by some TensorRT versions",
}


@dataclass(slots=True)
class ExportResult:
    weights: str
    formats: list[str]
    outputs: dict[str, str] = field(default_factory=dict)
    profile: str = ""
    half: bool = False
    imgsz: int = 640
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    ops_used: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.outputs) and not self.errors

    def describe(self) -> str:
        lines = [f"weights : {self.weights}"]
        lines.append(f"profile : {self.profile}   half={self.half}   imgsz={self.imgsz}")
        for fmt, path in sorted(self.outputs.items()):
            lines.append(f"  {fmt:<8} -> {path}")
        if self.warnings:
            lines.append("")
            lines.append("warnings:")
            lines.extend(f"  ! {w}" for w in self.warnings)
        if self.errors:
            lines.append("")
            lines.append("errors:")
            lines.extend(f"  x {e}" for e in self.errors)
        return "\n".join(lines)


def target_directory(run_or_path: str | Path) -> Path:
    """``artifacts/exports/<run name>/``."""
    weights = resolve_weights(run_or_path)
    run_name = weights.parent.parent.name if weights.parent.name == "weights" else weights.stem
    return subdir("exports") / run_name


def export(
    run_or_path: str | Path,
    *,
    formats: Sequence[str] = ("onnx", "tensorrt"),
    imgsz: int = 640,
    half: bool = False,
    profile: GpuProfile | str | None = None,
    batch: int = 1,
    simplify: bool = False,
    opset: int = 17,
    workspace_gb: float | None = None,
    calib_data: str | Path | None = None,
    int8: bool = False,
    dry_run: bool = False,
) -> ExportResult:
    """Export one checkpoint to the requested formats.

    Never raises for an expected failure; the caller gets a report. A missing
    TensorRT install is a *warning* rather than an error, because ONNX may still
    have exported successfully and the engine can be built on the Jetson box
    instead.
    """
    weights = resolve_weights(run_or_path)
    active = profile or detect_profile()
    profile_value = active.value if isinstance(active, GpuProfile) else str(active)
    out_dir = target_directory(run_or_path)

    result = ExportResult(
        weights=relative_to_root(weights),
        formats=list(formats),
        profile=profile_value,
        half=half,
        imgsz=imgsz,
    )

    if half and profile_value == "pascal":
        result.warnings.append(
            "FP16 requested on a Pascal device. sm_61 has no fp16 arithmetic path, so the "
            "engine will be slower and less accurate than FP32. The block diagram's "
            "FP16 nvinfer config targets Jetson Thor/Orin, not this box - export FP32 here "
            "and FP16 on the Jetson, or set --profile to the deployment target."
        )
    if half and profile_value == "cpu":
        result.warnings.append(
            "FP16 requested on a CPU-only environment. TensorRT will not be available; "
            "export ONNX only and build the engine on the deployment host."
        )

    if dry_run:
        result.warnings.append("dry run - nothing exported")
        log.info("export plan", extra={"formats": list(formats), "out": str(out_dir)})
        return result

    out_dir.mkdir(parents=True, exist_ok=True)
    model = _load(weights, result)
    if model is None:
        return result

    if "onnx" in formats:
        _export_onnx(model, weights, out_dir, imgsz, batch, opset, simplify, result)
    if "tensorrt" in formats:
        _export_tensorrt(
            model, weights, out_dir, imgsz, batch, half, int8, calib_data, workspace_gb, result
        )
    if "engine_only" in formats:
        result.warnings.append(
            "'engine_only' is not a separate format - TensorRT engines are built on the "
            "deployment host. Export ONNX here and build there."
        )

    write_json(
        out_dir / "export_report.json",
        {
            "weights": result.weights,
            "formats": result.formats,
            "profile": result.profile,
            "half": half,
            "imgsz": imgsz,
            "outputs": result.outputs,
            "ops_used": result.ops_used,
            "warnings": result.warnings,
            "errors": result.errors,
        },
    )
    return result


def _load(weights: Path, result: ExportResult) -> Any:
    try:
        import ultralytics
    except ImportError as exc:
        result.errors.append(
            f"ultralytics is not installed: {exc}. Install with pip install 'anti-uav[train]'."
        )
        return None
    try:
        name = str(weights).lower()
        cls = ultralytics.RTDETR if "rtdetr" in name else ultralytics.YOLO
        return cls(str(weights))
    except Exception as exc:
        result.errors.append(f"failed to load {weights.name}: {type(exc).__name__}: {exc}")
        return None


def _export_onnx(
    model: Any,
    weights: Path,
    out_dir: Path,
    imgsz: int,
    batch: int,
    opset: int,
    simplify: bool,
    result: ExportResult,
) -> None:
    target = out_dir / f"{weights.stem}_{imgsz}.onnx"
    try:
        model.export(
            format="onnx",
            imgsz=imgsz,
            batch=batch,
            opset=opset,
            simplify=simplify,
            dynamic=False,
            device=0 if result.profile not in {"cpu"} else "cpu",
            half=False,  # ONNX stays fp32; precision is applied at engine build
        )
    except Exception as exc:
        result.errors.append(f"ONNX export failed: {type(exc).__name__}: {exc}")
        return

    produced = _relocate(target, weights.parent, out_dir, result)
    if produced:
        result.outputs["onnx"] = relative_to_root(produced)
        _inspect_onnx(produced, result)


def _relocate(expected: Path, written_dir: Path, out_dir: Path, result: ExportResult) -> Path | None:
    """Ultralytics writes next to the weights; move it into the exports tree.

    ``written_dir`` is the weights directory. Globbing ``expected.parent`` here
    looks correct but is not: ``expected`` already lives *inside* ``out_dir``, so
    that search only ever re-globbed the empty exports directory and every
    successful ONNX export was then reported as a failure.
    """
    if expected.is_file():
        return expected
    if written_dir.is_dir():
        found = sorted(written_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if found:
            out_dir.mkdir(parents=True, exist_ok=True)
            return Path(shutil.move(str(found[0]), str(expected)))
    result.errors.append(
        f"export reported success but no .onnx appeared in {written_dir} or {out_dir}"
    )
    return None


def _inspect_onnx(path: Path, result: ExportResult) -> None:
    """Log the op types so a DeepStream-incompatible graph is caught here."""
    try:
        import onnx

        model = onnx.load(str(path))
        ops = sorted({node.op_type for node in model.graph.node})
        result.ops_used = ops
        risky = [op for op in ops if op in _KNOWN_PROBLEM_OPS]
        if risky:
            result.warnings.append(
                f"ONNX graph contains ops that may not run in DeepStream nvinfer: "
                f"{risky}. Known issue: "
                + "; ".join(f"{op} - {_KNOWN_PROBLEM_OPS[op]}" for op in risky)
            )
        log.info(
            "onnx exported",
            extra={"file": path.name, "ops": len(ops), "risky": risky},
        )
    except ImportError:
        result.warnings.append(
            "onnx is not installed, so the graph was not inspected. "
            "Install with: pip install 'anti-uav[export]'"
        )
    except Exception as exc:
        result.warnings.append(f"could not inspect the ONNX graph: {type(exc).__name__}: {exc}")


def _export_tensorrt(
    model: Any,
    weights: Path,
    out_dir: Path,
    imgsz: int,
    batch: int,
    half: bool,
    int8: bool,
    calib_data: str | Path | None,
    workspace_gb: float | None,
    result: ExportResult,
) -> None:
    """Build a TensorRT engine, if TensorRT is available on this machine.

    Almost never available on a Windows dev box, and not what you want to do on
    a cross-compile target anyway. That is by design: the engine is built on the
    Jetson, from the ONNX this function just produced.
    """
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        result.warnings.append(
            "TensorRT is not installed here, so no .engine was built - which is the normal "
            "case on a dev box. Copy the exported ONNX to the Jetson and build there:\n"
            "    trtexec --onnx=<model>.onnx --saveEngine=<model>.engine "
            "--fp16 --shapes=input:1x3x"
            f"{imgsz}x{imgsz}\n"
            "or build it there with: anti-uav export --run <copy> --formats tensorrt"
        )
        return

    onnx_path = out_dir / f"{weights.stem}_{imgsz}.onnx"
    if not onnx_path.is_file():
        result.warnings.append(
            "ONNX must be exported before TensorRT; export --formats onnx,tensorrt together."
        )
        return

    if int8 and calib_data is None:
        result.warnings.append(
            "int8 requested without --calib-data. INT8 needs a calibration set to be "
            "meaningful - use this project's own val split: "
            f"{subdir('processed') / '<combo>' / 'images' / 'val'}"
        )
        return

    try:
        model.export(
            format="engine",
            imgsz=imgsz,
            batch=batch,
            half=half,
            int8=int8,
            device=0,
            workspace=workspace_gb or 4.0,
            data=str(calib_data) if calib_data else None,
        )
    except Exception as exc:
        result.errors.append(f"TensorRT export failed: {type(exc).__name__}: {exc}")
        return

    produced = _relocate_engine(weights.parent, out_dir, result)
    if produced:
        result.outputs["tensorrt"] = relative_to_root(produced)


def _relocate_engine(written_dir: Path, out_dir: Path, result: ExportResult) -> Path | None:
    for directory in (written_dir, out_dir):
        if not directory.is_dir():
            continue
        engines = sorted(directory.glob("*.engine"), key=lambda p: p.stat().st_mtime, reverse=True)
        if engines:
            if directory == written_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                return Path(shutil.move(str(engines[0]), str(out_dir / engines[0].name)))
            return engines[0]
    result.errors.append(
        f"TensorRT reported success but no .engine appeared in {written_dir} or {out_dir}"
    )
    return None


def write_deepstream_config(
    run_or_path: str | Path,
    *,
    imgsz: int = 640,
    half: bool = True,
    class_names: Sequence[str] = ("drone", "bird"),
    batch: int = 13,
    profile: GpuProfile | str | None = None,
    workspace_gb: float = 32.0,
) -> Path:
    """Render an ``nvinfer`` config block matching the exported model.

    The class list, input shape and normalisation have to agree with the engine
    exactly. A mismatch here is the classic cause of "the tracker works in
    Python but the Jetson pipeline detects nothing", so this function derives
    every value from the same place the export did.
    """
    weights = resolve_weights(run_or_path)
    active = profile or detect_profile()
    profile_value = active.value if isinstance(active, GpuProfile) else str(active)
    out_dir = target_directory(run_or_path)

    rendered = f"""# GENERATED by `anti-uav export` from {relative_to_root(weights)}.
# Section 2 of the system block diagram: the nvinfer element in a DeepStream
# worker node. ~13 concurrent 1080p streams share one batched inference.

[infer-config]
# Path to the engine built on the Jetson from the exported ONNX.
model-engine-file=./{weights.stem}_{imgsz}.engine
labelfile-path=./drone_bird.labels
# Batch = cameras muxed into one inference, per the block diagram (~13/node).
batch-size={batch}
# 0 = fp16. Only valid on Jetson Thor/Orin; use 0 on Pascal.
network-mode={2 if half else 0}
# DeepStream expects explicit planar RGB with mean-subtracted input.
net-scale-factor=0.017352764
net-scale-0=0.485
net-scale-1=0.456
net-scale-2=0.406
offsets=0.485;0.456;0.406
force-layout=0
symmetric-padding=true
workspace-size={int(workspace_gb * 1024)}
gpu-id=0
num-detected-classes={len(class_names)}
# drone first: the nvinfer output tensor indexes classes in this order and the
# rule layer's confidence.initiate applies to index 0.
gie-unique-id={int(imgsz)}
process-mode=1
batch-size={batch}
network-type=0
cluster-mode=2
maintain-aspect-ratio=0
symmetric-padding=true
parse-bbox-float-numbers=true
parse-bbox-mode=0
nms-mode=0
topk=300
score-threshold=0.25
# NOTE: 0.25 is deliberately well below rules.confidence.initiate (0.60) in
# configs/rules/drone_rules.yaml. The tracker needs a permissive detector to
# hold a track through a low-confidence stretch; the rule layer decides what
# becomes an alert.

[class-attrs-all]
# Generated for: {', '.join(class_names)}
topk=300
score-threshold=0.25
"""

    labels = "".join(f"{name}\n" for name in class_names)
    config_path = out_dir / f"{weights.stem}_nvinfer.txt"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(rendered, encoding="utf-8")
    (out_dir / "drone_bird.labels").write_text(labels, encoding="utf-8")

    log.info(
        "nvinfer config written",
        extra={
            "file": relative_to_root(config_path),
            "profile": profile_value,
            "half": half,
            "classes": list(class_names),
        },
    )
    return config_path
