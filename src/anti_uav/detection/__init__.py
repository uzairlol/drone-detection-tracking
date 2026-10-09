"""Detection: training, inference, export, evaluation.

Two model families share this pipeline:

* ``yolo11n``    - the edge candidate, matching the ``nvinfer`` element in the
  system block diagram.
* ``rtdetr_x2``  - the accuracy reference.

Same data, same splits, same evaluator, same tracker. Only the recipe differs.
"""

from __future__ import annotations

from .evaluate import (
    EvalResult,
    evaluate_cross_dataset,
    evaluate_run,
    format_cross_table,
    format_result,
)
from .exporter import ExportResult, export, write_deepstream_config
from .matrix import MatrixPlan, build_matrix_plan, execute, format_plan, summary_table
from .predictor import Detection, Detector, FrameResult, overlay, resolve_weights
from .trainer import (
    TrainingResult,
    describe_plan,
    list_runs,
    plan_run,
    torch_environment,
    train,
)

__all__ = [
    "Detection",
    "Detector",
    "EvalResult",
    "ExportResult",
    "FrameResult",
    "MatrixPlan",
    "TrainingResult",
    "build_matrix_plan",
    "describe_plan",
    "evaluate_cross_dataset",
    "evaluate_run",
    "execute",
    "export",
    "format_cross_table",
    "format_plan",
    "format_result",
    "list_runs",
    "overlay",
    "plan_run",
    "resolve_weights",
    "summary_table",
    "torch_environment",
    "train",
    "write_deepstream_config",
]
