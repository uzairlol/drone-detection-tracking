"""Anti-UAV drone detection & tracking pipeline.

Two model families share one pipeline:

* ``yolo11n``    - YOLO11n (nano), the model the system block diagram puts behind
  ``nvinfer`` in each DeepStream worker node. Edge budget: ~3 ms FP16 on Jetson Thor.
* ``rtdetr_x2``  - RT-DETR-x2, transformer detector, evaluated as the accuracy
  ceiling rather than the edge candidate.

Layout::

    anti_uav.config      typed config schemas + loader
    anti_uav.data        download / ingest / convert / split / build / stats
    anti_uav.detection   training, inference, export, evaluation
    anti_uav.tracking    local MOT, global fusion, handoff + recovery, metrics
    anti_uav.rules       drone_rules.yaml threshold engine
    anti_uav.deploy      DeepStream + TensorRT artifacts
    anti_uav.api         FastAPI service backing the operator UI
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
