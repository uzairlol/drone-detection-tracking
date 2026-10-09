"""DeepStream and TensorRT deployment artifacts.

The deployment target is the edge fleet, not this development machine, so most of
this package *generates* configuration rather than running it:

``pipeline``     render the per-node ``deepstream-app`` pipeline
``probes``       render the on-edge pad / analytic probes
``nvtracker_nvdcf.cfg``  the tracker config (hand-written, heavily commented)
``exporter``     in :mod:`anti_uav.detection.exporter` - ONNX + TensorRT

The Python trackers are the *measured* stand-in; NvDCF is what would be deployed.
See the header of ``nvtracker_nvdcf.cfg`` for why those are not the same claim.
"""

from __future__ import annotations

from pathlib import Path

from .pipeline import (
    NodePipeline,
    cameras_for_node,
    nodes,
    render,
    render_all,
    rtsp_uri,
    summary,
)
from .probes import (
    ProbeState,
    make_analytics_probe,
    make_pad_probe,
    summarise,
    write_deepstream_probe_source,
)

#: Files shipped alongside the generated pipelines.
NVDC_PATH = Path(__file__).parent / "deepstream" / "nvtracker_nvdcf.cfg"


def deploy_dir() -> Path:
    """``artifacts/deploy`` - where rendered pipelines land."""
    from ..utils.paths import subdir

    return subdir("deploy")


def nvdcf_config_path() -> Path:
    return NVDC_PATH


__all__ = [
    "NVDC_PATH",
    "NodePipeline",
    "ProbeState",
    "cameras_for_node",
    "deploy_dir",
    "make_analytics_probe",
    "make_pad_probe",
    "nodes",
    "nvdcf_config_path",
    "render",
    "render_all",
    "rtsp_uri",
    "summarise",
    "summary",
    "write_deepstream_probe_source",
]
