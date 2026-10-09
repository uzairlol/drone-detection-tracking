"""Local (per-camera) trackers.

These are the runnable stand-ins for the block diagram's ``nvtracker`` element.
NvDCF itself ships only as a DeepStream config artifact
(``deploy/deepstream/nvtracker_nvdcf.cfg``) because it requires the DeepStream
plugin runtime and cannot be evaluated on a dev box.

Measured on this project's data, in increasing order of capability:

===========  ==============================  =========================================
tracker      what it adds                    when it is the right choice
===========  ==============================  =========================================
sort         nothing (IoU only)              the reference point; cheapest on Jetson
bytetrack    a low-confidence second pass    small targets that dip below threshold
botsort      a ReID appearance gate          crossings, and re-entry after occlusion
===========  ==============================  =========================================

Import :func:`build_tracker` rather than the classes directly, so a typo becomes a
useful error message instead of a confusing one later.
"""

from __future__ import annotations

from .base import BaseTracker, box_arrays, build_tracker, iou_cost
from .botsort import BotSortTracker, ablation_pair, cosine_similarity_matrix, fuse_costs
from .bytetrack import ByteTrackTracker
from .sort import IoUOnlyTracker, SortTracker, iou_thresholds_for, predict_next

#: All locally-runnable trackers, in the order the comparison table presents them.
LOCAL_TRACKERS: tuple[str, ...] = ("sort", "bytetrack", "botsort")

__all__ = [
    "LOCAL_TRACKERS",
    "BaseTracker",
    "BotSortTracker",
    "ByteTrackTracker",
    "IoUOnlyTracker",
    "SortTracker",
    "ablation_pair",
    "box_arrays",
    "build_tracker",
    "cosine_similarity_matrix",
    "fuse_costs",
    "iou_cost",
    "iou_thresholds_for",
    "predict_next",
]
