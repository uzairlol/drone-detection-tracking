"""Dataset acquisition backends.

One dispatcher per distribution channel, because the four datasets are
distributed four different ways. See :mod:`anti_uav.data.download.backends`.
"""

from __future__ import annotations

from .backends import (
    BaiduBackend,
    DownloadBackend,
    DownloadPlan,
    DownloadResult,
    GdriveBackend,
    GitBackend,
    HttpBackend,
    KaggleBackend,
    ManualBackend,
    ModelScopeBackend,
    backend_for,
    download,
    plan_for,
    verify_payload,
)

__all__ = [
    "BaiduBackend",
    "DownloadBackend",
    "DownloadPlan",
    "DownloadResult",
    "GdriveBackend",
    "GitBackend",
    "HttpBackend",
    "KaggleBackend",
    "ManualBackend",
    "ModelScopeBackend",
    "backend_for",
    "download",
    "plan_for",
    "verify_payload",
]