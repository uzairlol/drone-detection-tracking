"""The drone rule / threshold layer.

Section 4 of the system block diagram. ``drone_rules.yaml`` is the single,
centrally tunable source of every threshold here; the per-node DeepStream configs
and the global coordinator are both rendered from it.

The division of labour that makes this work: **the detector is permissive and this
layer is strict.** ``configs/app.yaml`` runs inference at ``conf_threshold: 0.25``
so the tracker can hold a track through a low-confidence stretch, and every
decision that matters is made here at ``confidence.initiate: 0.60``.
"""

from __future__ import annotations

from .engine import (
    RuleConfig,
    RuleEngine,
    RuleEvaluation,
    RuleOutcome,
    RuleResult,
    Severity,
    build_engine,
    format_evaluation,
    severity_order,
    validate_rules,
)

__all__ = [
    "RuleConfig",
    "RuleEngine",
    "RuleEvaluation",
    "RuleOutcome",
    "RuleResult",
    "Severity",
    "build_engine",
    "format_evaluation",
    "severity_order",
    "validate_rules",
]
