"""Metric extraction from an ultralytics validation result.

The counts are the only part of an evaluation report that can be silently wrong
in a way that looks like a broken model: ``images: 0  gt boxes: 0`` printed next
to a healthy mAP reads as a failed run, and a reader who trusts it concludes the
evaluation covered nothing. So the readers get pinned here against both layouts
ultralytics has shipped, using the attribute names from the real classes rather
than a hand-rolled stub that agrees with whatever the code happens to do.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from anti_uav.detection.evaluate import _count


class TestCount:
    def test_reads_the_modern_histogram_attributes(self) -> None:
        """Current ultralytics exposes per-class arrays on the metrics object."""
        metrics = SimpleNamespace(
            nt_per_class=np.array([8995, 11664]),
            nt_per_image=np.array([2600, 2690]),
        )
        assert _count(metrics, "nt_per_image", "nt") == 5290
        assert _count(metrics, "nt_per_class", "n_gt", "nt") == 20659

    def test_reads_the_legacy_scalar_attribute(self) -> None:
        """Older builds only had a scalar ``nt``, with no histogram at all."""
        assert _count(SimpleNamespace(nt=41), "nt_per_image", "nt") == 41

    def test_missing_attributes_report_zero_rather_than_raising(self) -> None:
        assert _count(SimpleNamespace(), "nt_per_image", "nt") == 0

    def test_a_present_but_null_histogram_is_skipped(self) -> None:
        """``DetMetrics`` initialises both counts to None before ``process()``.

        Reading None as 0 and stopping there would hide the scalar fallback, so a
        null value has to fall through to the next name rather than win.
        """
        metrics = SimpleNamespace(nt_per_image=None, nt_per_class=None, nt=7)
        assert _count(metrics, "nt_per_image", "nt") == 7
