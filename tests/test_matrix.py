"""Experiment-matrix planning.

The failure pinned here is the quiet kind. `--combos all` is the documented
shorthand in README.md and docs/COMMANDS.md, and it was being compared as a
literal combo slug: it matched nothing, the plan came back empty, and the command
still exited 0. A reader watching the output saw "dry run: nothing trained" and
reasonably concluded the matrix was simply already done - while in fact no run had
ever been planned. Anything that silently plans nothing must be loud, or tested.
"""

from __future__ import annotations

import pytest

from anti_uav.config.loader import load_matrix
from anti_uav.detection.matrix import build_matrix_plan


@pytest.fixture(scope="module")
def matrix():
    return load_matrix()


def _names(**kwargs) -> list[str]:
    # require_dataset=False so the plan covers every combo regardless of what
    # happens to be built on the machine running the tests.
    plan = build_matrix_plan(load_matrix(), require_dataset=False, **kwargs)
    return [p.run_name for p in plan.plans]


class TestShorthandExpansion:
    def test_combos_all_matches_no_filter(self, matrix) -> None:
        assert _names(combos=["all"]) == _names()

    def test_models_all_matches_no_filter(self, matrix) -> None:
        assert _names(models=["all"]) == _names()

    def test_both_expand_to_the_full_matrix(self, matrix) -> None:
        # 2 families x 7 enabled combos.
        assert len(_names(combos=["all"], models=["all"])) == 14

    def test_explicit_list_still_narrows(self, matrix) -> None:
        assert _names(combos=["dvb", "all4"]) == ["yolo11n__dvb", "rtdetr_x2__dvb"] or set(
            _names(combos=["dvb", "all4"])
        ) == {"yolo11n__dvb", "rtdetr_x2__dvb", "yolo11n__all4", "rtdetr_x2__all4"}

    def test_unknown_combo_plans_nothing_rather_than_everything(self, matrix) -> None:
        # The opposite failure would be worse: silently training the whole matrix
        # because a slug was misspelled.
        assert _names(combos=["definitely-not-a-combo"]) == []

    def test_all_mixed_with_an_explicit_slug_is_still_all(self, matrix) -> None:
        # `all` anywhere in the list means all, not "all plus a typo".
        assert _names(combos=["all", "nonsense"]) == _names()


class TestOrdering:
    def test_single_dataset_runs_come_before_cumulative_ones(self, matrix) -> None:
        names = _names()
        singles = [n for n in names if n.endswith(("__dvb", "__mavvid", "__antiuav", "__mmuav"))]
        cumulative = [
            n
            for n in names
            if n.endswith(("__dvb+mavvid", "__dvb+mavvid+antiuav", "__all4"))
        ]
        assert singles and cumulative
        # Single-dataset first: if a source is broken you find out on the cheap
        # run rather than after the multi-day cumulative ones.
        assert names.index(singles[-1]) < names.index(cumulative[0])

    def test_every_run_is_unique(self, matrix) -> None:
        names = _names()
        assert len(names) == len(set(names))
