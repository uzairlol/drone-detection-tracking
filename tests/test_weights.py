"""Pretrained base-weight resolution.

The failure these guard against is quiet and expensive: ultralytics fetches a
missing checkpoint from inside the trainer, so a machine with no route to the
release host fails *after* the dataset is built and the run has started. These
tests pin the resolution order and the reporting, with no network access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anti_uav.detection import weights


@pytest.fixture
def isolated_search(monkeypatch, tmp_path: Path):
    """Confine `locate` to one empty directory.

    The real search order ends at the project root, and this repo genuinely has a
    `yolo11n.pt` sitting in it. Without this, an "absent checkpoint" test would
    quietly find the real one and assert nothing useful.
    """
    monkeypatch.setattr(weights, "_search_dirs", lambda: [tmp_path])
    return tmp_path


class TestLocate:
    def test_absolute_path_is_honoured_as_given(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "custom.pt"
        checkpoint.write_bytes(b"weights")
        assert weights.locate(str(checkpoint)) == checkpoint

    def test_absolute_path_that_does_not_exist_resolves_to_none(self, tmp_path: Path) -> None:
        # Deliberately NOT falling back to a same-named file elsewhere: an
        # explicit path that is wrong must be reported wrong.
        assert weights.locate(str(tmp_path / "absent.pt")) is None

    def test_relative_explicit_path_does_not_fall_back(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "yolo11n.pt").write_bytes(b"weights")
        assert weights.locate("./missing.pt") is None

    def test_bare_filename_is_found_in_the_working_directory(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "yolo11n.pt").write_bytes(b"weights")
        assert weights.locate("yolo11n.pt") == tmp_path / "yolo11n.pt"

    def test_bare_filename_is_found_in_the_project_root(
        self, tmp_path: Path, monkeypatch, isolated_search: Path
    ) -> None:
        # Search order is cwd, then project root, then ultralytics' weights dir.
        # Here only the project root is searched, which stands in for it.
        elsewhere = isolated_search / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        (isolated_search / "rtdetr-x2.pt").write_bytes(b"weights")
        assert weights.locate("rtdetr-x2.pt") == isolated_search / "rtdetr-x2.pt"

    def test_absent_bare_filename_is_none(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert weights.locate("definitely-not-here.pt") is None


class TestTargets:
    def test_every_matrix_family_is_reported(self) -> None:
        found = weights.targets()
        families = {t.family for t in found}
        assert {"yolo11n", "rtdetr_x2"} <= families

    def test_unknown_family_is_rejected(self) -> None:
        # Silently reporting fewer targets than the matrix would train is exactly
        # the kind of quiet gap this module exists to close.
        with pytest.raises(KeyError, match="unknown model"):
            weights.targets(["yolo11n", "yolo99x"])

    def test_subset_filter_narrows_the_list(self) -> None:
        found = weights.targets(["yolo11n"])
        assert [t.family for t in found] == ["yolo11n"]

    def test_targets_are_deduplicated_by_checkpoint(self) -> None:
        # Two families may legitimately share one checkpoint; report it once.
        names = [t.name for t in weights.targets()]
        assert len(names) == len(set(names))

    def test_present_is_false_when_the_file_is_absent(self, isolated_search: Path) -> None:
        absent = [t for t in weights.targets() if not t.present]
        assert {t.name for t in absent} == {"yolo11n.pt", "rtdetr-x2.pt"}

    def test_present_is_true_once_the_file_lands(self, isolated_search: Path) -> None:
        (isolated_search / "yolo11n.pt").write_bytes(b"x" * 1024)
        found = {t.name: t for t in weights.targets()}
        assert found["yolo11n.pt"].present
        assert found["yolo11n.pt"].size_mb == pytest.approx(1024 / 1024**2, rel=1e-3)
        assert found["yolo11n.pt"].to_dict()["present"] is True

    def test_missing_helper_agrees_with_targets(self, isolated_search: Path) -> None:
        assert len(weights.missing()) == len(weights.targets())
        (isolated_search / "rtdetr-x2.pt").write_bytes(b"weights")
        assert [t.name for t in weights.missing()] == ["yolo11n.pt"]


class TestFetch:
    def test_dry_run_downloads_nothing(
        self, tmp_path: Path, monkeypatch, isolated_search: Path
    ) -> None:
        def explode(*_args, **_kwargs):  # pragma: no cover - must never run
            raise AssertionError("dry-run must not construct YOLO or hit the network")

        monkeypatch.setattr("ultralytics.YOLO", explode, raising=False)

        found, errors = weights.fetch(dry_run=True)
        assert errors == []
        assert all(not t.present for t in found)

    def test_everything_present_is_a_no_op(
        self, tmp_path: Path, monkeypatch, isolated_search: Path
    ) -> None:
        for name in ("yolo11n.pt", "rtdetr-x2.pt"):
            (isolated_search / name).write_bytes(b"weights")

        def explode(*_args, **_kwargs):  # pragma: no cover - must never run
            raise AssertionError("nothing to fetch, so YOLO must never be constructed")

        monkeypatch.setattr("ultralytics.YOLO", explode, raising=False)

        found, errors = weights.fetch()
        assert errors == []
        assert all(t.present for t in found)

    def test_a_failed_download_is_reported_not_raised(
        self, tmp_path: Path, monkeypatch, isolated_search: Path
    ) -> None:
        def boom(*_args, **_kwargs):
            raise ConnectionError("no route to the release host")

        monkeypatch.setattr("ultralytics.YOLO", boom, raising=False)

        found, errors = weights.fetch()
        assert len(errors) == len(found)
        assert all("ConnectionError" in message for message in errors)
        assert all(not t.present for t in found)
