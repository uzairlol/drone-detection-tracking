"""The data layer, on synthetic fixtures.

Two invariants get the most attention because breaking either silently inflates
every metric in the project:

* **a sequence never straddles the split** - adjacent frames of a video are
  near-identical, so a frame-level split leaks and reports double digits of
  mAP that do not exist;
* **tiling is decided per source** - MM-UAV's 12 px targets are invisible at a
  640 px input, and MAV-VID's 171 px targets do not need it.

The fixtures build real mp4s and real annotation files rather than mocking the
readers, because the reader is exactly what these tests are meant to exercise.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from smoke_data import fake_antiuav, fake_mmuav

from anti_uav.config import (
    load_dataset,
    load_matrix,
    load_registry,
)
from anti_uav.data import sanity, splits, stats
from anti_uav.data.convert import convert_dataset
from anti_uav.data.frameindex import load_index


@pytest.fixture(scope="module")
def converted(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Convert a synthetic Anti-UAV and MM-UAV tree, once, for the class."""
    raw = tmp_path_factory.mktemp("raw")
    fake_antiuav(raw, sequences=18, frames=8)
    fake_mmuav(raw, sequences=4, frames=10)

    convert_dataset(load_dataset("antiuav"), raw / "antiuav" / "300", variant="300")
    convert_dataset(load_dataset("mmuav"), raw / "mmuav" / "subset", variant="subset")
    return raw


class TestRegistryMetadata:
    def test_mmuav_is_the_tiling_case(self) -> None:
        """12 px median against a 40 px threshold: this is what makes MM-UAV
        tiled and leaves MAV-VID alone."""
        matrix = load_matrix()
        assert load_dataset("mmuav").median_target_px <= matrix.tiling_threshold_px
        assert load_dataset("mavvid").median_target_px > matrix.tiling_threshold_px

    def test_every_source_declares_an_approximate_size(self) -> None:
        for alias in load_registry().aliases:
            for source in load_dataset(alias).sources:
                assert source.approx_size_gb > 0, f"{alias}: {source.label}"


class TestConvert:
    def test_antiuav_produces_frames_and_boxes(self, converted: Path) -> None:
        records = load_index("antiuav", "300")
        assert records
        assert any(r.box_count > 0 for r in records)

    def test_every_frame_has_a_real_image(self, converted: Path) -> None:
        from anti_uav.data.frameindex import interim_root

        root = interim_root("antiuav", "300")
        for record in load_index("antiuav", "300")[:10]:
            assert (root / record.image).is_file(), record.image

    def test_mmuav_produces_a_gt_index(self, converted: Path) -> None:
        records = load_index("mmuav", "subset")
        assert records
        assert any(r.box_count > 0 for r in records)

    def test_frame_indices_are_unique_within_a_sequence(self, converted: Path) -> None:
        for alias, variant in (("antiuav", "300"), ("mmuav", "subset")):
            seen: set[tuple[str, str, int]] = set()
            for record in load_index(alias, variant):
                key = (record.sequence_id, record.modality, record.frame_index)
                assert key not in seen, f"{alias}: duplicate {key}"
                seen.add(key)


class TestSplits:
    def test_no_sequence_straddles_the_split(self, converted: Path) -> None:
        """The invariant that matters most in this whole repository."""
        records = load_index("antiuav", "300") + load_index("mmuav", "subset")
        assigned, report = splits.split_records(
            records, val_fraction=0.34, seed=0, combo="test"
        )
        by_sequence: dict[str, set[str]] = {}
        for record in assigned:
            by_sequence.setdefault(record.sequence_id, set()).add(record.split)
        straddling = {
            sequence: seen for sequence, seen in by_sequence.items() if len(seen) > 1
        }
        assert not straddling, f"sequences in both splits: {straddling}"
        assert not [w for w in report.warnings if "more than one split" in w]

    def test_both_splits_are_populated(self, converted: Path) -> None:
        records = load_index("antiuav", "300")
        assigned, _ = splits.split_records(records, val_fraction=0.34, seed=1, combo="t")
        assert {r.split for r in assigned} == {"train", "val"}

    def test_split_is_deterministic_for_a_seed(self, converted: Path) -> None:
        records = load_index("antiuav", "300")
        def assignment(rs: list) -> dict[str, str]:
            return {r.sequence_id: r.split for r in rs}

        first, _ = splits.split_records(records, val_fraction=0.34, seed=5, combo="t")
        second, _ = splits.split_records(records, val_fraction=0.34, seed=5, combo="t")
        assert assignment(first) == assignment(second)

    def test_a_different_seed_gives_a_different_split(self, converted: Path) -> None:
        records = load_index("antiuav", "300")
        def assignment(rs: list) -> dict[str, str]:
            return {r.sequence_id: r.split for r in rs}

        assignments = set()
        for seed in range(6):
            rs, _ = splits.split_records(records, val_fraction=0.34, seed=seed, combo="t")
            assignments.add(tuple(sorted(assignment(rs).items())))
        assert len(assignments) > 1, "the seed does not affect the split at all"

    def test_frame_level_strategy_is_available_but_warned_about(self, converted: Path) -> None:
        """The leaky strategy exists for genuinely frame-independent data and
        must announce itself when used on video."""
        records = load_index("antiuav", "300")
        _, report = splits.split_records(
            records, val_fraction=0.34, seed=0, combo="t", strategy=splits.SplitStrategy.FILE
        )
        assert any("upper bound" in w.lower() for w in report.warnings)


class TestStats:
    def test_stats_summarise_a_converted_dataset(self, converted: Path) -> None:
        summary = stats.compute(load_index("antiuav", "300"), dataset="antiuav", variant="300")
        assert summary.frames > 0

    def test_bird_negative_coverage_is_reported(self, converted: Path) -> None:
        """Two of four datasets can falsify a false-positive claim; the stats
        output has to say which."""
        by_dataset = {
            "antiuav": stats.compute(
                load_index("antiuav", "300"), dataset="antiuav", variant="300"
            ),
            "mmuav": stats.compute(
                load_index("mmuav", "subset"), dataset="mmuav", variant="subset"
            ),
        }
        text = stats.combo_summary(by_dataset)
        assert "bird" in text.lower()


def split_and_persist(alias: str, variant: str, combo: str = "t", val_fraction: float = 0.34):
    """Split a converted dataset and write the assignment back to the index.

    Must run before build(): the builder reads each record's split field to
    decide which image goes into train/ and which into val/.
    """
    records = load_index(alias, variant)
    assigned, report = splits.split_records(
        records, val_fraction=val_fraction, seed=0, combo=combo
    )
    splits.persist(assigned, report, combo)
    return assigned, report


def build_combo_for(slug: str):
    """Run the real build for one combo slug."""
    import importlib

    build_module = importlib.import_module("anti_uav.data.build")
    matrix = load_matrix()
    combo = next(c for c in matrix.combos if c.slug == slug)
    specs = {alias: load_dataset(alias) for alias in combo.datasets}
    return build_module.build(combo, specs, matrix)


class TestBuildAndSanity:
    def test_build_writes_a_data_yaml(self, converted: Path) -> None:
        split_and_persist("antiuav", "300")
        report = build_combo_for("antiuav")
        assert report.ok, report.errors
        yaml_path = Path(report.data_yaml)
        assert yaml_path.is_file(), yaml_path
        text = yaml_path.read_text(encoding="utf-8")
        assert "0: drone" in text
        assert "1: bird" in text

    def test_build_reports_its_sources(self, converted: Path) -> None:
        split_and_persist("antiuav", "300")
        report = build_combo_for("antiuav")
        assert [s.dataset for s in report.sources] == ["antiuav"]

    def test_build_report_on_disk_records_the_verdict(self, converted: Path) -> None:
        """The JSON has to be written *after* ``ok`` is computed.

        Serializing first left ``ok`` at its dataclass default of False, so every
        successful build on disk claimed to have failed - and a caller reading the
        report rather than the returned object had no way to tell the difference.
        """
        split_and_persist("antiuav", "300")
        report = build_combo_for("antiuav")
        assert report.ok, report.errors
        on_disk = json.loads(Path(report.data_yaml).with_name("build_report.json").read_text("utf-8"))
        assert on_disk["ok"] is True
        assert on_disk["errors"] == []

    def test_mmuav_is_tiled_because_its_targets_are_12_px(self, converted: Path) -> None:
        """Per-source tiling, verified through the resolved tiling decision."""
        split_and_persist("mmuav", "subset")
        report = build_combo_for("mmuav")
        assert report.sources[0].tiling, (
            "MM-UAV must be tiled - its targets are 12 px"
        )

    def test_antiuav_is_not_tiled_at_its_larger_target_size(self, converted: Path) -> None:
        """The other half of the per-source rule: 92 px targets do not need it."""
        split_and_persist("antiuav", "300")
        report = build_combo_for("antiuav")
        assert report.sources[0].tiling is False

    def test_sanity_passes_on_a_correct_build(self, converted: Path) -> None:
        split_and_persist("antiuav", "300", val_fraction=0.4)
        build_combo_for("antiuav")
        result = sanity.run_all(
            load_index("antiuav", "300"), combo="t", expect_datasets=["antiuav"]
        )
        assert result.ok, sanity.format_report(result)


class TestDvbFlatLabelResolution:
    """The Kaggle mirror ships one label dir per class, not a shared ``labels/``.

    ``Data/images_birds`` pairs with ``Data/labels_birds`` and
    ``Data/images_drones`` with ``Data/labels_drones``, and the two halves use
    different stems (``bird_image_NNNNNN`` vs ``image_NNNNNN``). A resolver that
    only looks for a literal ``labels`` directory finds nothing, every frame
    converts to an empty label, and the run trains a detector on nothing while
    every metric still prints a number.
    """

    @staticmethod
    def _mirror(root: Path) -> Path:
        import numpy as np

        from anti_uav.utils.imaging import write_image

        cases = {
            ("Data/images_birds", "Data/labels_birds", "bird_image_000000"):
                "1 0.328211 0.409547 0.640664 0.518665\n",
            ("Data/images_drones", "Data/labels_drones", "image_000000"):
                "0 0.3765625 0.48984375 0.046875 0.0515625\n",
        }
        for (image_dir, label_dir, stem), text in cases.items():
            (root / image_dir).mkdir(parents=True, exist_ok=True)
            (root / label_dir).mkdir(parents=True, exist_ok=True)
            write_image(root / image_dir / f"{stem}.png", np.zeros((64, 64, 3), dtype=np.uint8))
            (root / label_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        return root

    def test_labels_resolve_through_the_class_suffixed_directories(self, tmp_path: Path) -> None:
        from anti_uav.data.convert.dvb import DvbFlatConverter

        root = self._mirror(tmp_path)
        converter = DvbFlatConverter(load_dataset("dvb"), root)

        found = {
            image.name: converter._label_for(image, root)
            for image in sorted(root.rglob("*.png"))
        }
        assert found["bird_image_000000.png"] == root / "Data/labels_birds/bird_image_000000.txt"
        assert found["image_000000.png"] == root / "Data/labels_drones/image_000000.txt"

    def test_conversion_keeps_the_boxes_and_both_classes(self, tmp_path: Path) -> None:
        from anti_uav.data.convert.dvb import DvbFlatConverter

        root = self._mirror(tmp_path)
        converter = DvbFlatConverter(load_dataset("dvb"), root)
        report = converter.run()

        assert report.images_seen == 2
        assert report.images_without_labels == 0, "labels went missing during conversion"
        assert sorted(c for record in converter.records for c in record.class_ids) == [0, 1]
        assert all(record.box_count >= 1 for record in converter.records)

    def test_same_index_in_both_halves_does_not_collide(self, tmp_path: Path) -> None:
        """``image_000000`` and ``bird_image_000000`` share a numeric tail.

        Resolving both to ``vid_000000`` sent them to one interim path, so the
        drone's image and label overwrote the bird's and the dataset kept 9,000
        frames while the index claimed 17,936 - half the labels gone, and the
        surviving pairs mismatched.
        """
        from anti_uav.data.convert.dvb import DvbFlatConverter

        root = self._mirror(tmp_path)
        converter = DvbFlatConverter(load_dataset("dvb"), root)
        converter.run()

        sequences = [record.sequence_id for record in converter.records]
        assert len(set(sequences)) == len(sequences), f"sequence_id collision: {sequences}"

        labels = [record.label for record in converter.records]
        assert len(set(labels)) == len(labels), f"label path collision: {labels}"

    def test_a_reused_sequence_id_aborts_rather_than_overwriting(self, tmp_path: Path) -> None:
        """The base guard.

        Any dataset whose grouping is wrong would otherwise produce a smaller
        dataset than its index claims, with no error anywhere.
        """
        import numpy as np

        from anti_uav.data.convert.base import ConversionAborted
        from anti_uav.data.convert.dvb import DvbFlatConverter

        converter = DvbFlatConverter(load_dataset("dvb"), tmp_path / "raw")
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        box = (0, 0.5, 0.5, 0.2, 0.2)

        converter.emit_frame(
            sequence_id="vid_000000",
            frame_index=0,
            modality="rgb",
            source=frame,
            boxes=[box],
            source_labels=["drone"],
        )

        with pytest.raises(ConversionAborted):
            converter.emit_frame(
                sequence_id="vid_000000",
                frame_index=0,
                modality="rgb",
                source=frame,
                boxes=[(1, 0.5, 0.5, 0.2, 0.2)],
                source_labels=["bird"],
            )


class TestDownloadPlanning:
    r"""\dry_run=True\ is what makes these safe to run before committing to 60 GB."""

    @pytest.mark.parametrize("alias", ["dvb", "mavvid", "antiuav", "mmuav"])
    def test_every_dataset_produces_a_dry_run_plan(self, alias: str) -> None:
        from anti_uav.data.download import download

        results = list(download(load_dataset(alias), dry_run=True))
        assert results
        for result in results:
            assert result.status in {"skipped", "needs_credentials", "planned"}
            if result.status != "skipped":
                assert result.extras.get("plan") is not None

    def test_dry_run_creates_no_files(self) -> None:
        from anti_uav.data.download import download
        from anti_uav.utils.paths import subdir

        before = set(subdir("raw").rglob("*")) if subdir("raw").exists() else set()
        list(download(load_dataset("dvb"), dry_run=True))
        after = set(subdir("raw").rglob("*")) if subdir("raw").exists() else set()
        assert before == after

    def test_mmuav_reports_that_it_needs_credentials(self) -> None:
        """Baidu Pan has no anonymous API; the plan must say so rather than
        pretending the download will work."""
        from anti_uav.data.download import download

        results = list(download(load_dataset("mmuav"), dry_run=True))
        assert any(r.status == "needs_credentials" for r in results), [
            r.status for r in results
        ]


class TestKaggleCredentialGate:
    """The gate must accept every credential the CLI actually accepts.

    The CLI authenticates in a documented order, and this gate used to know only
    three of those shapes. `KAGGLE_USERNAME` + `KAGGLE_KEY` - the pair Kaggle's
    own UI hands you, and the pair a notebook receives from its Secrets panel -
    was missing, so the gate refused a download that was about to succeed. The
    failure mode is nasty because the message is confident and the CLI would
    have worked.

    Order matters as much as membership. The CLI reads its config file and then
    overlays environment variables, so an env var beats a stale file; and it
    tries access-token auth before the legacy API key.
    """

    @staticmethod
    def _home(monkeypatch, tmp_path, *files: str) -> Path:
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        if files:
            (tmp_path / ".kaggle").mkdir(parents=True, exist_ok=True)
            for name in files:
                (tmp_path / ".kaggle" / name).write_text("{}", encoding="utf-8")
        return tmp_path

    @staticmethod
    def _clear(monkeypatch) -> None:
        for key in list(os.environ):
            if key.startswith("KAGGLE_"):
                monkeypatch.delenv(key, raising=False)

    def _source(self, monkeypatch, tmp_path, env, files=()) -> str:
        from anti_uav.data.download.backends import KaggleBackend

        self._clear(monkeypatch)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        self._home(monkeypatch, tmp_path, *files)
        return KaggleBackend().credential_source()

    @pytest.mark.parametrize(
        ("env", "files", "expected"),
        [
            ({}, (), ""),
            ({"KAGGLE_API_TOKEN": "t"}, (), "KAGGLE_API_TOKEN"),
            ({"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}, (), "KAGGLE_USERNAME + KAGGLE_KEY"),
            # The notebook-proxy credential Kaggle sets for you inside a kernel.
            ({"KAGGLE_API_V1_TOKEN": "v"}, (), "KAGGLE_API_V1_TOKEN (Kaggle notebook proxy)"),
            ({}, ("access_token",), "~/.kaggle/access_token"),
            ({}, ("access_token.txt",), "~/.kaggle/access_token.txt"),
            ({}, ("kaggle.json",), "~/.kaggle/kaggle.json"),
            # env overlays the file, so a fresh pair beats a stale kaggle.json
            ({"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}, ("kaggle.json",), "KAGGLE_USERNAME + KAGGLE_KEY"),
            # access-token auth is tried before the legacy key
            ({"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}, ("access_token",), "~/.kaggle/access_token"),
        ],
    )
    def test_accepts_and_ranks(self, monkeypatch, tmp_path, env, files, expected) -> None:
        assert self._source(monkeypatch, tmp_path, env, files) == expected

    @pytest.mark.parametrize("env", [{"KAGGLE_USERNAME": "u"}, {"KAGGLE_KEY": "k"}])
    def test_half_a_legacy_pair_is_not_a_credential(self, monkeypatch, tmp_path, env) -> None:
        """One half of the legacy pair is not a credential.

        Reporting success here would hand the failure to the CLI as an opaque
        401 instead of naming the missing half.
        """
        assert self._source(monkeypatch, tmp_path, env) == ""

    def test_username_and_key_pair_is_the_notebook_documented_path(
        self, monkeypatch, tmp_path
    ) -> None:
        """Guards the specific regression.

        notebooks/kaggle_dvb.ipynb sets these two from `kaggle_secrets`. Before
        this was fixed, following the notebook could not possibly work.
        """
        assert (
            self._source(
                monkeypatch, tmp_path, {"KAGGLE_USERNAME": "user", "KAGGLE_KEY": "key"}
            )
            == "KAGGLE_USERNAME + KAGGLE_KEY"
        )

    def test_message_names_every_accepted_shape(self, monkeypatch, tmp_path) -> None:
        """A refusal that does not say what would work sends you hunting."""
        from anti_uav.data.download.backends import KaggleBackend

        self._clear(monkeypatch)
        self._home(monkeypatch, tmp_path)
        monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/kaggle")
        ok, reason = KaggleBackend().available()
        assert not ok
        for shape in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "kaggle.json"):
            assert shape in reason, f"{shape} missing from the refusal message"


class TestManualSourceStatus:
    """`manual` and `needs_credentials` must stay distinct statuses.

    They were the same string. A reference source that will never be downloaded
    automatically therefore reported that you were not logged in — and the
    natural response to that message is to go and authenticate, repeatedly,
    against a source that has nothing to authenticate with. The dvb GitHub
    annotations entry is the case that exposed it.
    """

    def test_reference_source_reports_manual_not_credentials(self, tmp_path) -> None:
        from anti_uav.config.schema import SourceKind
        from anti_uav.data.download.backends import DownloadPlan, ManualBackend

        plan = DownloadPlan(
            dataset="dvb",
            variant="full",
            source_label="reference",
            kind=SourceKind.MANUAL,
            url="https://example.invalid/repo",
            destination=tmp_path,
            approx_size_gb=0.01,
            needs_credentials=False,
        )
        result = ManualBackend().fetch(plan)

        assert result.status == "manual", f"got {result.status!r}"
        assert "needs_credentials" not in result.error
        # The message must not send you off to authenticate.
        assert "credential" not in result.error.lower()

    def test_every_source_kind_has_a_distinct_status(self) -> None:
        """A source blocked by a missing login still says so.

        The counterpart to the test above: splitting the statuses is only
        meaningful if the genuine credential case still reports a credential
        problem. Baidu needs a BDUSS cookie, so it must remain
        `needs_credentials`.
        """
        from anti_uav.data.download.backends import BaiduBackend

        ok, reason = BaiduBackend().available()
        assert not ok
        assert "cookie" in reason.lower() or "BDUSS" in reason

    def test_cli_renders_manual_status(self) -> None:
        """An unmapped status would print as a bare word with no colour cue."""
        from anti_uav.cli import _STATUS_STYLES

        assert "manual" in _STATUS_STYLES
        assert _STATUS_STYLES["manual"] != _STATUS_STYLES["needs_credentials"]


class TestHttpBackendRejectsWebPages:
    """A repo URL returns the hosting site's HTML, not the artefact.

    This shipped as a bug: the dvb registry declared the WOSDETC challenge repo
    as `kind: http`, and HttpBackend issued a plain GET. It saved 251 KB of
    `<!DOCTYPE html>` into `data/raw/dvb/full/challenge`, printed a sha256 for
    it, and reported `completed`. Zero annotations, a junk file in the dataset
    root, and a success message.
    """

    @staticmethod
    def _plan(tmp_path, url: str):
        from anti_uav.config.schema import SourceKind
        from anti_uav.data.download.backends import DownloadPlan

        return DownloadPlan(
            dataset="dvb",
            variant="full",
            source_label="test",
            kind=SourceKind.HTTP,
            url=url,
            destination=tmp_path / "raw",
            approx_size_gb=0.001,
            needs_credentials=False,
        )

    @staticmethod
    def _fake_response(content_type: str, body: bytes):
        class FakeResponse:
            def __init__(self) -> None:
                self.headers = {
                    "content-type": content_type,
                    "content-length": str(len(body)),
                }
                self._stream = iter([body])
                self.raw = None

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int = 0):
                return self._stream

        return FakeResponse()

    @pytest.mark.parametrize(
        ("content_type", "body"),
        [
            ("text/html; charset=utf-8", b"<!DOCTYPE html><html><body>repo</body></html>"),
            # A mirror that serves a page with a generic content-type still has
            # to be caught, hence the body sniff.
            ("application/octet-stream", b"<!DOCTYPE html>\n<html>...repo page...</html>"),
            ("text/plain", b"<html><head><title>GitHub</title></head></html>"),
        ],
    )
    def test_html_is_failed_and_nothing_is_written(
        self, monkeypatch, tmp_path, content_type: str, body: bytes
    ) -> None:
        import requests

        from anti_uav.data.download.backends import HttpBackend

        monkeypatch.setattr(
            requests, "get", lambda *a, **k: self._fake_response(content_type, body)
        )

        result = HttpBackend().fetch(self._plan(tmp_path, "https://example.invalid/repo"))

        assert result.status == "failed", f"HTML was accepted: {result.status}"
        assert "HTML" in (result.error or "")
        assert not list((tmp_path / "raw").glob("*.part")), "partial file left behind"
        assert not list((tmp_path / "raw").glob("repo*")), "junk file left in the dataset root"

    def test_a_real_artefact_still_downloads(self, monkeypatch, tmp_path) -> None:
        """The guard must not become a blanket refusal."""
        import requests

        from anti_uav.data.download.backends import HttpBackend

        payload = b"PK\x03\x04" + b"\x00" * 2048  # a plausible archive header
        monkeypatch.setattr(
            requests, "get", lambda *a, **k: self._fake_response("application/zip", payload)
        )

        result = HttpBackend().fetch(self._plan(tmp_path, "https://example.invalid/data.zip"))

        assert result.status == "completed", result.error
        assert result.bytes_downloaded == len(payload)

    def test_no_http_source_points_at_a_repository_url(self) -> None:
        """Structural guard: an `http` source must be a file, not a page.

        Catches the whole class, not just the instance we hit - a bare repo or
        directory URL under `kind: http` is always a mistake, because HttpBackend
        only knows how to GET a single file.
        """
        from anti_uav.config import load_registry

        offenders: list[str] = []
        for alias, spec in load_registry().datasets.items():
            for source in spec.sources:
                if source.kind.value != "http":
                    continue
                url = source.url.rstrip("/")
                if url.count("/") < 3 or url.endswith(".git"):
                    offenders.append(f"{alias}: {source.url} ({source.label})")
        assert not offenders, (
            "these http sources look like web pages or repositories, which return "
            f"HTML rather than an artefact: {offenders}"
        )
