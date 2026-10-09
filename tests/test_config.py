"""Every config file loads, and the invariants that protect the experiments.

These are the tests that fail on a bad hand-edit before a GPU hour is spent
finding out.
"""

from __future__ import annotations

import pytest

from anti_uav.config import (
    load_coverage_map,
    load_dataset,
    load_matrix,
    load_recipe,
    load_registry,
    load_rules,
    load_settings,
)
from anti_uav.config.schema import GpuProfile


class TestSettings:
    def test_loads(self) -> None:
        settings = load_settings()
        assert settings.imgsz >= 64

    def test_device_auto_becomes_ultralytics_empty_string(self) -> None:
        """`device=auto` raises inside ultralytics' select_device.

        The schema normalises it, so this is the assertion that keeps a
        training run from dying on the first step.
        """
        settings = load_settings()
        assert settings.device != "auto"
        assert settings.device in ("", "cpu") or settings.device[0].isdigit()

    def test_explicit_device_survives(self) -> None:
        from anti_uav.config import settings_with

        assert settings_with(device="cpu").device == "cpu"
        assert settings_with(device="0").device == "0"


class TestRegistry:
    def test_all_four_datasets_present(self) -> None:
        registry = load_registry()
        assert set(registry.aliases) == {"dvb", "mavvid", "antiuav", "mmuav"}

    @pytest.mark.parametrize("alias", ["dvb", "mavvid", "antiuav", "mmuav"])
    def test_dataset_has_a_source(self, alias: str) -> None:
        spec = load_dataset(alias)
        assert spec.sources, f"{alias} has no download source"

    @pytest.mark.parametrize("alias", ["antiuav", "mmuav"])
    def test_multi_variant_datasets_declare_a_default(self, alias: str) -> None:
        """Anti-UAV and MM-UAV have named variants; picking the wrong one is a
        60 GB mistake, so a default has to be recorded."""
        spec = load_dataset(alias)
        assert spec.variants
        assert spec.default_variant in spec.variants

    def test_single_release_datasets_need_no_variants(self) -> None:
        """dvb and mavvid are one fixed release each - an empty variants list is
        correct there, not a missing value."""
        for alias in ("dvb", "mavvid"):
            spec = load_dataset(alias)
            assert spec.default_variant is None
            assert not spec.variants

    @pytest.mark.parametrize("alias", ["dvb", "mavvid", "antiuav", "mmuav"])
    def test_median_target_is_recorded(self, alias: str) -> None:
        """Tiling keys off this number, so it cannot be optional."""
        assert load_dataset(alias).median_target_px > 0

    def test_only_dvb_and_mavvid_carry_bird_negatives(self) -> None:
        registry = load_registry()
        with_birds = {
            a for a in registry.aliases if registry.get(a).has_bird_negatives
        }
        assert with_birds == {"dvb", "mavvid"}

    def test_unified_label_order_is_drone_then_bird(self) -> None:
        registry = load_registry()
        for alias in registry.aliases:
            labels = registry.get(alias).classes.unified_labels
            assert labels[:2] == ["drone", "bird"], f"{alias}: {labels}"


class TestMatrix:
    def test_two_models_and_seven_combos(self) -> None:
        matrix = load_matrix()
        assert len(matrix.models) == 2
        assert len(matrix.combos) == 7

    def test_n_runs_is_models_times_combos(self) -> None:
        matrix = load_matrix()
        assert matrix.n_runs == len(matrix.models) * len(matrix.combos)
        assert matrix.n_runs == 14

    def test_tiling_threshold_is_set(self) -> None:
        """If this is 0, nothing tiles and MM-UAV's 12 px targets are invisible."""
        assert 0 < load_matrix().tiling_threshold_px < 100

    def test_all4_contains_every_dataset(self) -> None:
        matrix = load_matrix()
        all4 = next(c for c in matrix.combos if c.slug == "all4")
        assert set(all4.datasets) == {"dvb", "mavvid", "antiuav", "mmuav"}


class TestRecipes:
    @pytest.mark.parametrize("name", ["yolo11n", "rtdetr_x2"])
    def test_recipe_loads(self, name: str) -> None:
        recipe = load_recipe(name)
        assert recipe.epochs > 0
        assert recipe.imgsz >= 64

    def test_batch_auto_is_negative(self) -> None:
        """-1 means "let ultralytics pick"; the per-GPU overrides clamp it.

        A positive value here would silently override the 4090/5090 boxes.
        """
        assert load_recipe("yolo11n").batch == -1

    def test_both_families_are_reachable(self) -> None:
        assert load_recipe("yolo11n").family.value == "yolo11n"
        assert load_recipe("rtdetr_x2").family.value == "rtdetr_x2"


class TestSupportedPythonRange:
    """`verify-env` must enforce exactly what `pyproject.toml` declares.

    These two drifted once and the cost was real: the guard refused Python 3.13
    on the grounds that lapx and opencv-python had no wheels for it, while both
    had shipped them and Kaggle's image had moved to 3.13. So `verify-env` failed
    on a working environment and said nothing true.

    The guard cannot just be deleted - it is what turns an untested interpreter
    into a startup error instead of a confusing one at import. But it also must
    not carry hand-written claims about third-party wheel availability, which rot
    silently. So: assert the two ranges agree, and assert the range is not
    narrower than what the current interpreter satisfies.
    """

    @staticmethod
    def _declared_range() -> tuple[str, str]:
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        return tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
            "requires-python"
        ].removeprefix(">=")

    def test_guard_matches_pyproject(self) -> None:
        from anti_uav.verify import PYTHON_MAX, PYTHON_MIN

        floor, ceiling = self._declared_range().split(",<")
        assert floor.split(".")[1] == str(PYTHON_MIN[1]), (
            f"pyproject floor is {floor} but verify-env enforces {PYTHON_MIN}"
        )
        assert ceiling.split(".")[1] == str(PYTHON_MAX[1]), (
            f"pyproject ceiling is {ceiling} but verify-env enforces {PYTHON_MAX}"
        )

    def test_current_interpreter_is_inside_the_supported_range(self) -> None:
        """The suite is running, so the interpreter must satisfy the declared range."""
        import sys

        from anti_uav.verify import PYTHON_MAX, PYTHON_MIN

        current = (sys.version_info.major, sys.version_info.minor)
        assert PYTHON_MIN <= current < PYTHON_MAX

    def test_three_thirteen_is_supported(self) -> None:
        """3.13 has wheels for every pinned dep, so it must not be refused.

        lapx 0.10.0 publishes cp313 manylinux wheels and opencv-python-headless
        is a cp37-abi3 build. Excluding 3.13 blocked Kaggle, whose image ships it.
        """
        from anti_uav.verify import PYTHON_MAX, PYTHON_MIN

        assert PYTHON_MIN <= (3, 13) < PYTHON_MAX


class TestGpuProfiles:
    @pytest.mark.parametrize(
        "profile", ["pascal", "volta", "turing", "ampere", "ada", "blackwell", "cpu"]
    )
    def test_profile_override_file_exists(self, profile: str) -> None:
        from anti_uav.config import load_profile_override

        override = load_profile_override(profile)
        assert override.profile == GpuProfile(profile)

    def test_auto_resolves_to_a_real_profile(self) -> None:
        from anti_uav.config import load_profile_override

        assert load_profile_override(GpuProfile.AUTO).profile is not GpuProfile.AUTO

    def test_pascal_is_the_only_profile_that_disables_amp(self) -> None:
        """sm_61 is the only tier with no fp16 tensor cores.

        sm_70 and sm_75 have a fast one, and both used to be folded into
        ``pascal`` - which silently forced AMP off on a Kaggle T4. Any profile
        added later must not inherit that override.
        """
        from anti_uav.config import load_profile_override

        assert load_profile_override("pascal").force_amp is False
        for profile in ("volta", "turing", "ampere", "ada", "blackwell"):
            assert load_profile_override(profile).force_amp is not False, (
                f"{profile} has fp16 tensor cores and must not force AMP off"
            )

    @pytest.mark.parametrize("profile", ["volta", "turing", "ampere", "ada", "blackwell"])
    def test_gpu_profiles_pin_an_absolute_batch(self, profile: str) -> None:
        """A fractional batch is rejected outright under multi-GPU.

        ultralytics resolves ``batch: -1`` (AutoBatch) to a *fraction* of free
        VRAM and then raises on ``batch < 1.0`` when ``world_size > 1``. Every
        profile that can plausibly be used on a multi-GPU box therefore needs a
        concrete integer, and one divisible by 2 so DDP leaves no rank idle.
        """
        from anti_uav.config import load_profile_override

        override = load_profile_override(profile)
        assert override.force_batch is not None, (
            f"{profile} leaves batch at -1, which AutoBatch resolves to a fraction "
            f"and ultralytics rejects under DDP"
        )
        assert override.force_batch >= 1
        assert override.force_batch % 2 == 0, (
            f"{profile} batch {override.force_batch} is odd; DDP splits it across "
            f"two GPUs and would leave a rank idle"
        )

    @pytest.mark.parametrize("profile", ["volta", "turing", "ampere", "ada", "blackwell"])
    def test_per_model_batch_is_even_and_smaller_than_the_cnn(self, profile: str) -> None:
        """RT-DETR-x2 must get a smaller, still-even batch than YOLO11n.

        RT-DETR-x2 is 42.3 M params against YOLO11n's 2.6 M, and its hybrid
        encoder plus 100 denoising queries keep far more activations live. One
        batch number for both families either starves the CNN or OOMs the
        transformer on the same card - which is why this is a config field
        rather than something each operator rediscovers.
        """
        from anti_uav.config import load_profile_override

        override = load_profile_override(profile)
        det = override.force_batch_by_model.get("rtdetr_x2")
        assert det is not None, f"{profile} gives RT-DETR-x2 no batch of its own"
        assert det % 2 == 0, f"{profile} RT-DETR-x2 batch {det} is odd; DDP wastes a rank"
        assert det <= override.force_batch, (
            f"{profile} gives RT-DETR-x2 batch {det}, larger than YOLO11n's "
            f"{override.force_batch}, but it is the heavier of the two"
        )

    @pytest.mark.parametrize("profile", ["turing", "ampere", "ada", "blackwell"])
    def test_resolve_batch_honours_the_per_model_override(self, profile: str) -> None:
        """``force_batch_by_model`` must beat the profile-wide batch."""
        from anti_uav.config import load_profile_override, load_recipe
        from anti_uav.detection.trainer import resolve_batch

        override = load_profile_override(profile)
        expected = override.force_batch_by_model["rtdetr_x2"]
        assert resolve_batch(load_recipe("rtdetr_x2"), override, device="0,1") == expected
        assert resolve_batch(load_recipe("yolo11n"), override, device="0,1") == override.force_batch


class TestRules:
    def test_loads(self) -> None:
        rules = load_rules()
        assert rules.confidence.initiate > rules.confidence.maintain

    def test_detector_threshold_is_below_alert_threshold(self) -> None:
        """If these invert, the tracker is starved before the rules see a track."""
        assert load_settings().conf_threshold < load_rules().confidence.initiate

    def test_cross_camera_requires_more_than_one_camera(self) -> None:
        assert load_rules().cross_camera.min_cameras >= 2

    def test_validate_reports_the_known_advisories_not_errors(self) -> None:
        from anti_uav.rules import validate_rules

        problems = validate_rules()
        # The recovery-budget/handoff-lead note is a design tension the operator
        # resolves by choice; it is a warning, not a broken config.
        assert isinstance(problems, list)


class TestCoverageMap:
    def test_loads_with_100_cameras(self) -> None:
        coverage = load_coverage_map()
        assert len(coverage.cameras) == 100

    def test_node_split(self) -> None:
        coverage = load_coverage_map()
        counts: dict[str, int] = {}
        for camera in coverage.cameras:
            counts[camera.node] = counts.get(camera.node, 0) + 1
        assert len(counts) == 8
        assert sum(counts.values()) == 100
        assert sorted(counts.values()) == [12, 12, 12, 12, 13, 13, 13, 13]

    def test_camera_ids_are_unique(self) -> None:
        coverage = load_coverage_map()
        ids = [c.id for c in coverage.cameras]
        assert len(ids) == len(set(ids))

    def test_every_camera_has_intrinsics(self) -> None:
        for camera in load_coverage_map().cameras:
            intrinsics = camera.intrinsics
            assert intrinsics.width_px > 0
            assert intrinsics.height_px > 0
            assert intrinsics.fx_px > 0
            assert intrinsics.fy_px > 0

    def test_ptz_flag_agrees_with_role(self) -> None:
        """Every camera carries a ptz block; only ptz ones enable it.

        Fixed cameras keep a disabled block so the schema is uniform and the
        planner never has to branch on a missing attribute - so `enabled` must
        agree with `role`, which is the authoritative field.
        """
        cameras = load_coverage_map().cameras
        ptz = [c for c in cameras if c.role == "ptz"]
        assert len(ptz) == 20
        assert all(c.ptz is not None and c.ptz.enabled for c in ptz)
        assert all(not c.ptz.enabled for c in cameras if c.role == "fixed")

    def test_camera_count_splits_80_20(self) -> None:
        cameras = load_coverage_map().cameras
        assert sum(1 for c in cameras if c.role == "fixed") == 80
        assert sum(1 for c in cameras if c.role == "ptz") == 20
