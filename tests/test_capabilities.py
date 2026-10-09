"""The per-dataset evidential contract.

Four public datasets do not support the same conclusions: only two carry birds, so
only those can turn a precision number into evidence about false positives, and
only one has multi-object identity, so only that one makes IDF1 meaningful. Those
facts live in the registry; this module is the single derivation, and these tests
pin the derivation so a fifth dataset cannot quietly acquire an unstated exemption.
"""

from __future__ import annotations

import pytest

from anti_uav.config.loader import load_matrix, load_registry
from anti_uav.data.capabilities import (
    CLAIM_IDENTITY,
    CLAIM_PRECISION,
    CLAIM_TRACKING,
    all_source_capabilities,
    capabilities_for_run,
    combo_capabilities,
    source_capabilities_for,
)

SOURCES = ["dvb", "mavvid", "antiuav", "mmuav"]


class TestSourceCapabilities:
    @pytest.mark.parametrize("alias", SOURCES)
    def test_every_registered_source_resolves(self, alias: str) -> None:
        caps = source_capabilities_for(alias)
        assert caps.alias == alias
        assert caps.median_target_px and caps.median_target_px > 0

    def test_the_registry_and_the_contract_agree_on_birds(self) -> None:
        """The two bird sources, not one. This is the fact most easily misremembered."""
        with_birds = {
            a for a in SOURCES if source_capabilities_for(a).can_falsify_precision
        }
        assert with_birds == {"dvb", "mavvid"}

    def test_only_mmuav_supports_identity_metrics(self) -> None:
        """Anti-UAV labels every box id=1: track ids present, identity metrics vacuous."""
        with_identity = {a for a in SOURCES if source_capabilities_for(a).can_score_identity}
        assert with_identity == {"mmuav"}

    def test_track_ids_are_weaker_than_identity(self) -> None:
        """antiuav has MOT identity but one target per sequence."""
        antiuav = source_capabilities_for("antiuav")
        assert antiuav.has_track_ids is True
        assert antiuav.can_score_tracking is True
        assert antiuav.can_score_identity is False

    def test_only_antiuav_carries_visibility_flags(self) -> None:
        with_vis = {a for a in SOURCES if source_capabilities_for(a).can_score_visibility}
        assert with_vis == {"antiuav"}

    def test_tiling_agrees_with_the_registry_override(self) -> None:
        """MM-UAV's frames are 640x360, so a 640 px tile magnifies nothing."""
        mmuav = source_capabilities_for("mmuav")
        assert mmuav.requires_tiling is True
        assert mmuav.tile_size == 256
        assert source_capabilities_for("mavvid").requires_tiling is False
        assert source_capabilities_for("antiuav").requires_tiling is False
        assert source_capabilities_for("dvb").requires_tiling is True
        assert source_capabilities_for("dvb").tile_size == 640

    def test_combo_tiling_matches_what_build_actually_resolves(self) -> None:
        """The contract must not disagree with resolve_tiling, or it is decoration.

        Tiling is decided per combo x source: the dataset's own override wins,
        then the combo's force_tiling, then the median-size rule. The report has to
        follow all three or a table will claim a source is untiled while build
        tiles it.
        """
        from anti_uav.data.build import resolve_tiling

        registry = load_registry()
        matrix = load_matrix()
        for spec in matrix.combos:
            caps = combo_capabilities(spec.slug, tuple(spec.datasets))
            for alias in spec.datasets:
                resolved = resolve_tiling(registry.get(alias), spec, matrix)
                assert caps.tiling[alias] is resolved.enabled, f"{spec.slug}/{alias}"

    def test_a_combo_can_override_the_dataset_default(self) -> None:
        """force_tiling is per combo, so the dataset-level default is only a default.

        Uses mavvid, which has no tiling_override, so force_tiling actually applies.
        """
        forced_on = combo_capabilities("demo-forced", ("mavvid",), force_tiling=True)
        assert forced_on.tiled_sources == ("mavvid",)
        forced_off = combo_capabilities("demo-off", ("mavvid",), force_tiling=False)
        assert forced_off.tiled_sources == ()

    def test_a_dataset_override_beats_combo_force_tiling(self) -> None:
        """MM-UAV's 256 px override is the dataset's own decision and outranks
        force_tiling, exactly as resolve_tiling's precedence says."""
        off = combo_capabilities("demo", ("mmuav",), force_tiling=False)
        assert off.tiled_sources == ("mmuav",)

    def test_unknown_source_is_an_error(self) -> None:
        with pytest.raises(KeyError):
            source_capabilities_for("nope")


class TestComboCapabilities:
    def test_single_source_combos_inherit_only_that_source(self) -> None:
        dvb = combo_capabilities("dvb", ("dvb",))
        assert dvb.can_claim(CLAIM_PRECISION) is True
        assert dvb.can_claim(CLAIM_IDENTITY) is False
        assert dvb.can_claim(CLAIM_TRACKING) is False

    def test_a_combo_unions_its_sources_claims(self) -> None:
        """all4 can falsify false positives because dvb and mavvid are in it."""
        all4 = combo_capabilities("all4", ("dvb", "mavvid", "antiuav", "mmuav"))
        assert set(all4.precision_sources) == {"dvb", "mavvid"}
        assert set(all4.identity_sources) == {"mmuav"}
        assert set(all4.tracking_sources) == {"antiuav", "mmuav"}

    def test_drone_only_combos_cannot_claim_precision(self) -> None:
        for slug, sources in (("antiuav", ("antiuav",)), ("mmuav", ("mmuav",))):
            caps = combo_capabilities(slug, sources)
            assert caps.can_claim(CLAIM_PRECISION) is False, slug
            assert caps.warnings(), slug

    def test_the_warning_names_a_dataset_that_can_supply_the_evidence(self) -> None:
        caps = combo_capabilities("mmuav", ("mmuav",))
        joined = " ".join(caps.warnings())
        assert "bird" in joined
        assert "dvb" in joined and "mavvid" in joined

    def test_single_target_combo_warns_about_identity_not_about_tracking(self) -> None:
        caps = combo_capabilities("antiuav", ("antiuav",))
        assert caps.can_claim(CLAIM_TRACKING) is True
        assert caps.can_claim(CLAIM_IDENTITY) is False
        joined = " ".join(caps.warnings())
        assert "IDF1" in joined

    def test_no_track_ids_combo_says_so_plainly(self) -> None:
        caps = combo_capabilities("dvb", ("dvb",))
        assert caps.can_claim(CLAIM_TRACKING) is False
        assert "no track ids" in " ".join(caps.warnings())

    def test_a_combo_with_nothing_missing_warns_about_nothing(self) -> None:
        all4 = combo_capabilities("all4", ("dvb", "mavvid", "antiuav", "mmuav"))
        assert all4.warnings() == []

    def test_mixed_tiling_is_flagged_as_not_comparable(self) -> None:
        mixed = combo_capabilities("all4", ("dvb", "mavvid", "antiuav", "mmuav"))
        assert mixed.requires_tiling is True
        assert set(mixed.tiled_sources) == {"dvb", "mmuav"}
        assert mixed.is_comparable_across_combos is False

    def test_uniform_tiling_is_comparable(self) -> None:
        assert combo_capabilities("mmuav", ("mmuav",)).is_comparable_across_combos is True
        assert combo_capabilities("dvb", ("dvb",)).is_comparable_across_combos is True

    def test_every_matrix_combo_is_describable(self) -> None:
        for spec in load_matrix().combos:
            caps = combo_capabilities(spec.slug, tuple(spec.datasets))
            assert caps.sources, spec.slug
            assert caps.claims(), spec.slug


class TestRunMetadata:
    def test_a_run_knows_what_it_can_prove(self) -> None:
        caps = capabilities_for_run({"combo": "all4"})
        assert caps is not None
        assert caps.can_claim(CLAIM_PRECISION) is True
        assert caps.can_claim(CLAIM_IDENTITY) is True

    def test_a_drone_only_run_knows_it_cannot_prove_precision(self) -> None:
        caps = capabilities_for_run({"combo": "mmuav"})
        assert caps is not None
        assert caps.can_claim(CLAIM_PRECISION) is False

    @pytest.mark.parametrize("meta", [{}, {"combo": None}, {"combo": "not-a-combo"}])
    def test_unusable_metadata_returns_none_rather_than_raising(self, meta: dict) -> None:
        assert capabilities_for_run(meta) is None


class TestReporting:
    def test_summary_row_names_the_supporting_sources(self) -> None:
        row = combo_capabilities("dvb+mavvid", ("dvb", "mavvid")).summary_row()
        assert "dvb,mavvid" in row

    def test_explain_mentions_the_bird_sources_and_the_missing_identity(self) -> None:
        text = combo_capabilities("dvb", ("dvb",)).explain()
        assert "NOT SUPPORTED" in text
        assert CLAIM_PRECISION in text

    def test_all_source_capabilities_covers_the_registry(self) -> None:
        caps = all_source_capabilities()
        assert set(caps) == set(load_registry().aliases)

    def test_the_capability_report_renders_without_wrapping_itself(self) -> None:
        """matrix list prints through a Rich console that hard-wraps at ~80 cols."""
        from anti_uav.detection.matrix import capability_report

        for line in capability_report().splitlines():
            assert len(line) <= 80, f"line would wrap in the console: {line!r}"

    def test_report_names_every_source_and_every_combo(self) -> None:
        from anti_uav.detection.matrix import capability_report

        text = capability_report()
        for alias in SOURCES:
            assert alias in text, alias
        for spec in load_matrix().combos:
            assert spec.slug in text, spec.slug
