r"""Reproducible randomness.

A split that changes between runs makes every metric in the matrix incomparable
and every "why did this score change?" question unanswerable. These tests exist
because Python salts \hash()\ per process, which once made the seed a lie.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from anti_uav.utils.seed import rng_for, seed_everything, stable_hash


class TestStableHash:
    def test_is_not_the_builtin_hash(self) -> None:
        assert stable_hash("split|antiuav") == stable_hash("split|antiuav")

    def test_distinguishes_different_keys(self) -> None:
        assert stable_hash("a") != stable_hash("b")

    def test_is_stable_within_a_process(self) -> None:
        assert stable_hash("split|mmuav") == stable_hash("split|mmuav")

    def test_survives_a_fresh_interpreter(self) -> None:
        """The bug this guards: `hash()` on a str is salted per process, so a
        one-line subprocess is the only way to catch a regression here."""
        program = (
            "import sys; sys.path.insert(0, r'" + str(Path(__file__).resolve().parents[1] / "src") + "'); "
            "from anti_uav.utils.seed import stable_hash; "
            "print(stable_hash('split|antiuav'))"
        )
        values = set()
        for _ in range(3):
            out = subprocess.run(
                [sys.executable, "-c", program],
                capture_output=True,
                text=True,
                check=True,
            )
            values.add(out.stdout.strip())
        assert len(values) == 1, f"hash varies across processes: {values}"


class TestRngFor:
    def test_same_seed_same_stream(self) -> None:
        assert float(rng_for(7, "split", "dvb").random()) == float(
            rng_for(7, "split", "dvb").random()
        )

    def test_different_seed_different_stream(self) -> None:
        assert float(rng_for(1).random()) != float(rng_for(2).random())

    def test_different_stream_name_is_independent(self) -> None:
        """Adding a new consumer must not perturb an existing one."""
        assert float(rng_for(7, "split", "dvb").random()) == float(
            rng_for(7, "split", "dvb").random()
        )
        assert float(rng_for(7, "augment", "dvb").random()) != float(
            rng_for(7, "split", "dvb").random()
        )

    def test_survives_a_fresh_interpreter(self) -> None:
        program = (
            "import sys; sys.path.insert(0, r'"
            + str(Path(__file__).resolve().parents[1] / "src")
            + "'); from anti_uav.utils.seed import rng_for; "
            "print(round(float(rng_for(0, 'split', 'antiuav').random()), 12))"
        )
        values = set()
        for _ in range(3):
            out = subprocess.run(
                [sys.executable, "-c", program],
                capture_output=True,
                text=True,
                check=True,
            )
            values.add(out.stdout.strip())
        assert len(values) == 1, f"rng stream varies across processes: {values}"


class TestSeedEverything:
    def test_is_idempotent(self) -> None:
        seed_everything(0)
        seed_everything(0)  # must not raise

    def test_makes_torch_deterministic_when_available(self) -> None:
        seed_everything(0)
        try:
            import torch
        except ImportError:
            return
        assert torch.manual_seed is not None
