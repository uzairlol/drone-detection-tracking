"""End-to-end smoke test of the data layer on synthetic data.

Builds a tiny fake Anti-UAV and fake MM-UAV tree, runs convert -> split ->
build -> sanity, and asserts the invariants hold. This exercises the real code
paths without needing a 60 GB download.

Not part of the pytest suite; run manually with the ml interpreter.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anti_uav.config import load_dataset, load_matrix
from anti_uav.data import sanity, splits, stats
from anti_uav.data.build import build as build_combo
from anti_uav.data.convert import convert_dataset
from anti_uav.data.frameindex import load_index

# The scratch data root, resolved lazily rather than at import time.
#
# Two things make this fiddly. utils.paths.data_root() is lru_cache'd, so the
# override has to be in place before anything calls it - but smoke_cross_eval.py
# imports fake_antiuav from this module, and it has already pointed
# ANTI_UAV_DATA_DIR at its own temp dir. Assigning unconditionally here would
# silently relocate that script's fixture tree underneath it, so this adopts
# whatever root is already set and only creates one when running standalone.
#
# Skipping this entirely is what made the missing-source assertion below
# unfalsifiable: the script read the developer's real data/interim, where dvb
# genuinely exists, and wrote its `smoke` and `smoke2` builds into the real
# data/processed.
_DATA_ROOT: Path | None = None
_OWNS_DATA_ROOT = False


def data_root() -> Path:
    global _DATA_ROOT, _OWNS_DATA_ROOT
    if _DATA_ROOT is None:
        existing = os.environ.get("ANTI_UAV_DATA_DIR")
        if existing:
            _DATA_ROOT = Path(existing)
        else:
            _DATA_ROOT = Path(tempfile.mkdtemp(prefix="anti_uav_smoke_"))
            os.environ["ANTI_UAV_DATA_DIR"] = str(_DATA_ROOT)
            _OWNS_DATA_ROOT = True
    return _DATA_ROOT

RNG = np.random.default_rng(7)


def write_video(path: Path, frames: list, fps: int = 25) -> None:
    """Encode frames to a real mp4 so the decode + stride path is exercised."""
    import cv2

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def make_frame(width: int, height: int) -> np.ndarray:
    """Sky-like gradient with a bright blob, so tiling has something to find."""
    img = np.full((height, width, 3), 150, dtype=np.uint8)
    img[:, :, 0] = np.linspace(90, 200, width, dtype=np.uint8)[None, :]
    img[:, :, 2] = np.linspace(220, 140, width, dtype=np.uint8)[None, :]
    cy = height // 2 + int(RNG.integers(-height // 6, height // 6))
    cx = width // 2 + int(RNG.integers(-width // 6, width // 6))
    s = max(3, min(width, height) // 40)
    img[max(0, cy - s) : cy + s, max(0, cx - s) : cx + s] = 245
    return img


def fake_antiuav(root: Path, sequences: int = 6, frames: int = 12) -> Path:
    """``<seq>/<seq>_rgb.mp4``-style names with a matching ``<seq>.json``."""
    from anti_uav.utils.imaging import write_image  # noqa: F401  (used by fake_mmuav)

    base = root / "antiuav" / "300"
    for seq in range(1, sequences + 1):
        seq_dir = base / f"seq_{seq:03d}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        payload = {"frames": {}}
        rendered: list = []
        for frame in range(1, frames + 1):
            rendered.append(make_frame(640, 360))
            visible = frame % 4 != 0  # every 4th frame the target is absent
            if visible:
                payload["frames"][str(frame)] = [
                    {
                        "xmin": 300 + frame,
                        "ymin": 170,
                        "xmax": 330 + frame,
                        "ymax": 195,
                        "v": 1.0,
                        "id": 1,
                    }
                ]
            else:
                payload["frames"][str(frame)] = []
        # A REAL mp4, so the decoder + stride path is exercised rather than mocked.
        write_video(seq_dir / f"seq_{seq:03d}_rgb.mp4", rendered)
        (seq_dir / "seq_000.json").write_text(json.dumps(payload), encoding="utf-8")
    return base


def fake_mmuav(root: Path, sequences: int = 4, frames: int = 10) -> Path:
    """Publisher layout: ``<seq>/rgb_frame/*.jpg`` + ``<seq>/gt_rgb/gt.txt``."""
    from anti_uav.utils.imaging import write_image

    base = root / "mmuav" / "subset" / "train"
    for seq in range(1, sequences + 1):
        seq_dir = base / f"{seq:04d}"
        (seq_dir / "rgb_frame").mkdir(parents=True, exist_ok=True)
        (seq_dir / "gt_rgb").mkdir(parents=True, exist_ok=True)
        rows = []
        for frame in range(1, frames + 1):
            write_image(seq_dir / "rgb_frame" / f"{frame:04d}.jpg", make_frame(640, 360))
            if frame % 3:
                rows.append(
                    f"{frame},1,290.{frame},160,305,170,1,1,1.0"
                )
        (seq_dir / "gt_rgb" / "gt.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return base.parent.parent


def main() -> int:
    tmp = data_root()
    raw = tmp / "raw"
    failures: list[str] = []

    try:
        print("== building synthetic sources ==")
        fake_antiuav(raw)
        mmuav_raw = fake_mmuav(raw)

        print("\n== convert antiuav ==")
        spec_au = load_dataset("antiuav")
        rep_au = convert_dataset(spec_au, raw / "antiuav" / "300", variant="300")
        print(
            f"  ok={rep_au.ok} frames={rep_au.frames_written} boxes={rep_au.boxes_written} "
            f"seqs={rep_au.sequences} dropped={rep_au.boxes_dropped}"
        )
        print(f"  classes={rep_au.class_histogram}")
        print(f"  sizes={rep_au.frame_size_histogram}")
        # stride=3 in the registry, so 12 source frames -> 4 kept per sequence.
        expected_frames = 6 * -(-12 // 3)
        assert rep_au.ok and rep_au.frames_written == expected_frames, (
            f"expected {expected_frames} frames (6 seqs x ceil(12/stride 3)), "
            f"got {rep_au.frames_written}"
        )
        assert rep_au.boxes_written > 0, "no boxes survived anti-uav conversion"
        assert rep_au.class_histogram["drone"] > 0, "drone boxes missing from histogram"
        assert rep_au.class_histogram["bird"] == 0, "anti-uav should carry no bird boxes"
        assert not rep_au.errors, rep_au.errors

        print("\n== convert mmuav ==")
        spec_mm = load_dataset("mmuav")
        rep_mm = convert_dataset(
            spec_mm, mmuav_raw, variant="subset", modalities=["rgb"], max_sequences=4
        )
        print(
            f"  ok={rep_mm.ok} frames={rep_mm.frames_written} boxes={rep_mm.boxes_written} "
            f"seqs={rep_mm.sequences}"
        )
        assert rep_mm.ok and rep_mm.frames_written > 0

        print("\n== stats ==")
        s_au = stats.compute(
            load_index("antiuav", "300"), dataset="antiuav", variant="300"
        )
        s_mm = stats.compute(
            load_index("mmuav", "subset"), dataset="mmuav", variant="subset"
        )
        print("\n".join("  " + l for l in stats.format_stats(s_au).splitlines()[:9]))
        print("  ---")
        print("\n".join("  " + l for l in stats.format_stats(s_mm).splitlines()[:9]))
        print()
        print(
            "\n".join(
                "  " + l for l in stats.combo_summary({"antiuav": s_au, "mmuav": s_mm}).splitlines()
            )
        )

        print("\n== split ==")
        records = load_index("antiuav", "300") + load_index("mmuav", "subset")
        assigned, srep = splits.split_records(
            records, val_fraction=0.34, seed=0, combo="smoke"
        )
        print("\n".join("  " + l for l in splits.format_report(srep).splitlines()))
        assert not [w for w in srep.warnings if "more than one split" in w]
        # Persist writes the split back into each interim index. `build` reads
        # splits from disk, so this step is mandatory, not optional.
        splits.persist(assigned, srep, "smoke")

        print("\n== sanity ==")
        sreport = sanity.run_all(
            assigned, combo="smoke", expect_datasets=["antiuav", "mmuav"]
        )
        print("\n".join("  " + l for l in sanity.format_report(sreport).splitlines()))
        # The fixture has no birds at all, so the bird check MUST warn. That
        # warning is the feature working, not a failure.
        assert any(c.name == "bird_negatives" for c in sreport.warnings), (
            "expected the missing-bird-negatives warning"
        )
        assert all(c.passed for c in sreport.checks), [
            c.name for c in sreport.failures
        ]

        print("\n== build combo (tiling forced on for mmuav) ==")
        matrix = load_matrix()
        combo = next(c for c in matrix.enabled_combos() if c.slug == "all4")
        combo = combo.model_copy(update={"datasets": ["antiuav", "mmuav"], "slug": "smoke"})
        variants = {"antiuav": "300", "mmuav": "subset"}
        brep = build_combo(
            combo,
            {"antiuav": spec_au, "mmuav": spec_mm},
            matrix,
            variants=variants,
            clean=True,
        )
        print(f"  ok={brep.ok} train={brep.total_train_frames} val={brep.total_val_frames}")
        print(f"  classes={brep.class_counts}")
        for s in brep.sources:
            print(
                f"  source {s.dataset}: tiling={s.tiling} tile={s.tile_size} "
                f"in={s.frames_in} out={s.frames_out} train={s.train_frames} val={s.val_frames}"
            )
        print("  warnings:")
        for w in brep.warnings:
            print(f"    ! {w}")
        assert brep.ok, brep.errors
        assert brep.total_train_frames > brep.total_val_frames, "tiling should expand train"

        yaml_text = (Path(brep.output_dir) / "data.yaml").read_text(encoding="utf-8")
        print("\n  data.yaml:")
        print("\n".join("    " + l for l in yaml_text.splitlines()))

        print("\n== missing-source degradation ==")
        combo_missing = combo.model_copy(update={"datasets": ["antiuav", "dvb"], "slug": "smoke2"})
        brep2 = build_combo(
            combo_missing, {"antiuav": spec_au, "dvb": load_dataset("dvb")}, matrix,
            variants={"antiuav": "300"},
        )
        print(f"  ok={brep2.ok} warnings={len(brep2.warnings)}")
        for w in brep2.warnings:
            print(f"    ! {w[:120]}")
        assert brep2.ok, "a missing source must degrade, not crash"
        assert brep2.warnings, "a missing source must warn loudly"

        print("\n" + "=" * 70)
        print("ALL DATA-LAYER SMOKE CHECKS PASSED")
        print("=" * 70)
        return 0
    except AssertionError as exc:
        failures.append(str(exc))
        print(f"\nASSERTION FAILED: {exc}")
        return 1
    finally:
        # Only tear down a directory this process created. When imported by
        # smoke_cross_eval.py the root belongs to that script, and deleting it
        # would pull the ground out from under it.
        if _OWNS_DATA_ROOT:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
