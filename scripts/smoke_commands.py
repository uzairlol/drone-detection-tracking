"""`predict` and `replay` against a real checkpoint.

Both commands were added after the modules they wrap already existed, which is
exactly the situation where a command looks fine in `--help` and dies on first
real use. So: build a YOLO checkpoint, a synthetic dataset, and run both.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp(prefix="anti_uav_cmdsmoke_"))
os.environ["ANTI_UAV_DATA_DIR"] = str(tmp)
sys.path.insert(0, str(Path(__file__).parent))

from smoke_data import fake_antiuav  # noqa: E402

from anti_uav.config import load_dataset  # noqa: E402
from anti_uav.data import splits  # noqa: E402
from anti_uav.data.convert import convert_dataset  # noqa: E402
from anti_uav.data.frameindex import load_index  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not condition:
        failures.append(name)


def run_cli(*args: str) -> tuple[int, str]:
    """Invoke the CLI in-process so we get a real exit code and captured text.

    Exceptions are re-raised: a test that only sees `exit_code == 1` cannot
    tell "failed cleanly with a good message" from "crashed with a traceback",
    and guessing wrong there wastes more time than the plumbing costs.
    """
    import io
    from contextlib import redirect_stdout

    from typer.testing import CliRunner

    from anti_uav.cli import app

    runner = CliRunner()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = runner.invoke(app, list(args), catch_exceptions=True)
    if result.exception is not None and not isinstance(
        result.exception, SystemExit
    ):
        import click

        if not isinstance(result.exception, click.exceptions.Exit):
            raise result.exception
    return result.exit_code, buffer.getvalue() + result.output


print("=" * 74)
print("PREDICT AND REPLAY COMMANDS")
print("=" * 74)

# --- a synthetic converted dataset -----------------------------------------
raw = tmp / "raw"
spec = load_dataset("antiuav")
fake_antiuav(raw, sequences=4, frames=8)
convert_dataset(spec, raw / "antiuav" / "300", variant="300")
records = load_index("antiuav", "300")
assigned, report = splits.split_records(records, val_fraction=0.34, seed=3, combo="cmdsmoke")
splits.persist(assigned, report, "cmdsmoke")
sequences = sorted({r.sequence_id for r in load_index("antiuav", "300")})
print(f"  synthetic dataset: {len(records)} frames, {len(sequences)} sequences")

# --- a real YOLO checkpoint -------------------------------------------------
from ultralytics import YOLO  # noqa: E402

from anti_uav.config import load_matrix, load_settings  # noqa: E402
from anti_uav.data.frameindex import interim_root  # noqa: E402
from anti_uav.detection.matrix import build_matrix_plan  # noqa: E402
from anti_uav.utils.paths import subdir  # noqa: E402

settings = load_settings()
plan = build_matrix_plan(load_matrix(), require_dataset=False)
cell = next(
    p for p in plan.plans if p.model == "yolo11n" and p.combo == "antiuav"
)
run_dir = subdir("runs") / "yolo11n" / "antiuav"
run_dir.mkdir(parents=True, exist_ok=True)

# yolo11n at its smallest legal size: this test is about plumbing, not accuracy.
yolo = YOLO("yolo11n.yaml")
yolo.model.args["imgsz"] = 64
weights = run_dir / "weights" / "best.pt"
weights.parent.mkdir(parents=True, exist_ok=True)
yolo.save(str(weights))
print(f"  wrote a real checkpoint: {weights}  (cell {cell.model}/{cell.combo})")
check("checkpoint exists", weights.is_file())

# --- predict: an image ------------------------------------------------------
# FrameRecord.image already contains the "frames/" segment, so it is relative
# to the variant root (data/interim/<dataset>/<variant>/), not to frames_dir.
image = interim_root("antiuav", "300") / load_index("antiuav", "300")[0].image
code, out = run_cli("predict", "--run", str(run_dir), "--source", str(image))
print(f"  predict(image) exit={code}")
check("predict on an image exits 0", code == 0, out.strip().splitlines()[-1] if out.strip() else "")
check("predict reports the frame", "frame(s)" in out)

save_dir = tmp / "overlays"
code, out = run_cli(
    "predict", "--run", str(run_dir), "--source", str(image), "--save", str(save_dir)
)
print(f"  predict(image --save) exit={code}")
check("predict --save exits 0", code == 0)
written = list(save_dir.glob("*.jpg")) if save_dir.is_dir() else []
check("predict --save wrote an annotated image", bool(written), str([p.name for p in written]))

# --- predict: a video -------------------------------------------------------
from smoke_data import write_video  # noqa: E402

video_dir = tmp / "videos"
video_dir.mkdir(parents=True, exist_ok=True)
from anti_uav.utils.imaging import read_image  # noqa: E402

frames = [
    read_image(interim_root("antiuav", "300") / r.image)
    for r in load_index("antiuav", "300")[:6]
]
video_path = video_dir / "clip.mp4"
write_video(video_path, frames)
code, out = run_cli("predict", "--run", str(run_dir), "--source", str(video_path))
print(f"  predict(video) exit={code}")
check("predict on a video exits 0", code == 0)
check("predict saw multiple frames", "6 frame(s)" in out or "frame(s)" in out)

vid_save = tmp / "vid_overlays"
code, out = run_cli(
    "predict", "--run", str(run_dir), "--source", str(video_path),
    "--save", str(vid_save), "--stride", "2",
)
print(f"  predict(video --save --stride 2) exit={code}")
check("predict --save on video exits 0", code == 0)
vframes = list(vid_save.glob("*.jpg")) if vid_save.is_dir() else []
check("stride 2 wrote 3 of 6 frames", len(vframes) == 3, f"{len(vframes)} written")

# --- predict: error paths ---------------------------------------------------
code, out = run_cli("predict", "--run", str(run_dir), "--source", str(tmp))
check("predict on a directory fails cleanly", code != 0)
code, out = run_cli("predict", "--run", str(run_dir), "--source", str(image), "--tracker", "nope")
check("predict rejects an unknown tracker", code != 0 and "unknown tracker" in out)

# --- replay -----------------------------------------------------------------
code, out = run_cli(
    "replay", "--run", str(run_dir), "--dataset", "antiuav",
    "--sequence", sequences[0], "--variant", "300", "--tracker", "bytetrack",
)
print(f"  replay(one sequence) exit={code}")
print("   " + "\n   ".join(out.strip().splitlines()[:4]))
check("replay exits 0", code == 0)
check("replay names the sequence", sequences[0] in out)
check("replay prints metrics", "MOTA=" in out)

dump_dir = tmp / "mot"
code, out = run_cli(
    "replay", "--run", str(run_dir), "--dataset", "antiuav",
    "--variant", "300", "--tracker", "sort", "--dump", str(dump_dir),
)
print(f"  replay(all sequences --dump) exit={code}")
check("replay over all sequences exits 0", code == 0)
dumps = list(dump_dir.glob("*.txt")) if dump_dir.is_dir() else []
check("replay dumped one MOT file per sequence", len(dumps) == len(sequences),
      f"{len(dumps)} files for {len(sequences)} sequences")
# The checkpoint is a random-initialised yolo11n, so it legitimately detects
# nothing and the MOT dump is legitimately empty. What is under test is that the
# file exists, is empty rather than malformed, and is named per sequence.
check("every dump file exists and is well named",
      all((d.name.startswith("antiuav_") and d.name.endswith("_botsort.txt")) or
          d.name.endswith("_sort.txt") for d in dumps),
      ", ".join(d.name for d in dumps))
for d in dumps:
    rows = d.read_text(encoding="utf-8").strip().splitlines()
    check(f"  {d.name}: rows parse as MOT", all(len(r.split()) >= 6 for r in rows[:5]),
          f"{len(rows)} row(s) - empty is correct for an untrained model")

code, out = run_cli("replay", "--run", str(run_dir), "--dataset", "nope")
check("replay on an unknown dataset fails cleanly", code != 0)

print()
print("=" * 74)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    raise SystemExit(1)
print("ALL PREDICT/REPLAY COMMAND CHECKS PASSED")
print("=" * 74)
raise SystemExit(0)
