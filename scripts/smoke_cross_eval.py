"""Cross-dataset evaluation YAML builder, on a synthetic split.

`--cross` is the flag that makes the experiment matrix meaningful (a model
trained on all four datasets still has to be scored per dataset), and it runs
before any real weights exist, so it is worth proving the path works.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

tmp = Path(tempfile.mkdtemp(prefix="anti_uav_crosseval_"))
os.environ["ANTI_UAV_DATA_DIR"] = str(tmp)

sys.path.insert(0, str(Path(__file__).parent))

from anti_uav.config import load_dataset  # noqa: E402
from anti_uav.data import splits  # noqa: E402
from anti_uav.data.convert import convert_dataset  # noqa: E402
from anti_uav.data.frameindex import load_index, write_index  # noqa: E402
from anti_uav.detection import evaluate as ev  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not condition:
        failures.append(name)


print("=" * 74)
print("CROSS-DATASET EVALUATION")
print("=" * 74)

raw = tmp / "raw"

spec = load_dataset("antiuav")

# Reuse smoke_data's fake Anti-UAV builder so this test and the data-layer test
# exercise the same fixture layout (real mp4 + real JSON annotations).
from smoke_data import fake_antiuav  # noqa: E402

fake_antiuav(raw, sequences=6, frames=12)
convert_dataset(spec, raw / "antiuav" / "300", variant="300")

records = load_index("antiuav", "300")
assigned, report = splits.split_records(records, val_fraction=0.34, seed=7, combo="smoke")
splits.persist(assigned, report, "smoke")
n_val = sum(1 for r in load_index("antiuav", "300") if r.split == "val")
print(f"  split: {len(records)} frames, {n_val} held out for val")
check("the split produced a val set", n_val > 0, f"{n_val} val frames")

# --- the thing under test ---------------------------------------------------
data_yaml = ev._cross_eval_yaml("antiuav", spec)
check("_cross_eval_yaml returns a path for a split dataset", data_yaml is not None, str(data_yaml))
assert data_yaml is not None
check("  data.yaml exists on disk", data_yaml.is_file())

text = data_yaml.read_text(encoding="utf-8")
print("\n  ---- generated data.yaml ----")
for line in text.splitlines():
    print(f"  {line}")
print("  ----\n")

check("  header names the one-source combo", "cross-antiuav" in text)
check("  header lists the source", "antiuav" in text)
check("  drone is class 0", "0: drone" in text)
check("  bird is class 1", "1: bird" in text)
check("  path is absolute", "path:" in text and str(tmp).replace("\\", "/") in text)
check("  train and val both declared", "train: images/train" in text and "val: images/val" in text)
check(
    "  val images were linked in",
    (data_yaml.parent / "images" / "val").is_dir(),
    str(data_yaml.parent / "images" / "val"),
)

# --- the None path: nothing held out ---------------------------------------
recs = load_index("antiuav", "300")
write_index("antiuav", "300", [replace(r, split="train") for r in recs])
check("_cross_eval_yaml returns None when no val split exists",
      ev._cross_eval_yaml("antiuav", spec) is None)

print()
print("=" * 74)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    raise SystemExit(1)
print("ALL CROSS-EVAL SMOKE CHECKS PASSED")
print("=" * 74)
raise SystemExit(0)
