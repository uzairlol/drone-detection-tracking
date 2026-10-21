"""Run the trained detector over a folder of unlabelled video clips and summarise.

    python scripts/sweep_clips.py data/unlabeled/drone_clips data/unlabeled/bird_clips

There are no ground-truth boxes in these clips, so this cannot produce precision,
recall, mAP or F1. What it *can* do is answer the weaker question that still
matters: does the detector fire on the right kind of thing?

A clip filed under ``drone_clips`` should produce drone detections and few bird
detections; a clip under ``bird_clips`` should produce bird detections and few
drone detections. The last one is the falsifiable direction - a bird called a
drone is the failure the deployed system exists to prevent.

So this reports counts, not scores, and a clip that is mislabelled at the clip
level (stock footage of "drone" that actually shows a bird, or a bird clip with
a drone in frame) will show up as an apparent error that is not one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anti_uav.config.schema import TrackerName
from anti_uav.detection.predictor import Detector

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def sweep(
    detector: Detector,
    folder: Path,
    tracker: TrackerName | None,
    stride: int,
    max_frames: int,
    conf: float,
) -> dict[str, object]:
    clips = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    rows: list[dict[str, object]] = []

    for index, clip in enumerate(clips, 1):
        drones = birds = frames = 0
        track_ids: set[int] = set()
        drone_track_ids: set[int] = set()
        peak_conf: dict[str, float] = {"drone": 0.0, "bird": 0.0}

        for result in detector.predict_stream(
            clip, tracker=tracker, stride=stride, max_frames=max_frames, conf=conf
        ):
            frames += 1
            for det in result.drone_detections:
                drones += 1
                peak_conf["drone"] = max(peak_conf["drone"], det.confidence)
                if det.track_id is not None:
                    drone_track_ids.add(det.track_id)
            for det in result.bird_detections:
                birds += 1
                peak_conf["bird"] = max(peak_conf["bird"], det.confidence)
            track_ids.update(result.tracks)

        rows.append(
            {
                "clip": clip.name,
                "frames": frames,
                "drone_dets": drones,
                "bird_dets": birds,
                "tracks": len(track_ids),
                "drone_tracks": len(drone_track_ids),
                "drone_peak_conf": round(peak_conf["drone"], 3),
                "bird_peak_conf": round(peak_conf["bird"], 3),
            }
        )
        print(
            f"[{index:>2}/{len(clips)}] {clip.name[:38]:<38} "
            f"frames={frames:>4} drones={drones:>4} birds={birds:>4} "
            f"tracks={len(track_ids):>3}"
        )

    return {
        "folder": folder.name,
        "clips": len(clips),
        "total_frames": sum(int(r["frames"]) for r in rows),
        "drone_dets": sum(int(r["drone_dets"]) for r in rows),
        "bird_dets": sum(int(r["bird_dets"]) for r in rows),
        "clips_with_drone": sum(1 for r in rows if int(r["drone_dets"]) > 0),
        "clips_with_bird": sum(1 for r in rows if int(r["bird_dets"]) > 0),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", help="folders of clips to sweep")
    parser.add_argument("--run", default="artifacts/runs/yolo11n/yolo11n__dvb")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=15, help="sample every Nth frame")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument(
        "--tracker",
        default="none",
        choices=["none", *(t.value for t in TrackerName)],
        help="'none' is per-frame detection with no association",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default="artifacts/sweeps/unlabeled_clips.json")
    args = parser.parse_args()

    tracker = None if args.tracker == "none" else TrackerName(args.tracker)

    summaries = []
    with Detector(args.run, conf=args.conf, device=args.device) as detector:
        print(f"weights={detector.weights.name} classes={detector.class_names} "
              f"conf={args.conf} stride={args.stride} tracker={args.tracker}\n")
        for folder in args.folders:
            path = Path(folder)
            print(f"--- {path} ---")
            summaries.append(
                sweep(detector, path, tracker, args.stride, args.max_frames, args.conf)
            )
            print()

    print("=" * 96)
    print(f"{'folder':<14}{'clips':>6}{'frames':>8}{'drones':>8}{'birds':>8}"
          f"{'clips w/ drone':>16}{'clips w/ bird':>15}")
    print("-" * 96)
    for s in summaries:
        print(
            f"{s['folder']!s:<14}{s['clips']:>6}{s['total_frames']:>8}"
            f"{s['drone_dets']:>8}{s['bird_dets']:>8}"
            f"{s['clips_with_drone']:>16}{s['clips_with_bird']:>15}"
        )
    print("=" * 96)
    print("Counts, not scores: these clips carry no ground-truth boxes, so")
    print("precision/recall/mAP cannot be computed here. A drone detection in a")
    print("bird clip is a candidate false positive worth eyeballing, but stock")
    print("footage may genuinely contain the other subject.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"run": args.run, "conf": args.conf, "stride": args.stride,
             "tracker": args.tracker, "summaries": summaries},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
