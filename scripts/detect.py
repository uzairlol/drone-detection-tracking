"""Load the trained YOLO11n drone/bird model and detect in an image or a video.

    python scripts/detect.py <image-or-video> [--conf 0.25] [--tracker botsort]

Wraps anti_uav.detection.predictor.Detector, which is what `anti-uav predict`
calls under the hood, so this path and the CLI path use the same weights
resolution, thresholds and class names.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anti_uav.config.schema import TrackerName
from anti_uav.detection.predictor import Detector
from anti_uav.utils.imaging import list_images

# Accepts a run name, a run directory, or a .pt file directly.
RUN = "artifacts/runs/yolo11n/yolo11n__dvb"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="image, video, or a directory of frames")
    parser.add_argument("--run", default=RUN, help="run name, run dir, or .pt path")
    parser.add_argument("--conf", type=float, default=None, help="detection threshold")
    parser.add_argument(
        "--tracker",
        default="none",
        choices=[t.value for t in TrackerName],
        help="associate detections over time; 'none' is per-frame detection",
    )
    parser.add_argument("--device", default=None, help="e.g. 0, cpu")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    source = Path(args.source)
    video = source.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    tracker = None if args.tracker == "none" else TrackerName(args.tracker)

    with Detector(
        args.run, conf=args.conf, device=args.device, imgsz=args.imgsz
    ) as detector:
        print(
            f"loaded {detector.weights.name}  "
            f"classes={detector.class_names}  conf={detector.conf}"
        )

        if video:
            frames = 0
            totals = {"drone": 0, "bird": 0}
            for result in detector.predict_stream(
                source, tracker=tracker, max_frames=args.max_frames
            ):
                frames += 1
                totals["drone"] += len(result.drone_detections)
                totals["bird"] += len(result.bird_detections)
                if frames <= 5 or frames % 50 == 0:
                    tracked = len(result.tracks)
                    print(
                        f"  frame {result.frame_index:>5}  "
                        f"drones={len(result.drone_detections):>3}  "
                        f"birds={len(result.bird_detections):>3}  "
                        f"tracks={tracked:>3}"
                    )
            print(
                f"\n{frames} frames  "
                f"drone detections={totals['drone']}  "
                f"bird detections={totals['bird']}"
            )
        else:
            # predict_stream takes video only - frames_from_video rejects a
            # directory - so walk the images here and feed them one at a time.
            paths = [source] if source.is_file() else list_images(source)
            if args.max_frames is not None:
                paths = paths[: args.max_frames]
            frames = 0
            totals = {"drone": 0, "bird": 0}
            for path in paths:
                result = detector.predict_image(path)
                frames += 1
                totals["drone"] += len(result.drone_detections)
                totals["bird"] += len(result.bird_detections)
                if frames <= 5 or frames % 50 == 0:
                    print(
                        f"  {path.name:<20} "
                        f"drones={len(result.drone_detections):>3}  "
                        f"birds={len(result.bird_detections):>3}"
                    )
            print(
                f"\n{frames} images  "
                f"drone detections={totals['drone']}  "
                f"bird detections={totals['bird']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
