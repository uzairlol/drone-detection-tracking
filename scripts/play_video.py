"""Play a video (or loop a folder of videos) with live drone/bird detection boxes and a metrics strip.

    python scripts/play_video.py data/unlabeled/bird_clips/video_2.mp4
    python scripts/play_video.py clip.mp4 --tracker botsort --conf 0.25
    python scripts/play_video.py data/unlabeled/drone_clips --random   # random clip, loops until quit
    python scripts/play_video.py data/unlabeled/drone_clips            # every clip in order, loops until quit

Barebones on purpose: a window, boxes, and a text strip. No UI framework.

Controls: q or ESC to quit, SPACE to pause, +/- to change the sampling stride.

On a slow GPU this cannot hit real-time. Inference is the bottleneck, not drawing,
so `--stride N` runs the model on every Nth frame and reuses that result in
between - the video still plays at wall-clock speed, and the boxes stay glued to
their target for N frames instead of flickering. `--every-frame` opts back into
running the model on every frame when you want the true cost.
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from pathlib import Path

import cv2

from anti_uav.config.schema import TrackerName
from anti_uav.detection.predictor import Detector, FrameResult, overlay

BGR = {
    "drone": (48, 48, 235),
    "bird": (40, 176, 64),
    "unknown": (200, 200, 200),
}


def draw_hud(
    canvas,
    *,
    frame_no: int,
    total: int,
    fps: float,
    stride: int,
    infer_ms: float,
    counts: Counter,
    running: Counter,
    tracks: int,
    source_fps: float,
) -> None:
    """Bottom-left metrics strip. Keeps out of the way of the boxes."""
    rows = [
        f"{source_fps:.0f}fps src | play {fps:5.1f}fps | infer {infer_ms:5.1f}ms | stride {stride}",
        f"frame {frame_no}/{total}  |  this frame: {counts['drone']} drone  {counts['bird']} bird"
        f"  |  live tracks {tracks}",
        f"cumulative: {running['drone']} drone  {running['bird']} bird",
    ]
    height = 12 + 20 * len(rows)
    cv2.rectangle(canvas, (0, canvas.shape[0] - height), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
    y = canvas.shape[0] - height + 22
    for row in rows:
        cv2.putText(canvas, row, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20


def collect_videos(source: Path, random_pick: bool) -> list[Path]:
    """Return the list of videos to play. A single file returns [file]; a folder returns all videos in it."""
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if source.is_file():
        return [source]
    if not source.is_dir():
        return []
    videos = sorted(p for p in source.iterdir() if p.suffix.lower() in exts)
    if random_pick and videos:
        return [random.choice(videos)]
    return videos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="video file, or a folder to play/loop all clips from")
    parser.add_argument("--run", default="artifacts/runs/yolo11n/yolo11n__dvb")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--tracker", default="none", choices=[t.value for t in TrackerName])
    parser.add_argument("--stride", type=int, default=4, help="run the model every Nth frame (biggest latency lever)")
    parser.add_argument("--every-frame", action="store_true", help="disable stride sampling")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--scale", type=float, default=1.0, help="window scale, 1.0 = native (auto-capped to fit 1280x720)")
    parser.add_argument("--random", action="store_true", help="pick a random video from the given folder")
    parser.add_argument("--fast", action="store_true", help="speed preset: stride 16, imgsz 256, tracker none, GPU")
    args = parser.parse_args()

    if args.fast:
        args.stride = 16
        args.imgsz = 256
        args.tracker = "none"
        args.device = "0"
        args.conf = 0.25

    source = Path(args.video)
    videos = collect_videos(source, args.random)
    if not videos:
        print(f"no videos found at: {source}")
        return 1

    tracker = None if args.tracker == "none" else TrackerName(args.tracker)
    stride = 1 if args.every_frame else max(1, args.stride)
    running: Counter = Counter()
    paused = False
    frame_no = 0
    cached: FrameResult | None = None
    window = "anti-uav"
    index = 0

    print(f"{len(videos)} video(s) queued  tracker={args.tracker} conf={args.conf} stride={stride}")
    print("q/ESC quit | SPACE pause | +/- stride")

    with Detector(args.run, conf=args.conf, device=args.device, imgsz=args.imgsz) as detector:
        play_start = time.perf_counter()
        played = 0

        while True:  # loop over videos until q/ESC
            video = videos[index % len(videos)]
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                print(f"could not open {video}, skipping")
                index += 1
                continue

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            # fit the window to a 1080p-friendly max (never upscale small videos)
            max_w, max_h = 1280, 720
            fit_scale = 1.0
            if src_w and src_h:
                fit_scale = min(max_w / src_w, max_h / src_h, 1.0)
            effective_scale = min(args.scale, fit_scale)

            print(
                f"[{index % len(videos) + 1}/{len(videos)}] {video.name}  {total} frames @ {source_fps:.0f}fps  "
                f"{src_w}x{src_h}"
            )

            frame_no = 0
            cached = None

            def present(
                canvas: FrameResult,
                frame,
                frame_no: int,
                infer_ms: float,
                total_frames: int,
                # Bound here rather than read from the enclosing scope: these are
                # per-video values, and a closure that captured the loop
                # variable would read whatever the *next* iteration set. The
                # function is redefined each time so it would still be correct
                # today, but that is a coincidence, not a guarantee.
                scale: float = effective_scale,
                fps: float = source_fps,
            ):
                """Draw, show, and pump the keyboard. Returns False to quit."""
                nonlocal played, paused, stride
                for det in canvas.detections:
                    running[det.class_name] += 1
                view = overlay(frame, canvas, colors=BGR)
                if scale != 1.0:
                    view = cv2.resize(view, None, fx=scale, fy=scale)
                draw_hud(
                    view,
                    frame_no=frame_no,
                    total=total_frames,
                    fps=played / max(1e-6, time.perf_counter() - play_start),
                    stride=stride,
                    infer_ms=infer_ms,
                    counts=Counter(d.class_name for d in canvas.detections),
                    running=running,
                    tracks=len(canvas.tracks),
                    source_fps=fps,
                )
                cv2.imshow(window, view)
                played += 1

                delay = 1.0 / fps if not paused else 0.0
                if delay:
                    time.sleep(max(0.0, delay - (time.perf_counter() - play_start) / max(1, played)))

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    return False
                if key == ord(" "):
                    paused = not paused
                if key in (ord("+"), ord("=")):
                    stride = max(1, stride - 1)
                if key == ord("-"):
                    stride += 1
                return True

            if tracker is not None:
                # Tracking needs consecutive frames, so the stream is consumed once in
                # order. Sampling happens inside the stream rather than around it.
                for result in detector.predict_stream(video, tracker=tracker, conf=args.conf):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, result.frame_index)
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if not present(result, frame, result.frame_index + 1, 0.0, total):
                        cap.release()
                        return 0
            else:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_no += 1
                    infer_ms = 0.0
                    if frame_no % stride == 0 or cached is None:
                        t0 = time.perf_counter()
                        cached = detector.predict_image(frame, conf=args.conf)
                        infer_ms = (time.perf_counter() - t0) * 1000.0
                    if not present(cached, frame, frame_no, infer_ms, total):
                        cap.release()
                        return 0

            cap.release()
            index += 1

    cv2.destroyAllWindows()
    print(
        f"\ndone: {frame_no} frames  "
        f"drone={running['drone']}  bird={running['bird']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
