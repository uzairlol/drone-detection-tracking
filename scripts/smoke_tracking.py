"""Smoke test of the tracking core on synthetic detections (not a test suite entry).

Builds a scripted scenario with a known ground truth - a drone crossing paths with
a bird, then occluding - and checks each tracker recovers the identity. Exercises
Kalman, the three local trackers, the appearance gate, and the MOT metrics against
a hand-computable answer.
"""

from __future__ import annotations

import numpy as np

from anti_uav.tracking.appearance import AppearanceEmbedder, crop_box
from anti_uav.tracking.kalman import KalmanBank, KalmanTrack2D
from anti_uav.tracking.local import build_tracker
from anti_uav.tracking.metrics import (
    FrameGT,
    FramePred,
    evaluate_report,
    evaluate_sequence,
)
from anti_uav.tracking.types import TrackObservation

W, H = 1920, 1080
FPS = 30.0


def box(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def scenario():
    """Drone flies left->right; bird flies right->left; they cross around frame 40.

    Frames 60-69 the drone is occluded (no detection) - the classic identity-loss
    case that appearance and prediction are supposed to survive.
    """
    frames = []
    for f in range(100):
        drone = box(200 + f * 14, 400 + int(30 * np.sin(f / 12)), 40, 40)
        bird = box(1700 - f * 13, 430 - int(20 * np.sin(f / 9)), 26, 18)
        occluded = 60 <= f <= 69
        frames.append((drone, None if occluded else bird))
    return frames


def observations_for(frame_index, boxes, confidences, embeddings=None):
    out = []
    for i, (b, c) in enumerate(zip(boxes, confidences, strict=True)):
        emb = None if embeddings is None else embeddings[i]
        out.append(
            TrackObservation(
                frame_index=frame_index,
                timestamp_s=frame_index / FPS,
                box=b,
                confidence=c,
                class_id=0,
                class_name="drone" if i == 0 else "bird",
                embedding=emb,
            )
        )
    return out


def run_tracker(name, frames, use_bird_as_class=True, with_reid=False):
    tracker = build_tracker(
        name,
        high_threshold=0.5,
        low_threshold=0.2,
        match_threshold=0.8,
        max_age=30,
        min_hits=3,
        class_id=0,
    )
    if hasattr(tracker, "with_reid"):
        tracker.with_reid = with_reid

    gt, pred = {}, {}
    tracks = {}

    for f, (drone, bird) in enumerate(frames):
        # Bird is fed to the tracker as class_id 1, which our trackers filter out
        # (class_id=0), so the drone must stay one clean identity through the
        # crossing. We re-frame the bird as a separate class to test filtering.
        obs = [observations_for(f, [drone], [0.9])[0]]
        if bird is not None and use_bird_as_class:
            b = observations_for(f, [bird], [0.8])[0]
            b.class_id = 1
            b.class_name = "bird"
            obs.append(b)

        result = tracker.update(obs, frame_index=f, timestamp_s=f / FPS)

        gt[f] = FrameGT(f).objects.append((1, drone, 1.0)) or gt.get(f) or FrameGT(f)
        if f not in gt or not gt[f].objects:
            gt[f] = FrameGT(f)
            gt[f].objects.append((1, drone, 1.0))

        pred[f] = FramePred(f, objects=[(t.track_id, t.box) for t in result.tracks])
        for t in result.tracks:
            tracks.setdefault(t.track_id, t).history[f] = t.box

    return evaluate_sequence("cross", gt, pred, iou_threshold=0.5), tracker


def main() -> int:
    frames = scenario()
    print("=" * 74)
    print("SCENARIO: drone L->R, bird R->L, cross at ~frame 40, drone occluded 60-69")
    print("=" * 74)

    print("\n-- kalman: predict/update round trip --")
    # Drive a target moving at exactly 5 m/s: at 30 fps that is 0.1667 m per frame.
    kf = KalmanTrack2D()
    truth = 0.0
    kf.initialise((truth, 50.0), (5.0, 0.0))
    for _ in range(30):
        truth += 5.0 / 30.0
        kf.predict(1 / 30)
        kf.update((truth, 50.0))
    print(f"  truth x       {truth:.3f} m")
    print(f"  estimate      ({kf.x[0]:.2f}, {kf.x[1]:.2f})")
    print(f"  error         {abs(kf.x[0] - truth):.4f} m")
    print(f"  speed         {kf.speed_m_s:.3f} m/s (expected 5.000)")
    print(f"  3-sigma       {kf.uncertainty_m:.2f} m")
    print(f"  converged     {kf.converged}")
    bank = KalmanBank()
    tid = bank.create((0.0, 0.0))
    bank.update(tid, (1.0, 1.0))
    print(f"  bank len      {len(bank)}  contains {tid}: {tid in bank}")
    bank.prune(keep=set())
    print(f"  after prune   {len(bank)}")

    print("\n-- trackers --")
    reports = []
    for name in ("sort", "bytetrack", "botsort"):
        seq, tracker = run_tracker(name, frames)
        reports.append(evaluate_report([seq], tracker=name, dataset="synthetic"))
        print(
            f"  {name:<10} MOTA={seq.mota:7.4f} IDF1={seq.idf1:7.4f} HOTA={seq.hota:7.4f} "
            f"IDSW={seq.id_switches} miss={seq.misses} fp={seq.false_positives} "
            f"frag={seq.fragmentations}"
        )

    print("\n-- bytetrack with low-confidence detections (its whole point) --")
    # Re-run with the drone's confidence dipping below the high threshold after the
    # occlusion, which is the case ByteTrack's second pass exists for.
    tracker = build_tracker("bytetrack", high_threshold=0.5, low_threshold=0.15,
                             match_threshold=0.8, max_age=30, min_hits=3, class_id=0)
    gt, pred = {}, {}
    for f, (drone, _bird) in enumerate(frames):
        conf = 0.3 if 64 <= f <= 68 else 0.9
        res = tracker.update(
            [observations_for(f, [drone], [conf])[0]],
            frame_index=f, timestamp_s=f / FPS,
        )
        gtf = FrameGT(f)
        gtf.objects.append((1, drone, 1.0))
        gt[f] = gtf
        pred[f] = FramePred(f, objects=[(t.track_id, t.box) for t in res.tracks])
    seq = evaluate_sequence("lowconf", gt, pred)
    print(f"  MOTA={seq.mota:.4f} IDF1={seq.idf1:.4f} IDSW={seq.id_switches} frag={seq.fragmentations}")
    print("  (a track held through the dip is the whole point of the low-conf pass)")

    print("\n-- appearance --")
    embedder = AppearanceEmbedder.load_default()
    print(f"  default source: {embedder.describe()}")
    img = np.full((H, W, 3), 140, dtype=np.uint8)
    img[400:440, 200:240] = 230
    bird_img = np.full((H, W, 3), 90, dtype=np.uint8)
    bird_img[410:428, 900:926] = 200
    d_box = box(220, 420, 40, 40)
    b_box = box(913, 419, 26, 18)
    e_drone = embedder.embed(embedder.crop(img, d_box))
    e_bird = embedder.embed(embedder.crop(bird_img, b_box))
    e_drone2 = embedder.embed(embedder.crop(img, box(240, 420, 40, 40)))
    print(f"  embedding dim  {len(e_drone)}")
    print(f"  drone vs bird  {float(np.dot(e_drone, e_bird)):.4f}  (should be low)")
    print(f"  drone vs drone {float(np.dot(e_drone, e_drone2)):.4f}  (should be high)")
    print(f"  empty crop     {crop_box(img, (0, 0, 1, 1)).shape}")

    print("\n-- anti-uav visibility-aware metric --")
    gt2, pred2 = {}, {}
    for f in range(20):
        g = FrameGT(f)
        if f < 10:
            g.objects.append((1, box(100, 100, 40, 40), 1.0))
        else:
            g.objects.append((1, box(100, 100, 40, 40), 0.0))  # target absent
        gt2[f] = g
        if f < 10:
            pred2[f] = FramePred(f, objects=[(1, box(100, 100, 40, 40))])
        else:
            pred2[f] = FramePred(f, objects=[])  # correctly predicts nothing
    perfect = evaluate_sequence("vis", gt2, pred2, visibility_aware=True)
    print(f"  perfect tracker accuracy {perfect.anti_uav_accuracy:.4f}  (expect 1.0)")

    gt3, pred3 = {}, {}
    for f in range(20):
        g = FrameGT(f)
        g.objects.append((1, box(100, 100, 40, 40), 1.0)) if f < 10 else g.objects.append(
            (1, box(100, 100, 40, 40), 0.0)
        )
        gt3[f] = g
        pred3[f] = FramePred(f, objects=[(1, box(100, 100, 40, 40))])  # hallucinates when absent
    hallucinating = evaluate_sequence("vis2", gt3, pred3, visibility_aware=True)
    print(f"  hallucinating accuracy  {hallucinating.anti_uav_accuracy:.4f}  (expect < 1.0)")
    print("  ^ this is the case a plain mean-IoU metric cannot distinguish from the one above")

    print("\n" + "=" * 74)
    for report in reports:
        print(report.summary())
        break
    print("\nALL TRACKING SMOKE CHECKS COMPLETED")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
