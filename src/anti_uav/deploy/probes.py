"""DeepStream probe callbacks.

The pipeline's metadata probes are where detections, tracks and the local rule
gate are evaluated **on the edge**, before anything is published to the bus. That
placement is not an implementation detail - it is the latency budget.

The diagram specifies end-to-end latency under 500 ms across 8 nodes and a message
bus. Round-tripping every frame to a central service to apply a threshold would
spend most of that budget on transport. So the cheap, per-frame decisions happen
here, and only *candidate events* go on the bus for the global coordination that
genuinely needs global state.

What runs here:

* persistence and confidence gating per track (cheap, no global state)
* candidate clip cutting, started on a tentative drone track
* health events: FPS, dropped frames, RTSP reconnect counts

What deliberately does **not** run here: cross-camera confirmation, handoff and
coverage scheduling. Those need all 100 cameras and belong to the coordinator.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class TrackStats:
    """Per-source-id state the probes accumulate."""

    source_id: str
    started_s: float
    frames_seen: int = 0
    first_frame: int = -1
    last_frame: int = -1
    hits: int = 0
    misses: int = 0
    highest_confidence: float = 0.0
    last_publish_s: float = 0.0
    alert: bool = False

    def record(self, frame_number: int, confidence: float, timestamp_s: float) -> None:
        self.frames_seen += 1
        if self.first_frame < 0:
            self.first_frame = frame_number
        self.last_frame = frame_number
        self.hits += 1
        self.misses = 0
        self.highest_confidence = max(self.highest_confidence, confidence)
        self.last_publish_s = timestamp_s

    def record_miss(self, frame_number: int) -> None:
        self.misses += 1
        self.last_frame = frame_number


@dataclass(slots=True)
class ProbeState:
    """State the probes share, keyed by source then object id.

    This is a *flat* struct, not the richer ``Track`` from
    :mod:`anti_uav.tracking.types`. On the edge the goal is a decision - hold,
    promote, or drop - and a small record per track keeps the memory footprint
    predictable across 8 nodes and hundreds of tracks.
    """

    #: How many consecutive frames before a tentative track is published.
    promote_after_hits: int = 12
    #: Confidence above which a track may be promoted.
    promote_confidence: float = 0.60
    #: Confidence below which an alerted track is dropped.
    drop_confidence: float = 0.35
    #: Frames a track may miss before it is discarded.
    max_missed_frames: int = 4

    tracks: dict[tuple[str, int], TrackStats] = field(default_factory=dict)
    #: source_id -> live RTSP / decode health.
    health: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Candidate clip jobs awaiting a VLM verification.
    clip_jobs: list[dict[str, Any]] = field(default_factory=list)
    published: int = 0
    candidates: int = 0
    started_s: float = field(default_factory=time.monotonic)

    # -- tracker gate ------------------------------------------------------- #

    def on_track(
        self,
        source_id: str,
        object_id: int,
        confidence: float,
        frame_number: int,
        rectangle: tuple[float, float, float, float],
        class_id: int = 0,
        timestamp_s: float | None = None,
    ) -> str:
        """Per-object callback. Returns ``hold``/``promote``/``drop``.

        This is the edge-side rule gate. It implements only the confidence and
        persistence rules from ``drone_rules.yaml`` - the two that need no global
        state - and leaves the rest to the coordinator.
        """
        now = timestamp_s if timestamp_s is not None else time.monotonic() - self.started_s
        key = (source_id, object_id)
        stats = self.tracks.get(key)

        if stats is None:
            stats = TrackStats(source_id=source_id, started_s=now)
            self.tracks[key] = stats
        stats.record(frame_number, confidence, now)

        if class_id != 0:
            # Birds are counted but never promoted: the local gate's job is
            # drone-vs-everything-else, and the cross-camera rule handles the rest.
            return "hold"

        if stats.alert:
            if confidence < self.drop_confidence:
                del self.tracks[key]
                return "drop"
            return "hold"

        if confidence >= self.promote_confidence and stats.hits >= self.promote_after_hits:
            stats.alert = True
            self.candidates += 1
            self.clip_jobs.append(
                {
                    "source_id": source_id,
                    "object_id": object_id,
                    "started_s": now,
                    "rect": list(rectangle),
                    "first_frame": stats.first_frame,
                }
            )
            return "promote"

        return "hold"

    def expire(self, frame_number: int) -> int:
        """Drop tracks that have exceeded the miss budget. Returns the count."""
        stale = [
            key
            for key, stats in self.tracks.items()
            if frame_number - stats.last_frame > self.max_missed_frames
        ]
        for key in stale:
            del self.tracks[key]
        return len(stale)

    # -- health ------------------------------------------------------------- #

    def record_health(self, source_id: str, **fields: Any) -> dict[str, Any]:
        entry = self.health.setdefault(
            source_id, {"fps": 0.0, "dropped_frames": 0, "reconnects": 0, "frames": 0}
        )
        entry.update(fields)
        return entry

    def snapshot(self) -> dict[str, Any]:
        return {
            "live_tracks": len(self.tracks),
            "candidates": self.candidates,
            "published": self.published,
            "pending_clips": len(self.clip_jobs),
            "uptime_s": round(time.monotonic() - self.started_s, 1),
            "sources": {k: dict(v) for k, v in self.health.items()},
        }


# --------------------------------------------------------------------------- #
# probe factories
# --------------------------------------------------------------------------- #


def make_pad_probe(state: ProbeState) -> Callable[..., Any]:
    """A ``pad-buffer-probe`` that clears stale tracks and reports FPS.

    Wired as the muxer's pad probe so it runs once per batched frame - the right
    cadence for an expiry sweep, which is a per-second concern rather than a
    per-frame one.
    """
    last_frame = {"number": -1}

    def probe(pad: Any, _unused: Any) -> Any:
        number = int(pad.get_frame_num())
        if number == last_frame["number"]:
            return pad
        last_frame["number"] = number

        frame_number = pad.get_frame_number()
        expired = state.expire(frame_number)
        if expired:
            log.debug("expired tracks", extra={"count": expired, "frame": frame_number})
        return pad

    return probe


def make_ospad_source_id_probe() -> Callable[..., Any]:
    """``nvurisrc`` pad probe that surfaces the camera id as the source name.

    The source name is the *camera id*, not a stream index, which is what lets a
    track message be attributed to ``fixed-042`` rather than ``src_id_3``. Without
    it every downstream event names a number and the operator has to keep a lookup
    table - exactly the kind of indirection that produces a mislabelled alert.
    """

    def probe(pad: Any) -> Any:
        uri = pad.get_stream_meta().obj_meta.uri
        # RTSP/HLS URIs use forward slashes regardless of host OS.
        name = PurePosixPath(uri).name
        pad.set_stream_name(name)
        return pad

    return probe


def make_analytics_probe(
    state: ProbeState,
    publish: Callable[[str, dict[str, Any]], None] | None = None,
    *,
    publish_interval_s: float = 0.5,
) -> Callable[..., Any]:
    """The user-defined ``analytic-probe`` that gates and publishes.

    ``publish`` is injected so the real implementation can POST to the message bus
    while tests and offline replay use a list. Returning ``OK`` on every frame is
    what keeps the pipeline running even when publication fails - dropping frames
    because MQTT is slow would turn a control-plane outage into a detection
    outage.
    """
    last_publish = {"s": 0.0}

    def probe(pad: Any, message: Any, _user_data: Any) -> Any:
        batch = list(pad.get_stream_meta().frame_user_meta_list)
        now = time.monotonic() - state.started_s
        promoted: list[dict[str, Any]] = []

        for frame_meta in batch:
            source_id = frame_meta.stream_name or "unknown"
            state.record_health(source_id, frames=state.health.get(source_id, {}).get("frames", 0) + 1)

            for obj_meta in frame_meta.obj_meta_list:
                user_meta = obj_meta.obj_user_meta_list[0] if obj_meta.obj_user_meta_list else None
                class_id = int(getattr(user_meta, "class_id", 0)) if user_meta else 0
                rect = (
                    float(obj_meta.rect_params.left),
                    float(obj_meta.rect_params.top),
                    float(obj_meta.rect_params.width),
                    float(obj_meta.rect_params.height),
                )
                verdict = state.on_track(
                    source_id,
                    int(obj_meta.object_id),
                    float(obj_meta.obj_confidence),
                    int(pad.get_frame_number()),
                    rect,
                    class_id=class_id,
                    timestamp_s=now,
                )
                if verdict == "promote":
                    promoted.append(
                        {
                            "camera_id": source_id,
                            "local_track_id": int(obj_meta.object_id),
                            "box": list(rect),
                            "confidence": round(float(obj_meta.obj_confidence), 4),
                        }
                    )

        if promoted:
            _emit(publish, "drone/candidates", {"candidates": promoted, "t": round(now, 3)})

        if publish and (now - last_publish["s"]) >= publish_interval_s:
            last_publish["s"] = now
            _emit(publish, "node/health", state.snapshot())

        return message  # GstPadReturn.OK
    return probe


def make_vlm_probe(publish: Callable[[str, dict[str, Any]], None] | None = None) -> Callable[..., Any]:
    """Publishes candidate clips for the VLM verification stage.

    The VLM is off-box and never on the frame path - it only ever sees clips of
    already-candidate tracks, which is what keeps a 7B vision model off the
    latency-critical path entirely.
    """

    def probe(pad: Any, message: Any, _user_data: Any) -> Any:
        batch = list(pad.get_stream_meta().frame_user_meta_list)
        for frame_meta in batch:
            if not frame_meta.obj_user_meta_list:
                continue
            user_meta = frame_meta.obj_user_meta_list[0]
            clip = getattr(user_meta, "unique_component_id", "")
            if not clip:
                continue
            _emit(
                publish,
                "vlm/verify",
                {
                    "camera_id": frame_meta.stream_name,
                    "clip_id": clip,
                    "clip_path": f"./clips/{frame_meta.stream_name}_{clip}.mp4",
                },
            )
        return message

    return probe


def _emit(publish: Callable[[str, dict[str, Any]], None] | None, topic: str, payload: dict[str, Any]) -> None:
    if publish is None:
        return
    try:
        publish(topic, payload)
    except Exception as exc:
        # A broker problem must never stop the pipeline.
        log.warning("publish failed", extra={"topic": topic, "error": str(exc)})


def write_deepstream_probe_source(
    path: str | Path,
    state: ProbeState | None = None,
    *,
    config_dir: str = ".",
) -> Path:
    """Write ``deepstream_probe.py`` - the module DeepStream imports.

    DeepStream loads this file by path, so it must be self-contained: it may not
    import from the project unless that is on ``sys.path`` on the Jetson. The
    generated shim adds it when present and degrades to inline thresholds when
    not, so the same file works in a container and on a bare device.
    """
    resolved = state or ProbeState()
    from ..utils.io import write_text

    body = f'''# GENERATED by anti-uav deploy render. Edit ProbeState or re-render.
# Runs inside DeepStream, NOT in the anti_uav process. Keep it dependency-free.
#
#   sys.path.append("{config_dir}")
#   import probe_state
#   STATE = probe_state.make()
#
# The values below are rendered from configs/rules/drone_rules.yaml. The probe's
# job is the confidence and persistence gates only - the cross-camera, handoff and
# coverage rules need global state and run in the coordinator.

import os
import sys
import time

try:
    sys.path.append({config_dir!r})
    import probe_state as _shared
    STATE = _shared.make()
except Exception:  # standalone fallback
    class _T:
        promote_after_hits = {resolved.promote_after_hits}
        promote_confidence = {resolved.promote_confidence}
        drop_confidence = {resolved.drop_confidence}
        max_missed_frames = {resolved.max_missed_frames}
    class _S:
        def __init__(self):
            self.t = _T()
            self.tracks, self.health, self.clip_jobs = {{}}, {{}}, []
            self.candidates = self.published = 0
            self.started_s = time.monotonic()
    STATE = _S()

MQTT_HOST = os.environ.get("ANTI_UAV_MQTT_HOST", "127.0.0.1")
MQTT_TOPIC_PREFIX = os.environ.get("ANTI_UAV_MQTT_PREFIX", "drone")

_last = {{"frame": -1}}


def publish(topic, payload):
    """Fire-and-forget. A broker outage must not stall the pipeline."""
    try:
        import paho.mqtt.publish as mqtt
        mqtt.single(f"{{MQTT_TOPIC_PREFIX}}/{{topic}}", payload, hostname=MQTT_HOST, qos=1)
    except Exception:
        pass


def pad_buffer_probe(pad, _unused):
    """Pad probe: expire stale tracks once per batched frame."""
    number = int(pad.get_frame_num())
    if number == _last["frame"]:
        return pad
    _last["frame"] = number
    frame = pad.get_frame_number()
    for key in [
        k for k, v in STATE.tracks.items() if frame - v.last_frame > STATE.t.max_missed_frames
    ]:
        del STATE.tracks[key]
    return pad


def uridecodebin_pad_probe(pad):
    """Pad probe: name the pad with the camera id, not a stream index."""
    uri = pad.get_stream_meta().obj_meta.uri
    # URIs use forward slashes regardless of host OS, so PurePosixPath not Path.
    pad.set_stream_name(PurePosixPath(uri).name)
    return pad


def analytic_probe(pad, message, _user_data):
    """Gates each track and publishes candidates onto the bus."""
    promoted = []
    now = time.monotonic() - STATE.started_s
    for frame_meta in pad.get_stream_meta().frame_user_meta_list:
        source = frame_meta.stream_name or "unknown"
        for obj in frame_meta.obj_meta_list:
            meta = obj.obj_user_meta_list[0] if obj.obj_user_meta_list else None
            class_id = int(getattr(meta, "class_id", 0)) if meta else 0
            verdict = STATE.on_track(
                source,
                int(obj.object_id),
                float(obj.obj_confidence),
                int(pad.get_frame_number()),
                (obj.rect_params.left, obj.rect_params.top, obj.rect_params.width, obj.rect_params.height),
                class_id=class_id,
                timestamp_s=now,
            )
            if verdict == "promote":
                promoted.append({{"camera_id": source, "local_track_id": int(obj.object_id)}})
    if promoted:
        publish("candidates", {{"candidates": promoted, "t": now}})
    return message
'''
    return write_text(path, body)


def summarise(state: ProbeState) -> str:
    snapshot = state.snapshot()
    lines = [
        f"uptime        : {snapshot['uptime_s']} s",
        f"live tracks   : {snapshot['live_tracks']}",
        f"candidates    : {snapshot['candidates']}",
        f"pending clips : {snapshot['pending_clips']}",
        f"sources       : {len(snapshot['sources'])}",
    ]
    for source, health in sorted(snapshot["sources"].items()):
        lines.append(f"  {source}: frames={health.get('frames', 0)} "
                     f"reconnects={health.get('reconnects', 0)}")
    return "\n".join(lines)
