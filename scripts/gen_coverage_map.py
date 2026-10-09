"""Generate ``configs/coverage/coverage_map.yaml`` for the 100-camera fleet.

The system block diagram specifies a concrete fleet: 80 fixed AXIS P3275-LVE
overview cameras, 20 AXIS V5925 PTZ detail cameras, sharded across 8 headless
DeepStream worker nodes at ~13 cameras each. Hand-maintaining 100 camera entries
is not realistic, so the map is generated from a small set of site parameters and
committed. Re-run after changing anything:

    uv run python scripts/gen_coverage_map.py

Deliberately synthesised. Intrinsics come from each model's published field of
view rather than a calibration target, and every position is a plausible-looking
site coordinate. Replace both before the coverage planner's decisions mean
anything operational - see docs/DATASETS.md -> "Coverage map provenance".
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anti_uav.config.loader import config_path

# --- site geometry (metres, site-local ENU: +x east, +y north, +z up) --------
SITE_WIDTH_M = 820.0
SITE_DEPTH_M = 640.0
POLE_HEIGHT_M = 6.0
PTZ_HEIGHT_M = 9.0

N_FIXED = 80
N_PTZ = 20
N_NODES = 8

# Published fields of view for the two camera models in the block diagram.
P3275_LVE = {  # 8 MP, 16:9, ~97 deg horizontal at full wide
    "model": "AXIS P3275-LVE",
    "width_px": 3840,
    "height_px": 2160,
    "hfov_deg": 97.0,
    "note": "8 MP fixed overview, H.264/H.265 Zipstream",
}
V5925 = {  # 1080p60, 30x optical zoom; wide end ~90 deg
    "model": "AXIS V5925",
    "width_px": 1920,
    "height_px": 1080,
    "hfov_deg": 90.0,
    "note": "1080p60 PTZ detail, 30x optical, ONVIF G/M/S + VISCA over IP",
}

# A protected asset for each entry in the coverage planner's priority queue.
PROTECTED_ZONES = [
    {
        "id": "pz-substation-north",
        "rect": (120.0, 430.0, 260.0, 540.0),
        "priority": 10,
        "height_m": 18.0,
        "description": "HV substation - highest-value asset on site",
    },
    {
        "id": "pz-tank-farm",
        "rect": (540.0, 300.0, 700.0, 430.0),
        "priority": 8,
        "height_m": 14.0,
        "description": "Fuel tank farm",
    },
    {
        "id": "pz-control-building",
        "rect": (330.0, 60.0, 450.0, 160.0),
        "priority": 9,
        "height_m": 12.0,
        "description": "Site control building",
    },
    {
        "id": "pz-perimeter-east",
        "rect": (740.0, 100.0, 810.0, 560.0),
        "priority": 6,
        "height_m": 12.0,
        "description": "East perimeter - most common ingress vector",
    },
    {
        "id": "pz-approach-west",
        "rect": (10.0, 180.0, 90.0, 480.0),
        "priority": 7,
        "height_m": 20.0,
        "description": "West approach corridor",
    },
]

GEOFENCES = [
    {
        "id": "gf-critical-infrastructure",
        "rect": (100.0, 40.0, 720.0, 560.0),
        "action": "escalate",
        "height_m": 0.0,
        "description": "Everything over the asset area escalates to critical",
    },
    {
        "id": "gf-no-fly-advisory",
        "rect": (-120.0, -120.0, 940.0, 760.0),
        "action": "alert",
        "height_m": 0.0,
        "description": "Outer advisory ring",
    },
    {
        "id": "gf-tower-exclusion",
        "rect": (240.0, 240.0, 300.0, 300.0),
        "action": "suppress",
        "height_m": 60.0,
        "description": "Communications mast - structure-induced false positives",
    },
]


def fx_from_hfov(width_px: int, hfov_deg: float) -> float:
    """Pinhole focal length in pixels from the horizontal field of view."""
    return (width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def build_intrinsics(spec: dict, hfov_deg: float | None = None) -> dict:
    width = spec["width_px"]
    height = spec["height_px"]
    fov = hfov_deg if hfov_deg is not None else spec["hfov_deg"]
    focal = fx_from_hfov(width, fov)
    return {
        "fx_px": round(focal, 2),
        "fy_px": round(focal, 2),
        "cx_px": width / 2.0,
        "cy_px": height / 2.0,
        "width_px": width,
        "height_px": height,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        "hfov_deg": round(fov, 2),
    }


def rect_to_polygon(rect: tuple[float, float, float, float]) -> list[float]:
    """``(min_x, min_y, max_x, max_y)`` -> flat ``[x0,y0, x1,y1, ...]``."""
    min_x, min_y, max_x, max_y = rect
    return [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]


def perimeter_positions(count: int) -> list[tuple[float, float, float, float]]:
    """Evenly spaced positions around the site boundary.

    Returns ``(x, y, yaw_deg, pitch_deg)``. Each camera looks inward at the site
    centre and tilts up ~8 deg: it is watching the sky above the perimeter, not
    the fence line.
    """
    cx, cy = SITE_WIDTH_M / 2.0, SITE_DEPTH_M / 2.0
    half_w, half_d = SITE_WIDTH_M / 2.0, SITE_DEPTH_M / 2.0

    corners = [
        (0.0, 0.0),
        (half_w * 2.0, 0.0),
        (half_w * 2.0, half_d * 2.0),
        (0.0, half_d * 2.0),
    ]
    # Segment lengths so spacing is uniform rather than per-side uniform.
    lengths = [
        math.dist(corners[0], corners[1]),
        math.dist(corners[1], corners[2]),
        math.dist(corners[2], corners[3]),
        math.dist(corners[3], corners[0]),
    ]
    perimeter = sum(lengths)

    out: list[tuple[float, float, float, float]] = []
    for i in range(count):
        target = perimeter * (i + 0.5) / count
        walked = 0.0
        for index, length in enumerate(lengths):
            if walked + length >= target:
                t = (target - walked) / length
                ax, ay = corners[index]
                bx, by = corners[(index + 1) % 4]
                x = ax + (bx - ax) * t
                y = ay + (by - ay) * t
                break
            walked += length
        else:  # pragma: no cover - float edge case
            x, y = corners[0]

        dx, dy = cx - x, cy - y
        yaw = math.degrees(math.atan2(dx, dy)) % 360.0
        out.append((x, y, round(yaw, 2), -8.0))
    return out


def interior_positions(count: int) -> list[tuple[float, float, float, float]]:
    """A 2-row interior lattice looking outward at the protected assets."""
    cx, cy = SITE_WIDTH_M / 2.0, SITE_DEPTH_M / 2.0
    cols = math.ceil(math.sqrt(count * SITE_WIDTH_M / SITE_DEPTH_M)) or 1
    rows = math.ceil(count / cols)

    margin_x, margin_y = 150.0, 150.0
    xs = [
        margin_x + (SITE_WIDTH_M - 2 * margin_x) * (c + 0.5) / cols for c in range(cols)
    ]
    ys = [margin_y + (SITE_DEPTH_M - 2 * margin_y) * (r + 0.5) / rows for r in range(rows)]

    out: list[tuple[float, float, float, float]] = []
    for index in range(count):
        x, y = xs[index % cols], ys[(index // cols) % rows]
        yaw = math.degrees(math.atan2(cx - x, cy - y)) % 360.0
        out.append((x, y, round(yaw, 2), -10.0))
    return out


def ptz_positions(count: int) -> list[tuple[float, float, float, float]]:
    """PTZ mounts interleaved along the perimeter, at pole height."""
    cx, cy = SITE_WIDTH_M / 2.0, SITE_DEPTH_M / 2.0
    out: list[tuple[float, float, float, float]] = []
    for i in range(count):
        angle = 2.0 * math.pi * (i / count)
        rx, ry = SITE_WIDTH_M * 0.34, SITE_DEPTH_M * 0.34
        x = cx + rx * math.cos(angle)
        y = cy + ry * math.sin(angle)
        yaw = math.degrees(math.atan2(cx - x, cy - y)) % 360.0
        out.append((x, y, round(yaw, 2), -6.0))
    return out


def assign_nodes(camera_ids: list[str]) -> dict[str, str]:
    """Shard across the 8 worker nodes, ~13 cameras each, evenly.

    Round-robin over a shuffled-but-deterministic order so no node ends up with
    all the PTZ cameras (which would make one node's failure far more damaging
    than the block diagram's "nodes are independent failure domains" assumes).
    """
    buckets: dict[str, list[str]] = {f"node-{i:02d}": [] for i in range(N_NODES)}
    for index, camera_id in enumerate(camera_ids):
        buckets[f"node-{index % N_NODES:02d}"].append(camera_id)
    return {cam: node for node, cams in buckets.items() for cam in cams}


def build_map() -> dict:
    cameras: list[dict] = []

    n_perimeter = 40
    positions = perimeter_positions(n_perimeter) + interior_positions(N_FIXED - n_perimeter)

    for index, (x, y, yaw, pitch) in enumerate(positions, start=1):
        cameras.append(
            {
                "id": f"fixed-{index:03d}",
                "node": "node-00",
                "role": "fixed",
                "model": P3275_LVE["model"],
                "mount": {
                    "position_m": [round(x, 2), round(y, 2), POLE_HEIGHT_M],
                    "yaw_deg": yaw,
                    "pitch_deg": pitch,
                    "roll_deg": 0.0,
                },
                "intrinsics": build_intrinsics(P3275_LVE),
                # No horizon_y: a 6 m pole looking at a flat site has its horizon
                # above the top of frame. The rule layer derives it from the
                # mount pitch when this is null.
                "horizon_y": None,
                "enabled": True,
                "tags": ["overview", f"pos-{'perimeter' if index <= n_perimeter else 'interior'}"],
            }
        )

    for index, (x, y, yaw, pitch) in enumerate(ptz_positions(N_PTZ), start=1):
        cameras.append(
            {
                "id": f"ptz-{index:03d}",
                "node": "node-00",
                "role": "ptz",
                "model": V5925["model"],
                "mount": {
                    "position_m": [round(x, 2), round(y, 2), PTZ_HEIGHT_M],
                    "yaw_deg": yaw,
                    "pitch_deg": pitch,
                    "roll_deg": 0.0,
                },
                "intrinsics": build_intrinsics(V5925),
                "horizon_y": None,
                "enabled": True,
                "tags": ["detail", "ptz"],
                "ptz": {
                    "enabled": True,
                    "pan_min_deg": -170.0,
                    "pan_max_deg": 170.0,
                    "tilt_min_deg": -90.0,
                    "tilt_max_deg": 90.0,
                    "zoom_min": 1.0,
                    "zoom_max": 30.0,
                    "max_pan_speed_deg_s": 30.0,
                    "max_tilt_speed_deg_s": 20.0,
                    # ~1.2 s to physically settle after a move. This is what the
                    # 2.0 s handoff lead in drone_rules.yaml has to cover.
                    "settle_time_s": 1.2,
                    "overshoot_ratio": 0.08,
                },
            }
        )

    node_of = assign_nodes([cam["id"] for cam in cameras])
    for cam in cameras:
        cam["node"] = node_of[cam["id"]]

    return {
        "schema_version": 1,
        "frame": "site-local-ENU",
        "default_ground_plane_z_m": 0.0,
        "nominal_range_m": 320.0,
        "cameras": cameras,
        "protected_zones": [
            {
                "id": zone["id"],
                "polygon": rect_to_polygon(zone["rect"]),
                "priority": zone["priority"],
                "height_m": zone["height_m"],
                "description": zone["description"],
            }
            for zone in PROTECTED_ZONES
        ],
        "geofences": [
            {
                "id": fence["id"],
                "polygon": rect_to_polygon(fence["rect"]),
                "action": fence["action"],
                "height_m": fence["height_m"],
                "description": fence["description"],
            }
            for fence in GEOFENCES
        ],
        "overlap_zones": [],
        "notes": (
            f"Generated by scripts/gen_coverage_map.py. {N_FIXED} fixed + {N_PTZ} PTZ "
            f"across {N_NODES} worker nodes over a {SITE_WIDTH_M:.0f} x {SITE_DEPTH_M:.0f} m site. "
            "Intrinsics are derived from published fields of view, NOT from a calibration "
            "target, and positions are synthetic. Replace both with survey data before "
            "trusting the coverage planner or the handoff risk estimator. "
            "overlap_zones is empty on purpose - the runtime computes pairwise FOV "
            "intersection on load and caches it."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(config_path("coverage", "coverage_map.yaml")),
        help="destination path",
    )
    parser.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = parser.parse_args()

    data = build_map()
    header = (
        "# -------------------------------------------------------------------------\n"
        f"# Coverage map - {len(data['cameras'])} cameras across {N_NODES} DeepStream worker nodes.\n"
        "#\n"
        "# GENERATED FILE. Edit scripts/gen_coverage_map.py and re-run:\n"
        "#     uv run python scripts/gen_coverage_map.py\n"
        "#\n"
        "# Section 3 of the system block diagram. Consumed by:\n"
        "#   * tracking/coordination/coverage.py  - which cameras see where\n"
        "#   * tracking/coordination/scheduler.py - keeps >=1 camera per protected zone\n"
        "#   * tracking/coordination/risk.py      - FOV exit prediction\n"
        "#   * tracking/coordination/handoff.py   - scored receiver selection\n"
        "#   * tracking/coordination/ptz.py       - ONVIF absolute-move targets\n"
        "#   * rules/engine.py                    - above-horizon + geofence tests\n"
        "#\n"
        "# GitOps-managed: this file plus drone_rules.yaml are the only two inputs\n"
        "# the per-node DeepStream configs are rendered from.\n"
        "# -------------------------------------------------------------------------\n"
    )

    body = yaml.safe_dump(data, sort_keys=False, default_flow_style=False, width=100, allow_unicode=True)
    rendered = header + "\n" + body

    if args.stdout:
        print(rendered)
        return 0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")
    print(f"wrote {out} ({len(data['cameras'])} cameras, {len(data['protected_zones'])} zones)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
