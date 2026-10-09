"""API smoke test using FastAPI's TestClient (not a test suite entry)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from anti_uav.api.app import app

client = TestClient(app)
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}{('  ' + detail) if detail else ''}")
    if not condition:
        failures.append(name)


print("=" * 74)
print("API")
print("=" * 74)

r = client.get("/api/health")
check("GET /api/health", r.status_code == 200, f"status {r.status_code}")
health = r.json()
print(f"       status={health['status']} profile={health['profile']} runs={health['runs']}")
print(f"       warnings={health['warnings']}")

r = client.get("/api/datasets")
check("GET /api/datasets", r.status_code == 200, f"status {r.status_code}")
datasets = r.json()
check("  4 datasets", len(datasets) == 4, f"got {len(datasets)}")
birdless = [d["alias"] for d in datasets if not d["has_bird_negatives"]]
check("  drone-only flagged", set(birdless) == {"antiuav", "mmuav"}, str(birdless))
for d in datasets:
    print(f"       {d['alias']:9s} sources={len(d['sources'])} "
          f"median_px={d['median_target_px']} birds={d['has_bird_negatives']}")

r = client.get("/api/runs")
check("GET /api/runs", r.status_code == 200, f"status {r.status_code} (empty is fine)")
runs = r.json()
print(f"       {len(runs)} run(s)")

r = client.get("/api/matrix")
check("GET /api/matrix", r.status_code == 200, f"status {r.status_code}")
matrix = r.json()
check("  14 cells (2 models x 7 combos)", len(matrix["cells"]) == 14, f"got {len(matrix['cells'])}")
print(f"       models={matrix['models']}")
print(f"       combos={matrix['combos']}")
missing = [c["map50"] for c in matrix["cells"] if c["map50"] is None]
check("  untrained cells report null, not 0", all(m is None for m in missing),
      f"{len(missing)} of {len(matrix['cells'])} untrained")

r = client.get("/api/rules")
check("GET /api/rules", r.status_code == 200, f"status {r.status_code}")
rules = r.json()
check("  thresholds present", "confidence" in rules["rules"], str(list(rules["rules"])[:5]))
print(f"       initiate={rules['rules']['confidence']['initiate']} "
      f"maintain={rules['rules']['confidence']['maintain']}")
if rules["problems"]:
    for p in rules["problems"]:
        print(f"       problem: {p}")

r = client.post("/api/rules/explain", json={
    "box": [400, 200, 424, 224],
    "confidence": 0.85,
    "hits": 30,
    "duration_s": 2.0,
    "camera_id": "fixed-001",
    "image_height_px": 1080,
    "global_id": 7,
    "velocity_m_s": [12.0, 0.0],
    "ground_xy": [300.0, 300.0],
    "ground_z": 25.0,
    "sightings": {
        "7": [
            {"camera_id": "fixed-001", "node": "node-00", "t": 1.9, "ground_xy": [300, 300]},
            {"camera_id": "fixed-040", "node": "node-00", "t": 1.9, "ground_xy": [301, 300]},
        ]
    },
})
check("POST /api/rules/explain", r.status_code == 200, f"status {r.status_code}")
if r.status_code == 200:
    result = r.json()
    print(f"       alerted={result['alerted']} severity={result['severity']} "
          f"first_failure={result['first_failure']}")
    for g in result["gates"]:
        print(f"         [{g['outcome']:<7}] {g['name']:<14} {g['detail'][:60]}")
    check("  a strong track alerts", result["alerted"], f"gates={[g['outcome'] for g in result['gates']]}")

r = client.post("/api/rules/explain", json={
    "box": [400, 1010, 418, 1020],
    "confidence": 0.30,
    "hits": 4,
    "duration_s": 0.2,
    "camera_id": "fixed-001",
    "image_height_px": 1080,
    "global_id": None,
    "velocity_m_s": [0.0, 0.0],
    "ground_xy": [300.0, 300.0],
    "ground_z": 25.0,
})
if r.status_code == 200:
    weak = r.json()
    failing = [g["name"] for g in weak["gates"] if g["outcome"] == "fail"]
    print(f"\n  bird-like track: alerted={weak['alerted']} failing={failing}")
    check("  a bird-like track is rejected", not weak["alerted"] and len(failing) >= 3, str(failing))
    check("  no global identity blocks the alert",
          "cross_camera" in failing, str(failing))

r = client.get("/api/coverage")
check("GET /api/coverage", r.status_code == 200, f"status {r.status_code}")
if r.status_code == 200:
    cov = r.json()
    s = cov["summary"]
    print(f"       cameras={s['cameras']} fixed={s['fixed']} ptz={s['ptz']} "
          f"nodes={s['nodes']} overlaps={s['overlap_zones']}")
    print(f"       protected={s['protected_zones']} uncovered={s['uncovered_zones']} "
          f"all_covered={s['all_zones_covered']}")
    check("  100 cameras", s["cameras"] == 100, str(s["cameras"]))
    check("  80 fixed / 20 PTZ", s["fixed"] == 80 and s["ptz"] == 20, f"{s['fixed']}/{s['ptz']}")
    check("  every zone covered", s["all_zones_covered"], str(s["uncovered_zones"]))
    print(f"       fixed-covered zones: {s['fixed_covered_zones']}")
    print(f"       PTZ-only zones:       {s['ptz_only_zones']}")
    check("  fixed + PTZ coverage accounts for all 5 zones",
          len(s["fixed_covered_zones"]) + len(s["ptz_only_zones"]) == 5,
          f"{len(s['fixed_covered_zones'])} + {len(s['ptz_only_zones'])}")

r = client.get("/api/tracking?dataset=mmuav")
check("GET /api/tracking", r.status_code == 200, f"status {r.status_code}")
if r.status_code == 200:
    print(f"       note: {r.json()['note'][:80]}")

r = client.get("/api/samples?combo=all4")
check("GET /api/samples", r.status_code == 200, f"status {r.status_code}")
if r.status_code == 200:
    print(f"       note: {r.json()['note'][:90]}")

r = client.get("/")
check("GET / (the UI)", r.status_code == 200, f"status {r.status_code}")
check("  references the vendored chart.js",
      "chart.umd.min.js" in r.text and "/static/app.js" in r.text)
check("  no CDN references",
      "http://" not in r.text.replace('xmlns="http://', '').replace("http://www", ""),
      "offline requirement")

r = client.get("/static/app.js")
check("GET /static/app.js", r.status_code == 200, f"status {r.status_code}")
r = client.get("/static/app.css")
check("GET /static/app.css", r.status_code == 200, f"status {r.status_code}")
r = client.get("/static/vendor/chart.umd.min.js")
check("GET chart.umd.min.js", r.status_code == 200,
      f"status {r.status_code}, {len(r.content):,} bytes")

r = client.get("/api/runs/does-not-exist")
check("unknown run -> 404", r.status_code == 404, f"status {r.status_code}")
r = client.get("/api/datasets/nope/stats")
check("unknown dataset -> 404", r.status_code == 404, f"status {r.status_code}")
r = client.post("/api/predict", json={"run": "nope", "image_path": "nope.jpg"})
check("bad predict -> 4xx", 400 <= r.status_code < 500, f"status {r.status_code}")

print()
print("=" * 74)
if failures:
    print(f"{len(failures)} API CHECK(S) FAILED: {failures}")
    raise SystemExit(1)
print("ALL API SMOKE CHECKS PASSED")
print("=" * 74)
raise SystemExit(0)
