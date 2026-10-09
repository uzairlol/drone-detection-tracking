# data/raw/antiuav/600 — PLACEHOLDER EMPTY (fallback, not the default)

Drop zone for **Anti-UAV600**. IR only, ModelScope.

## Fill it with

```powershell
anti-uav download --dataset antiuav --variant 600
```

From ModelScope (`ly261666/3rd_Anti-UAV`). **No account needed at all** — this is
the one Anti-UAV source that authenticates with nothing, which is why it is
registered as the fallback if Google Drive is blocked on your network.

## Before you bother

**Prefer variant 300.** 600 is IR only and ~120 GB, twice the size. A run on 600
is **not comparable** to a run on 300.

The combo for it is present but **disabled** in `configs/matrix.yaml`
(`antiuav600`, commented out at the bottom of the file). To use it, uncomment
that block and set `enabled: true` — the intent is recorded there rather than
lost. Keep it in its own combo.

```powershell
anti-uav convert --dataset antiuav --variant 600
```

Dataset card: `configs/datasets/registry.yaml` -> `datasets.antiuav.sources`
(`kind: modelscope`)