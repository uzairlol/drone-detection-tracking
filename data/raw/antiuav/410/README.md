# data/raw/antiuav/410 — PLACEHOLDER EMPTY (fallback, not the default)

Drop zone for **Anti-UAV410**. IR only, Baidu Pan.

## Fill it with

```powershell
anti-uav download --dataset antiuav --variant 410
```

Needs a Baidu session cookie — see `data/raw/_credentials/README.md`. The command
reports `needs_credentials` and prints the share link and extraction code rather
than failing obscurely.

## Before you bother

**Prefer variant 300.** 410 is IR only and roughly 40 GB, and the registry keeps
it as a fallback for when the 300 Google Drive link is dead. A run on 410 is
**not comparable** to a run on 300 — different modality, different size. If you
end up using it, give it its own combo; do not fold it into `all4` and read the
row as if it meant the same thing.

```powershell
anti-uav convert --dataset antiuav --variant 410
```

Dataset card: `configs/datasets/registry.yaml` -> `datasets.antiuav.sources`
(`kind: baidu`)