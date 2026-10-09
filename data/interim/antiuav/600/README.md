# data/interim/antiuav/600 — write with `anti-uav convert --dataset antiuav --variant 600`

Produced from `data/raw/antiuav/600/` (ModelScope, IR only). See `../README.md`.

**Fallback variant, and the largest download in the matrix at ~120 GB.** Prefer
`300`. Nothing here is comparable to a 300 result.

The combo for it exists but is **disabled** in `configs/matrix.yaml` (`antiuav600`,
commented out at the bottom). Uncomment and set `enabled: true` to use it.

No account is needed to fetch this one, which is the only reason it is registered
— it is the fallback that still works when Google Drive is blocked.