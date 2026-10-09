# data/interim/antiuav/410 — write with `anti-uav convert --dataset antiuav --variant 410`

Produced from `data/raw/antiuav/410/` (Baidu Pan, IR only). See `../README.md`.

**Fallback variant.** `300` is the default and the only one with both RGB and IR.
A number derived from 410 is not comparable to one derived from 300 — different
modality, different size. Keep it in its own combo.

The registry declares `default_variant: "300"` for antiuav, so plain
`anti-uav build --combo antiuav` reads `300`. Naming `410` explicitly everywhere
is the only way to be sure which one a result came from.

No visibility-flag guarantee beyond what the release ships, and no bird negatives.