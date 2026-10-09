# data/interim/mmuav/full — write with `anti-uav convert --dataset mmuav --variant full`

Produced from `data/raw/mmuav/full/` (RGB + IR + Event). See `../README.md`.

**You almost certainly want `../subset` instead.** `full` is ~400 GB extracted;
the 150-sequence RGB subset is ~12 GB and is enough for every experiment in the
matrix.

Use `full` only if you need the IR channel or the complete sequence list. Note the
source excludes `**/event_frame/**` and `**/sot_groundtruth/**` by default —
pass `--modalities rgb,ir` to keep IR.

The registry defaults mmuav to `subset`, so a plain `anti-uav build --combo mmuav`
reads the subset. Naming `full` explicitly everywhere is the only way to be sure
which one a number came from.