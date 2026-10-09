# data/raw/mmuav/full — PLACEHOLDER EMPTY (only if you have the disk)

Drop zone for the **complete** MM-UAV release: RGB + IR + Event.

## You probably do not want this

**~400 GB extracted.** Use `../subset/README.md` instead. The 150-sequence RGB
subset is ~12 GB and is enough for every experiment in the matrix.

Use `full` only if you need the IR channel or the full sequence list.

## Fill it with

```powershell
anti-uav download --dataset mmuav --variant full
```

Same Baidu Pan constraints as the subset — cookie, captcha, throttling. Budget
several attempts. `docs/DATASETS.md` has the disk arithmetic.

## Excluded by default

The source's `exclude_globs` drops two directories unless you ask for them
explicitly, because together they are a large fraction of a payload that the RGB
detector cannot use:

- `**/event_frame/**` — ~1/3 of the download, unusable for RGB training
- `**/sot_groundtruth/**` — single-object tracking GT, not needed (MM-UAV ships
  multi-object MOT GT, which is what `track-eval` reads)

Pass `--modalities rgb,ir` to keep the IR channel.

Then:

```powershell
anti-uav convert --dataset mmuav --variant full
```

Note that `configs/datasets/registry.yaml` sets `default_variant: subset` for
mmuav, so plain `anti-uav build --combo mmuav` reads the subset. Naming `full`
explicitly everywhere is the only way to be sure which one a number came from.