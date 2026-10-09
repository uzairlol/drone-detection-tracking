# data/manifests — what actually landed, as opposed to what was published

Machine-written. Every stage that measures something drops a JSON manifest here,
and these files are the record of ground truth about your data.

## What writes here

| file | written by | records |
| --- | --- | --- |
| `stats_<dataset>_<variant>.json` | `anti-uav stats` | frame counts, target-size distribution, class histogram, bird-negative table |
| `dedup_<dataset>_<variant>.json` | `anti-uav dedup` | pHash near-duplicate clusters spanning the split |
| `<dataset>.json` | `anti-uav dedup` | per-source dedup summary |

## Why the published figures are not the truth

`configs/datasets/registry.yaml` carries the **publishers'** numbers —
`approx_frames: 104760` for dvb, `2800000` for mmuav. They are useful for
capacity planning and for spotting a download that silently truncated.

They are not what you have. Conversion drops frames (focus scoring, unreadable
sources, stride), subsetting cuts sequences, and a Baidu transfer that died at 80%
still leaves a directory that looks populated. The manifests are what the tooling
actually observed, so when a metric disagrees with expectation, read these before
suspecting the model.

## The bird-negative table

`anti-uav stats` prints, per dataset, whether bird negatives exist at all. This
matters more than it sounds: three of the four datasets are drone-only, so a
precision number measured on them cannot falsify a false-positive claim. The
evaluator attaches that caveat to the metric rather than leaving you to remember,
and this manifest is where you check whether the caveat applies to a run.