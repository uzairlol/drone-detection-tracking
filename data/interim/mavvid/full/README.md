# data/interim/mavvid/full — write with `anti-uav convert --dataset mavvid`

Produced from `data/raw/mavvid/full/`. See `../README.md` for the index format.

Expected: `index.jsonl`, `convert_report.json`, `frames/<sequence>/rgb/*.jpg`,
`labels/<sequence>/rgb/*.txt`.

## What to check

`convert_report.json` records which label spellings were observed. MAV-VID has
shipped with `drone`, `Drone` and `0` across releases; the converter normalises
case and numeric ids, then asserts every observed label was mapped. If a label
had no mapping the conversion reports it rather than dropping the box — read that
line before trusting a frame count.

`anti-uav stats --dataset mavvid` should show the largest median target in the
matrix (~171 px) and no tiling requirement.