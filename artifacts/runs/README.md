# artifacts/runs — training runs

`artifacts/runs/<family>/<family>__<combo>/`, e.g. `yolo11n/yolo11n__all4/`.

Produced by `anti-uav train` or `anti-uav matrix run`. 14 runs are planned: two
model families (YOLO11n, RT-DETR-x2) across seven combos.

Per run: `args.yaml`, `results.csv` (the training curve the API and UI read),
`train_batch*.jpg` previews, `weights/best.pt`, `weights/last.pt`.

`weights/best.pt` is the skip/resume signal — the planner skips any run that
already has one. See `../README.md`.