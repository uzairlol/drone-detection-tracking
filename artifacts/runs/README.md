# artifacts/runs — training runs

`artifacts/runs/<family>/<family>__<combo>/`, e.g. `yolo11n/yolo11n__all8/`.

Produced by `anti-uav train` or `anti-uav matrix run`. 28 runs are planned: two
model families (YOLO11n, RT-DETR-x2) across fourteen combos. **At handover only
`yolo11n/yolo11n__dvb/` exists**; the other 27 are still to run. See
[`../../HANDOVER.md`](../../HANDOVER.md).

Per run: `args.yaml`, `results.csv` (the training curve the API and UI read),
`train_batch*.jpg` previews, `weights/best.pt`, `weights/last.pt`.

`weights/best.pt` is the skip/resume signal — the planner skips any run that
already has one. See `../README.md`.