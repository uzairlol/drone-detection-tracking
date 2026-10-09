# artifacts/ — generated output. Nothing here is an input.

```
artifacts/
  runs/      training runs: args.yaml, results.csv, weights/, batch previews
  exports/   ONNX, TensorRT engines, nvinfer configs, label files
  tracks/    MOT row dumps from replay / track-eval
  preds/     annotated frames written by `anti-uav predict`
  deploy/    rendered DeepStream pipelines, one per node
```

All of it is reproducible and all of it is gitignored. `data/` and `artifacts/`
are the only mutable trees; `ANTI_UAV_DATA_DIR` relocates both together.

## runs/

`artifacts/runs/<family>/<family>__<combo>/`

```powershell
anti-uav matrix list                              # what exists
anti-uav list-runs                               # same, with a weights column
anti-uav evaluate --run artifacts/runs/yolo11n/all4 --cross
```

The `weights/best.pt` file is the resume/skip signal: the matrix planner and
`scripts/train_matrix.ps1` both skip any run whose `best.pt` already exists, so
re-running after a crash or an interruption is always safe. If a run "did not
happen", check for that file before re-launching.

## exports/

Written by `anti-uav export`. ONNX, a TensorRT engine, and the matching `nvinfer`
config with the two class names the pipeline declares (`drone`, `bird`).

FP16 is unsafe on Pascal and the profile system will say so rather than produce
an engine that silently loses precision.

## deploy/

Written by `anti-uav deploy render`. One DeepStream pipeline per node, plus the
on-edge probe source. It warns per node when the engine file is missing and
prints the exact `trtexec` command to build it — it does not fail the render,
because the engine is normally built on the target device.

## Deliberately not tracked

Weights, ONNX and engines are all re-fetchable or re-derivable, so `.gitignore`
excludes `*.pt`, `*.onnx`, `*.engine` repo-wide. A stray `git add -A` cannot
commit a 100 GB tree or a set of trained weights.