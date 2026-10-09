# artifacts/deploy — rendered DeepStream pipelines

Written by `anti-uav deploy render`:

```powershell
anti-uav deploy render --output artifacts/deploy --rtsp-host 10.0.0.5 --engine ./model.engine
anti-uav deploy render --node node-00     # one node only, while iterating
```

One `pipeline_node-NN.txt` per camera node, plus `deepstream_probe.py`, the
on-edge probe source.

A missing engine file produces a **warning per node** listing the exact `trtexec`
command to build it — it does not fail the render, because the engine is normally
built on the target device rather than on the training box.

The rendered configs declare exactly two classes, `drone` and `bird`, matching
`unified_labels` in `configs/datasets/registry.yaml`.