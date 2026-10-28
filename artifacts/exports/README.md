# artifacts/exports — ONNX, TensorRT, nvinfer configs

Written by `anti-uav export`:

```powershell
anti-uav export --run artifacts/runs/yolo11n/yolo11n__all4 --formats onnx,tensorrt --fp16
```

Per run, when requested: `best_<imgsz>.onnx`, `best_<imgsz>.engine`,
`best_nvinfer.txt`, a `drone_bird.labels` file, and `export_report.json` recording
what was produced and anything that failed.

> **At handover only the ONNX export exists** (`yolo11n__dvb/best_640.onnx`).
> There is no TensorRT engine; build it on the target device. See
> [`../../HANDOVER.md`](../../HANDOVER.md).

FP16 is unsafe on Pascal (sm_61 has no half-throughput path) and the profile
system will refuse rather than emit an engine that quietly loses precision.