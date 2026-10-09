# artifacts/exports — ONNX, TensorRT, nvinfer configs

Written by `anti-uav export`:

```powershell
anti-uav export --run artifacts/runs/yolo11n/all4 --formats onnx,tensorrt --fp16
```

Per run: `best_<imgsz>.onnx`, `best_<imgsz>.engine`, `best_nvinfer.txt`, a
`drone_bird.labels` file, and `export_report.json` recording what was produced and
anything that failed.

FP16 is unsafe on Pascal (sm_61 has no half-throughput path) and the profile
system will refuse rather than emit an engine that quietly loses precision.