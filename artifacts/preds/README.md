# artifacts/preds — annotated frames from `anti-uav predict`

```powershell
anti-uav predict --run artifacts/runs/yolo11n/all4 --source frame.jpg --save out/
```

Draws boxes, class labels and tracker ids onto frames so you can look at what the
detector actually did. Useful for the question a confusion matrix cannot answer:
is this a miss, or a box on the wrong object?

Note that `predict` also *prints* the per-detection table rather than a count,
because 40 detections at conf 0.25 and 3 at 0.8 are completely different
situations.