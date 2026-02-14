# Kepler ROV

A robotics team based out of Aptos High School. We build an underwater ROV to compete in the MATE ROV competition.

## Laptop CV worker

A standalone laptop-side CV worker is provided in `laptop_cv_worker.py`.

It keeps control (`top-side.py`) and CV processing separate by:
- reading Pi MJPEG frames in a reader thread,
- always running inference on only the newest frame (drops stale frames),
- printing JSON lines with detections and per-frame timing (`capture_ms`, `infer_ms`, `effective_fps`).

Example:

```bash
python laptop_cv_worker.py \
  --mjpeg-url http://192.168.42.42:5001/video/Camera%201 \
  --model /path/to/yolov8n.onnx \
  --conf-threshold 0.35
```
