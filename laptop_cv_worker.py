#!/usr/bin/env python3
"""Laptop-side CV worker for ROV camera MJPEG feeds.

Responsibilities:
- Connect to the Pi MJPEG endpoint.
- Decode frames with OpenCV in a dedicated reader thread.
- Run lightweight object detection (YOLO ONNX via OpenCV DNN).
- Emit detection results with per-frame performance metrics.

Example:
    python laptop_cv_worker.py \
      --mjpeg-url http://192.168.42.42:5001/video/Camera%201 \
      --model /path/to/yolov8n.onnx \
      --conf-threshold 0.35
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.request import Request, urlopen

import cv2
import numpy as np


@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]


@dataclass
class FramePacket:
    frame_id: int
    frame_bgr: np.ndarray
    capture_ms: float
    captured_at: float


class LatestFrameBuffer:
    """Thread-safe single-slot frame buffer.

    Writer always overwrites with the newest frame.
    Reader can fetch the latest packet and skip stale ones automatically.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packet: Optional[FramePacket] = None
        self._frame_counter = 0

    def put(self, frame_bgr: np.ndarray, capture_ms: float) -> None:
        with self._lock:
            self._frame_counter += 1
            self._packet = FramePacket(
                frame_id=self._frame_counter,
                frame_bgr=frame_bgr,
                capture_ms=capture_ms,
                captured_at=time.monotonic(),
            )

    def get_latest(self) -> Optional[FramePacket]:
        with self._lock:
            return self._packet


class MjpegReaderThread(threading.Thread):
    """Continuously reads an MJPEG stream and updates a latest-frame buffer."""

    def __init__(self, mjpeg_url: str, buffer: LatestFrameBuffer, timeout_s: float = 5.0) -> None:
        super().__init__(daemon=True)
        self.mjpeg_url = mjpeg_url
        self.buffer = buffer
        self.timeout_s = timeout_s
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        request = Request(self.mjpeg_url, headers={"User-Agent": "rov-cv-worker/1.0"})

        while not self._stop_event.is_set():
            try:
                with urlopen(request, timeout=self.timeout_s) as response:
                    byte_buffer = bytearray()
                    while not self._stop_event.is_set():
                        chunk = response.read(4096)
                        if not chunk:
                            raise ConnectionError("MJPEG stream ended")
                        byte_buffer.extend(chunk)

                        start = byte_buffer.find(b"\xff\xd8")
                        end = byte_buffer.find(b"\xff\xd9")
                        if start == -1 or end == -1 or end <= start:
                            continue

                        jpg_bytes = bytes(byte_buffer[start : end + 2])
                        del byte_buffer[: end + 2]

                        capture_t0 = time.perf_counter()
                        frame_np = np.frombuffer(jpg_bytes, dtype=np.uint8)
                        frame_bgr = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                        capture_ms = (time.perf_counter() - capture_t0) * 1000.0

                        if frame_bgr is None:
                            continue

                        self.buffer.put(frame_bgr=frame_bgr, capture_ms=capture_ms)
            except Exception as exc:  # Retry loop for stream interruptions
                print(f"[reader] stream error: {exc}; reconnecting in 1.0s")
                time.sleep(1.0)


class YoloOnnxDetector:
    """Lightweight YOLO ONNX detector using OpenCV DNN."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float,
        nms_threshold: float,
        input_size: int,
    ) -> None:
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        h, w = frame_bgr.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame_bgr,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        pred = self.net.forward()

        return self._postprocess(pred, frame_w=w, frame_h=h)

    def _postprocess(self, pred: np.ndarray, frame_w: int, frame_h: int) -> List[Detection]:
        # Common YOLO ONNX outputs are often [1, N, 5+nc] or [1, 5+nc, N].
        arr = np.squeeze(pred)

        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)

        if arr.ndim == 2 and arr.shape[0] < arr.shape[1] and arr.shape[0] <= 8:
            arr = arr.T

        boxes_xywh: List[List[float]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        for row in arr:
            if row.shape[0] < 6:
                continue

            if row.shape[0] == 6:
                x1, y1, x2, y2, conf, class_id = row.tolist()
                if conf < self.conf_threshold:
                    continue
                x = max(0, min(frame_w - 1, int(x1)))
                y = max(0, min(frame_h - 1, int(y1)))
                w = max(1, min(frame_w - x, int(x2 - x1)))
                h = max(1, min(frame_h - y, int(y2 - y1)))
                boxes_xywh.append([x, y, w, h])
                confidences.append(float(conf))
                class_ids.append(int(class_id))
                continue

            cx, cy, bw, bh, obj = row[:5]
            if obj < 1e-6:
                continue

            class_scores = row[5:]
            class_id = int(np.argmax(class_scores))
            cls_conf = float(class_scores[class_id])
            conf = float(obj * cls_conf)
            if conf < self.conf_threshold:
                continue

            scale_x = frame_w / float(self.input_size)
            scale_y = frame_h / float(self.input_size)

            x = int((cx - bw / 2.0) * scale_x)
            y = int((cy - bh / 2.0) * scale_y)
            w = int(bw * scale_x)
            h = int(bh * scale_y)

            x = max(0, min(frame_w - 1, x))
            y = max(0, min(frame_h - 1, y))
            w = max(1, min(frame_w - x, w))
            h = max(1, min(frame_h - y, h))

            boxes_xywh.append([x, y, w, h])
            confidences.append(conf)
            class_ids.append(class_id)

        if not boxes_xywh:
            return []

        indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences, self.conf_threshold, self.nms_threshold)
        if len(indices) == 0:
            return []

        detections: List[Detection] = []
        flat_indices = np.array(indices).reshape(-1).tolist()
        for idx in flat_indices:
            x, y, w, h = boxes_xywh[idx]
            detections.append(
                Detection(
                    class_id=class_ids[idx],
                    confidence=float(confidences[idx]),
                    bbox_xyxy=(int(x), int(y), int(x + w), int(y + h)),
                )
            )

        return detections


def run_worker(args: argparse.Namespace) -> None:
    frame_buffer = LatestFrameBuffer()
    reader = MjpegReaderThread(args.mjpeg_url, frame_buffer, timeout_s=args.stream_timeout)
    detector = YoloOnnxDetector(
        model_path=args.model,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        input_size=args.input_size,
    )

    reader.start()
    print(f"[worker] connected reader thread for: {args.mjpeg_url}")

    last_processed_id = 0
    frame_count = 0
    fps_window_start = time.monotonic()

    try:
        while True:
            packet = frame_buffer.get_latest()
            if packet is None or packet.frame_id == last_processed_id:
                time.sleep(0.001)
                continue

            # Producer/consumer behavior: always process newest frame; stale frame IDs are skipped.
            last_processed_id = packet.frame_id

            infer_t0 = time.perf_counter()
            detections = detector.infer(packet.frame_bgr)
            infer_ms = (time.perf_counter() - infer_t0) * 1000.0
            filtered_detections = [det for det in detections if det.confidence >= args.conf_threshold]
            count = len(filtered_detections)

            frame_vis = packet.frame_bgr.copy()
            for det in filtered_detections:
                x1, y1, x2, y2 = det.bbox_xyxy
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det.class_id} {det.confidence:.2f}"
                text_y = y1 - 8 if y1 > 16 else y1 + 18
                cv2.putText(
                    frame_vis,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame_vis,
                f"{args.count_label}: {count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if args.show_preview:
                if args.preview_scale != 1.0:
                    frame_vis = cv2.resize(
                        frame_vis,
                        dsize=None,
                        fx=args.preview_scale,
                        fy=args.preview_scale,
                        interpolation=cv2.INTER_LINEAR,
                    )

                cv2.imshow("ROV CV Worker", frame_vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("[worker] preview exit requested, stopping...")
                    break

            frame_count += 1
            elapsed = time.monotonic() - fps_window_start
            effective_fps = frame_count / elapsed if elapsed > 0 else 0.0

            result = {
                "frame_id": packet.frame_id,
                "capture_ms": round(packet.capture_ms, 3),
                "infer_ms": round(infer_ms, 3),
                "effective_fps": round(effective_fps, 2),
                "frame_age_ms": round((time.monotonic() - packet.captured_at) * 1000.0, 3),
                "detections": [
                    {
                        "class_id": det.class_id,
                        "confidence": round(det.confidence, 4),
                        "bbox_xyxy": list(det.bbox_xyxy),
                    }
                    for det in filtered_detections
                ],
                "count": count,
            }
            print(json.dumps(result, separators=(",", ":")))
    except KeyboardInterrupt:
        print("\n[worker] interrupted, stopping...")
    finally:
        reader.stop()
        if args.show_preview:
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Laptop-side MJPEG CV worker")
    parser.add_argument("--mjpeg-url", required=True, help="MJPEG URL (e.g. http://<pi>:5001/video/Camera%201)")
    parser.add_argument("--model", required=True, help="Path to lightweight ONNX detector model")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--nms-threshold", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--input-size", type=int, default=640, help="Square detector input size")
    parser.add_argument("--stream-timeout", type=float, default=5.0, help="MJPEG socket read timeout (seconds)")
    parser.add_argument("--show-preview", action="store_true", help="Show an annotated OpenCV preview window")
    parser.add_argument("--preview-scale", type=float, default=1.0, help="Display scale factor for preview window")
    parser.add_argument("--count-label", default="Green crabs", help="Summary label text shown in preview overlay")
    return parser.parse_args()


if __name__ == "__main__":
    run_worker(parse_args())
