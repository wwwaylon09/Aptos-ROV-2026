#!/usr/bin/env python3
"""Laptop-side CV worker for ROV camera MJPEG feeds.

Responsibilities:
- Connect to the Pi MJPEG endpoint.
- Decode frames with OpenCV in a dedicated reader thread.
- Run lightweight object detection (YOLO ONNX via OpenCV DNN).
- Emit detection results with per-frame performance metrics.

Example:
    # Edit configuration variables in this file, then run:
    python laptop_cv_worker.py
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import Request, urlopen

import cv2
import numpy as np


# Worker configuration
MJPEG_URL = "http://192.168.42.42:5001/video/Camera%201"
MODEL_PATH = "crab_detection_v1.onnx"
MODEL_NUM_CLASSES = 3
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.45
INPUT_SIZE = 640
STREAM_TIMEOUT_S = 5.0
SHOW_PREVIEW = False
PREVIEW_SCALE = 1.0
COUNT_LABEL = "Green crabs"


def resolve_model_path(model_path: str) -> Path:
    """Resolve model path across Windows/WSL/POSIX environments."""

    raw_path = model_path.strip().strip('"').strip("'")
    expanded = os.path.expandvars(os.path.expanduser(raw_path))
    path = Path(expanded)

    if path.exists():
        return path

    if os.name != "nt" and len(expanded) >= 3 and expanded[1] == ":" and expanded[2] in ("\\", "/"):
        drive = expanded[0].lower()
        tail = expanded[3:].replace("\\", "/")
        wsl_path = Path(f"/mnt/{drive}/{tail}")
        if wsl_path.exists():
            return wsl_path

    return path


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
        num_classes: Optional[int] = None,
    ) -> None:
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.num_classes = num_classes

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

        if isinstance(pred, (list, tuple)):
            if not pred:
                return []
            if len(pred) > 1:
                print(f"[detector] model returned {len(pred)} outputs; using the first output tensor")
            pred = pred[0]

        has_objectness: Optional[bool] = None
        non_batch_dims = [int(d) for d in pred.shape if int(d) != 1]
        if len(non_batch_dims) == 2:
            feature_dim = min(non_batch_dims)
            if self.num_classes is not None and feature_dim == self.num_classes + 4:
                has_objectness = False
            elif self.num_classes is not None and feature_dim == self.num_classes + 5:
                has_objectness = True
            elif feature_dim == 84:  # YOLOv8 style: [cx, cy, w, h, class_scores...]
                has_objectness = False
            elif feature_dim == 85:  # YOLOv5/7 style: [cx, cy, w, h, obj, class_scores...]
                has_objectness = True

        return self._postprocess(pred, frame_w=w, frame_h=h, has_objectness=has_objectness)

    def _postprocess(
        self,
        pred: np.ndarray,
        frame_w: int,
        frame_h: int,
        has_objectness: Optional[bool] = None,
    ) -> List[Detection]:
        # Common YOLO ONNX outputs are often [1, N, 5+nc], [1, 5+nc, N], [1, N, 4+nc], or [1, 4+nc, N].
        arr = np.squeeze(pred)

        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)

        if arr.ndim == 2 and arr.shape[0] < arr.shape[1] and arr.shape[0] <= 256:
            arr = arr.T

        boxes_xywh: List[List[float]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        coord_max = float(np.max(np.abs(arr[:, :4]))) if arr.ndim == 2 and arr.shape[1] >= 4 else 0.0
        coords_are_normalized = coord_max <= 2.0

        def _clamp_box(x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
            x = max(0, min(frame_w - 1, int(x)))
            y = max(0, min(frame_h - 1, int(y)))
            w = max(1, min(frame_w - x, int(w)))
            h = max(1, min(frame_h - y, int(h)))
            return x, y, w, h

        for row in arr:
            if row.shape[0] < 6:
                continue

            if row.shape[0] == 6:
                x1, y1, x2, y2, conf, class_id = row.tolist()
                if conf < self.conf_threshold:
                    continue
                if coords_are_normalized:
                    x1 *= frame_w
                    x2 *= frame_w
                    y1 *= frame_h
                    y2 *= frame_h
                x, y, w, h = _clamp_box(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                boxes_xywh.append([x, y, w, h])
                confidences.append(float(conf))
                class_ids.append(int(class_id))
                continue

            b0, b1, b2, b3 = [float(v) for v in row[:4]]

            if has_objectness is True:
                obj = float(row[4])
                if obj < 1e-6:
                    continue

                class_scores = row[5:]
                class_id = int(np.argmax(class_scores))
                cls_conf = float(class_scores[class_id])
                conf = float(obj * cls_conf)
            else:
                class_scores = row[4:]
                class_id = int(np.argmax(class_scores))
                conf = float(class_scores[class_id])

            if conf < self.conf_threshold:
                continue

            scale_x = frame_w / float(self.input_size)
            scale_y = frame_h / float(self.input_size)

            if coords_are_normalized:
                candidates = [
                    _clamp_box(int((b0 - b2 / 2.0) * frame_w), int((b1 - b3 / 2.0) * frame_h), int(b2 * frame_w), int(b3 * frame_h)),
                    _clamp_box(int(b0 * frame_w), int(b1 * frame_h), int((b2 - b0) * frame_w), int((b3 - b1) * frame_h)),
                    _clamp_box(int(b0 * frame_w), int(b1 * frame_h), int(b2 * frame_w), int(b3 * frame_h)),
                ]
            else:
                candidates = [
                    _clamp_box(int((b0 - b2 / 2.0) * scale_x), int((b1 - b3 / 2.0) * scale_y), int(b2 * scale_x), int(b3 * scale_y)),
                    _clamp_box(int(b0 * scale_x), int(b1 * scale_y), int((b2 - b0) * scale_x), int((b3 - b1) * scale_y)),
                    _clamp_box(int(b0 * scale_x), int(b1 * scale_y), int(b2 * scale_x), int(b3 * scale_y)),
                ]

            x, y, w, h = max(candidates, key=lambda box: box[2] * box[3])

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


def run_worker() -> None:
    resolved_model_path = resolve_model_path(MODEL_PATH)
    if not resolved_model_path.exists():
        raise FileNotFoundError(
            "Model file not found. "
            f"Configured MODEL_PATH={MODEL_PATH!r}, resolved path={str(resolved_model_path)!r}."
        )

    frame_buffer = LatestFrameBuffer()
    reader = MjpegReaderThread(MJPEG_URL, frame_buffer, timeout_s=STREAM_TIMEOUT_S)
    detector = YoloOnnxDetector(
        model_path=str(resolved_model_path),
        conf_threshold=CONF_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        input_size=INPUT_SIZE,
        num_classes=MODEL_NUM_CLASSES,
    )

    reader.start()
    print(f"[worker] connected reader thread for: {MJPEG_URL}")

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
            filtered_detections = [det for det in detections if det.confidence >= CONF_THRESHOLD]
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
                f"{COUNT_LABEL}: {count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if SHOW_PREVIEW:
                if PREVIEW_SCALE != 1.0:
                    frame_vis = cv2.resize(
                        frame_vis,
                        dsize=None,
                        fx=PREVIEW_SCALE,
                        fy=PREVIEW_SCALE,
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
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()



if __name__ == "__main__":
    run_worker()
