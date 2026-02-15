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

import importlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

import cv2
import numpy as np

try:
    import onnxruntime
except ImportError:
    onnxruntime = None


# Worker configuration
MJPEG_URL = "http://192.168.42.42:5001/video/Camera%201"
MODEL_PATH = "crab_detection_v2.onnx"
MODEL_NUM_CLASSES = 3
CLASS_NAMES: Sequence[str] = (
    # Update with your model's species labels in class-index order.
    "Green",
    "Jonah",
    "Rock",
)
DETECTOR_BACKEND = "auto"  # one of: "auto", "opencv", "onnxruntime"
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
INPUT_SIZE = 640  # fallback/default when model input shape is dynamic
STREAM_TIMEOUT_S = 5.0
SHOW_PREVIEW = True
PREVIEW_SCALE = 1.0
COUNT_LABEL = "Green Crabs"
GREEN_CLASS_NAMES = {"green"}
RED_CLASS_NAMES = {"jonah", "rock"}


def detection_color_bgr(class_name: str) -> Tuple[int, int, int]:
    """Return display color for a detected crab species in BGR format."""

    normalized = class_name.strip().lower()
    if normalized in GREEN_CLASS_NAMES:
        return (0, 255, 0)
    if normalized in RED_CLASS_NAMES:
        return (0, 0, 255)
    return (255, 255, 0)


def is_green_crab(class_name: str) -> bool:
    """True when detection represents a green crab species."""

    return class_name.strip().lower() in GREEN_CLASS_NAMES


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
    class_name: str
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
    """YOLO ONNX detector with OpenCV DNN + ONNX Runtime fallback."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float,
        nms_threshold: float,
        input_size: int,
        num_classes: Optional[int] = None,
        backend: str = "auto",
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.num_classes = num_classes
        self.class_name_by_id: Dict[int, str] = {
            idx: name for idx, name in enumerate(class_names or [])
        }

        if backend not in ("auto", "opencv", "onnxruntime"):
            raise ValueError(f"Unsupported backend {backend!r}; expected auto/opencv/onnxruntime")

        self.backend_preference = backend
        self.active_backend = "opencv" if backend in ("auto", "opencv") else "onnxruntime"
        self.net: Optional[cv2.dnn.Net] = None
        self.ort_session = None
        self.ort_input_name: Optional[str] = None
        self.ort_input_hw: Tuple[int, int] = (self.input_size, self.input_size)

        if self.active_backend == "opencv":
            self.net = cv2.dnn.readNetFromONNX(model_path)
        else:
            self._init_onnxruntime()

    def _init_onnxruntime(self) -> None:
        if onnxruntime is None:
            raise RuntimeError(
                "onnxruntime is not installed. Install it with: pip install onnxruntime"
            )

        self.ort_session = onnxruntime.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )
        ort_input = self.ort_session.get_inputs()[0]
        self.ort_input_name = ort_input.name

        shape = ort_input.shape
        if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.ort_input_hw = (int(shape[2]), int(shape[3]))
            if self.ort_input_hw[0] != self.input_size or self.ort_input_hw[1] != self.input_size:
                print(
                    "[detector] overriding input size from model metadata: "
                    f"{self.input_size} -> {self.ort_input_hw[0]}x{self.ort_input_hw[1]}"
                )

    def _looks_like_opencv_reshape_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return "reshape" in msg and "computeshapebyreshapemask" in msg

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        if self.active_backend == "onnxruntime":
            return self._infer_onnxruntime(frame_bgr)

        try:
            return self._infer_opencv(frame_bgr)
        except cv2.error as exc:
            should_fallback = (
                self.backend_preference == "auto"
                and self._looks_like_opencv_reshape_error(exc)
            )
            if not should_fallback:
                raise

            print("[detector] OpenCV DNN failed with reshape error; switching to onnxruntime backend")
            self._init_onnxruntime()
            self.active_backend = "onnxruntime"
            return self._infer_onnxruntime(frame_bgr)

    def _infer_opencv(self, frame_bgr: np.ndarray) -> List[Detection]:
        if self.net is None:
            raise RuntimeError("OpenCV DNN backend is not initialized")

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

        has_objectness = self._infer_objectness_layout(pred)
        return self._postprocess(pred, frame_w=w, frame_h=h, has_objectness=has_objectness)

    def _infer_onnxruntime(self, frame_bgr: np.ndarray) -> List[Detection]:
        if self.ort_session is None or self.ort_input_name is None:
            raise RuntimeError("onnxruntime backend is not initialized")

        h, w = frame_bgr.shape[:2]
        in_h, in_w = self.ort_input_hw
        resized = cv2.resize(frame_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        outputs = self.ort_session.run(None, {self.ort_input_name: blob})
        if not outputs:
            return []

        pred = outputs[0]
        has_objectness = self._infer_objectness_layout(pred)
        return self._postprocess(pred, frame_w=w, frame_h=h, has_objectness=has_objectness)

    def _infer_objectness_layout(self, pred: np.ndarray) -> Optional[bool]:
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
        return has_objectness

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

            if self.active_backend == "onnxruntime":
                in_h, in_w = self.ort_input_hw
                scale_x = frame_w / float(in_w)
                scale_y = frame_h / float(in_h)
            else:
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
                    class_name=self.class_name_by_id.get(class_ids[idx], str(class_ids[idx])),
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
    if CLASS_NAMES and MODEL_NUM_CLASSES != len(CLASS_NAMES):
        print(
            "[worker] warning: MODEL_NUM_CLASSES does not match len(CLASS_NAMES): "
            f"{MODEL_NUM_CLASSES} != {len(CLASS_NAMES)}"
        )
    detector = YoloOnnxDetector(
        model_path=str(resolved_model_path),
        conf_threshold=CONF_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        input_size=INPUT_SIZE,
        num_classes=MODEL_NUM_CLASSES,
        backend=DETECTOR_BACKEND,
        class_names=CLASS_NAMES,
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
            count = sum(1 for det in filtered_detections if is_green_crab(det.class_name))

            frame_vis = packet.frame_bgr.copy()
            for det in filtered_detections:
                x1, y1, x2, y2 = det.bbox_xyxy
                box_color = detection_color_bgr(det.class_name)
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), box_color, 2)
                label = f"{det.class_name} {det.confidence:.2f}"
                text_y = y1 - 8 if y1 > 16 else y1 + 18
                cv2.putText(
                    frame_vis,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
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
                        "class_name": det.class_name,
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
