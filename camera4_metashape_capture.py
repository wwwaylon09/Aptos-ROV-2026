#!/usr/bin/env python3
"""Capture Camera 4 frames on demand, then run an automated Metashape workflow.

Workflow:
1) Wait for a start command from stdin.
2) Capture frames from Camera 4 into a session folder.
3) Wait for a stop command from stdin.
4) Build a photogrammetry model via Metashape (direct API or CLI script runner).

Example:
    python camera4_metashape_capture.py \
      --stream-url "http://192.168.42.42:5001/video/Camera%204" \
      --output-root ./photogrammetry_sessions \
      --metashape-executable "C:/Program Files/Agisoft/Metashape Pro/metashape.exe"
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional



@dataclass
class CaptureConfig:
    stream_url: str
    output_root: str
    session_name: Optional[str]
    start_command: str
    stop_command: str
    auto_start: bool
    preview: bool
    capture_fps: float
    capture_duration_s: float
    image_format: str
    jpeg_quality: int
    png_compression: int
    max_frames: int
    min_laplacian_variance: float
    dedupe_mean_absdiff_threshold: float
    min_frames_for_model: int


@dataclass
class MetashapeConfig:
    run_metashape: bool
    metashape_executable: str
    metashape_headless: bool
    project_filename: str
    model_filename: str
    model_format: str
    image_glob: str
    match_downscale: int
    keypoint_limit: int
    tiepoint_limit: int
    generic_preselection: bool
    reference_preselection: bool
    filter_stationary_points: bool
    guided_matching: bool
    sequential_preselection: bool
    adaptive_fitting: bool
    camera_fit_f: bool
    camera_fit_cxcy: bool
    camera_fit_k1k2k3: bool
    camera_fit_k4: bool
    camera_fit_p1p2: bool
    camera_fit_b1b2: bool
    reset_alignment: bool
    optimize_cameras: bool
    depth_downscale: int
    depth_filter_mode: str
    max_neighbors: int
    reuse_depth: bool
    face_count: str
    interpolation: str
    calculate_vertex_colors: bool
    build_texture: bool
    texture_size: int
    texture_count: int
    blending_mode: str
    ghosting_filter: bool
    export_crs: str
    min_aligned_cameras: int


def parse_args() -> tuple[CaptureConfig, MetashapeConfig]:
    parser = argparse.ArgumentParser(
        description="Capture Camera 4 frames on command, then run Metashape photogrammetry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python camera4_metashape_capture.py --auto-start --capture-duration 120\n"
            "  python camera4_metashape_capture.py --no-auto-start  # type start/stop in stdin"
        ),
    )

    # Capture controls
    parser.add_argument("--stream-url", default="http://192.168.42.42:5001/video/Camera%204")
    parser.add_argument("--output-root", default="./photogrammetry_sessions")
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--start-command", default="start")
    parser.add_argument("--stop-command", default="stop")
    parser.add_argument(
        "--auto-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start capture immediately. Disable this to wait for the start command on stdin.",
    )
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capture-fps", type=float, default=2.0, help="Saved frame rate target.")
    parser.add_argument(
        "--capture-duration",
        type=float,
        default=0.0,
        help="Automatically stop after N seconds (0 disables).",
    )
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--png-compression", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = unlimited")
    parser.add_argument(
        "--min-laplacian-variance",
        type=float,
        default=0.0,
        help="Reject blurry frames under this threshold (0 disables).",
    )
    parser.add_argument(
        "--dedupe-mean-absdiff-threshold",
        type=float,
        default=0.0,
        help="Reject near-duplicate frames under this mean absolute pixel difference (0 disables).",
    )
    parser.add_argument(
        "--min-frames-for-model",
        type=int,
        default=40,
        help="Require at least this many captured frames before launching Metashape.",
    )

    # Metashape controls
    default_headless = not sys.platform.startswith("win")
    parser.add_argument("--run-metashape", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--metashape-executable",
        default="metashape",
        help="Path to Metashape executable used for -r script mode when Python API is unavailable.",
    )
    parser.add_argument(
        "--metashape-headless",
        action=argparse.BooleanOptionalAction,
        default=default_headless,
        help="Use headless/offscreen Metashape mode (off by default on Windows).",
    )
    parser.add_argument("--project-filename", default="project.psx")
    parser.add_argument("--model-filename", default="model.obj")
    parser.add_argument("--model-format", choices=["obj", "ply", "stl", "fbx", "glb"], default="obj")
    parser.add_argument("--image-glob", default="*.jpg")

    parser.add_argument("--match-downscale", type=int, default=1)
    parser.add_argument("--keypoint-limit", type=int, default=40000)
    parser.add_argument("--tiepoint-limit", type=int, default=4000)
    parser.add_argument("--generic-preselection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-preselection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter-stationary-points", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--guided-matching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--sequential-preselection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sequential pair preselection when available in Metashape API.",
    )

    parser.add_argument("--adaptive-fitting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-fit-f", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-fit-cxcy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-fit-k1k2k3", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-fit-k4", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--camera-fit-p1p2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-fit-b1b2", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reset-alignment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--optimize-cameras",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run Optimize Cameras after alignment (disabled by default).",
    )

    parser.add_argument("--depth-downscale", type=int, default=2)
    parser.add_argument(
        "--depth-filter-mode",
        choices=["NoFiltering", "MildFiltering", "ModerateFiltering", "AggressiveFiltering"],
        default="ModerateFiltering",
    )
    parser.add_argument("--max-neighbors", type=int, default=16)
    parser.add_argument("--reuse-depth", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument(
        "--face-count",
        choices=["LowFaceCount", "MediumFaceCount", "HighFaceCount", "CustomFaceCount"],
        default="HighFaceCount",
    )
    parser.add_argument(
        "--interpolation",
        choices=["DisabledInterpolation", "EnabledInterpolation", "Extrapolated"],
        default="EnabledInterpolation",
    )
    parser.add_argument("--calculate-vertex-colors", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--build-texture", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--texture-count", type=int, default=1)
    parser.add_argument(
        "--blending-mode",
        choices=["MosaicBlending", "AverageBlending", "MaxIntensity", "MinIntensity", "DisabledBlending"],
        default="MosaicBlending",
    )
    parser.add_argument("--ghosting-filter", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--export-crs", default="")
    parser.add_argument(
        "--min-aligned-cameras",
        type=int,
        default=4,
        help="Fail early if fewer than this many cameras align before depth/model build.",
    )

    args = parser.parse_args()

    capture_cfg = CaptureConfig(
        stream_url=args.stream_url,
        output_root=args.output_root,
        session_name=args.session_name,
        start_command=args.start_command.strip(),
        stop_command=args.stop_command.strip(),
        auto_start=args.auto_start,
        preview=args.preview,
        capture_fps=max(args.capture_fps, 0.1),
        capture_duration_s=max(args.capture_duration, 0.0),
        image_format=args.image_format,
        jpeg_quality=max(0, min(args.jpeg_quality, 100)),
        png_compression=max(0, min(args.png_compression, 9)),
        max_frames=max(args.max_frames, 0),
        min_laplacian_variance=max(args.min_laplacian_variance, 0.0),
        dedupe_mean_absdiff_threshold=max(args.dedupe_mean_absdiff_threshold, 0.0),
        min_frames_for_model=max(args.min_frames_for_model, 3),
    )

    metashape_cfg = MetashapeConfig(
        run_metashape=args.run_metashape,
        metashape_executable=args.metashape_executable,
        metashape_headless=args.metashape_headless,
        project_filename=args.project_filename,
        model_filename=args.model_filename,
        model_format=args.model_format,
        image_glob=args.image_glob,
        match_downscale=args.match_downscale,
        keypoint_limit=args.keypoint_limit,
        tiepoint_limit=args.tiepoint_limit,
        generic_preselection=args.generic_preselection,
        reference_preselection=args.reference_preselection,
        filter_stationary_points=args.filter_stationary_points,
        guided_matching=args.guided_matching,
        sequential_preselection=args.sequential_preselection,
        adaptive_fitting=args.adaptive_fitting,
        camera_fit_f=args.camera_fit_f,
        camera_fit_cxcy=args.camera_fit_cxcy,
        camera_fit_k1k2k3=args.camera_fit_k1k2k3,
        camera_fit_k4=args.camera_fit_k4,
        camera_fit_p1p2=args.camera_fit_p1p2,
        camera_fit_b1b2=args.camera_fit_b1b2,
        reset_alignment=args.reset_alignment,
        optimize_cameras=args.optimize_cameras,
        depth_downscale=max(args.depth_downscale, 1),
        depth_filter_mode=args.depth_filter_mode,
        max_neighbors=args.max_neighbors,
        reuse_depth=args.reuse_depth,
        face_count=args.face_count,
        interpolation=args.interpolation,
        calculate_vertex_colors=args.calculate_vertex_colors,
        build_texture=args.build_texture,
        texture_size=args.texture_size,
        texture_count=args.texture_count,
        blending_mode=args.blending_mode,
        ghosting_filter=args.ghosting_filter,
        export_crs=args.export_crs,
        min_aligned_cameras=max(args.min_aligned_cameras, 1),
    )

    if capture_cfg.image_format == "png" and metashape_cfg.image_glob == "*.jpg":
        metashape_cfg.image_glob = "*.png"

    return capture_cfg, metashape_cfg


def input_thread_worker(input_queue: queue.Queue[str]) -> None:
    while True:
        try:
            line = input().strip()
        except EOFError:
            return
        input_queue.put(line)


def log(message: str) -> None:
    print(message, flush=True)



def get_cv2():
    import cv2  # type: ignore

    return cv2


def frame_is_sharp_enough(frame, threshold: float) -> bool:
    if threshold <= 0:
        return True
    cv2 = get_cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance >= threshold


def frame_is_unique_enough(frame, last_saved, threshold: float) -> bool:
    if threshold <= 0 or last_saved is None:
        return True
    cv2 = get_cv2()
    diff = cv2.absdiff(frame, last_saved)
    return float(diff.mean()) >= threshold


def ensure_session_dirs(output_root: Path, session_name: Optional[str]) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = session_name or f"camera4_session_{timestamp}"
    session_dir = output_root / name
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return session_dir, images_dir


def capture_on_command(capture_cfg: CaptureConfig) -> tuple[Path, Path, int]:
    cv2 = get_cv2()
    output_root = Path(capture_cfg.output_root).resolve()
    session_dir, images_dir = ensure_session_dirs(output_root, capture_cfg.session_name)

    input_queue: queue.Queue[str] = queue.Queue()
    threading.Thread(target=input_thread_worker, args=(input_queue,), daemon=True).start()

    if not capture_cfg.auto_start:
        if not sys.stdin.isatty():
            log("stdin is not interactive; forcing --auto-start so capture can proceed.")
            capture_cfg.auto_start = True
        else:
            log(f"Type '{capture_cfg.start_command}' to begin capture.")
            log(f"Type '{capture_cfg.stop_command}' to stop capture and launch processing.")

    if not capture_cfg.auto_start:
        while True:
            cmd = input_queue.get()
            if cmd.lower() == capture_cfg.start_command.lower():
                break
            log(f"Ignoring command '{cmd}'. Waiting for '{capture_cfg.start_command}'.")

    cap = cv2.VideoCapture(capture_cfg.stream_url)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open stream: {capture_cfg.stream_url}")

    log("Capture started.")

    interval_s = 1.0 / capture_cfg.capture_fps
    last_save_time = 0.0
    frame_idx = 0
    saved_frames = 0
    last_saved_frame = None

    if capture_cfg.preview:
        log("Preview window open. Press 'q' in the preview window to stop as well.")

    capture_started_at = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            frame_idx += 1
            now = time.monotonic()

            if capture_cfg.preview:
                preview = frame.copy()
                cv2.putText(
                    preview,
                    f"Saved: {saved_frames}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Camera 4 Capture", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log("Preview stop key pressed.")
                    break

            while True:
                try:
                    cmd = input_queue.get_nowait()
                except queue.Empty:
                    break
                if cmd.lower() == capture_cfg.stop_command.lower():
                    log("Stop command received.")
                    return session_dir, images_dir, saved_frames
                log(f"Ignoring command '{cmd}' while capturing.")

            if capture_cfg.capture_duration_s and (now - capture_started_at) >= capture_cfg.capture_duration_s:
                log(f"Capture duration reached ({capture_cfg.capture_duration_s:.1f}s).")
                break

            if now - last_save_time < interval_s:
                continue
            if not frame_is_sharp_enough(frame, capture_cfg.min_laplacian_variance):
                continue
            if not frame_is_unique_enough(frame, last_saved_frame, capture_cfg.dedupe_mean_absdiff_threshold):
                continue

            filename = f"frame_{saved_frames:06d}.{capture_cfg.image_format}"
            out_path = images_dir / filename

            if capture_cfg.image_format == "jpg":
                params = [cv2.IMWRITE_JPEG_QUALITY, capture_cfg.jpeg_quality]
            else:
                params = [cv2.IMWRITE_PNG_COMPRESSION, capture_cfg.png_compression]

            if not cv2.imwrite(str(out_path), frame, params):
                log(f"Failed to save {out_path}")
                continue

            saved_frames += 1
            last_saved_frame = frame.copy()
            last_save_time = now

            if saved_frames % 10 == 0:
                log(f"Saved {saved_frames} frames...")

            if capture_cfg.max_frames and saved_frames >= capture_cfg.max_frames:
                log(f"Reached max frames ({capture_cfg.max_frames}).")
                break

        return session_dir, images_dir, saved_frames
    finally:
        cap.release()
        if capture_cfg.preview:
            cv2.destroyAllWindows()


def count_aligned_cameras(chunk) -> int:
    return sum(1 for cam in chunk.cameras if cam.transform is not None)


def ensure_sufficient_alignment(chunk, minimum: int) -> None:
    aligned = count_aligned_cameras(chunk)
    if aligned < minimum:
        raise RuntimeError(
            f"Only {aligned} cameras aligned. Need at least {minimum} aligned cameras for reliable depth/map model build. "
            "Capture more frames with stronger overlap and motion parallax, then retry."
        )


def match_photos_with_optional_sequential(chunk, cfg):
    import Metashape  # type: ignore

    match_kwargs = dict(
        downscale=cfg.match_downscale,
        generic_preselection=cfg.generic_preselection,
        reference_preselection=cfg.reference_preselection,
        filter_stationary_points=cfg.filter_stationary_points,
        guided_matching=cfg.guided_matching,
        keypoint_limit=cfg.keypoint_limit,
        tiepoint_limit=cfg.tiepoint_limit,
    )

    if cfg.sequential_preselection and hasattr(Metashape, "SequentialPreselection"):
        try:
            chunk.matchPhotos(
                reference_preselection_mode=Metashape.SequentialPreselection,
                **match_kwargs,
            )
            return
        except TypeError:
            pass

    chunk.matchPhotos(**match_kwargs)


def run_metashape_direct(images_dir: Path, session_dir: Path, cfg: MetashapeConfig) -> None:
    import Metashape  # type: ignore

    doc = Metashape.Document()
    project_path = session_dir / cfg.project_filename
    doc.save(str(project_path))

    chunk = doc.addChunk()
    image_paths = sorted(str(p) for p in images_dir.glob(cfg.image_glob))
    if not image_paths:
        raise RuntimeError(f"No images matched {cfg.image_glob} in {images_dir}")

    chunk.addPhotos(image_paths)
    match_photos_with_optional_sequential(chunk, cfg)
    chunk.alignCameras(
        reset_alignment=cfg.reset_alignment,
        adaptive_fitting=cfg.adaptive_fitting,
    )

    if cfg.optimize_cameras:
        chunk.optimizeCameras(
            fit_f=cfg.camera_fit_f,
            fit_cx=cfg.camera_fit_cxcy,
            fit_cy=cfg.camera_fit_cxcy,
            fit_k1=cfg.camera_fit_k1k2k3,
            fit_k2=cfg.camera_fit_k1k2k3,
            fit_k3=cfg.camera_fit_k1k2k3,
            fit_k4=cfg.camera_fit_k4,
            fit_p1=cfg.camera_fit_p1p2,
            fit_p2=cfg.camera_fit_p1p2,
            fit_b1=cfg.camera_fit_b1b2,
            fit_b2=cfg.camera_fit_b1b2,
        )

    ensure_sufficient_alignment(chunk, cfg.min_aligned_cameras)

    chunk.buildDepthMaps(
        downscale=cfg.depth_downscale,
        filter_mode=getattr(Metashape, cfg.depth_filter_mode),
        max_neighbors=cfg.max_neighbors,
        reuse_depth=cfg.reuse_depth,
    )
    try:
        chunk.buildModel(
            source_data=Metashape.DepthMapsData,
            face_count=getattr(Metashape, cfg.face_count),
            interpolation=getattr(Metashape, cfg.interpolation),
            vertex_colors=cfg.calculate_vertex_colors,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "No cameras with depth maps in working volume" in message:
            aligned = count_aligned_cameras(chunk)
            raise RuntimeError(
                f"Metashape could not build depth-map model because no usable depth maps were generated. "
                f"Aligned cameras: {aligned}. Try collecting 50-100 images with stronger overlap/parallax "
                "and avoid near-stationary frames."
            ) from exc
        raise

    if cfg.build_texture:
        try:
            chunk.buildUV(
                mapping_mode=Metashape.GenericMapping,
                page_count=cfg.texture_count,
                texture_size=cfg.texture_size,
            )
        except TypeError:
            chunk.buildUV()

        chunk.buildTexture(
            blending_mode=getattr(Metashape, cfg.blending_mode),
            ghosting_filter=cfg.ghosting_filter,
        )

    model_path = session_dir / cfg.model_filename
    model_format_enum = {
        "obj": Metashape.ModelFormatOBJ,
        "ply": Metashape.ModelFormatPLY,
        "stl": Metashape.ModelFormatSTL,
        "fbx": Metashape.ModelFormatFBX,
        "glb": Metashape.ModelFormatGLTF,
    }[cfg.model_format]

    kwargs = {}
    if cfg.export_crs:
        kwargs["crs"] = Metashape.CoordinateSystem(cfg.export_crs)

    chunk.exportModel(str(model_path), format=model_format_enum, **kwargs)
    doc.save()


def metashape_runner_script_text() -> str:
    return '''import json
import sys
from pathlib import Path

import Metashape

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text(encoding="utf-8"))
session_dir = Path(config["session_dir"])
images_dir = Path(config["images_dir"])
cfg = config["metashape"]

project_path = session_dir / cfg["project_filename"]
doc = Metashape.Document()
doc.save(str(project_path))
chunk = doc.addChunk()

def count_aligned_cameras(chunk):
    return sum(1 for cam in chunk.cameras if cam.transform is not None)

def ensure_sufficient_alignment(chunk, minimum):
    aligned = count_aligned_cameras(chunk)
    if aligned < minimum:
        raise RuntimeError(
            f"Only {aligned} cameras aligned. Need at least {minimum} aligned cameras for reliable depth/map model build. "
            "Capture more frames with stronger overlap and motion parallax, then retry."
        )

image_paths = sorted(str(p) for p in images_dir.glob(cfg["image_glob"]))
if not image_paths:
    raise RuntimeError(f"No images matched {cfg['image_glob']} in {images_dir}")

def match_photos_with_optional_sequential(chunk, cfg):
    match_kwargs = dict(
        downscale=cfg["match_downscale"],
        generic_preselection=cfg["generic_preselection"],
        reference_preselection=cfg["reference_preselection"],
        filter_stationary_points=cfg["filter_stationary_points"],
        guided_matching=cfg["guided_matching"],
        keypoint_limit=cfg["keypoint_limit"],
        tiepoint_limit=cfg["tiepoint_limit"],
    )

    if cfg["sequential_preselection"] and hasattr(Metashape, "SequentialPreselection"):
        try:
            chunk.matchPhotos(
                reference_preselection_mode=Metashape.SequentialPreselection,
                **match_kwargs,
            )
            return
        except TypeError:
            pass

    chunk.matchPhotos(**match_kwargs)

chunk.addPhotos(image_paths)
match_photos_with_optional_sequential(chunk, cfg)
chunk.alignCameras(
    reset_alignment=cfg["reset_alignment"],
    adaptive_fitting=cfg["adaptive_fitting"],
)

if cfg["optimize_cameras"]:
    chunk.optimizeCameras(
        fit_f=cfg["camera_fit_f"],
        fit_cx=cfg["camera_fit_cxcy"],
        fit_cy=cfg["camera_fit_cxcy"],
        fit_k1=cfg["camera_fit_k1k2k3"],
        fit_k2=cfg["camera_fit_k1k2k3"],
        fit_k3=cfg["camera_fit_k1k2k3"],
        fit_k4=cfg["camera_fit_k4"],
        fit_p1=cfg["camera_fit_p1p2"],
        fit_p2=cfg["camera_fit_p1p2"],
        fit_b1=cfg["camera_fit_b1b2"],
        fit_b2=cfg["camera_fit_b1b2"],
    )

ensure_sufficient_alignment(chunk, cfg["min_aligned_cameras"])

chunk.buildDepthMaps(
    downscale=cfg["depth_downscale"],
    filter_mode=getattr(Metashape, cfg["depth_filter_mode"]),
    max_neighbors=cfg["max_neighbors"],
    reuse_depth=cfg["reuse_depth"],
)
try:
    chunk.buildModel(
        source_data=Metashape.DepthMapsData,
        face_count=getattr(Metashape, cfg["face_count"]),
        interpolation=getattr(Metashape, cfg["interpolation"]),
        vertex_colors=cfg["calculate_vertex_colors"],
    )
except RuntimeError as exc:
    message = str(exc)
    if "No cameras with depth maps in working volume" in message:
        aligned = count_aligned_cameras(chunk)
        raise RuntimeError(
            f"Metashape could not build depth-map model because no usable depth maps were generated. "
            f"Aligned cameras: {aligned}. Try collecting 50-100 images with stronger overlap/parallax "
            "and avoid near-stationary frames."
        ) from exc
    raise

if cfg["build_texture"]:
    try:
        chunk.buildUV(
            mapping_mode=Metashape.GenericMapping,
            page_count=cfg["texture_count"],
            texture_size=cfg["texture_size"],
        )
    except TypeError:
        chunk.buildUV()

    chunk.buildTexture(
        blending_mode=getattr(Metashape, cfg["blending_mode"]),
        ghosting_filter=cfg["ghosting_filter"],
    )

model_path = session_dir / cfg["model_filename"]
model_format_enum = {
    "obj": Metashape.ModelFormatOBJ,
    "ply": Metashape.ModelFormatPLY,
    "stl": Metashape.ModelFormatSTL,
    "fbx": Metashape.ModelFormatFBX,
    "glb": Metashape.ModelFormatGLTF,
}[cfg["model_format"]]

kwargs = {}
if cfg["export_crs"]:
    kwargs["crs"] = Metashape.CoordinateSystem(cfg["export_crs"])

chunk.exportModel(str(model_path), format=model_format_enum, **kwargs)
doc.save()
print(f"Model exported to: {model_path}")
'''




def resolve_metashape_executable(configured_value: str) -> tuple[str, list[str]]:
    """Resolve the Metashape executable path and return attempted candidates."""
    attempted: list[str] = []

    if configured_value:
        attempted.append(configured_value)
        found = shutil.which(configured_value)
        if found:
            return found, attempted

        configured_path = Path(configured_value).expanduser()
        if configured_path.exists():
            return str(configured_path), attempted

    if sys.platform.startswith("win"):
        windows_candidates = [
            Path(r"C:/Program Files/Agisoft/Metashape Pro/metashape.exe"),
            Path(r"C:/Program Files/Agisoft/Metashape Pro/python/metashape.exe"),
            Path(r"C:/Program Files/Agisoft/Metashape Professional/metashape.exe"),
            Path(r"C:/Program Files/Agisoft/Metashape Professional/python/metashape.exe"),
        ]
        for candidate in windows_candidates:
            attempted.append(str(candidate))
            if candidate.exists():
                return str(candidate), attempted

    # Keep the original string to let subprocess surface platform-specific errors
    # after we've provided attempted-path diagnostics to the user.
    return configured_value, attempted

def run_metashape_via_cli(images_dir: Path, session_dir: Path, cfg: MetashapeConfig) -> None:
    exe, attempted_executables = resolve_metashape_executable(cfg.metashape_executable)

    with tempfile.TemporaryDirectory(prefix="metashape_auto_") as tmp:
        tmpdir = Path(tmp)
        script_path = tmpdir / "metashape_runner.py"
        config_path = tmpdir / "metashape_config.json"

        script_path.write_text(metashape_runner_script_text(), encoding="utf-8")
        config_payload = {
            "session_dir": str(session_dir),
            "images_dir": str(images_dir),
            "metashape": asdict(cfg),
        }
        config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

        base_cmd = [exe, "-r", str(script_path), str(config_path)]
        cmd = [exe, "-platform", "offscreen", "-r", str(script_path), str(config_path)] if cfg.metashape_headless else base_cmd

        log("Running Metashape CLI:")
        log(" ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as exc:
            attempted_text = "\n".join(f"  - {candidate}" for candidate in attempted_executables)
            raise RuntimeError(
                "Could not find a runnable Metashape executable.\n"
                "Set --metashape-executable to your metashape.exe path.\n"
                "Attempted:\n"
                f"{attempted_text}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            should_retry_without_offscreen = (
                cfg.metashape_headless and sys.platform.startswith("win") and exc.returncode != 0
            )
            if not should_retry_without_offscreen:
                raise

            log("Metashape offscreen mode failed on Windows; retrying without offscreen.")
            log(" ".join(base_cmd))
            subprocess.run(base_cmd, check=True)

        log("Running Metashape CLI:")
        log(" ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as exc:
            attempted_text = "\n".join(f"  - {candidate}" for candidate in attempted_executables)
            raise RuntimeError(
                "Could not find a runnable Metashape executable.\n"
                "Set --metashape-executable to your metashape.exe path.\n"
                "Attempted:\n"
                f"{attempted_text}"
            ) from exc



def run_metashape(images_dir: Path, session_dir: Path, cfg: MetashapeConfig) -> None:
    if not cfg.run_metashape:
        log("Metashape run disabled via --no-run-metashape.")
        return

    try:
        import Metashape  # noqa: F401

        log("Using direct Metashape Python API in current interpreter.")
        run_metashape_direct(images_dir, session_dir, cfg)
        return
    except Exception as exc:
        log(f"Direct Metashape API unavailable ({exc}). Falling back to CLI script mode.")

    run_metashape_via_cli(images_dir, session_dir, cfg)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    capture_cfg, metashape_cfg = parse_args()

    session_dir, images_dir, saved_frames = capture_on_command(capture_cfg)
    log(f"Capture stopped. Saved frames: {saved_frames}")
    log(f"Session directory: {session_dir}")

    if saved_frames < capture_cfg.min_frames_for_model:
        log(
            f"Not enough frames for photogrammetry. Need at least {capture_cfg.min_frames_for_model} "
            f"(captured {saved_frames})."
        )
        return 1

    run_metashape(images_dir, session_dir, metashape_cfg)
    log("Workflow complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
