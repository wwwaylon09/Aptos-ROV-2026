#!/usr/bin/env python3
"""Run an automated Metashape workflow from a known-good image folder.

This script is intended for validating Metashape processing independent of live camera
capture. Point it at a folder of photos that already reconstruct well and it will run
alignment, depth-map generation, model building, and optional texturing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class MetashapeConfig:
    metashape_executable: str
    metashape_headless: bool
    image_glob: str
    project_filename: str
    model_filename: str
    model_format: str
    match_downscale: int
    keypoint_limit: int
    tiepoint_limit: int
    generic_preselection: bool
    reference_preselection: bool
    filter_stationary_points: bool
    guided_matching: bool
    sequential_preselection: bool
    adaptive_fitting: bool
    reset_alignment: bool
    min_aligned_cameras: int
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


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> tuple[Path, Path, MetashapeConfig]:
    parser = argparse.ArgumentParser(
        description="Build a Metashape model from an existing image folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images-dir", required=True, help="Folder that contains input photos.")
    parser.add_argument(
        "--output-dir",
        default="./metashape_runs/latest",
        help="Directory where project/model outputs will be written.",
    )
    parser.add_argument("--image-glob", default="*.jpg", help="Glob pattern used to pick input files.")
    parser.add_argument(
        "--metashape-executable",
        default="metashape",
        help="Path to Metashape executable for CLI fallback mode.",
    )
    default_headless = not sys.platform.startswith("win")
    parser.add_argument(
        "--metashape-headless",
        action=argparse.BooleanOptionalAction,
        default=default_headless,
        help="Use -platform offscreen in CLI fallback mode.",
    )

    parser.add_argument("--project-filename", default="project.psx")
    parser.add_argument("--model-filename", default="model.obj")
    parser.add_argument("--model-format", choices=["obj", "ply", "stl", "fbx", "glb"], default="obj")

    parser.add_argument("--match-downscale", type=int, default=1)
    parser.add_argument("--keypoint-limit", type=int, default=40000)
    parser.add_argument("--tiepoint-limit", type=int, default=4000)
    parser.add_argument("--generic-preselection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-preselection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter-stationary-points", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--guided-matching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sequential-preselection", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--adaptive-fitting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-alignment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-aligned-cameras", type=int, default=4)

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

    args = parser.parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {images_dir}")

    cfg = MetashapeConfig(
        metashape_executable=args.metashape_executable,
        metashape_headless=args.metashape_headless,
        image_glob=args.image_glob,
        project_filename=args.project_filename,
        model_filename=args.model_filename,
        model_format=args.model_format,
        match_downscale=max(args.match_downscale, 1),
        keypoint_limit=max(args.keypoint_limit, 0),
        tiepoint_limit=max(args.tiepoint_limit, 0),
        generic_preselection=args.generic_preselection,
        reference_preselection=args.reference_preselection,
        filter_stationary_points=args.filter_stationary_points,
        guided_matching=args.guided_matching,
        sequential_preselection=args.sequential_preselection,
        adaptive_fitting=args.adaptive_fitting,
        reset_alignment=args.reset_alignment,
        min_aligned_cameras=max(args.min_aligned_cameras, 1),
        depth_downscale=max(args.depth_downscale, 1),
        depth_filter_mode=args.depth_filter_mode,
        max_neighbors=max(args.max_neighbors, 1),
        reuse_depth=args.reuse_depth,
        face_count=args.face_count,
        interpolation=args.interpolation,
        calculate_vertex_colors=args.calculate_vertex_colors,
        build_texture=args.build_texture,
        texture_size=max(args.texture_size, 128),
        texture_count=max(args.texture_count, 1),
        blending_mode=args.blending_mode,
        ghosting_filter=args.ghosting_filter,
    )

    return images_dir, output_dir, cfg


def resolve_metashape_executable(configured_value: str) -> tuple[str, list[str]]:
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

    return configured_value, attempted


def runner_script_text() -> str:
    return '''import json
import sys
from pathlib import Path

import Metashape

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text(encoding="utf-8"))
images_dir = Path(config["images_dir"])
output_dir = Path(config["output_dir"])
cfg = config["metashape"]


def count_aligned_cameras(chunk):
    return sum(1 for cam in chunk.cameras if cam.transform is not None)


def ensure_sufficient_alignment(chunk, minimum):
    aligned = count_aligned_cameras(chunk)
    if aligned < minimum:
        raise RuntimeError(
            f"Only {aligned} cameras aligned. Need at least {minimum} aligned cameras for reliable model build."
        )


project_path = output_dir / cfg["project_filename"]
doc = Metashape.Document()
doc.save(str(project_path))
chunk = doc.addChunk()

image_paths = sorted(str(p) for p in images_dir.glob(cfg["image_glob"]))
if not image_paths:
    raise RuntimeError(f"No images matched {cfg['image_glob']} in {images_dir}")

match_kwargs = dict(
    downscale=cfg["match_downscale"],
    generic_preselection=cfg["generic_preselection"],
    reference_preselection=cfg["reference_preselection"],
    filter_stationary_points=cfg["filter_stationary_points"],
    guided_matching=cfg["guided_matching"],
    keypoint_limit=cfg["keypoint_limit"],
    tiepoint_limit=cfg["tiepoint_limit"],
)

chunk.addPhotos(image_paths)
if cfg["sequential_preselection"] and hasattr(Metashape, "SequentialPreselection"):
    try:
        chunk.matchPhotos(reference_preselection_mode=Metashape.SequentialPreselection, **match_kwargs)
    except TypeError:
        chunk.matchPhotos(**match_kwargs)
else:
    chunk.matchPhotos(**match_kwargs)

chunk.alignCameras(
    reset_alignment=cfg["reset_alignment"],
    adaptive_fitting=cfg["adaptive_fitting"],
)

ensure_sufficient_alignment(chunk, cfg["min_aligned_cameras"])

chunk.buildDepthMaps(
    downscale=cfg["depth_downscale"],
    filter_mode=getattr(Metashape, cfg["depth_filter_mode"]),
    max_neighbors=cfg["max_neighbors"],
    reuse_depth=cfg["reuse_depth"],
)

chunk.buildModel(
    source_data=Metashape.DepthMapsData,
    face_count=getattr(Metashape, cfg["face_count"]),
    interpolation=getattr(Metashape, cfg["interpolation"]),
    vertex_colors=cfg["calculate_vertex_colors"],
)

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

model_path = output_dir / cfg["model_filename"]
model_format_enum = {
    "obj": Metashape.ModelFormatOBJ,
    "ply": Metashape.ModelFormatPLY,
    "stl": Metashape.ModelFormatSTL,
    "fbx": Metashape.ModelFormatFBX,
    "glb": Metashape.ModelFormatGLTF,
}[cfg["model_format"]]
chunk.exportModel(str(model_path), format=model_format_enum)
doc.save()
print(f"Model exported to: {model_path}")
'''


def run_metashape_via_cli(images_dir: Path, output_dir: Path, cfg: MetashapeConfig) -> None:
    exe, attempted_executables = resolve_metashape_executable(cfg.metashape_executable)

    with tempfile.TemporaryDirectory(prefix="metashape_images_") as tmp:
        tmpdir = Path(tmp)
        script_path = tmpdir / "runner.py"
        config_path = tmpdir / "config.json"

        script_path.write_text(runner_script_text(), encoding="utf-8")
        config_path.write_text(
            json.dumps(
                {
                    "images_dir": str(images_dir),
                    "output_dir": str(output_dir),
                    "metashape": asdict(cfg),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

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
            retry_without_offscreen = cfg.metashape_headless and sys.platform.startswith("win")
            if not retry_without_offscreen:
                raise

            log("Metashape offscreen mode failed on Windows; retrying without offscreen.")
            log(" ".join(base_cmd))
            subprocess.run(base_cmd, check=True)


def run_metashape(images_dir: Path, output_dir: Path, cfg: MetashapeConfig) -> None:
    try:
        import Metashape  # noqa: F401

        log("Metashape Python API detected in current interpreter.")
    except Exception as exc:
        log(f"Direct API unavailable ({exc}). Using CLI script mode.")

    run_metashape_via_cli(images_dir, output_dir, cfg)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    images_dir, output_dir, cfg = parse_args()

    matched_images = sorted(images_dir.glob(cfg.image_glob))
    if not matched_images:
        log(f"No images matched '{cfg.image_glob}' in {images_dir}")
        return 1

    log(f"Input images: {len(matched_images)}")
    log(f"Output directory: {output_dir}")

    run_metashape(images_dir, output_dir, cfg)
    log("Workflow complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
