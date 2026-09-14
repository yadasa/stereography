from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .models import DepthEngine
from .gaussian4d import TemporalGaussianRenderer


ProgressCallback = Callable[[float, str, str], None]


@dataclass(frozen=True)
class RenderSettings:
    model: str
    render_method: str
    eye_separation: float
    depth_strength: float
    convergence: float
    temporal_smoothing: float
    resolution: str
    gaussian_scale: float = 1.35


def _target_size(width: int, height: int, preset: str) -> tuple[int, int]:
    limits = {"720p": (1280, 720), "1080p": (1920, 1080)}
    if preset == "source":
        return width - width % 2, height - height % 2
    max_width, max_height = limits[preset]
    scale = min(1.0, max_width / width, max_height / height)
    target_width = max(2, int(round(width * scale)) // 2 * 2)
    target_height = max(2, int(round(height * scale)) // 2 * 2)
    return target_width, target_height


def _disparity(depth: np.ndarray, settings: RenderSettings) -> np.ndarray:
    # Positive disparity brings nearer-than-convergence pixels forward.
    return (depth - settings.convergence) * settings.eye_separation * settings.depth_strength


def depth_warp(frame: np.ndarray, depth: np.ndarray, settings: RenderSettings) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    shift = _disparity(depth, settings) * 0.5
    left = cv2.remap(frame, grid_x + shift, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    right = cv2.remap(frame, grid_x - shift, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return left, right


def _forward_splat(frame: np.ndarray, depth: np.ndarray, offset: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.int32)
    target_x = np.rint(xx + offset).astype(np.int32)
    valid = (target_x >= 0) & (target_x < width)

    source_pixels = frame.reshape(-1, 3)
    depth_flat = depth.reshape(-1)
    target_flat = (yy * width + np.clip(target_x, 0, width - 1)).reshape(-1)
    valid_flat = valid.reshape(-1)
    source_ids = np.nonzero(valid_flat)[0]
    targets = target_flat[valid_flat]

    # Sort far-to-near inside each target pixel and retain the last (nearest) sample.
    order = np.lexsort((depth_flat[source_ids], targets))
    sorted_targets = targets[order]
    sorted_sources = source_ids[order]
    keep = np.r_[sorted_targets[1:] != sorted_targets[:-1], True]
    chosen_targets = sorted_targets[keep]
    chosen_sources = sorted_sources[keep]

    output = np.zeros((height * width, 3), dtype=np.uint8)
    occupied = np.zeros(height * width, dtype=np.uint8)
    output[chosen_targets] = source_pixels[chosen_sources]
    occupied[chosen_targets] = 1
    output = output.reshape(height, width, 3)
    holes = (1 - occupied.reshape(height, width)) * 255
    if holes.any():
        output = cv2.inpaint(output, holes, 3, cv2.INPAINT_TELEA)
    return output


def point_cloud_warp(frame: np.ndarray, depth: np.ndarray, settings: RenderSettings) -> tuple[np.ndarray, np.ndarray]:
    shift = _disparity(depth, settings) * 0.5
    return _forward_splat(frame, depth, -shift), _forward_splat(frame, depth, shift)


def _gaussian_renderer(settings: RenderSettings) -> TemporalGaussianRenderer:
    return TemporalGaussianRenderer(
        gaussian_scale=settings.gaussian_scale,
        temporal_smoothing=settings.temporal_smoothing,
        eye_separation=settings.eye_separation,
        depth_strength=settings.depth_strength,
        convergence=settings.convergence,
    )


def stereo_frame(
    frame: np.ndarray,
    depth: np.ndarray,
    settings: RenderSettings,
    gaussian_renderer: TemporalGaussianRenderer | None = None,
) -> np.ndarray:
    if settings.render_method == "depth-warp":
        left, right = depth_warp(frame, depth, settings)
    elif settings.render_method == "point-cloud":
        left, right = point_cloud_warp(frame, depth, settings)
    elif settings.render_method == "gaussian-4d":
        renderer = gaussian_renderer or _gaussian_renderer(settings)
        left, right = renderer.render(frame, depth, _disparity(depth, settings))
    else:
        raise ValueError(f"Unsupported rendering method: {settings.render_method}")
    return np.concatenate((left, right), axis=1)


def _encode_for_browser(silent_path: Path, source_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(silent_path), "-i", str(source_path),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "160k", "-shortest", str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=600)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        shutil.copyfile(silent_path, output_path)


def process_video(
    source_path: Path,
    output_path: Path,
    settings: RenderSettings,
    engine: DepthEngine,
    progress: ProgressCallback,
) -> dict[str, float | int | str]:
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not decode this video. Try an H.264 MP4 or WebM file.")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if frame_count else 0.0
    if duration > 90.5:
        capture.release()
        raise ValueError("This first workflow accepts videos up to 90 seconds.")
    if width < 2 or height < 2:
        capture.release()
        raise ValueError("The video has invalid dimensions.")

    target_width, target_height = _target_size(width, height, settings.resolution)
    silent_path = output_path.with_name("silent.mp4")
    writer = cv2.VideoWriter(
        str(silent_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (target_width * 2, target_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not initialize the local MP4 encoder.")

    previous_depth: np.ndarray | None = None
    gaussian_renderer = _gaussian_renderer(settings) if settings.render_method == "gaussian-4d" else None
    processed = 0
    inference_ms = 0.0
    temporal_reuse = 0.0
    gaussian_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != (target_width, target_height):
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            depth, elapsed_ms = engine.infer(frame, settings.model)
            inference_ms += elapsed_ms
            if previous_depth is not None and settings.temporal_smoothing > 0:
                depth = (
                    settings.temporal_smoothing * previous_depth
                    + (1.0 - settings.temporal_smoothing) * depth
                ).astype(np.float32)
            previous_depth = depth
            writer.write(stereo_frame(frame, depth, settings, gaussian_renderer))
            if gaussian_renderer is not None:
                temporal_reuse += gaussian_renderer.last_temporal_reuse
                gaussian_count += gaussian_renderer.last_gaussian_count
            processed += 1
            ratio = processed / max(frame_count, processed)
            progress(
                0.05 + ratio * 0.85,
                f"Rendering frame {processed}{f' / {frame_count}' if frame_count else ''}",
                f"{settings.model} · {settings.render_method} · {target_width}×{target_height} per eye",
            )
    finally:
        capture.release()
        writer.release()

    if processed == 0:
        raise ValueError("No frames could be decoded from this video.")
    progress(0.93, "Encoding browser preview…", "Muxing the original audio when available.")
    _encode_for_browser(silent_path, source_path, output_path)
    silent_path.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    progress(1.0, "Complete", f"{processed} frames rendered locally in {elapsed:.1f}s.")
    result = {
        "output_kind": "video",
        "media_type": "video/mp4",
        "elapsed_seconds": elapsed,
        "frames": processed,
        "fps": fps,
        "width_per_eye": target_width,
        "height": target_height,
        "average_inference_ms": inference_ms / processed,
    }
    if gaussian_renderer is not None:
        result.update({
            "average_gaussians_per_frame": int(gaussian_count / processed),
            "average_temporal_reuse_pct": temporal_reuse / processed * 100.0,
            "gaussian_mode": "monocular-4d-lite",
        })
    return result


def process_image(
    source_path: Path,
    output_path: Path,
    settings: RenderSettings,
    engine: DepthEngine,
    progress: ProgressCallback,
) -> dict[str, float | int | str]:
    started = time.perf_counter()
    frame = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("OpenCV could not decode this image. Try a JPG, PNG, or WebP file.")
    height, width = frame.shape[:2]
    if width * height > 50_000_000:
        raise ValueError("Keep still images under 50 megapixels for this workflow.")

    target_width, target_height = _target_size(width, height, settings.resolution)
    if (width, height) != (target_width, target_height):
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    progress(0.12, "Inferring image depth…", f"{settings.model} · {target_width}×{target_height}")
    depth, inference_ms = engine.infer(frame, settings.model)
    progress(0.68, "Synthesizing eye views…", f"{settings.render_method} · {target_width}×{target_height} per eye")
    gaussian_renderer = _gaussian_renderer(settings) if settings.render_method == "gaussian-4d" else None
    stereo = stereo_frame(frame, depth, settings, gaussian_renderer)
    if not cv2.imwrite(str(output_path), stereo, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError("Could not encode the stereoscopic PNG.")

    elapsed = time.perf_counter() - started
    progress(1.0, "Complete", f"Stereo image rendered locally in {elapsed:.1f}s.")
    result = {
        "output_kind": "image",
        "media_type": "image/png",
        "elapsed_seconds": elapsed,
        "frames": 1,
        "width_per_eye": target_width,
        "height": target_height,
        "average_inference_ms": inference_ms,
    }
    if gaussian_renderer is not None:
        result.update({
            "average_gaussians_per_frame": gaussian_renderer.last_gaussian_count,
            "average_temporal_reuse_pct": 0.0,
            "gaussian_mode": "single-frame-gaussian",
        })
    return result
