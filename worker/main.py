from __future__ import annotations

import base64
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .models import MODEL_SPECS, DepthEngine
from .pipeline import RenderSettings, process_image, process_video


LAB_ROOT = Path(os.getenv("STEREO_LAB_WORKDIR", Path.home() / ".stereo-lab")).resolve()
LAB_ROOT.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_ACTIVE_JOBS = 1
ALLOWED_ORIGINS = [
    "https://stereo-depth-lab-keith.h2vmw6.chatgpt.site",
    "https://stereo-depth-lab-keith.lunar-scout-5657.chatgpt.site",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


@dataclass
class Job:
    id: str
    directory: Path
    source_path: Path
    output_path: Path
    settings: RenderSettings
    media_kind: str
    status: str = "queued"
    progress: float = 0.0
    stage: str = "Queued"
    detail: str = "Waiting for the local GPU."
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    elapsed_seconds: float = 0.0
    result: dict = field(default_factory=dict)

    def public(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "detail": self.detail,
            "error": self.error,
            "created_at": self.created_at,
            "elapsed_seconds": self.elapsed_seconds,
            "settings": asdict(self.settings),
            "media_kind": self.media_kind,
            **self.result,
        }


app = FastAPI(title="Stereo Depth Lab Worker", version="1.0.0", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


@app.middleware("http")
async def private_network_access(request: Request, call_next):
    response = await call_next(request)
    if request.headers.get("access-control-request-private-network") == "true":
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


engine = DepthEngine()
jobs: dict[str, Job] = {}
jobs_lock = threading.RLock()


def _cleanup_old_jobs() -> None:
    cutoff = time.time() - 2 * 60 * 60
    with jobs_lock:
        expired = [job_id for job_id, job in jobs.items() if job.created_at < cutoff and job.status != "processing"]
        for job_id in expired:
            shutil.rmtree(jobs[job_id].directory, ignore_errors=True)
            del jobs[job_id]


def _active_jobs() -> int:
    return sum(job.status in {"queued", "processing"} for job in jobs.values())


def _run_job(job: Job) -> None:
    started = time.perf_counter()

    def report(value: float, stage: str, detail: str) -> None:
        with jobs_lock:
            job.progress = float(max(0.0, min(1.0, value)))
            job.stage = stage
            job.detail = detail

    try:
        with jobs_lock:
            job.status = "processing"
        processor = process_image if job.media_kind == "image" else process_video
        result = processor(job.source_path, job.output_path, job.settings, engine, report)
        with jobs_lock:
            job.result = result
            job.elapsed_seconds = float(result["elapsed_seconds"])
            job.status = "complete"
            job.progress = 1.0
    except Exception as exc:  # job errors are intentionally surfaced to the private UI
        logging.exception("Stereo render %s failed", job.id)
        with jobs_lock:
            job.status = "failed"
            job.error = str(exc)
            job.stage = "Render failed"
            job.detail = "Review the worker terminal for the full local traceback."
            job.elapsed_seconds = time.perf_counter() - started


@app.get("/api/health")
def health() -> dict:
    device_label = "Diagnostic pipeline"
    if not engine.fake_mode:
        if engine.cuda:
            import torch

            device_label = torch.cuda.get_device_name(0)
        else:
            device_label = "CPU"
    return {
        "status": "ok",
        "private": True,
        "cuda": engine.cuda,
        "device": device_label,
        "models": [asdict(spec) for spec in MODEL_SPECS.values()],
        "loaded_models": engine.loaded_models,
        "active_jobs": _active_jobs(),
        "renderers": ["depth-warp", "point-cloud", "gaussian-4d"],
    }


@app.post("/api/jobs")
async def create_job(
    media: UploadFile | None = File(None),
    video: UploadFile | None = File(None),
    model: str = Form(...),
    render_method: str = Form(...),
    eye_separation: float = Form(...),
    depth_strength: float = Form(...),
    convergence: float = Form(...),
    temporal_smoothing: float = Form(...),
    resolution: str = Form(...),
    gaussian_scale: float = Form(1.35),
) -> dict:
    _cleanup_old_jobs()
    upload = media or video
    if upload is None:
        raise HTTPException(422, "Upload a video or image.")
    if model not in MODEL_SPECS:
        raise HTTPException(422, "Choose a supported depth model.")
    if render_method not in {"depth-warp", "point-cloud", "gaussian-4d"}:
        raise HTTPException(422, "Choose a supported view-synthesis method.")
    if resolution not in {"720p", "1080p", "source"}:
        raise HTTPException(422, "Choose 720p, 1080p, or source resolution.")
    if not 0 <= eye_separation <= 64:
        raise HTTPException(422, "Eye separation must be between 0 and 64 pixels.")
    if not 0 <= depth_strength <= 2:
        raise HTTPException(422, "Depth strength must be between 0 and 2.")
    if not 0 <= convergence <= 1 or not 0 <= temporal_smoothing <= 0.95:
        raise HTTPException(422, "Convergence or smoothing is outside its safe range.")
    if not 0.55 <= gaussian_scale <= 3.0:
        raise HTTPException(422, "Gaussian scale must be between 0.55 and 3.0 pixels.")
    with jobs_lock:
        if _active_jobs() >= MAX_ACTIVE_JOBS:
            raise HTTPException(409, "The local GPU is already processing another file.")

    job_id = uuid.uuid4().hex
    directory = LAB_ROOT / job_id
    directory.mkdir(parents=True, exist_ok=False)
    suffix = Path(upload.filename or "source.mp4").suffix.lower()
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    video_suffixes = {".mp4", ".mov", ".webm", ".m4v", ".avi"}
    content_type = (upload.content_type or "").lower()
    if suffix in image_suffixes or content_type.startswith("image/"):
        media_kind = "image"
        if suffix not in image_suffixes:
            suffix = ".png"
    elif suffix in video_suffixes or content_type.startswith("video/"):
        media_kind = "video"
        if suffix not in video_suffixes:
            suffix = ".mp4"
    else:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(422, "Choose an MP4, MOV, WebM, JPG, PNG, or WebP file.")
    source_path = directory / f"source{suffix}"
    written = 0
    try:
        with source_path.open("wb") as destination:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Keep the local source under 512 MB.")
                destination.write(chunk)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        await upload.close()

    settings = RenderSettings(
        model=model,
        render_method=render_method,
        eye_separation=eye_separation,
        depth_strength=depth_strength,
        convergence=convergence,
        temporal_smoothing=temporal_smoothing,
        resolution=resolution,
        gaussian_scale=gaussian_scale,
    )
    output_path = directory / ("stereo-sbs.png" if media_kind == "image" else "stereo-sbs.mp4")
    job = Job(job_id, directory, source_path, output_path, settings, media_kind)
    with jobs_lock:
        jobs[job_id] = job
    threading.Thread(target=_run_job, args=(job,), daemon=True, name=f"stereo-job-{job_id[:8]}").start()
    return {"id": job_id, "status": job.status}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Render job not found.")
    return job.public()


@app.get("/api/jobs/{job_id}/output")
def get_output(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Render job not found.")
    if job.status != "complete" or not job.output_path.exists():
        raise HTTPException(409, "The render is not complete.")
    media_type = str(job.result.get("media_type", "application/octet-stream"))
    filename = "stereo-sbs.png" if job.media_kind == "image" else "stereo-sbs.mp4"
    return FileResponse(job.output_path, media_type=media_type, filename=filename)


@app.post("/api/compare")
async def compare_depth_models(frame: UploadFile = File(...)) -> dict:
    payload = await frame.read(12 * 1024 * 1024 + 1)
    await frame.close()
    if len(payload) > 12 * 1024 * 1024:
        raise HTTPException(413, "Comparison frame is too large.")
    encoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if encoded is None:
        raise HTTPException(422, "Could not decode the comparison frame.")

    result = {}
    for model_id in MODEL_SPECS:
        depth, elapsed_ms = engine.infer(encoded, model_id)
        color = engine.colorize(depth)
        ok, png = cv2.imencode(".png", color)
        if not ok:
            raise HTTPException(500, "Could not encode a depth preview.")
        result[model_id] = {
            "image": base64.b64encode(png.tobytes()).decode("ascii"),
            "milliseconds": elapsed_ms,
            "device": engine.device,
        }
    return result
