from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    repository: str


MODEL_SPECS = {
    "depth-anything-v2-small": ModelSpec(
        id="depth-anything-v2-small",
        label="Depth Anything V2 Small",
        repository="depth-anything/Depth-Anything-V2-Small-hf",
    ),
    "midas-hybrid": ModelSpec(
        id="midas-hybrid",
        label="MiDaS Hybrid",
        repository="Intel/dpt-hybrid-midas",
    ),
}


class DepthEngine:
    """Lazy, thread-safe relative-depth inference for the two lab backends."""

    def __init__(self) -> None:
        self._models: dict[str, tuple[object, object]] = {}
        self._lock = threading.RLock()
        self.fake_mode = os.getenv("STEREO_LAB_FAKE_MODEL", "0") == "1"
        if self.fake_mode:
            self.device = "diagnostic"
            self.cuda = False
        else:
            import torch

            self.cuda = bool(torch.cuda.is_available())
            self.device = "cuda" if self.cuda else "cpu"

    @property
    def loaded_models(self) -> list[str]:
        return list(self._models)

    def _load(self, model_id: str) -> tuple[object, object]:
        if model_id not in MODEL_SPECS:
            raise ValueError(f"Unsupported depth model: {model_id}")
        with self._lock:
            if model_id in self._models:
                return self._models[model_id]

            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            spec = MODEL_SPECS[model_id]
            processor = AutoImageProcessor.from_pretrained(spec.repository)
            dtype = torch.float16 if self.cuda else torch.float32
            model = AutoModelForDepthEstimation.from_pretrained(
                spec.repository,
                torch_dtype=dtype,
            ).to(self.device)
            model.eval()
            self._models[model_id] = (processor, model)
            return processor, model

    @staticmethod
    def _normalize(depth: np.ndarray) -> np.ndarray:
        finite = np.isfinite(depth)
        if not finite.any():
            return np.zeros(depth.shape, dtype=np.float32)
        low, high = np.percentile(depth[finite], (2.0, 98.0))
        if high - low < 1e-7:
            return np.zeros(depth.shape, dtype=np.float32)
        normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
        return normalized.astype(np.float32)

    def infer(self, frame_bgr: np.ndarray, model_id: str) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        if self.fake_mode:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            depth = cv2.GaussianBlur(gray, (0, 0), 7.0)
            return self._normalize(depth), (time.perf_counter() - started) * 1000

        import torch
        import torch.nn.functional as functional

        processor, model = self._load(model_id)
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if self.cuda:
            inputs = {key: value.half() if value.is_floating_point() else value for key, value in inputs.items()}
        with torch.inference_mode():
            prediction = model(**inputs).predicted_depth
            prediction = functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = prediction.float().cpu().numpy()
        return self._normalize(depth), (time.perf_counter() - started) * 1000

    @staticmethod
    def colorize(depth: np.ndarray) -> np.ndarray:
        plane = np.clip(depth * 255, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(plane, cv2.COLORMAP_TURBO)

