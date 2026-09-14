from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GaussianFrameState:
    color: np.ndarray
    depth: np.ndarray
    gray: np.ndarray
    confidence: np.ndarray


class TemporalGaussianRenderer:
    """A practical monocular 4D Gaussian splat renderer.

    Each source pixel seeds a depth-aware Gaussian in screen-space. The fourth
    dimension is represented by a persistent, optical-flow-advected Gaussian
    layer with a temporal covariance controlled by temporal_smoothing.

    This is intentionally not a trained multi-view 4DGS reconstruction. It is a
    deterministic preview renderer designed for casual monocular uploads.
    """

    def __init__(
        self,
        gaussian_scale: float,
        temporal_smoothing: float,
        eye_separation: float,
        depth_strength: float,
        convergence: float,
    ) -> None:
        self.gaussian_scale = float(np.clip(gaussian_scale, 0.55, 3.0))
        self.temporal_smoothing = float(np.clip(temporal_smoothing, 0.0, 0.95))
        self.disparity_scale = float(eye_separation * depth_strength)
        self.convergence = float(convergence)
        self.state: GaussianFrameState | None = None
        self.last_temporal_reuse = 0.0
        self.last_gaussian_count = 0

    @staticmethod
    def _flow_warp(previous: GaussianFrameState, current_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Backward flow maps each current pixel to its source in the prior frame.
        flow = cv2.calcOpticalFlowFarneback(
            current_gray,
            previous.gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=19,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        height, width = current_gray.shape
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        map_x = grid_x + flow[..., 0]
        map_y = grid_y + flow[..., 1]
        warped_color = cv2.remap(previous.color, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        warped_depth = cv2.remap(previous.depth, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        warped_confidence = cv2.remap(previous.confidence, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return warped_color, warped_depth, warped_confidence

    @staticmethod
    def _edge_confidence(depth: np.ndarray) -> np.ndarray:
        gradient_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gradient_x, gradient_y)
        return np.exp(-2.25 * magnitude).astype(np.float32)

    @staticmethod
    def _scatter(
        color: np.ndarray,
        depth: np.ndarray,
        opacity: np.ndarray,
        horizontal_offset: np.ndarray,
        numerator: np.ndarray,
        denominator: np.ndarray,
    ) -> None:
        height, width = depth.shape
        yy, xx = np.indices((height, width), dtype=np.int32)
        projected_x = xx.astype(np.float32) + horizontal_offset
        base_x = np.floor(projected_x).astype(np.int32)
        fraction = projected_x - base_x

        # Exponential depth weighting approximates visibility: near Gaussians
        # dominate far Gaussians that land on the same screen-space footprint.
        visibility = np.exp(3.25 * (depth - 1.0)).astype(np.float32)
        source_color = color.astype(np.float32) / 255.0
        pixel_count = height * width

        for target_x, bilinear_weight in ((base_x, 1.0 - fraction), (base_x + 1, fraction)):
            valid = (target_x >= 0) & (target_x < width)
            indices = (yy[valid] * width + target_x[valid]).astype(np.int64)
            weights = (opacity[valid] * visibility[valid] * bilinear_weight[valid]).astype(np.float64)
            denominator.reshape(-1)[:] += np.bincount(indices, weights=weights, minlength=pixel_count)
            for channel in range(3):
                values = weights * source_color[..., channel][valid]
                numerator[..., channel].reshape(-1)[:] += np.bincount(indices, weights=values, minlength=pixel_count)

    def _render_eye(
        self,
        current_color: np.ndarray,
        current_depth: np.ndarray,
        current_opacity: np.ndarray,
        current_offset: np.ndarray,
        history: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
        direction: float,
    ) -> np.ndarray:
        height, width = current_depth.shape
        numerator = np.zeros((height, width, 3), dtype=np.float64)
        denominator = np.zeros((height, width), dtype=np.float64)
        self._scatter(current_color, current_depth, current_opacity, direction * current_offset, numerator, denominator)

        if history is not None:
            history_color, history_depth, history_opacity, history_offset = history
            self._scatter(history_color, history_depth, history_opacity, direction * history_offset, numerator, denominator)

        # Projected covariance is anisotropic: stereo parallax expands more on x
        # than y. Filtering numerator and opacity implements Gaussian splatting.
        sigma_x = self.gaussian_scale * 1.35
        sigma_y = self.gaussian_scale * 0.72
        numerator = cv2.GaussianBlur(numerator, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
        denominator = cv2.GaussianBlur(denominator, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
        rendered = numerator / np.maximum(denominator[..., None], 1e-7)
        rendered = np.clip(rendered * 255.0, 0, 255).astype(np.uint8)

        holes = (denominator < 8e-4).astype(np.uint8) * 255
        if holes.any():
            rendered = cv2.inpaint(rendered, holes, 3, cv2.INPAINT_TELEA)
        return rendered

    def render(self, frame: np.ndarray, depth: np.ndarray, disparity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge_confidence = self._edge_confidence(depth)
        current_opacity = (0.25 + 0.75 * edge_confidence).astype(np.float32)
        history_layer: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

        if self.state is not None and self.temporal_smoothing > 0:
            warped_color, warped_depth, warped_confidence = self._flow_warp(self.state, current_gray)
            photometric_error = np.mean(np.abs(warped_color.astype(np.float32) - frame.astype(np.float32)), axis=2)
            match_confidence = np.exp(-photometric_error / 26.0).astype(np.float32)

            # A broad scene cut invalidates temporal Gaussians instead of
            # smearing them into the new shot.
            if float(np.mean(photometric_error)) < 58.0:
                sigma_t = 0.3 + 2.7 * self.temporal_smoothing
                temporal_kernel = np.exp(-0.5 / (sigma_t * sigma_t))
                history_opacity = (
                    self.temporal_smoothing
                    * temporal_kernel
                    * warped_confidence
                    * match_confidence
                ).astype(np.float32)
                history_offset = (warped_depth - self.convergence) * self.disparity_scale * 0.5
                history_layer = (warped_color, warped_depth, history_opacity, history_offset)
                self.last_temporal_reuse = float(np.mean(history_opacity))

        half_disparity = disparity * 0.5
        left = self._render_eye(frame, depth, current_opacity, half_disparity, history_layer, -1.0)
        right = self._render_eye(frame, depth, current_opacity, half_disparity, history_layer, 1.0)

        if history_layer is None:
            fused_color = frame.astype(np.float32)
            fused_depth = depth.astype(np.float32)
            confidence = current_opacity
            self.last_temporal_reuse = 0.0
        else:
            history_color, history_depth, history_opacity, _ = history_layer
            blend = np.clip(history_opacity[..., None], 0.0, 0.88)
            fused_color = (frame.astype(np.float32) * (1.0 - blend) + history_color.astype(np.float32) * blend)
            fused_depth = (depth * (1.0 - blend[..., 0]) + history_depth * blend[..., 0]).astype(np.float32)
            confidence = np.clip(current_opacity + history_opacity * 0.35, 0.0, 1.0)

        self.state = GaussianFrameState(
            color=np.clip(fused_color, 0, 255).astype(np.uint8),
            depth=fused_depth,
            gray=current_gray,
            confidence=confidence.astype(np.float32),
        )
        self.last_gaussian_count = int(depth.size * (2 if history_layer is not None else 1))
        return left, right
