from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ["STEREO_LAB_FAKE_MODEL"] = "1"

import cv2
import numpy as np

from worker.models import DepthEngine
from worker.pipeline import RenderSettings, depth_warp, point_cloud_warp, process_image, process_video


class StereoPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((72, 128, 3), dtype=np.uint8)
        self.frame[:, :64] = (30, 60, 220)
        self.frame[:, 64:] = (180, 210, 40)
        self.depth = np.tile(np.linspace(0, 1, 128, dtype=np.float32), (72, 1))
        self.settings = RenderSettings(
            model="depth-anything-v2-small",
            render_method="depth-warp",
            eye_separation=20,
            depth_strength=1,
            convergence=0.5,
            temporal_smoothing=0.7,
            resolution="source",
        )

    def test_both_view_synthesis_paths_preserve_eye_shape(self) -> None:
        for renderer in (depth_warp, point_cloud_warp):
            left, right = renderer(self.frame, self.depth, self.settings)
            self.assertEqual(left.shape, self.frame.shape)
            self.assertEqual(right.shape, self.frame.shape)
            self.assertFalse(np.array_equal(left, right))

    def test_short_video_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            output = root / "stereo.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (128, 72))
            self.assertTrue(writer.isOpened())
            for index in range(8):
                frame = np.roll(self.frame, index * 3, axis=1)
                writer.write(frame)
            writer.release()

            events = []
            result = process_video(source, output, self.settings, DepthEngine(), lambda *args: events.append(args))
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1000)
            self.assertEqual(result["frames"], 8)
            self.assertEqual(result["width_per_eye"], 128)
            self.assertEqual(events[-1][0], 1.0)

    def test_still_image_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            output = root / "stereo.png"
            self.assertTrue(cv2.imwrite(str(source), self.frame))

            events = []
            result = process_image(source, output, self.settings, DepthEngine(), lambda *args: events.append(args))
            rendered = cv2.imread(str(output), cv2.IMREAD_COLOR)
            self.assertIsNotNone(rendered)
            self.assertEqual(rendered.shape, (72, 256, 3))
            self.assertEqual(result["output_kind"], "image")
            self.assertEqual(result["frames"], 1)
            self.assertEqual(events[-1][0], 1.0)


if __name__ == "__main__":
    unittest.main()
