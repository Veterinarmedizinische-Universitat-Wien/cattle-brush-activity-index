"""End-to-end run on small synthetic videos (no real recordings needed).

Two filename-timestamped clips in the same hour, separated by a 60 s gap. A
textured patch inside the ROI "rotates" (shifts) during known intervals.
"""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import cv2
import numpy as np
import pandas as pd

import brush_rotation_activity_index as bri

FPS = 10
W, H = 640, 360
ROI = np.array([[200, 100], [200, 260], [440, 260], [440, 100]], np.int32)


def write_clip(path, seconds, active):
    rng = np.random.default_rng(0)
    texture = rng.integers(20, 160, (200, 280, 3), dtype=np.uint8)
    background = np.full((H, W, 3), 90, np.uint8)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV cannot write mp4v here")
    shift = 0
    for i in range(seconds * FPS):
        t = i / FPS
        if any(a <= t < b for a, b in active):
            shift += 7
        frame = background.copy()
        patch = np.roll(texture, shift, axis=1)
        frame[80:280, 180:460] = patch
        noise = rng.integers(-2, 3, frame.shape, dtype=np.int16)
        writer.write(np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    writer.release()


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.inp = os.path.join(cls.tmp, "in")
        os.makedirs(cls.inp)
        # clip 1: 12:00:00-12:04:00, clip 2 starts 12:05:00 (60 s gap)
        write_clip(os.path.join(cls.inp, "2024-04-25_12-00-00.mp4"), 240, [(30, 60), (150, 170)])
        write_clip(os.path.join(cls.inp, "2024-04-25_12-05-00.mp4"), 240, [(40, 80)])
        # untimestamped video -> standalone relative-time run
        write_clip(os.path.join(cls.inp, "no_time.mp4"), 240, [(100, 130)])

        args = bri.build_arg_parser().parse_args([
            "--input", cls.inp, "--output", os.path.join(cls.tmp, "out"),
            "--experiment-name", "e2e",
        ])
        cls.ctx = bri.context_from_args(args)
        cls.ctx.roi_polygon = ROI
        cls.ctx.roi_reference_size = (W, H)
        bri.run(cls.inp, cls.ctx)
        cls.out = cls.ctx.output_folder

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_reports_written(self):
        for name in ("rotation_summary.xlsx", "hour_bucket_summary.xlsx",
                     "file_timestamp_report.xlsx", "clip_timing_diagnostics.xlsx"):
            self.assertTrue(os.path.exists(os.path.join(self.out, name)), name)
        self.assertTrue(os.listdir(os.path.join(self.out, "annotated_videos")))

    def test_gap_resets_clip_join(self):
        hour = pd.read_excel(os.path.join(self.out, "hour_bucket_summary.xlsx"))
        self.assertEqual(int(hour.loc[0, "Clip Join Resets"]), 1)
        self.assertEqual(int(hour.loc[0, "Placeholder Frames (excluded from normalization refs)"]), 2)

    def test_detected_bouts_match_active_intervals(self):
        rot = pd.read_excel(os.path.join(self.out, "rotation_summary.xlsx"))
        starts = pd.to_datetime(rot["Bout Start"], format="%d.%m.%Y %H:%M:%S.%f")
        ends = pd.to_datetime(rot["Bout End"], format="%d.%m.%Y %H:%M:%S.%f")
        base = pd.Timestamp("2024-04-25 12:00:00")
        got = [((s - base).total_seconds(), (e - base).total_seconds()) for s, e in zip(starts, ends)]
        expected = [(30, 60), (150, 170), (340, 380)]
        self.assertEqual(len(got), len(expected), got)
        for (gs, ge), (es, ee) in zip(got, expected):
            self.assertAlmostEqual(gs, es, delta=1.5)
            self.assertAlmostEqual(ge, ee, delta=1.5)

    def test_standalone_relative_video(self):
        rel = os.path.join(self.out, "standalone_relative", "no_time", "rotation_summary.xlsx")
        rot = pd.read_excel(rel)
        self.assertEqual(len(rot), 1)
        self.assertTrue(str(rot.loc[0, "Bout Start"]).startswith("00:01:"))


if __name__ == "__main__":
    unittest.main()
