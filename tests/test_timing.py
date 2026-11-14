"""Timestamp handling: DVR clock repairs and filename fallback."""
import sys
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import io_utils


def dvr_times(n, start=datetime(2024, 4, 25, 0, 0, 0), fps=25.0):
    return [start + timedelta(seconds=i / fps) for i in range(n)]


class TimingTests(unittest.TestCase):
    def test_median_positive_step_ignores_repeats_and_gaps(self):
        ts = dvr_times(10)
        ts.insert(5, ts[4])                                  # duplicate
        ts.append(ts[-1] + timedelta(seconds=5))             # recording gap
        self.assertEqual(io_utils.median_positive_step(ts), timedelta(seconds=0.04))
        self.assertIsNone(io_utils.median_positive_step(ts[:1]))

    def test_backward_timestamp_repaired_on_dvr_clock(self):
        frags = ["00:00:00.000", "00:00:00.040", "00:00:00.080", "00:00:00.060", "00:00:00.160"]
        ts = io_utils.reconstruct_dvr_timestamps(date(2024, 4, 25), frags, fps=30.0)
        self.assertEqual(ts[3] - ts[2], timedelta(seconds=0.04))   # not 1/30 s

    def test_short_dvr_stream_extended_on_dvr_clock(self):
        meta = {"timestamps": dvr_times(98), "camera_id": 1, "overlay_id": None}
        with patch.object(io_utils, "extract_dvr_metadata", return_value=meta):
            out = io_utils.get_video_timestamps_and_ids("clip.avi", fps=30.0, total_frames=100)
        self.assertEqual(len(out["timestamps"]), 100)
        self.assertEqual(out["timestamps"][-1] - out["timestamps"][-2], timedelta(seconds=0.04))

    def test_clock_fps_from_dvr_not_container(self):
        info = {"fps": 30.0, "length": 50, "frame_width": 1280, "frame_height": 720}
        meta = {"timestamps": dvr_times(50), "camera_id": 1, "overlay_id": None}
        with patch.object(io_utils, "get_video_basic_info", return_value=info), \
             patch.object(io_utils, "extract_dvr_metadata", return_value=meta):
            clip = io_utils.get_clip_start_info("clip.avi")
        self.assertEqual(clip["timestamp_source"], "dvr")
        self.assertAlmostEqual(clip["actual_fps"], 25.0, places=3)
        self.assertEqual(clip["fps"], 30.0)

    def test_filename_fallback_uses_container_rate(self):
        # A re-encoded copy without the DVR stream falls back to the filename.
        info = {"fps": 30.0, "length": 31, "frame_width": 1280, "frame_height": 720}
        with patch.object(io_utils, "get_video_basic_info", return_value=info), \
             patch.object(io_utils, "extract_dvr_metadata", return_value=None):
            clip = io_utils.get_clip_start_info("2024-04-25_12-00-00_480p.mp4")
        self.assertEqual(clip["timestamp_source"], "filename")
        self.assertEqual(clip["actual_fps"], 30.0)
        self.assertEqual(clip["timestamps"][30] - clip["timestamps"][0], timedelta(seconds=1))


if __name__ == "__main__":
    unittest.main()
