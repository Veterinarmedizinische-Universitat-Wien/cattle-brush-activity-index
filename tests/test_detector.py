"""Unit tests for the detector, normalisation, ROI handling and run context."""
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import numpy as np

import brush_rotation_activity_index as bri
import config


class ImportSideEffectTests(unittest.TestCase):
    def test_import_creates_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC)] + sys.path))
            subprocess.run(
                [sys.executable, "-c", "import brush_rotation_activity_index"],
                cwd=tmp, env=env, check=True,
            )
            self.assertEqual(os.listdir(tmp), [])


class RoiTests(unittest.TestCase):
    def test_no_reference_size_means_native_coordinates(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        polygon = np.array([[100, 100], [100, 300], [400, 300], [400, 100]], np.int32)
        _, scaled = bri.prepare_analysis_frame(frame, polygon, None)
        np.testing.assert_array_equal(scaled, np.rint(polygon * 0.5).astype(np.int32))

    def test_custom_polygon_scaled_from_native_1080p(self):
        """A custom polygon on 1080p footage must not be scaled as if drawn on 720p."""
        frame = np.zeros((1080, 1920, 3), np.uint8)
        polygon = np.array([[300, 300], [300, 600], [900, 600], [900, 300]], np.int32)
        _, scaled = bri.prepare_analysis_frame(frame, polygon, None)
        np.testing.assert_array_equal(scaled, np.rint(polygon / 3).astype(np.int32))

    def test_reference_size_applied_without_resizing(self):
        frame = np.zeros((1080, 1920, 3), np.uint8)
        polygon = config.ROI_ALL["B"]
        with patch.object(bri, "PROCESSING_HEIGHT", None):
            out, scaled = bri.prepare_analysis_frame(frame, polygon, (1280, 720))
        self.assertEqual(out.shape[:2], (1080, 1920))
        np.testing.assert_array_equal(scaled, np.rint(polygon * 1.5).astype(np.int32))

    def test_cli_preset_uses_its_reference_size(self):
        args = bri.build_arg_parser().parse_args(["--roi-preset", "B", "--output", "/tmp/x"])
        ctx = bri.context_from_args(args)
        self.assertEqual(ctx.roi_reference_size, (1280, 720))
        np.testing.assert_array_equal(ctx.roi_polygon, config.ROI_ALL["B"])
        self.assertEqual(ctx.output_folder, os.path.join("/tmp/x", config.EXPERIMENT_NAME))


class NormalisationTests(unittest.TestCase):
    def test_placeholder_frames_excluded_from_references(self):
        rng = np.random.default_rng(1)
        real = rng.gamma(2.0, 1000.0, 5000).astype(np.float32)
        with_placeholders = np.concatenate([np.zeros(500, np.float32), real])
        valid = np.concatenate([np.zeros(500, bool), np.ones(5000, bool)])

        _, info_clean = bri.robust_log_normalize(real)
        norm, info_masked = bri.robust_log_normalize(with_placeholders, valid_mask=valid)
        _, info_unmasked = bri.robust_log_normalize(with_placeholders)

        self.assertAlmostEqual(info_masked["low_ref"], info_clean["low_ref"], places=5)
        self.assertAlmostEqual(info_masked["high_ref"], info_clean["high_ref"], places=5)
        self.assertNotAlmostEqual(info_unmasked["low_ref"], info_clean["low_ref"], places=2)
        self.assertEqual(len(norm), len(with_placeholders))
        self.assertTrue(np.all(norm[:500] == 0.0))

    def test_no_mask_matches_previous_behaviour(self):
        x = np.linspace(0, 5000, 1000, dtype=np.float32)
        a, _ = bri.robust_log_normalize(x)
        b, _ = bri.robust_log_normalize(x, valid_mask=np.ones(1000, bool))
        np.testing.assert_array_equal(a, b)


class ThresholdTests(unittest.TestCase):
    def test_reference_curve_and_percentage(self):
        fps = 10.0
        seg = int(10 * 60 * fps)
        signal = np.concatenate([np.full(seg, 0.5), np.full(seg, 1.0)]).astype(np.float32)
        thr, ref = bri.compute_segment_threshold_curve(signal, fps, return_reference=True)
        np.testing.assert_allclose(ref[:seg], 0.5)
        np.testing.assert_allclose(thr[:seg], 0.62 * 0.5, rtol=1e-6)
        self.assertEqual(bri.threshold_percentage_at(thr, ref, 0), 62)
        self.assertEqual(bri.threshold_percentage_at(thr, ref, seg + 1), 50)

    def test_default_return_is_unchanged(self):
        signal = np.ones(100, np.float32)
        out = bri.compute_segment_threshold_curve(signal, 10.0)
        self.assertIsInstance(out, np.ndarray)


class ShortBoutRuleTests(unittest.TestCase):
    fps = 10.0

    def bouts(self, *specs):
        """specs: (start_s, duration_s) -> frame periods."""
        return [(int(s * self.fps), int((s + d) * self.fps) - 1) for s, d in specs]

    def test_study_defaults_cannot_remove_detected_bouts(self):
        # Every detected bout is > MIN_BOUT_SECONDS, so "all < SHORT_BOUT_SECONDS"
        # is never true with the shipped defaults.
        periods = self.bouts((0, 4.1), (10, 4.2), (20, 4.5))
        out = bri.evaluate_window(periods, self.fps, window_seconds=40,
                                  threshold=config.SHORT_BOUT_SECONDS)
        self.assertEqual(out, periods)
        self.assertTrue(any("evaluate_window" in w for w in config.validate_config()))

    def test_rule_removes_short_cluster_when_threshold_exceeds_minimum(self):
        short = self.bouts((0, 4.5), (10, 4.5), (20, 4.5))
        isolated_long = self.bouts((300, 30))
        out = bri.evaluate_window(short + isolated_long, self.fps, window_seconds=40, threshold=6)
        self.assertEqual(out, isolated_long)

    def test_no_warning_when_rule_can_fire(self):
        with patch.object(config, "SHORT_BOUT_SECONDS", 6):
            self.assertFalse(any("evaluate_window" in w for w in config.validate_config()))


class ClipJoinTests(unittest.TestCase):
    def test_contiguity(self):
        t = datetime(2024, 4, 25, 12, 0, 0)
        self.assertTrue(bri.clips_are_contiguous(t, t + timedelta(seconds=0.04)))
        self.assertTrue(bri.clips_are_contiguous(t, t - timedelta(seconds=0.5)))  # overlap
        self.assertFalse(bri.clips_are_contiguous(t, t + timedelta(seconds=30)))
        self.assertFalse(bri.clips_are_contiguous(None, t))


if __name__ == "__main__":
    unittest.main()
