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
    @patch.object(bri, "PROCESSING_HEIGHT", 360)
    def test_no_reference_size_means_native_coordinates(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        polygon = np.array([[100, 100], [100, 300], [400, 300], [400, 100]], np.int32)
        _, scaled = bri.prepare_analysis_frame(frame, polygon, None)
        np.testing.assert_array_equal(scaled, np.rint(polygon * 0.5).astype(np.int32))

    @patch.object(bri, "PROCESSING_HEIGHT", 360)
    def test_custom_polygon_scaled_from_native_1080p(self):
        """A custom polygon on 1080p footage must not be scaled as if drawn on 720p."""
        frame = np.zeros((1080, 1920, 3), np.uint8)
        polygon = np.array([[300, 300], [300, 600], [900, 600], [900, 300]], np.int32)
        _, scaled = bri.prepare_analysis_frame(frame, polygon, None)
        np.testing.assert_array_equal(scaled, np.rint(polygon / 3).astype(np.int32))

    def test_reference_size_applied_without_resizing(self):
        frame = np.zeros((1080, 1920, 3), np.uint8)
        polygon = config.ROI_ALL["A2"]
        with patch.object(bri, "PROCESSING_HEIGHT", None):
            out, scaled = bri.prepare_analysis_frame(frame, polygon, (1280, 720))
        self.assertEqual(out.shape[:2], (1080, 1920))
        np.testing.assert_array_equal(scaled, np.rint(polygon * 1.5).astype(np.int32))

    def test_cli_preset_uses_its_reference_size(self):
        args = bri.build_arg_parser().parse_args(["--roi-preset", "A2", "--output", "/tmp/x"])
        ctx = bri.context_from_args(args)
        self.assertEqual(ctx.roi_reference_size, (1280, 720))
        np.testing.assert_array_equal(ctx.roi_polygon, config.ROI_ALL["A2"])
        self.assertEqual(ctx.output_folder, os.path.join("/tmp/x", config.EXPERIMENT_NAME))

    def test_configured_custom_roi_used_without_cli_override(self):
        custom = np.array([[10, 20], [10, 80], [90, 80], [90, 20]], np.int32)
        with patch.object(bri, "ROI", custom), patch.object(bri, "ROI_REFERENCE_SIZE", None):
            ctx = bri.context_from_args(bri.build_arg_parser().parse_args([]))
        np.testing.assert_array_equal(ctx.roi_polygon, custom)
        self.assertIsNone(ctx.roi_reference_size)  # native video resolution

    def test_configured_preset_roi_gets_its_reference_size(self):
        with patch.object(bri, "ROI", config.ROI_ALL["A2"]), patch.object(bri, "ROI_REFERENCE_SIZE", None):
            ctx = bri.context_from_args(bri.build_arg_parser().parse_args([]))
        np.testing.assert_array_equal(ctx.roi_polygon, config.ROI_ALL["A2"])
        self.assertEqual(ctx.roi_reference_size, (1280, 720))

    def test_explicit_reference_size_for_custom_roi(self):
        custom = np.array([[10, 20], [10, 80], [90, 80], [90, 20]], np.int32)
        with patch.object(bri, "ROI", custom), patch.object(bri, "ROI_REFERENCE_SIZE", (1280, 720)):
            ctx = bri.context_from_args(bri.build_arg_parser().parse_args([]))
        self.assertEqual(ctx.roi_reference_size, (1280, 720))

    def test_cli_preset_overrides_configured_roi(self):
        custom = np.array([[10, 20], [10, 80], [90, 80], [90, 20]], np.int32)
        with patch.object(bri, "ROI", custom):
            ctx = bri.context_from_args(bri.build_arg_parser().parse_args(["--roi-preset", "B5"]))
        np.testing.assert_array_equal(ctx.roi_polygon, config.ROI_ALL["B5"])
        self.assertEqual(ctx.roi_reference_size, (1920, 1080))


class GeometryTests(unittest.TestCase):
    def test_native_resolution_no_resize(self):
        geo = bri.build_analysis_geometry(1920, 1080, config.ROI_ALL["B1"], (1920, 1080), None)
        self.assertIsNone(geo.target_size)
        frame = np.zeros((1080, 1920, 3), np.uint8)
        self.assertIs(geo.prepare(frame), frame)  # no copy
        np.testing.assert_array_equal(geo.polygon, config.ROI_ALL["B1"])

    def test_never_upscales_or_copies_at_native_height(self):
        for target in (720, 1080):
            geo = bri.build_analysis_geometry(1280, 720, config.ROI_ALL["A2"], (1280, 720), target)
            self.assertIsNone(geo.target_size, target)

    def test_downscale_target_and_masks(self):
        geo = bri.build_analysis_geometry(1920, 1080, config.ROI_ALL["B1"], (1920, 1080), 360)
        self.assertEqual(geo.target_size, (640, 360))
        self.assertEqual(geo.mask_color.shape, (360, 640, 3))
        self.assertEqual(geo.mask_gray.shape, (360, 640))
        np.testing.assert_array_equal(geo.mask_gray, geo.mask_color[:, :, 0])

    def test_cached_masks_match_per_frame_masks(self):
        rng = np.random.default_rng(2)
        frame = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
        geo = bri.build_analysis_geometry(640, 360, config.ROI_ALL["A2"], (1280, 720), None)
        np.testing.assert_array_equal(
            bri.preprocess_frame_for_white(frame, geo.polygon, geo.mask_color),
            bri.preprocess_frame_for_white(frame, geo.polygon),
        )


class MidFileResolutionChangeTests(unittest.TestCase):
    def test_geometry_rebuilt_when_frame_size_changes(self):
        rng = np.random.default_rng(4)
        frames = [rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(5)]
        frames += [rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(5)]

        class FakeCapture:
            def __init__(self, *_):
                self.i = 0
            def isOpened(self):
                return True
            def read(self):
                if self.i >= len(frames):
                    return False, None
                self.i += 1
                return True, frames[self.i - 1]
            def release(self):
                pass

        t0 = datetime(2024, 4, 25, 12, 0, 0)
        clip = {
            "video_path": "fake.mp4", "video_file": "fake.mp4", "fps": 10.0,
            "actual_fps": 10.0, "frame_width": 1280, "frame_height": 720,
            "timestamps": [t0 + timedelta(seconds=i / 10) for i in range(10)],
            "timestamp_source": "filename", "camera_id": None, "stream_id": None,
            "overlay_id": None, "length": 10,
        }
        ctx = bri.RunContext(output_folder="unused", roi_polygon=config.ROI_ALL["A2"],
                             roi_reference_size=(1280, 720),
                             save_roi_preview=False, show_roi_preview=False)
        with patch.object(bri.cv2, "VideoCapture", FakeCapture), \
                patch("builtins.print") as printed:
            agg = bri.build_continuous_hour_signal(t0, [clip], ctx)

        self.assertEqual(len(agg["combined_diffs_raw"]), 10)
        warnings = [c.args[0] for c in printed.call_args_list if "rebuilding ROI geometry" in str(c.args)]
        self.assertEqual(len(warnings), 1)
        # first frame and the frame after the size change have no comparable previous ROI
        self.assertEqual(int((~agg["combined_valid"]).sum()), 2)


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


class ClipJoinTests(unittest.TestCase):
    def test_contiguity(self):
        t = datetime(2024, 4, 25, 12, 0, 0)
        self.assertTrue(bri.clips_are_contiguous(t, t + timedelta(seconds=0.04)))
        self.assertTrue(bri.clips_are_contiguous(t, t - timedelta(seconds=0.5)))  # overlap
        self.assertFalse(bri.clips_are_contiguous(t, t + timedelta(seconds=30)))
        self.assertFalse(bri.clips_are_contiguous(None, t))


if __name__ == "__main__":
    unittest.main()
