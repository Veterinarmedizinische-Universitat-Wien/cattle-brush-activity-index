"""Regression checks for per-farm timebase and ROI scaling."""
import sys
import unittest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from brush_rotation_activity_index import prepare_analysis_frame
from config import ROI_ALL, ROI_REFERENCE_SIZES

class ProcessingTests(unittest.TestCase):
    def test_hungary_roi(self):
        frame = np.zeros((720,1280,3),np.uint8)
        small, polygon = prepare_analysis_frame(frame,ROI_ALL["B"],ROI_REFERENCE_SIZES["B"])
        self.assertEqual(small.shape[:2],(360,640))
        np.testing.assert_array_equal(polygon, np.rint(ROI_ALL["B"] * 0.5).astype(np.int32))
    def test_krems_roi(self):
        frame = np.zeros((1080,1920,3),np.uint8)
        small, polygon = prepare_analysis_frame(frame,ROI_ALL["60"],ROI_REFERENCE_SIZES["60"])
        self.assertEqual(small.shape[:2],(360,640))
        np.testing.assert_array_equal(polygon, np.rint(ROI_ALL["60"] / 3).astype(np.int32))

if __name__ == "__main__":
    unittest.main()
