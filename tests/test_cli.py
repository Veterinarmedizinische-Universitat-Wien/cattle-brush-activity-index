"""Command-line options."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import brush_rotation_activity_index as bri
import config


class CliTests(unittest.TestCase):
    def parse(self, *argv):
        return bri.context_from_args(bri.build_arg_parser().parse_args(list(argv)))

    def test_processing_height_option(self):
        self.assertEqual(self.parse("--processing-height", "360").processing_height, 360)

    def test_native_resolution_by_default(self):
        self.assertEqual(self.parse().processing_height, config.PROCESSING_HEIGHT)

    def test_processing_height_kept_in_child_context(self):
        ctx = self.parse("--processing-height", "480")
        self.assertEqual(ctx.child("elsewhere").processing_height, 480)

    def test_every_replicate_has_a_preset(self):
        for name in [f"{farm}{i}" for farm in "AB" for i in range(1, 6)]:
            ctx = self.parse("--roi-preset", name)
            self.assertIn(name, config.ROI_REFERENCE_SIZES)
            self.assertEqual(ctx.roi_reference_size, config.ROI_REFERENCE_SIZES[name])


if __name__ == "__main__":
    unittest.main()
