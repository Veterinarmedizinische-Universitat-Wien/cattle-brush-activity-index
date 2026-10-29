"""Configuration for the cattle-brush activity-index analysis.

Edit paths, ROI selection, output switches, and study parameters here.
The detection algorithm itself remains in ``brush_rotation_activity_index.py``.
"""

import os
from datetime import datetime, timezone

import numpy as np

# ============================================================
# EXPERIMENT LABEL
# ============================================================

EXPERIMENT_NAME = "brush_index_v6"

# ============================================================
# ROI PRESETS
# ============================================================

ROI_ALL = {
    #"A":  np.array([[740,500],[780,630],[940,540],[900,430]], dtype=np.int32),
    "A":  np.array([[740,510],[770,620],[920,540],[880,450]], dtype=np.int32),    # 27-07-2024
    #"B":  np.array([[380,320],[450,420],[580,350],[530,245]], dtype=np.int32),
    "B":  np.array([[435,270],[470,370],[640,310],[600,210]], dtype=np.int32),    # 27-07-2024
    #"C":  np.array([[590,190],[530,280],[670,360],[720,270]], dtype=np.int32),
    "C":  np.array([[600,130],[590,230],[750,240],[760,140]], dtype=np.int32),    # 27-07-2024
    #"D":  np.array([[820,190],[780,290],[920,360],[960,250]], dtype=np.int32),
    "D":  np.array([[900,110],[860,210],[1000,280],[1040,170]], dtype=np.int32),  # 27-07-2024
    "E":  np.array([[660,300],[640,420],[790,440],[810,350]], dtype=np.int32),
    "44": np.array([[700, 60],[700,180],[1210,180],[1210, 60]], dtype=np.int32),
    "60": np.array([[1210,170],[1210, 50],[1600,170],[1600,290]], dtype=np.int32),
    "rpi cam": np.array([[500,200],[500,230],[650,230],[650,200]], dtype=np.int32),
}
# Active ROI: a preset from ROI_ALL or a custom np.int32 polygon.
# The --roi-preset command-line option overrides this for a single run.
ROI = ROI_ALL["44"]

# ============================================================
# SETTINGS
# ============================================================

FRAME_SKIP_ANNOTATION = 3
PROCESSING_HEIGHT = 360  # Resize BEFORE motion extraction; None disables
# Frame size (width, height) in which each ROI preset was drawn. ROI polygons are
# scaled from this size to the processed frame size.
ROI_REFERENCE_SIZES = {"A": (1280,720), "B": (1280,720), "C": (1280,720), "D": (1280,720), "E": (1280,720), "44": (1920,1080), "60": (1920,1080), "rpi cam": (1280,720)}
# Reference size (width, height) for the active ROI.
# None = automatic: a preset uses its entry in ROI_REFERENCE_SIZES; a custom
# polygon is taken to be in each video's native resolution. Set a tuple only for
# a custom polygon drawn on a frame of a different resolution.
ROI_REFERENCE_SIZE = None
DOWNSCALE = 0.5
OUTPUT_CODEC = "mp4v"
OUTPUT_EXT = ".mp4"

WRITE_ANNOTATED_VIDEO = True
WRITE_HOUR_PLOTS = True
SHOW_ROI_PREVIEW = False
SAVE_ROI_PREVIEW = True

SUPPORTED_FORMATS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg")

# Image preprocessing
USE_CLAHE = True
CLAHE_ONLY_IF_DARK = True
CLAHE_DARK_MEAN_THRESHOLD = 80
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
BLEND_ALPHA = 0.6

# Smoothing
SMOOTH_WINDOW = 5

# Robust normalization
ROBUST_NORM_LOW_PCT = 5
ROBUST_NORM_HIGH_PCT = 95
USE_LOG_COMPRESSION = True

# Legacy-style adaptive segment logic (on normalized signal)
LEGACY_SEGMENT_MINUTES = 10
LEGACY_SEGMENT_MAX_SWITCH = 0.90
LEGACY_LOW_PCT = 0.62
LEGACY_HIGH_PCT = 0.50

# Robust threshold reference
THRESHOLD_REF_PERCENTILE = 99.5

# Spike suppression for threshold-only signal
USE_THRESHOLD_SPIKE_SUPPRESSION = True
THRESHOLD_SPIKE_MIN_SECONDS = 10.0
THRESHOLD_SPIKE_HIGH_Q = 99.0
THRESHOLD_SPIKE_ISOLATION_GAP_SECONDS = 5.0

# Clip joins within an hour bucket: the previous-frame state is carried across
# consecutive clips only when the gap between the last frame of one clip and the
# first frame of the next is at most this many seconds. Larger gaps reset it so
# that unrelated frames are not differenced.
CLIP_JOIN_MAX_GAP_SECONDS = 2.0

# Bout rules
MIN_BOUT_SECONDS = 4
MAX_BOUT_SECONDS = 2000

# Legacy-style post-filters in normalized units
LEGACY_PIXEL_THRESHOLD_LOW = 0.26
LEGACY_PIXEL_THRESHOLD_HIGH = 0.40
LEGACY_CONTEXT_WINDOW_MINUTES = 4
LEGACY_CONTEXT_MAX_PIXEL = 0.45

# Clustered short-bout rule (evaluate_window). It removes bouts when every bout
# starting within SHORT_BOUT_WINDOW_SECONDS is shorter than SHORT_BOUT_SECONDS.
# Detection already requires bouts to be longer than MIN_BOUT_SECONDS, so this
# rule can only remove anything when SHORT_BOUT_SECONDS > MIN_BOUT_SECONDS.
# With the study defaults (4 and 4) it removes nothing; validate_config() warns.
SHORT_BOUT_WINDOW_SECONDS = 40
SHORT_BOUT_SECONDS = 4

# Merge nearby bouts to reduce fragmentation
MERGE_CLOSE_BOUTS = True
MERGE_GAP_SECONDS = 2.0

# Timestamp policy
REQUIRE_DVR_TIMESTAMPS = False

# Standalone fallback for videos without DVR or recognised filename timestamps.
# Such videos are processed independently and are never mixed with the
# absolute-time continuous-hour workflow.
ALLOW_STANDALONE_RELATIVE_TIME = True
RELATIVE_TIME_ANCHOR = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Diagnostics
PRINT_HOUR_SIGNAL_STATS = True
PRINT_THRESHOLD_SIGNAL_STATS = True

# ============================================================
# PATHS
# ============================================================
# Defaults point at the repository's examples/ folder. Override them with the
# --input/--output command-line options or the CBAI_INPUT/CBAI_OUTPUT
# environment variables. Output folders are created only when a run starts.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_folder = os.environ.get("CBAI_INPUT", os.path.join(REPO_ROOT, "examples", "input"))
base_output_folder = os.environ.get("CBAI_OUTPUT", os.path.join(REPO_ROOT, "examples", "output"))


def validate_config():
    """Return warnings for parameter combinations that silently disable a rule."""
    warnings = []
    if SHORT_BOUT_SECONDS <= MIN_BOUT_SECONDS:
        warnings.append(
            f"SHORT_BOUT_SECONDS ({SHORT_BOUT_SECONDS}) <= MIN_BOUT_SECONDS ({MIN_BOUT_SECONDS}): "
            "the clustered short-bout rule (evaluate_window) cannot remove any bout."
        )
    return warnings
