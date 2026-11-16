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

EXPERIMENT_NAME = "brush_activity_index"

# ============================================================
# ROI PRESETS
# ============================================================
# Brush ROI polygons (x, y) of the ten evaluation recordings, as used for the
# published results. Farm A polygons were drawn on 1280 x 720 frames, Farm B
# polygons on 1920 x 1080 frames (see ROI_REFERENCE_SIZES). For other cameras,
# set ROI below to your own np.int32 polygon and check the saved ROI preview.

ROI_ALL = {
    # Farm A (Lely Luna), 25.04.2024
    "A1": np.array([[740, 510], [770, 620], [920, 540], [880, 450]], dtype=np.int32),   # Pen 1, brush A
    "A2": np.array([[380, 320], [450, 420], [580, 350], [530, 245]], dtype=np.int32),   # Pen 1, brush B, ####################  (Phaidra sample video 1)##########################
    "A3": np.array([[820, 190], [780, 290], [920, 360], [960, 250]], dtype=np.int32),   # Pen 2, brush A
    "A4": np.array([[590, 190], [530, 280], [670, 360], [720, 270]], dtype=np.int32),   # Pen 2, brush B
    "A5": np.array([[660, 300], [640, 420], [790, 440], [810, 350]], dtype=np.int32),   # Pen 3
    # Farm B (DeLaval swinging brush)
    "B1": np.array([[700, 60], [700, 180], [1210, 180], [1210, 60]], dtype=np.int32),   # brush A, 16.11.2024
    "B2": np.array([[700, 60], [700, 180], [1210, 180], [1210, 60]], dtype=np.int32),   # brush A, 01.02.2025
    "B3": np.array([[1210, 170], [1210, 50], [1600, 170], [1600, 290]], dtype=np.int32),  # brush B, 01.02.2025
    "B4": np.array([[700, 60], [700, 180], [1210, 180], [1210, 60]], dtype=np.int32),   # brush A, 17.05.2025
    "B5": np.array([[1210, 170], [1210, 50], [1600, 170], [1600, 290]], dtype=np.int32),  # brush B, 17.05.2025, ####################  (Phaidra sample video 2)##########################
}

# Frame size (width, height) on which each preset was drawn. Polygons are scaled
# from this size to the analysed frame.
ROI_REFERENCE_SIZES = {
    **{k: (1280, 720) for k in ("A1", "A2", "A3", "A4", "A5")},
    **{k: (1920, 1080) for k in ("B1", "B2", "B3", "B4", "B5")},
}

# Active ROI: a preset from ROI_ALL or a custom np.int32 polygon.
# The --roi-preset command-line option overrides this for a single run.
ROI = ROI_ALL["A1"]

# ============================================================
# SETTINGS
# ============================================================

FRAME_SKIP_ANNOTATION = 3
# Analysis resolution. None (default) analyses frames at the recording's native
# resolution, as for the main evaluation results. Set e.g. 720, 480 or 360 to
# resize frames inside the pipeline before motion extraction (in-pipeline
# downscaling, as in the resolution benchmark); validate detection first.
PROCESSING_HEIGHT = None
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

# Adaptive segment threshold (on the normalized signal)
SEGMENT_MINUTES = 10
SEGMENT_HIGH_MOTION_REF = 0.90
LOW_MOTION_MULTIPLIER = 0.62
HIGH_MOTION_MULTIPLIER = 0.50

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

# Post-processing filters (normalized signal units)
PIXEL_THRESHOLD_LOW = 0.26
PIXEL_THRESHOLD_HIGH = 0.40
CONTEXT_WINDOW_MINUTES = 4
CONTEXT_MAX_PIXEL = 0.45

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

