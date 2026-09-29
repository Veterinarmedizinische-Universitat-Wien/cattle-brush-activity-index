# ============================================================
# CATTLE-BRUSH MECHANICAL-ROTATION ACTIVITY INDEX
# PUBLIC RESEARCH RELEASE
# ============================================================
# This script implements the cross-farm activity-index pipeline used for
# mechanical cattle-brush rotation detection. The detection logic and study
# parameter defaults are preserved from the supplied analysis script.
#
# Core sequence:
# 1) extract a continuous raw motion signal from a predefined brush ROI;
# 2) apply robust log/percentile normalization and temporal smoothing;
# 3) build a threshold-only signal with isolated short spikes suppressed;
# 4) estimate segment-specific thresholds using a high-percentile reference;
# 5) apply final detection to the original smoothed normalized signal; and
# 6) apply the study post-processing rules to obtain rotation intervals.
# ============================================================

import os
import re
import json
import argparse
import tempfile
import subprocess
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime, timedelta

# ============================================================
# EXPERIMENT LABEL
# ============================================================

EXPERIMENT_NAME = "legacy_continuous_hour_signal_cross_farm_normalized_v3_spike_robust_threshold"

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
DEFAULT_ROI_PRESET = "E"
ROI = ROI_ALL[DEFAULT_ROI_PRESET].copy()

# ============================================================
# SETTINGS
# ============================================================

FRAME_SKIP_ANNOTATION = 3
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

# NEW: robust threshold reference
THRESHOLD_REF_PERCENTILE = 99.5

# NEW: spike suppression for threshold-only signal
USE_THRESHOLD_SPIKE_SUPPRESSION = True
THRESHOLD_SPIKE_MIN_SECONDS = 10.0
THRESHOLD_SPIKE_HIGH_Q = 99.0
THRESHOLD_SPIKE_ISOLATION_GAP_SECONDS = 5.0

# Bout rules
MIN_BOUT_SECONDS = 4
MAX_BOUT_SECONDS = 2000

# Legacy-style post-filters in NORMALIZED units
LEGACY_PIXEL_THRESHOLD_LOW = 0.26
LEGACY_PIXEL_THRESHOLD_HIGH = 0.40
LEGACY_CONTEXT_WINDOW_MINUTES = 4
LEGACY_CONTEXT_MAX_PIXEL = 0.45

SHORT_BOUT_WINDOW_SECONDS = 40
SHORT_BOUT_SECONDS = 4

# Merge nearby bouts to reduce fragmentation
MERGE_CLOSE_BOUTS = True
MERGE_GAP_SECONDS = 2.0

# Timestamp policy
REQUIRE_DVR_TIMESTAMPS = False

# Diagnostics
PRINT_HOUR_SIGNAL_STATS = True
PRINT_THRESHOLD_SIGNAL_STATS = True

# ============================================================
# RUNTIME PATHS
# ============================================================
#
# Paths are configured from command-line arguments in configure_run().
# This removes machine-specific paths while leaving the analysis logic unchanged.

input_folder = None
base_output_folder = None
output_folder = None
annotated_video_folder = None
plot_folder = None
preview_folder = None

rotation_data = []
gap_report = []
hour_summary = []
file_timestamp_report = []

# ============================================================
# GENERAL UTILITIES
# ============================================================

def is_video_file(name: str) -> bool:
    return name.lower().endswith(SUPPORTED_FORMATS)

def safe_run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def check_ffmpeg_available() -> bool:
    try:
        result = safe_run(["ffmpeg", "-version"])
        return result.returncode == 0
    except Exception:
        return False

FFMPEG_AVAILABLE = check_ffmpeg_available()

def sanitize_filename(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", s)

def frame_to_time(frame, fps):
    total_seconds = int(frame / max(fps, 1e-6))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}m {seconds}s"

# ============================================================
# FILENAME PARSING
# ============================================================

def parse_timestamp_from_filename(filename: str):
    stem = os.path.splitext(os.path.basename(filename))[0]

    m = re.search(r"Event(\d{14})", stem)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        except Exception:
            pass

    candidates = []

    m = re.search(r"(\d{2}\.\d{2}\.\d{4})[-_ ](\d{6})", stem)
    if m:
        candidates.append((" ".join(m.groups()), "%d.%m.%Y %H%M%S"))

    for s in re.findall(r"(?<!\d)(\d{14})(?!\d)", stem):
        candidates.append((s, "%Y%m%d%H%M%S"))

    m = re.search(r"(\d{4}-\d{2}-\d{2})[-_ ](\d{2}-\d{2}-\d{2})", stem)
    if m:
        candidates.append((" ".join(m.groups()), "%Y-%m-%d %H-%M-%S"))

    m = re.search(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})", stem)
    if m:
        candidates.append(("".join(m.groups()), "%Y%m%d%H%M%S"))

    m = re.search(r"(\d{2}\.\d{2}\.\d{4})_(\d{6})", stem)
    if m:
        candidates.append((" ".join(m.groups()), "%d.%m.%Y %H%M%S"))

    for value, fmt in candidates:
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            pass

    return None

# ============================================================
# FFPROBE HELPERS
# ============================================================

def ffprobe_json(video_path):
    if not FFMPEG_AVAILABLE:
        return None

    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    result = safe_run(cmd)
    if result.returncode != 0:
        return None

    try:
        return json.loads(result.stdout)
    except Exception:
        return None

# ============================================================
# DVR METADATA EXTRACTION
# ============================================================

def _build_datetime(date_obj, hour_val, minute_val, second_val, micro_val):
    return datetime(
        year=date_obj.year,
        month=date_obj.month,
        day=date_obj.day,
        hour=hour_val,
        minute=minute_val,
        second=second_val,
        microsecond=micro_val
    )

def reconstruct_dvr_timestamps(date_obj, time_fragments, fps):
    timestamps = []

    if not time_fragments:
        return timestamps

    frame_interval = timedelta(seconds=1.0 / fps) if fps > 0 else timedelta(seconds=1 / 30)

    for frag in time_fragments:
        parts = frag.split(":")
        if len(parts) != 3:
            continue

        try:
            h = int(parts[0])
            m = int(parts[1])
            sec_float = float(parts[2])

            sec = int(sec_float)
            micros = int(round((sec_float - sec) * 1_000_000))

            if micros >= 1_000_000:
                sec += 1
                micros -= 1_000_000

            ts = _build_datetime(date_obj, h, m, sec, micros)

            if timestamps and ts < timestamps[-1]:
                ts = timestamps[-1] + frame_interval

            timestamps.append(ts)

        except Exception:
            continue

    return timestamps

def extract_dvr_metadata_from_stream(video_path, stream_index, fps):
    if not FFMPEG_AVAILABLE:
        return None

    tmp = tempfile.NamedTemporaryFile(delete=False).name
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-map", f"0:{stream_index}",
            "-c", "copy",
            "-f", "data",
            tmp
        ]
        result = safe_run(cmd)

        if result.returncode != 0:
            return None
        if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            return None

        with open(tmp, "rb") as f:
            data = f.read().decode(errors="ignore")

        id_match = re.search(r"#(\d+)-(\d+):\d+:\d+(?:\.\d+)?", data)
        camera_id = int(id_match.group(1)) if id_match else None
        overlay_id = None

        date_match = re.search(r"\((\d{4}/\d+/\d+)\)", data)
        if not date_match:
            return None

        date_obj = datetime.strptime(date_match.group(1), "%Y/%m/%d").date()
        time_fragments = re.findall(r"#\d+-((?:\d+:){2}\d+\.\d+)", data)
        if not time_fragments:
            return None

        timestamps = reconstruct_dvr_timestamps(date_obj, time_fragments, fps)
        if len(timestamps) == 0:
            return None

        return {
            "camera_id": camera_id,
            "overlay_id": overlay_id,
            "timestamps": timestamps
        }

    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

def extract_dvr_metadata(video_path, fps):
    probe = ffprobe_json(video_path)
    if probe is None:
        return None

    streams = probe.get("streams", [])
    candidate_indices = []

    for s in streams:
        idx = s.get("index")
        ctype = s.get("codec_type", "")
        if idx is None:
            continue
        if ctype != "video":
            candidate_indices.append(idx)

    if not candidate_indices:
        candidate_indices = [s.get("index") for s in streams if s.get("index") not in (None, 0)]

    for idx in candidate_indices:
        meta = extract_dvr_metadata_from_stream(video_path, idx, fps)
        if meta is not None:
            return meta

    return None

# ============================================================
# TIMESTAMP DECISION
# ============================================================

def build_timestamps_from_start(start_dt, fps, total_frames):
    if start_dt is None or total_frames <= 0:
        return []
    if fps <= 0:
        fps = 30.0
    return [start_dt + timedelta(seconds=i / fps) for i in range(total_frames)]

def get_video_timestamps_and_ids(video_path, fps, total_frames):
    meta = extract_dvr_metadata(video_path, fps)

    if meta and meta.get("timestamps"):
        timestamps = meta["timestamps"]
        print(f"Using DVR metadata timestamps: {os.path.basename(video_path)}")

        if len(timestamps) >= total_frames:
            timestamps = timestamps[:total_frames]
        else:
            print(
                f"Warning: DVR timestamps shorter than frame count "
                f"({len(timestamps)} < {total_frames}) in {os.path.basename(video_path)}"
            )
            if len(timestamps) >= 2:
                step = timestamps[-1] - timestamps[-2]
                if step.total_seconds() <= 0:
                    step = timedelta(seconds=1 / max(fps, 1e-6))
            else:
                step = timedelta(seconds=1 / max(fps, 1e-6))

            extended = list(timestamps)
            while len(extended) < total_frames:
                extended.append(extended[-1] + step)
            timestamps = extended

        return {
            "timestamps": timestamps,
            "camera_id": meta.get("camera_id"),
            "stream_id": None,
            "overlay_id": meta.get("overlay_id"),
            "timestamp_source": "dvr"
        }

    fname = parse_timestamp_from_filename(os.path.basename(video_path))
    if fname:
        print(f"Using filename timestamps: {os.path.basename(video_path)}")
        return {
            "timestamps": build_timestamps_from_start(fname, fps, total_frames),
            "camera_id": None,
            "stream_id": None,
            "overlay_id": None,
            "timestamp_source": "filename"
        }

    print(f"No DVR metadata found and filename timestamp missing: {os.path.basename(video_path)}")
    return None

def get_video_basic_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if length <= 0 or frame_width <= 0 or frame_height <= 0:
        return None

    return {
        "fps": float(fps),
        "length": int(length),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
    }

def get_clip_start_info(video_path):
    basic = get_video_basic_info(video_path)
    if basic is None:
        return None

    ts_info = get_video_timestamps_and_ids(video_path, basic["fps"], basic["length"])
    if ts_info is None or len(ts_info["timestamps"]) == 0:
        return None

    if REQUIRE_DVR_TIMESTAMPS and ts_info["timestamp_source"] != "dvr":
        print(f"Skipping non-DVR timestamp file: {os.path.basename(video_path)}")
        return None

    return {
        "video_path": video_path,
        "video_file": os.path.basename(video_path),
        "fps": basic["fps"],
        "length": basic["length"],
        "frame_width": basic["frame_width"],
        "frame_height": basic["frame_height"],
        "start_timestamp": ts_info["timestamps"][0],
        "timestamps": ts_info["timestamps"],
        "timestamp_source": ts_info["timestamp_source"],
        "camera_id": ts_info["camera_id"],
        "stream_id": ts_info["stream_id"],
        "overlay_id": ts_info["overlay_id"],
    }

def build_hour_buckets(folder: str):
    buckets = defaultdict(list)

    for file in sorted(os.listdir(folder)):
        if not is_video_file(file):
            continue

        video_path = os.path.join(folder, file)
        start_info = get_clip_start_info(video_path)
        if start_info is None:
            print(f"Skipping (no valid timestamp): {file}")
            continue

        ts = start_info["start_timestamp"]
        hour_key = ts.replace(minute=0, second=0, microsecond=0)
        buckets[hour_key].append(start_info)

        file_timestamp_report.append({
            "Video File": file,
            "Start Timestamp": ts.strftime("%d.%m.%Y %H:%M:%S.%f")[:-3],
            "Timestamp Source": start_info["timestamp_source"],
            "FPS": start_info["fps"],
            "Frame Count": start_info["length"],
        })

    for hour_key in buckets:
        buckets[hour_key].sort(key=lambda x: x["start_timestamp"])

    return buckets

# ============================================================
# DIAGNOSTICS
# ============================================================

def diagnose_timestamps(timestamps, fps):
    if len(timestamps) < 2:
        return {
            "container_fps": fps,
            "estimated_fps": None,
            "dropped_frames": 0,
            "duplicate_frames": 0,
            "timing_std": 0.0,
            "median_interval": None,
        }

    times = np.array([(t - timestamps[0]).total_seconds() for t in timestamps], dtype=float)
    dt = np.diff(times)

    median = float(np.median(dt))
    estimated_fps = None if median <= 0 else 1.0 / median
    dropped = int(np.sum(dt > median * 1.5))
    duplicates = int(np.sum(dt < median * 0.2))
    timing_std = float(np.std(dt))

    return {
        "container_fps": fps,
        "estimated_fps": estimated_fps,
        "dropped_frames": dropped,
        "duplicate_frames": duplicates,
        "timing_std": timing_std,
        "median_interval": median,
    }

def detect_timestamp_gaps(video_name, timestamps):
    if len(timestamps) < 2:
        return

    times = np.array([(t - timestamps[0]).total_seconds() for t in timestamps], dtype=float)
    dt = np.diff(times)
    median_interval = float(np.median(dt))

    if median_interval <= 0:
        return

    for i, gap in enumerate(dt):
        gap_type = None

        if gap > 60:
            gap_type = "Clock Jump"
        elif gap > median_interval * 1.5:
            gap_type = "Dropped Frames"
        elif gap < median_interval * 0.2:
            gap_type = "Duplicate Frame"
        elif abs(gap - median_interval) > median_interval * 0.5:
            gap_type = "Timing Jitter"

        if gap_type:
            gap_report.append({
                "Video File": video_name,
                "Frame Index": i,
                "Timestamp Before": timestamps[i].strftime("%H:%M:%S.%f"),
                "Timestamp After": timestamps[i + 1].strftime("%H:%M:%S.%f"),
                "Delta Seconds": float(gap),
                "Expected Interval": median_interval,
                "Gap Type": gap_type
            })

# ============================================================
# ROI / PREVIEW
# ============================================================

def display_first_frame_with_roi(video_path, roi_polygon, save_path=None, show=False):
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Could not open video for ROI preview: {video_path}")
        return

    ret, frame = vidcap.read()
    vidcap.release()

    if not ret:
        print(f"Error: Unable to read first frame of {os.path.basename(video_path)}")
        return

    frame_with_roi = cv2.polylines(frame.copy(), [roi_polygon], True, (0, 255, 0), 2)
    frame_with_roi = cv2.cvtColor(frame_with_roi, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(frame_with_roi)
    plt.title(f"First Frame with ROI - {os.path.basename(video_path)}")
    plt.axis("off")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close()

# ============================================================
# SIGNAL HELPERS
# ============================================================

def moving_average_with_tail(data, window_size):
    if len(data) == 0:
        return np.array([], dtype=np.float32)

    data = np.asarray(data, dtype=np.float32)
    window_size = max(1, min(window_size, len(data)))

    smoothed = np.convolve(
        data,
        np.ones(window_size, dtype=np.float32) / window_size,
        mode="valid"
    )

    tail_start = len(data) - (window_size - 1)
    if tail_start < len(data):
        tail_averages = [np.mean(data[i:]) for i in range(tail_start, len(data))]
        smoothed = np.concatenate((smoothed, np.asarray(tail_averages, dtype=np.float32)))

    return smoothed.astype(np.float32)

def robust_log_normalize(signal, low_pct=5, high_pct=95, use_log=True):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return np.array([], dtype=np.float32), {
            "transform": "none",
            "low_ref": None,
            "high_ref": None,
            "max_after_norm": None
        }

    x = signal.copy()
    if use_log:
        x = np.log1p(np.maximum(x, 0))
        transform = "log1p"
    else:
        transform = "linear"

    lo = float(np.percentile(x, low_pct))
    hi = float(np.percentile(x, high_pct))

    if hi - lo < 1e-8:
        norm = np.zeros_like(x, dtype=np.float32)
        return norm, {
            "transform": transform,
            "low_ref": lo,
            "high_ref": hi,
            "max_after_norm": 0.0
        }

    norm = (x - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, None).astype(np.float32)

    return norm, {
        "transform": transform,
        "low_ref": lo,
        "high_ref": hi,
        "max_after_norm": float(np.max(norm))
    }

def print_signal_stats(label, signal):
    if signal is None or len(signal) == 0:
        print(f"{label}: empty")
        return

    arr = np.asarray(signal, dtype=np.float32)
    print(
        f"{label}: min={np.min(arr):.4f}, "
        f"p5={np.percentile(arr, 5):.4f}, "
        f"p25={np.percentile(arr, 25):.4f}, "
        f"p50={np.percentile(arr, 50):.4f}, "
        f"p75={np.percentile(arr, 75):.4f}, "
        f"p95={np.percentile(arr, 95):.4f}, "
        f"max={np.max(arr):.4f}"
    )

# ============================================================
# FRAME PREPROCESSING / RAW SIGNAL
# ============================================================

def enhance_low_light(frame):
    if not USE_CLAHE:
        return frame

    if CLAHE_ONLY_IF_DARK and np.mean(frame) >= CLAHE_DARK_MEAN_THRESHOLD:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l = clahe.apply(l)

    enhanced_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def get_roi_from_polygon_legacy(frame, polygon):
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return cv2.bitwise_and(frame, mask)

def preprocess_frame_for_white_legacy(frame, polygon):
    roi = get_roi_from_polygon_legacy(frame, polygon)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    normalized_l = cv2.normalize(l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    _, thresholded_l = cv2.threshold(normalized_l, 220, 255, cv2.THRESH_BINARY_INV)
    return cv2.bitwise_and(roi, roi, mask=thresholded_l)

def blend_with_previous(current_roi, prev_roi, alpha=0.6):
    if prev_roi is None:
        return current_roi
    return cv2.addWeighted(current_roi, alpha, prev_roi, 1 - alpha, 0)

def compute_motion_score_sum_abs_diff(prev_roi, roi):
    frame_diff = cv2.absdiff(prev_roi, roi)
    return float(np.sum(frame_diff))

# ============================================================
# ROBUST THRESHOLD HELPERS
# ============================================================

def suppress_isolated_short_spikes(
    signal,
    fps,
    min_spike_seconds=10.0,
    high_q=99.0,
    isolation_gap_seconds=5.0
):
    """
    Suppress isolated short high-amplitude spikes for THRESHOLD ESTIMATION ONLY.

    Strategy:
    - find runs above a high percentile level
    - suppress only runs shorter than min_spike_seconds
    - do NOT suppress if they are close to another high run
    - replace with local median context, not zero

    This keeps sustained strong activity available for thresholding while
    preventing brief artifacts from inflating segment threshold.
    """
    x = np.asarray(signal, dtype=np.float32).copy()
    n = len(x)

    if n == 0:
        return x

    level = float(np.percentile(x, high_q))
    above = x >= level

    min_len = max(1, int(round(min_spike_seconds * max(1e-6, fps))))
    isolation_gap = max(1, int(round(isolation_gap_seconds * max(1e-6, fps))))

    runs = []
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        start = i
        while i < n and above[i]:
            i += 1
        end = i - 1
        runs.append((start, end))

    if not runs:
        return x

    for idx, (start, end) in enumerate(runs):
        run_len = end - start + 1

        prev_close = False
        next_close = False

        if idx > 0:
            prev_start, prev_end = runs[idx - 1]
            prev_close = (start - prev_end - 1) <= isolation_gap

        if idx < len(runs) - 1:
            next_start, next_end = runs[idx + 1]
            next_close = (next_start - end - 1) <= isolation_gap

        is_isolated = (not prev_close) and (not next_close)

        if run_len < min_len and is_isolated:
            left = max(0, start - min_len)
            right = min(n, end + 1 + min_len)

            context_left = x[left:start]
            context_right = x[end + 1:right]
            context = np.concatenate([context_left, context_right])

            if context.size > 0:
                replacement = float(np.median(context))
            else:
                replacement = float(np.median(x))

            x[start:end + 1] = replacement

    return x

# ============================================================
# LEGACY-STYLE DETECTOR HELPERS (NORMALIZED SIGNAL)
# ============================================================

def compute_segment_threshold_curve(
    smoothed_diffs,
    fps,
    segment_minutes=10,
    high_threshold=0.90,
    low_pct=0.62,
    high_pct=0.50,
    ref_percentile=99.5,
):
    """
    Robust segment threshold:
      - compute segment reference using a high percentile (not max)
      - if segment reference is 'strong' use high_pct, else low_pct
      - threshold = pct * seg_ref

    This makes the threshold robust to short outlier spikes.
    """
    signal = np.asarray(smoothed_diffs, dtype=np.float32)
    n = len(signal)
    if n == 0:
        return np.array([], dtype=np.float32)

    seg_frames = max(1, int(segment_minutes * 60 * max(1e-6, fps)))
    threshold_curve = np.zeros(n, dtype=np.float32)

    for start in range(0, n, seg_frames):
        end = min(start + seg_frames, n)
        seg = signal[start:end]

        if seg.size == 0:
            seg_ref = 0.0
        else:
            seg_ref = float(np.percentile(seg, ref_percentile))

        if seg_ref <= 0:
            seg_thr = 0.0
        else:
            pct = high_pct if seg_ref >= high_threshold else low_pct
            seg_thr = pct * seg_ref

        threshold_curve[start:end] = seg_thr

    return threshold_curve

def detect_periods_from_threshold_frame_based(signal, threshold_curve, fps, min_duration_seconds):
    periods = []
    in_event = False
    start_idx = None

    for i, value in enumerate(signal):
        thr = threshold_curve[i]
        if value > thr:
            if not in_event:
                in_event = True
                start_idx = i
        else:
            if in_event:
                in_event = False
                end_idx = i - 1
                duration = (end_idx - start_idx + 1) / max(1e-6, fps)
                if duration > min_duration_seconds:
                    periods.append((start_idx, end_idx))

    if in_event and start_idx is not None:
        end_idx = len(signal) - 1
        duration = (end_idx - start_idx + 1) / max(1e-6, fps)
        if duration > min_duration_seconds:
            periods.append((start_idx, end_idx))

    return periods

def merge_close_periods(rotation_periods, fps, max_gap_seconds=2.0):
    if not rotation_periods:
        return []

    max_gap_frames = int(max_gap_seconds * max(1e-6, fps))
    merged = [rotation_periods[0]]

    for start, end in rotation_periods[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 <= max_gap_frames:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged

def filter_low_movement_bouts(rotation_periods, smoothed_diffs, pixel_threshold=0.26):
    filtered_periods = []
    for start, end in rotation_periods:
        bout_diffs = smoothed_diffs[start:end + 1]
        if len(bout_diffs) == 0:
            continue
        if float(np.max(bout_diffs)) >= pixel_threshold:
            filtered_periods.append((start, end))
    return filtered_periods

def filter_contextual_false_positives_with_edges(
    rotation_periods,
    smoothed_diffs,
    pixel_threshold_low=0.26,
    pixel_threshold_high=0.40
):
    filtered_periods = []
    num_periods = len(rotation_periods)

    for i, (start, end) in enumerate(rotation_periods):
        bout_diffs = smoothed_diffs[start:end + 1]
        if len(bout_diffs) == 0:
            continue

        max_diff = float(np.max(bout_diffs))

        if pixel_threshold_low <= max_diff <= pixel_threshold_high:
            prev_valid = False
            next_valid = False

            if i > 0:
                prev_start, prev_end = rotation_periods[i - 1]
                prev_max_diff = float(np.max(smoothed_diffs[prev_start:prev_end + 1]))
                prev_valid = prev_max_diff > pixel_threshold_high

            if i < num_periods - 1:
                next_start, next_end = rotation_periods[i + 1]
                next_max_diff = float(np.max(smoothed_diffs[next_start:next_end + 1]))
                next_valid = next_max_diff > pixel_threshold_high

            if i == 0:
                if not next_valid:
                    continue
            elif i == num_periods - 1:
                if not prev_valid:
                    continue
            else:
                if not prev_valid and not next_valid:
                    continue

        filtered_periods.append((start, end))

    return filtered_periods

def is_within_threshold(bout_durations, threshold=4):
    return all(duration < threshold for duration in bout_durations)

def evaluate_window(rotation_periods, fps, window_seconds=40, threshold=4):
    filtered_bouts = []
    window_frames = int(window_seconds * max(1e-6, fps))

    for i, (start, end) in enumerate(rotation_periods):
        window_bouts = []
        for j in range(i, len(rotation_periods)):
            other_start, other_end = rotation_periods[j]
            if other_start <= start + window_frames:
                window_bouts.append((other_start, other_end))
            else:
                break

        bout_durations = [(e - s + 1) / max(1e-6, fps) for s, e in window_bouts]

        if is_within_threshold(bout_durations, threshold=threshold):
            surrounding_bouts = []

            if i > 0:
                ps, pe = rotation_periods[i - 1]
                if (pe - ps + 1) / max(1e-6, fps) <= 5:
                    surrounding_bouts.append((ps, pe))

            if i + 1 < len(rotation_periods):
                ns, ne = rotation_periods[i + 1]
                if (ne - ns + 1) / max(1e-6, fps) <= 5:
                    surrounding_bouts.append((ns, ne))

            filtered_bouts.extend(window_bouts + surrounding_bouts)

    filtered_set = set(filtered_bouts)
    return [bout for bout in rotation_periods if bout not in filtered_set]

def filter_low_movement_within_window(
    rotation_periods,
    smoothed_diffs,
    fps,
    window_minutes=4,
    max_pixel_threshold=0.45
):
    filtered_periods = []
    window_frames = int(window_minutes * 60 * max(1e-6, fps))

    for start, end in rotation_periods:
        window_start = max(0, start - window_frames)
        window_end = min(len(smoothed_diffs), end + window_frames)
        max_pixel_movement = float(np.max(smoothed_diffs[window_start:window_end]))
        if max_pixel_movement > max_pixel_threshold:
            filtered_periods.append((start, end))
    return filtered_periods

# ============================================================
# CONTINUOUS HOUR SIGNAL EXTRACTION
# ============================================================

def build_continuous_hour_signal(hour_key, clip_infos, roi_polygon):
    """
    Builds one continuous RAW motion signal for the whole hour, preserving
    prev_roi across clip boundaries exactly like a truly concatenated video.
    """
    combined_diffs = []
    combined_timestamps = []
    combined_video_index = []
    combined_local_frame = []
    clip_results = []

    prev_roi = None

    for clip_idx, clip_info in enumerate(clip_infos):
        video_path = clip_info["video_path"]
        video_file = clip_info["video_file"]
        fps = clip_info["fps"]
        frame_width = clip_info["frame_width"]
        frame_height = clip_info["frame_height"]
        timestamps = clip_info["timestamps"]
        timestamp_source = clip_info["timestamp_source"]

        diagnostics = diagnose_timestamps(timestamps, fps)
        detect_timestamp_gaps(video_file, timestamps)

        if SAVE_ROI_PREVIEW or SHOW_ROI_PREVIEW:
            preview_path = os.path.join(
                preview_folder,
                os.path.splitext(video_file)[0] + "_roi_preview.png"
            )
            display_first_frame_with_roi(
                video_path,
                roi_polygon,
                save_path=preview_path if SAVE_ROI_PREVIEW else None,
                show=SHOW_ROI_PREVIEW
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        clip_frame_diffs = []
        actual_timestamps = []
        local_idx_map = []

        total_frames_target = len(timestamps)
        pbar = tqdm(total=total_frames_target, desc=f"Continuous signal [{video_file}]")
        local_frame_idx = 0

        while local_frame_idx < total_frames_target:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                enhanced_frame = enhance_low_light(frame)
                preprocessed_frame = preprocess_frame_for_white_legacy(enhanced_frame, roi_polygon)
                gray_frame = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2GRAY)
                roi = get_roi_from_polygon_legacy(gray_frame, roi_polygon)
                roi = blend_with_previous(roi, prev_roi, alpha=BLEND_ALPHA)

                if prev_roi is not None:
                    movement_score = compute_motion_score_sum_abs_diff(prev_roi, roi)
                    clip_frame_diffs.append(movement_score)
                else:
                    clip_frame_diffs.append(0.0)

                prev_roi = roi

                actual_timestamps.append(timestamps[local_frame_idx])
                local_idx_map.append(local_frame_idx)

            except Exception as e:
                print(f"Error processing frame {local_frame_idx} in {video_file}: {e}")
                clip_frame_diffs.append(0.0)
                actual_timestamps.append(timestamps[local_frame_idx])
                local_idx_map.append(local_frame_idx)

            local_frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        if len(clip_frame_diffs) == 0:
            print(f"No usable frames in {video_file}")
            continue

        clip_result = {
            "clip_idx": clip_idx,
            "video_path": video_path,
            "video_file": video_file,
            "fps": fps,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "timestamps": actual_timestamps,
            "frame_diffs_raw": np.asarray(clip_frame_diffs, dtype=np.float32),
            "camera_id": clip_info["camera_id"],
            "stream_id": clip_info["stream_id"],
            "overlay_id": clip_info["overlay_id"],
            "timestamp_source": timestamp_source,
            "diagnostics": diagnostics,
            "roi_polygon": roi_polygon.copy(),
        }

        global_start_idx = len(combined_diffs)

        combined_diffs.extend(clip_frame_diffs)
        combined_timestamps.extend(actual_timestamps)
        combined_video_index.extend([clip_idx] * len(clip_frame_diffs))
        combined_local_frame.extend(local_idx_map)

        global_end_idx = len(combined_diffs) - 1

        clip_result["global_start_idx"] = global_start_idx
        clip_result["global_end_idx"] = global_end_idx

        clip_results.append(clip_result)

    return {
        "hour_key": hour_key,
        "clip_results": clip_results,
        "combined_diffs_raw": np.asarray(combined_diffs, dtype=np.float32),
        "combined_timestamps": combined_timestamps,
        "combined_video_index": np.asarray(combined_video_index, dtype=int) if combined_video_index else np.array([], dtype=int),
        "combined_local_frame": np.asarray(combined_local_frame, dtype=int) if combined_local_frame else np.array([], dtype=int),
    }

# ============================================================
# RUN FULL LEGACY-STYLE DETECTOR ON NORMALIZED HOUR SIGNAL
# ============================================================

def detect_rotation_periods_for_hour_legacy(combined_diffs_raw, fps_for_filters):
    # 1) normalize original signal
    normalized_diffs, norm_info = robust_log_normalize(
        combined_diffs_raw,
        low_pct=ROBUST_NORM_LOW_PCT,
        high_pct=ROBUST_NORM_HIGH_PCT,
        use_log=USE_LOG_COMPRESSION
    )

    # 2) smooth original signal
    smoothed_diffs = moving_average_with_tail(normalized_diffs, SMOOTH_WINDOW)

    if len(smoothed_diffs) == 0:
        return [], smoothed_diffs, np.array([], dtype=np.float32), normalized_diffs, norm_info, np.array([], dtype=np.float32)

    # 3) build threshold-only signal
    if USE_THRESHOLD_SPIKE_SUPPRESSION:
        threshold_signal = suppress_isolated_short_spikes(
            smoothed_diffs,
            fps=fps_for_filters,
            min_spike_seconds=THRESHOLD_SPIKE_MIN_SECONDS,
            high_q=THRESHOLD_SPIKE_HIGH_Q,
            isolation_gap_seconds=THRESHOLD_SPIKE_ISOLATION_GAP_SECONDS
        )
    else:
        threshold_signal = smoothed_diffs.copy()

    # 4) compute robust threshold from threshold-only signal
    threshold_curve = compute_segment_threshold_curve(
        smoothed_diffs=threshold_signal,
        fps=fps_for_filters,
        segment_minutes=LEGACY_SEGMENT_MINUTES,
        high_threshold=LEGACY_SEGMENT_MAX_SWITCH,
        low_pct=LEGACY_LOW_PCT,
        high_pct=LEGACY_HIGH_PCT,
        ref_percentile=THRESHOLD_REF_PERCENTILE
    )

    # 5) detect on original smoothed signal
    rotation_periods = detect_periods_from_threshold_frame_based(
        signal=smoothed_diffs,
        threshold_curve=threshold_curve,
        fps=fps_for_filters,
        min_duration_seconds=MIN_BOUT_SECONDS
    )

    # 6) legacy-style postprocessing unchanged
    if MERGE_CLOSE_BOUTS:
        rotation_periods = merge_close_periods(
            rotation_periods,
            fps=fps_for_filters,
            max_gap_seconds=MERGE_GAP_SECONDS
        )

    rotation_periods = filter_low_movement_bouts(
        rotation_periods,
        smoothed_diffs,
        pixel_threshold=LEGACY_PIXEL_THRESHOLD_LOW
    )

    rotation_periods = filter_contextual_false_positives_with_edges(
        rotation_periods,
        smoothed_diffs,
        pixel_threshold_low=LEGACY_PIXEL_THRESHOLD_LOW,
        pixel_threshold_high=LEGACY_PIXEL_THRESHOLD_HIGH
    )

    rotation_periods = evaluate_window(
        rotation_periods,
        fps_for_filters,
        window_seconds=SHORT_BOUT_WINDOW_SECONDS,
        threshold=SHORT_BOUT_SECONDS
    )

    rotation_periods = filter_low_movement_within_window(
        rotation_periods,
        smoothed_diffs,
        fps_for_filters,
        window_minutes=LEGACY_CONTEXT_WINDOW_MINUTES,
        max_pixel_threshold=LEGACY_CONTEXT_MAX_PIXEL
    )

    filtered_by_duration = []
    for s, e in rotation_periods:
        duration = (e - s + 1) / max(1e-6, fps_for_filters)
        if duration <= MAX_BOUT_SECONDS:
            filtered_by_duration.append((s, e))

    return filtered_by_duration, smoothed_diffs, threshold_curve, normalized_diffs, norm_info, threshold_signal

# ============================================================
# PLOT
# ============================================================

def save_hour_plot_legacy(
    hour_key,
    combined_timestamps,
    smoothed_diffs,
    threshold_curve,
    rotation_periods,
    out_path,
    threshold_signal=None
):
    if len(smoothed_diffs) == 0 or len(combined_timestamps) == 0:
        return

    n = min(len(smoothed_diffs), len(combined_timestamps))
    x = np.arange(n)

    plt.figure(figsize=(14, 6))
    plt.plot(x, smoothed_diffs[:n], color="red", label="Farm-Robust Motion Signal")

    if threshold_signal is not None and len(threshold_signal) == n:
        plt.plot(
            x,
            threshold_signal[:n],
            color="orange",
            alpha=0.45,
            linewidth=1.0,
            label="Threshold-Only Signal"
        )

    plt.plot(x, threshold_curve[:n], color="blue", linestyle="--", label="Robust Segment Threshold")

    num_ticks = min(12, n)
    tick_idx = np.linspace(0, n - 1, num_ticks).astype(int)
    tick_labels = [combined_timestamps[i].strftime("%H:%M:%S") for i in tick_idx]
    plt.xticks(tick_idx, tick_labels, rotation=45)

    for idx, (start, end) in enumerate(rotation_periods):
        if start < n and end < n:
            plt.axvspan(
                start,
                end,
                color="lightblue",
                alpha=0.30,
                label="Rotation Period" if idx == 0 else ""
            )

    plt.xlabel("Real Time")
    plt.ylabel("Normalized motion signal")
    plt.title(f"Hourly movement curve - {hour_key.strftime('%Y-%m-%d %H:00')}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ============================================================
# ANNOTATION HELPERS
# ============================================================

def build_rotation_mask(length, rotation_periods):
    mask = np.zeros(length, dtype=bool)
    for s, e in rotation_periods:
        s = max(0, s)
        e = min(length - 1, e)
        if s <= e:
            mask[s:e + 1] = True
    return mask

def build_cumulative_rotation_seconds_global(length, rotation_periods, fps):
    cumulative = np.zeros(length, dtype=np.float32)
    if length <= 0:
        return cumulative

    active = np.zeros(length + 1, dtype=np.int32)
    for s, e in rotation_periods:
        s = max(0, min(length - 1, s))
        e = max(0, min(length - 1, e))
        if s <= e:
            active[s] += 1
            active[e + 1] -= 1

    active = np.cumsum(active[:-1]) > 0
    increment = 1.0 / max(1e-6, fps)

    total = 0.0
    for i in range(length):
        if active[i]:
            total += increment
        cumulative[i] = total

    return cumulative

def compute_segment_percentage_for_frame(smoothed_diffs, threshold_curve, fps, idx):
    seg_frames = max(1, int(LEGACY_SEGMENT_MINUTES * 60 * max(1e-6, fps)))
    seg_start = (idx // seg_frames) * seg_frames
    seg_end = min(seg_start + seg_frames, len(smoothed_diffs))

    if seg_end <= seg_start:
        return 0

    seg_ref = float(np.percentile(smoothed_diffs[seg_start:seg_end], THRESHOLD_REF_PERCENTILE))
    thr = float(threshold_curve[idx])

    if seg_ref <= 0:
        return 0

    return int(round(100.0 * thr / seg_ref))

def write_annotated_video_for_clip_legacy(
    vr,
    out_path,
    hour_smoothed_diffs,
    hour_threshold_curve,
    hour_rotation_mask,
    hour_cumulative_rotation_seconds,
    fps_for_filters
):
    video_path = vr["video_path"]
    fps = vr["fps"]
    frame_width = vr["frame_width"]
    frame_height = vr["frame_height"]
    timestamps = vr["timestamps"]
    roi_polygon = vr["roi_polygon"]
    global_start = vr["global_start_idx"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not reopen video for annotation: {video_path}")
        return

    new_w = max(2, int(frame_width * DOWNSCALE))
    new_h = max(2, int(frame_height * DOWNSCALE))
    out_fps = max(1.0, fps / max(1, FRAME_SKIP_ANNOTATION))
    roi_polygon_scaled = (roi_polygon.astype(np.float32) * DOWNSCALE).astype(np.int32)

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*OUTPUT_CODEC),
        out_fps,
        (new_w, new_h)
    )

    total_frames = len(timestamps)
    pbar = tqdm(total=total_frames, desc=f"Annotating [{os.path.basename(video_path)}]")

    frame_idx = 0
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        global_idx = global_start + frame_idx

        if frame_idx % FRAME_SKIP_ANNOTATION != 0:
            frame_idx += 1
            pbar.update(1)
            continue

        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        frame = cv2.polylines(frame, [roi_polygon_scaled], True, (0, 255, 0), 2)

        sm_idx = global_idx - SMOOTH_WINDOW + 1
        sm_idx = max(0, min(len(hour_smoothed_diffs) - 1, sm_idx)) if len(hour_smoothed_diffs) > 0 else 0

        smoothed_score = float(hour_smoothed_diffs[sm_idx]) if len(hour_smoothed_diffs) > 0 else 0.0
        local_thr = float(hour_threshold_curve[sm_idx]) if len(hour_threshold_curve) > 0 else 0.0
        pct_used = compute_segment_percentage_for_frame(
            hour_smoothed_diffs,
            hour_threshold_curve,
            fps_for_filters,
            sm_idx
        ) if len(hour_smoothed_diffs) > 0 else 0

        time_text = timestamps[frame_idx].strftime("%d.%m.%Y %H:%M:%S.%f")[:-3]

        cv2.putText(
            frame,
            f"Norm Score: {smoothed_score:.3f}  Thr: {local_thr:.3f} ({pct_used}%)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA
        )

        total_rot_secs = int(hour_cumulative_rotation_seconds[min(global_idx, len(hour_cumulative_rotation_seconds) - 1)]) \
            if len(hour_cumulative_rotation_seconds) > 0 else 0
        mins, secs = divmod(total_rot_secs, 60)

        cv2.putText(
            frame,
            f"Total Rotation: {mins}m {secs}s",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 0), 2, cv2.LINE_AA
        )

        margin = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thickness = 2

        (w1, _), _ = cv2.getTextSize(time_text, font, scale, thickness)
        x = new_w - w1 - margin
        y = new_h - margin

        cv2.putText(frame, time_text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        if 0 <= global_idx < len(hour_rotation_mask) and hour_rotation_mask[global_idx]:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (new_w, new_h), (255, 0, 0), -1)
            cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

        writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    writer.release()
    cap.release()

# ============================================================
# HOUR PROCESSING
# ============================================================

def process_hour_bucket(hour_key, clip_infos):
    roi_polygon = np.asarray(ROI, dtype=np.int32)

    print("\n" + "=" * 80)
    print(f"PROCESSING HOUR BUCKET: {hour_key.strftime('%Y-%m-%d %H:00')}")
    print("=" * 80)

    hour_agg = build_continuous_hour_signal(hour_key, clip_infos, roi_polygon)

    if len(hour_agg["clip_results"]) == 0 or len(hour_agg["combined_diffs_raw"]) == 0:
        print("No valid videos in this hour bucket.")
        return

    fps_values = [vr["fps"] for vr in hour_agg["clip_results"] if vr["fps"] > 0]
    fps_for_filters = float(np.median(fps_values)) if len(fps_values) > 0 else 30.0

    if PRINT_HOUR_SIGNAL_STATS:
        print_signal_stats("RAW hour signal", hour_agg["combined_diffs_raw"])

    hour_rotation_periods, smoothed_diffs, threshold_curve, normalized_diffs, norm_info, threshold_signal = \
        detect_rotation_periods_for_hour_legacy(hour_agg["combined_diffs_raw"], fps_for_filters)

    if PRINT_HOUR_SIGNAL_STATS:
        print_signal_stats("NORMALIZED hour signal", normalized_diffs)
        print_signal_stats("SMOOTHED normalized signal", smoothed_diffs)
        print("Normalization info:", norm_info)

    if PRINT_THRESHOLD_SIGNAL_STATS:
        print_signal_stats("THRESHOLD-ONLY signal", threshold_signal)

    for start_g, end_g in hour_rotation_periods:
        duration = (end_g - start_g + 1) / max(1e-6, fps_for_filters)

        start_video_idx = int(hour_agg["combined_video_index"][start_g])
        end_video_idx = int(hour_agg["combined_video_index"][end_g])

        start_vr = next((vr for vr in hour_agg["clip_results"] if vr["clip_idx"] == start_video_idx), None)
        end_vr = next((vr for vr in hour_agg["clip_results"] if vr["clip_idx"] == end_video_idx), None)

        rotation_data.append({
            "Hour Bucket": hour_key.strftime("%Y-%m-%d %H:00"),
            "Start File": start_vr["video_file"] if start_vr else "",
            "End File": end_vr["video_file"] if end_vr else "",
            "Timestamp Source": start_vr["timestamp_source"] if start_vr else "",
            "Bout Start": hour_agg["combined_timestamps"][start_g].strftime("%d.%m.%Y %H:%M:%S.%f")[:-3],
            "Bout End": hour_agg["combined_timestamps"][end_g].strftime("%d.%m.%Y %H:%M:%S.%f")[:-3],
            "Duration (seconds, frame-based)": duration,
        })

    hour_summary.append({
        "Hour Bucket": hour_key.strftime("%Y-%m-%d %H:00"),
        "Num Files": len(hour_agg["clip_results"]),
        "Num Events": len(hour_rotation_periods),
        "First File": hour_agg["clip_results"][0]["video_file"],
        "Last File": hour_agg["clip_results"][-1]["video_file"],
        "Signal Frames": len(hour_agg["combined_diffs_raw"]),
        "Normalization Transform": norm_info["transform"],
        "Norm Low Ref": norm_info["low_ref"],
        "Norm High Ref": norm_info["high_ref"],
        "Norm Max": norm_info["max_after_norm"],
        "Threshold Ref Percentile": THRESHOLD_REF_PERCENTILE,
        "Spike Suppression Enabled": USE_THRESHOLD_SPIKE_SUPPRESSION,
        "Spike Min Seconds": THRESHOLD_SPIKE_MIN_SECONDS,
        "Spike High Q": THRESHOLD_SPIKE_HIGH_Q,
        "Spike Isolation Gap Seconds": THRESHOLD_SPIKE_ISOLATION_GAP_SECONDS,
    })

    if WRITE_HOUR_PLOTS:
        plot_path = os.path.join(plot_folder, f"{hour_key.strftime('%Y%m%d_%H00')}_movement.png")
        save_hour_plot_legacy(
            hour_key,
            hour_agg["combined_timestamps"],
            smoothed_diffs,
            threshold_curve,
            hour_rotation_periods,
            plot_path,
            threshold_signal=threshold_signal
        )

    hour_rotation_mask = build_rotation_mask(len(smoothed_diffs), hour_rotation_periods)
    hour_cumulative_rotation_seconds = build_cumulative_rotation_seconds_global(
        len(smoothed_diffs),
        hour_rotation_periods,
        fps_for_filters
    )

    if WRITE_ANNOTATED_VIDEO:
        for vr in hour_agg["clip_results"]:
            out_name = os.path.splitext(vr["video_file"])[0] + "_legacy_annotated" + OUTPUT_EXT
            out_path = os.path.join(annotated_video_folder, sanitize_filename(out_name))

            write_annotated_video_for_clip_legacy(
                vr=vr,
                out_path=out_path,
                hour_smoothed_diffs=smoothed_diffs,
                hour_threshold_curve=threshold_curve,
                hour_rotation_mask=hour_rotation_mask,
                hour_cumulative_rotation_seconds=hour_cumulative_rotation_seconds,
                fps_for_filters=fps_for_filters
            )

# ============================================================
# COMMAND-LINE CONFIGURATION
# ============================================================

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Detect mechanical cattle-brush rotation from fixed-camera video "
            "using the activity-index pipeline used in the associated study."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing input video files."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Base directory for analysis outputs."
    )
    parser.add_argument(
        "--roi-preset",
        default=DEFAULT_ROI_PRESET,
        choices=sorted(ROI_ALL.keys()),
        help=f"ROI preset to use (default: {DEFAULT_ROI_PRESET})."
    )
    parser.add_argument(
        "--experiment-name",
        default=EXPERIMENT_NAME,
        help="Name of the output subdirectory."
    )
    parser.add_argument(
        "--require-dvr-timestamps",
        action="store_true",
        help="Skip files that do not contain usable DVR timestamps."
    )
    parser.add_argument(
        "--no-annotated-video",
        action="store_true",
        help="Do not write annotated video outputs."
    )
    parser.add_argument(
        "--no-hour-plots",
        action="store_true",
        help="Do not write hourly signal plots."
    )
    parser.add_argument(
        "--show-roi-preview",
        action="store_true",
        help="Display ROI preview windows during processing."
    )
    parser.add_argument(
        "--no-roi-preview",
        action="store_true",
        help="Do not save ROI preview images."
    )
    return parser


def configure_run(args):
    """Configure paths and non-algorithmic run options from CLI arguments."""
    global ROI
    global EXPERIMENT_NAME
    global REQUIRE_DVR_TIMESTAMPS
    global WRITE_ANNOTATED_VIDEO
    global WRITE_HOUR_PLOTS
    global SHOW_ROI_PREVIEW
    global SAVE_ROI_PREVIEW
    global input_folder
    global base_output_folder
    global output_folder
    global annotated_video_folder
    global plot_folder
    global preview_folder

    input_folder = os.path.abspath(args.input)
    base_output_folder = os.path.abspath(args.output)

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input directory does not exist: {input_folder}")

    ROI = ROI_ALL[args.roi_preset].copy()
    EXPERIMENT_NAME = args.experiment_name
    REQUIRE_DVR_TIMESTAMPS = bool(args.require_dvr_timestamps)
    WRITE_ANNOTATED_VIDEO = not bool(args.no_annotated_video)
    WRITE_HOUR_PLOTS = not bool(args.no_hour_plots)
    SHOW_ROI_PREVIEW = bool(args.show_roi_preview)
    SAVE_ROI_PREVIEW = not bool(args.no_roi_preview)

    output_folder = os.path.join(base_output_folder, EXPERIMENT_NAME)
    annotated_video_folder = os.path.join(output_folder, "annotated_videos")
    plot_folder = os.path.join(output_folder, "plots")
    preview_folder = os.path.join(output_folder, "previews")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(annotated_video_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)
    os.makedirs(preview_folder, exist_ok=True)

    print(f"Input directory: {input_folder}")
    print(f"Output directory: {output_folder}")
    print(f"ROI preset: {args.roi_preset}")
    print(f"FFmpeg/ffprobe available: {FFMPEG_AVAILABLE}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_run(args)

    buckets = build_hour_buckets(input_folder)

    if len(buckets) == 0:
        print("No valid video files found.")
    else:
        for hour_key, clip_infos in sorted(buckets.items()):
            process_hour_bucket(hour_key, clip_infos)

    df = pd.DataFrame(rotation_data)
    excel_path = os.path.join(output_folder, "rotation_summary.xlsx")
    df.to_excel(excel_path, index=False)
    print("Summary saved:", excel_path)

    hour_df = pd.DataFrame(hour_summary)
    hour_excel_path = os.path.join(output_folder, "hour_bucket_summary.xlsx")
    hour_df.to_excel(hour_excel_path, index=False)
    print("Hour summary saved:", hour_excel_path)

    ts_df = pd.DataFrame(file_timestamp_report)
    ts_excel_path = os.path.join(output_folder, "file_timestamp_report.xlsx")
    ts_df.to_excel(ts_excel_path, index=False)
    print("File timestamp report saved:", ts_excel_path)

    if len(gap_report) > 0:
        gap_df = pd.DataFrame(gap_report)
        gap_csv = os.path.join(output_folder, "timestamp_gap_report.csv")
        gap_df.to_csv(gap_csv, index=False)
        print("Gap report saved:", gap_csv)
    else:
        print("No timestamp gaps detected.")
