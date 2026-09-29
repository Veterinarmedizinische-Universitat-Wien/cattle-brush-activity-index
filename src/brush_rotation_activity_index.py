# ============================================================
# FULLY PATCHED CROSS-FARM CODE
# CONTINUOUS HOUR SIGNAL + LEGACY-STYLE DETECTION/POSTPROCESSING
# USING ROBUST NORMALIZED SIGNAL FOR CROSS-FARM TRANSFER
# + DVR TIMESTAMPS FOR PLOTTING / EXPORT / ANNOTATION
#
# ROBUST THRESHOLD PATCH
# ----------------------
# This version keeps your motion-difference logic intact, but makes
# threshold determination robust against short extreme spikes.
#
# MAIN FIXES
# ----------
# 1) Compute the original raw motion signal exactly as before.
# 2) Normalize and smooth exactly as before.
# 3) Build a THRESHOLD-ONLY signal where isolated short spikes are suppressed.
# 4) Compute segment thresholds from that threshold-only signal.
# 5) Use a HIGH PERCENTILE segment reference instead of raw segment max.
# 6) Run final detection on the ORIGINAL smoothed signal.
#
# RESULT
# ------
# Short lone spikes (< ~10 sec) no longer inflate thresholds for an
# entire segment, while the original signal remains fully available
# for bout detection.
# ============================================================

import os
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    ALLOW_STANDALONE_RELATIVE_TIME,
    BLEND_ALPHA,
    CLAHE_CLIP_LIMIT,
    CLAHE_DARK_MEAN_THRESHOLD,
    CLAHE_ONLY_IF_DARK,
    CLAHE_TILE_GRID,
    DOWNSCALE,
    FRAME_SKIP_ANNOTATION,
    LEGACY_CONTEXT_MAX_PIXEL,
    LEGACY_CONTEXT_WINDOW_MINUTES,
    LEGACY_HIGH_PCT,
    LEGACY_LOW_PCT,
    LEGACY_PIXEL_THRESHOLD_HIGH,
    LEGACY_PIXEL_THRESHOLD_LOW,
    LEGACY_SEGMENT_MAX_SWITCH,
    LEGACY_SEGMENT_MINUTES,
    MAX_BOUT_SECONDS,
    MERGE_CLOSE_BOUTS,
    MERGE_GAP_SECONDS,
    MIN_BOUT_SECONDS,
    OUTPUT_CODEC,
    OUTPUT_EXT,
    PRINT_HOUR_SIGNAL_STATS,
    PRINT_THRESHOLD_SIGNAL_STATS,
    RELATIVE_TIME_ANCHOR,
    REQUIRE_DVR_TIMESTAMPS,
    ROBUST_NORM_HIGH_PCT,
    ROBUST_NORM_LOW_PCT,
    ROI,
    SAVE_ROI_PREVIEW,
    SHORT_BOUT_SECONDS,
    SHORT_BOUT_WINDOW_SECONDS,
    SHOW_ROI_PREVIEW,
    SMOOTH_WINDOW,
    THRESHOLD_REF_PERCENTILE,
    THRESHOLD_SPIKE_HIGH_Q,
    THRESHOLD_SPIKE_ISOLATION_GAP_SECONDS,
    THRESHOLD_SPIKE_MIN_SECONDS,
    USE_CLAHE,
    USE_LOG_COMPRESSION,
    USE_THRESHOLD_SPIKE_SUPPRESSION,
    WRITE_ANNOTATED_VIDEO,
    WRITE_HOUR_PLOTS,
    annotated_video_folder,
    input_folder,
    output_folder,
    plot_folder,
    preview_folder,
)
from io_utils import (
    ensure_directories,
    format_relative_timestamp,
    get_clip_start_info,
    is_video_file,
    sanitize_filename,
)

# Create the configured output directories once at startup.
ensure_directories(output_folder, annotated_video_folder, plot_folder, preview_folder)

rotation_data = []
gap_report = []
hour_summary = []
file_timestamp_report = []

# ============================================================
# INPUT GROUPING
# ============================================================

def build_hour_buckets(folder: str):
    """
    Build the original absolute-time hour buckets and keep untimestamped videos
    separate for standalone relative-time processing.

    Absolute-time videos (DVR or filename timestamp) retain the original workflow.
    Relative-time videos are returned separately and are never combined together.
    """
    buckets = defaultdict(list)
    relative_videos = []

    video_files = [
        file for file in sorted(os.listdir(folder))
        if is_video_file(file)
    ]

    allow_relative_time = (
        ALLOW_STANDALONE_RELATIVE_TIME
        and not REQUIRE_DVR_TIMESTAMPS
    )

    for file in video_files:
        video_path = os.path.join(folder, file)
        start_info = get_clip_start_info(
            video_path,
            allow_relative_time=allow_relative_time
        )

        if start_info is None:
            print(f"Skipping (no valid timestamp): {file}")
            continue

        if start_info["timestamp_source"] == "relative":
            print(
                f"Standalone relative-time analysis selected: {file} "
                f"(starts at 00:00:00)"
            )
            relative_videos.append(start_info)
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

    return buckets, relative_videos

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
    threshold_signal=None,
    source_label=None,
    relative_time=False
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
    if relative_time:
        tick_labels = [
            format_relative_timestamp(combined_timestamps[i], include_milliseconds=False)
            for i in tick_idx
        ]
    else:
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

    plt.xlabel("Relative Time" if relative_time else "Real Time")
    plt.ylabel("Normalized motion signal")

    if relative_time:
        title = "Movement curve - relative time"
    else:
        title = f"Hourly movement curve - {hour_key.strftime('%Y-%m-%d %H:00')}"
    if source_label:
        title += f"\nSource: {source_label}"
    plt.title(title)
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

        if vr.get("timestamp_source") == "relative":
            time_text = format_relative_timestamp(timestamps[frame_idx])
        else:
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

    is_relative_run = (
        len(hour_agg["clip_results"]) == 1
        and hour_agg["clip_results"][0].get("timestamp_source") == "relative"
    )

    for start_g, end_g in hour_rotation_periods:
        duration = (end_g - start_g + 1) / max(1e-6, fps_for_filters)

        start_video_idx = int(hour_agg["combined_video_index"][start_g])
        end_video_idx = int(hour_agg["combined_video_index"][end_g])

        start_vr = next((vr for vr in hour_agg["clip_results"] if vr["clip_idx"] == start_video_idx), None)
        end_vr = next((vr for vr in hour_agg["clip_results"] if vr["clip_idx"] == end_video_idx), None)

        bout_start_ts = hour_agg["combined_timestamps"][start_g]
        bout_end_ts = hour_agg["combined_timestamps"][end_g]

        if is_relative_run:
            hour_bucket_label = "relative"
            bout_start_text = format_relative_timestamp(bout_start_ts)
            bout_end_text = format_relative_timestamp(bout_end_ts)
        else:
            hour_bucket_label = hour_key.strftime("%Y-%m-%d %H:00")
            bout_start_text = bout_start_ts.strftime("%d.%m.%Y %H:%M:%S.%f")[:-3]
            bout_end_text = bout_end_ts.strftime("%d.%m.%Y %H:%M:%S.%f")[:-3]

        rotation_data.append({
            "Hour Bucket": hour_bucket_label,
            "Start File": start_vr["video_file"] if start_vr else "",
            "End File": end_vr["video_file"] if end_vr else "",
            "Timestamp Source": start_vr["timestamp_source"] if start_vr else "",
            "Bout Start": bout_start_text,
            "Bout End": bout_end_text,
            "Duration (seconds, frame-based)": duration,
        })

    hour_summary.append({
        "Hour Bucket": "relative" if is_relative_run else hour_key.strftime("%Y-%m-%d %H:00"),
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
        if is_relative_run:
            source_stem = os.path.splitext(hour_agg["clip_results"][0]["video_file"])[0]
            source_label = source_stem
            plot_path = os.path.join(
                plot_folder,
                f"{sanitize_filename(source_stem)}_movement.png"
            )
        else:
            source_label = None
            plot_path = os.path.join(
                plot_folder,
                f"{hour_key.strftime('%Y%m%d_%H00')}_movement.png"
            )

        save_hour_plot_legacy(
            hour_key,
            hour_agg["combined_timestamps"],
            smoothed_diffs,
            threshold_curve,
            hour_rotation_periods,
            plot_path,
            threshold_signal=threshold_signal,
            source_label=source_label,
            relative_time=is_relative_run
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
# REPORT WRITING
# ============================================================

def write_reports(report_output_folder):
    os.makedirs(report_output_folder, exist_ok=True)

    df = pd.DataFrame(rotation_data)
    excel_path = os.path.join(report_output_folder, "rotation_summary.xlsx")
    df.to_excel(excel_path, index=False)
    print("Summary saved:", excel_path)

    hour_df = pd.DataFrame(hour_summary)
    hour_excel_path = os.path.join(report_output_folder, "hour_bucket_summary.xlsx")
    hour_df.to_excel(hour_excel_path, index=False)
    print("Hour summary saved:", hour_excel_path)

    ts_df = pd.DataFrame(file_timestamp_report)
    ts_excel_path = os.path.join(report_output_folder, "file_timestamp_report.xlsx")
    ts_df.to_excel(ts_excel_path, index=False)
    print("File timestamp report saved:", ts_excel_path)

    if len(gap_report) > 0:
        gap_df = pd.DataFrame(gap_report)
        gap_csv = os.path.join(report_output_folder, "timestamp_gap_report.csv")
        gap_df.to_csv(gap_csv, index=False)
        print("Gap report saved:", gap_csv)
    else:
        print("No timestamp gaps detected.")

def process_standalone_relative_video(clip_info):
    """Process one untimestamped video independently with its own reports."""
    global output_folder, annotated_video_folder, plot_folder, preview_folder
    global rotation_data, gap_report, hour_summary, file_timestamp_report

    original_paths = (
        output_folder,
        annotated_video_folder,
        plot_folder,
        preview_folder,
    )
    original_data = (
        rotation_data,
        gap_report,
        hour_summary,
        file_timestamp_report,
    )

    source_stem = sanitize_filename(os.path.splitext(clip_info["video_file"])[0])
    standalone_root = os.path.join(
        original_paths[0],
        "standalone_relative",
        source_stem
    )

    try:
        output_folder = standalone_root
        annotated_video_folder = os.path.join(output_folder, "annotated_videos")
        plot_folder = os.path.join(output_folder, "plots")
        preview_folder = os.path.join(output_folder, "previews")

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(annotated_video_folder, exist_ok=True)
        os.makedirs(plot_folder, exist_ok=True)
        os.makedirs(preview_folder, exist_ok=True)

        rotation_data = []
        gap_report = []
        hour_summary = []
        file_timestamp_report = [{
            "Video File": clip_info["video_file"],
            "Start Timestamp": "00:00:00.000",
            "Timestamp Source": "relative",
            "FPS": clip_info["fps"],
            "Frame Count": clip_info["length"],
        }]

        print("\n" + "=" * 80)
        print(f"STANDALONE RELATIVE-TIME VIDEO: {clip_info['video_file']}")
        print("Relative time starts at 00:00:00 (internal anchor: 1970-01-01 00:00:00 UTC)")
        print("=" * 80)

        process_hour_bucket(RELATIVE_TIME_ANCHOR, [clip_info])
        write_reports(output_folder)

    finally:
        (
            output_folder,
            annotated_video_folder,
            plot_folder,
            preview_folder,
        ) = original_paths
        (
            rotation_data,
            gap_report,
            hour_summary,
            file_timestamp_report,
        ) = original_data

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    buckets, relative_videos = build_hour_buckets(input_folder)

    if len(buckets) == 0 and len(relative_videos) == 0:
        print("No valid video files found.")

    # Original absolute-time workflow remains unchanged.
    if len(buckets) > 0:
        for hour_key, clip_infos in sorted(buckets.items()):
            process_hour_bucket(hour_key, clip_infos)

        write_reports(output_folder)

    # Untimestamped videos are never combined. Each receives its own output folder
    # and its own rotation/timestamp/hour reports using relative time from 00:00:00.
    for clip_info in relative_videos:
        process_standalone_relative_video(clip_info)

