"""Video, timestamp, filename, and filesystem utilities.

This module contains input/output support only. It does not implement the
activity-index detection algorithm or alter its thresholds/post-processing.
"""

import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timedelta
from functools import lru_cache

import cv2

from config import (
    RELATIVE_TIME_ANCHOR,
    REQUIRE_DVR_TIMESTAMPS,
    SUPPORTED_FORMATS,
)


def ensure_directories(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def is_video_file(name: str) -> bool:
    return name.lower().endswith(SUPPORTED_FORMATS)


def safe_run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


@lru_cache(maxsize=1)
def check_ffmpeg_available() -> bool:
    """Checked on first use rather than at import time."""
    try:
        result = safe_run(["ffmpeg", "-version"])
        return result.returncode == 0
    except Exception:
        return False


def sanitize_filename(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", s)


def frame_to_time(frame, fps):
    total_seconds = int(frame / max(fps, 1e-6))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}m {seconds}s"


def format_relative_timestamp(timestamp, include_milliseconds=True):
    """Format the synthetic 1970 UTC anchor as elapsed video time."""
    elapsed = max(0.0, (timestamp - RELATIVE_TIME_ANCHOR).total_seconds())
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    if include_milliseconds:
        milliseconds = int(round((elapsed - int(elapsed)) * 1000))
        if milliseconds >= 1000:
            seconds += 1
            milliseconds -= 1000
            if seconds >= 60:
                minutes += 1
                seconds -= 60
                if minutes >= 60:
                    hours += 1
                    minutes -= 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


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
    if not check_ffmpeg_available():
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


def median_positive_step(timestamps):
    """Median positive interval between consecutive timestamps (timedelta), or None."""
    steps = sorted(
        (b - a).total_seconds() for a, b in zip(timestamps, timestamps[1:]) if b > a
    )
    if not steps:
        return None
    mid = len(steps) // 2
    median = steps[mid] if len(steps) % 2 else (steps[mid - 1] + steps[mid]) / 2
    return timedelta(seconds=median)


def reconstruct_dvr_timestamps(date_obj, time_fragments, fps):
    timestamps = []

    if not time_fragments:
        return timestamps

    # Used only when a timestamp goes backwards: step forward by the last
    # positive DVR interval seen, falling back to the container rate.
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
            elif timestamps and ts > timestamps[-1]:
                frame_interval = ts - timestamps[-1] if (ts - timestamps[-1]).total_seconds() < 1 else frame_interval

            timestamps.append(ts)

        except Exception:
            continue

    return timestamps


def extract_dvr_metadata_from_stream(video_path, stream_index, fps):
    if not check_ffmpeg_available():
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
# TIMESTAMP DECISION / BASIC VIDEO INFO
# ============================================================

def build_timestamps_from_start(start_dt, fps, total_frames):
    if start_dt is None or total_frames <= 0:
        return []
    if fps <= 0:
        fps = 30.0
    return [start_dt + timedelta(seconds=i / fps) for i in range(total_frames)]


def get_video_timestamps_and_ids(video_path, fps, total_frames, allow_relative_time=False):
    meta = extract_dvr_metadata(video_path, fps)

    if meta and meta.get("timestamps"):
        timestamps = meta["timestamps"]
        print(f"Using DVR metadata timestamps: {os.path.basename(video_path)}")

        if len(timestamps) >= total_frames:
            timestamps = timestamps[:total_frames]
        else:
            # The DVR stream commonly lacks the timestamp of the last frame; a
            # shortfall of one frame is expected and not reported.
            if total_frames - len(timestamps) > 1:
                print(
                    f"Warning: DVR timestamps shorter than frame count "
                    f"({len(timestamps)} < {total_frames}) in {os.path.basename(video_path)}"
                )
            # Extend on the DVR clock, not the container frame rate.
            step = median_positive_step(timestamps) or timedelta(seconds=1 / max(fps, 1e-6))

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
        print(
            f"Using filename timestamps (no DVR stream; frame times at the container "
            f"rate of {fps:.3f} fps): {os.path.basename(video_path)}"
        )
        return {
            "timestamps": build_timestamps_from_start(fname, fps, total_frames),
            "camera_id": None,
            "stream_id": None,
            "overlay_id": None,
            "timestamp_source": "filename"
        }

    if allow_relative_time:
        print(
            f"No DVR metadata or recognised filename timestamp: {os.path.basename(video_path)}. "
            f"Using relative time from 00:00:00 for standalone video."
        )
        return {
            "timestamps": build_timestamps_from_start(RELATIVE_TIME_ANCHOR, fps, total_frames),
            "camera_id": None,
            "stream_id": None,
            "overlay_id": None,
            "timestamp_source": "relative"
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


def get_clip_start_info(video_path, allow_relative_time=False, require_dvr=None):
    if require_dvr is None:
        require_dvr = REQUIRE_DVR_TIMESTAMPS

    basic = get_video_basic_info(video_path)
    if basic is None:
        return None

    ts_info = get_video_timestamps_and_ids(
        video_path,
        basic["fps"],
        basic["length"],
        allow_relative_time=allow_relative_time
    )
    if ts_info is None or len(ts_info["timestamps"]) == 0:
        return None

    if require_dvr and ts_info["timestamp_source"] != "dvr":
        print(f"Skipping non-DVR timestamp file: {os.path.basename(video_path)}")
        return None

    # Estimate sampling rate from independent DVR timestamps, not container FPS.
    # Filename-based timing remains an assumption and is labeled accordingly.
    timestamps = ts_info["timestamps"]
    step = median_positive_step(timestamps) if ts_info["timestamp_source"] == "dvr" else None
    actual_fps = 1 / step.total_seconds() if step else basic["fps"]

    return {
        "actual_fps": actual_fps,
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
