# Cattle Brush Activity Index for Mechanical Rotation Detection

Research code for detecting **mechanical rotation of cattle grooming brushes** from fixed-camera video using a lightweight CPU-based activity index.

The code measures brush motion. Used alone, its output therefore represents brush/resource-level rotation. Individual-level brush-use monitoring would require a separate animal-identification/contact layer.

## Repository structure

```text
src/
  brush_rotation_activity_index.py   # detection pipeline and analysis orchestration
  config.py                          # paths, ROI presets, parameters, output switches
  io_utils.py                        # video/timestamp/filename/file-system utilities
examples/
  input/                             # local test video location; videos are git-ignored
  output/                            # local generated output; includes report schema only
docs/
  CONFIGURATION.md
  METHODS_TO_CODE_MAP.md
  RELEASE_CHECKLIST.md
```

The detector itself remains in one main script so that the manuscript workflow can be followed from signal extraction through post-processing. Configuration and input/timestamp utilities are separated only for readability; the detection thresholds and post-processing logic are unchanged by this refactor.

## Quick start

1. Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Install FFmpeg separately and confirm that `ffmpeg` and `ffprobe` are available on the command line if DVR metadata extraction is required.

3. Edit `src/config.py`:
   - `input_folder`
   - `base_output_folder`
   - `ROI` / ROI preset
   - optional output switches

4. Run:

```bash
python src/brush_rotation_activity_index.py
```

See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for the timing modes and filename formats.

## Processing sequence

1. Read video and obtain timing information.
2. Apply the predefined brush ROI and image preprocessing.
3. Calculate frame-to-frame absolute pixel differences and sum them within the ROI.
4. Apply log transformation and 5th–95th percentile normalization.
5. Smooth the normalized signal with a 5-frame moving average.
6. Build a threshold-only signal by suppressing isolated short high-amplitude spikes.
7. Divide the signal into 10-min segments and use the 99.5th percentile as the segment reference.
8. Apply the study-specific dynamic multiplier (0.50 for reference values >= 0.90; 0.62 otherwise).
9. Detect candidate rotation intervals on the original smoothed normalized signal.
10. Apply temporal and contextual post-processing and export the final mechanical-rotation intervals.

See [`docs/METHODS_TO_CODE_MAP.md`](docs/METHODS_TO_CODE_MAP.md) for the function-level mapping.

## Timestamp handling

Timing is resolved in this order:

1. **DVR metadata** — absolute date/time reconstructed from the supported DVR metadata stream.
2. **Recognised filename timestamp** — absolute start time parsed from the filename, with frame times generated from decoded FPS.
3. **Relative time** — if neither source exists and `ALLOW_STANDALONE_RELATIVE_TIME = True`, that video is processed independently from `00:00:00`.

Relative-time videos are never combined with each other or with the absolute-time hourly sequence. Each receives its own output folder and its own rotation/timestamp/hour reports. Internally the relative mode uses `1970-01-01 00:00:00 UTC` only as a computational anchor; user-facing output is formatted as elapsed time.

Supported filename timestamp patterns include:

```text
Event20240425123045.mp4
20240425123045.mp4
25.04.2024-123045.mp4
25.04.2024_123045.mp4
2024-04-25_12-30-45.mp4
2024_04_25_12_30_45.mp4
```

If `REQUIRE_DVR_TIMESTAMPS = True`, non-DVR timing sources are rejected.

## Outputs

### Absolute-time/DVR or filename workflow

The original hourly workflow can generate:

- `rotation_summary.xlsx` — detected rotation intervals, source file(s), timestamp source, and duration
- `hour_bucket_summary.xlsx` — hourly signal/threshold summary
- `file_timestamp_report.xlsx` — timestamp source and basic video metadata
- `timestamp_gap_report.csv` — timing anomalies when detected
- `plots/` — hourly normalized motion/threshold plots
- `previews/` — ROI preview images
- `annotated_videos/` — optional annotated videos

### Standalone relative-time workflow

Every untimestamped video receives a separate directory under:

```text
standalone_relative/<source_video_name>/
```

with its own `rotation_summary.xlsx`, `hour_bucket_summary.xlsx`, `file_timestamp_report.xlsx`, plot, ROI preview, and optional annotated video. Relative bout times are written as elapsed `HH:MM:SS.sss` values.
