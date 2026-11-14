# Cattle Brush Activity Index

Detects **mechanical rotation of cattle grooming brushes** in fixed-camera video
with a lightweight, CPU-only frame-differencing activity index. It runs on a
desktop or on an edge device such as a Raspberry Pi 5.

The output describes brush rotation, not which cow used the brush. Individual
brush use needs an additional animal-identification or contact layer.

## Installation

```bash
python -m pip install -r requirements.txt
```

Install [FFmpeg](https://ffmpeg.org/) and make sure `ffmpeg` and `ffprobe` are
on the command line. They are needed to read DVR timestamps.

## Usage

```bash
python src/brush_rotation_activity_index.py --input path/to/videos --output path/to/results --roi-preset A1
```

- `--roi-preset` selects one of the brush ROIs in `src/config.py` (`A1`–`A5`,
  `B1`–`B5`, the ten evaluation recordings). For other cameras, set `ROI` in
  `src/config.py` to your own polygon.
- Settings in `src/config.py` are the defaults; command-line options override
  them for one run. See [`docs/CLI_HELP.txt`](docs/CLI_HELP.txt) for all options.
- Always check the saved ROI preview before a long run.

## Method

1. Read the video and its frame timestamps.
2. Mask the brush ROI and preprocess the frame.
3. Sum absolute frame-to-frame pixel differences within the ROI.
4. Log-transform and normalise to the 5th–95th percentiles.
5. Smooth with a 5-frame moving average.
6. Suppress short isolated spikes in a copy used only for threshold estimation.
7. Compute an adaptive threshold per 10-min segment (99.5th percentile × 0.50 or 0.62).
8. Detect rotation intervals on the smoothed signal.
9. Merge gaps < 2 s, remove bouts < 4 s and apply the contextual filters.

Every step and parameter is mapped to code in
[`docs/METHODS_TO_CODE_MAP.md`](docs/METHODS_TO_CODE_MAP.md).

## Timing

Frame times come from, in order:

1. **DVR timestamps** embedded in the video (frame rate measured from them);
2. **a timestamp in the file name**, with frame times at the container frame rate;
3. **relative time** from 00:00:00, for videos with neither (analysed separately).

With DVR timestamps, the frame rate used for the time windows is measured from
them; otherwise the container frame rate is used. To analyse at a lower
resolution, use `--processing-height`, which resizes inside the pipeline and
keeps the original video and its timestamps.

Details: [`docs/TIMING.md`](docs/TIMING.md).

## Outputs

Written to `<output>/<experiment name>/` as UTF-8 CSV:

| File | Content |
|---|---|
| `rotation_summary.csv` | Detected rotation intervals: start, end, source clips, timestamp source |
| `hour_bucket_summary.csv` | Per-hour signal and threshold summary |
| `file_timestamp_report.csv` | Timestamp source, container and clock frame rate per video |
| `clip_timing_diagnostics.csv` | Per-clip frame-rate estimate, dropped/duplicate frames |
| `timestamp_gap_report.csv` | Timing anomalies |

plus hourly plots (`plots/`), ROI previews (`previews/`) and optional annotated
videos (`annotated_videos/`). Use `Bout End − Bout Start` for bout durations.

## Repository layout

```text
src/        brush_rotation_activity_index.py (pipeline), config.py, io_utils.py
tests/      unit and end-to-end tests
docs/       CLI_HELP.txt, CONFIGURATION.md, METHODS_TO_CODE_MAP.md, TIMING.md
examples/   local input/output folders and the output schema
```

## Citation and license

See [`CITATION.cff`](CITATION.cff). Released under the [MIT License](LICENSE).
