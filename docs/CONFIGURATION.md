# Configuration and timing modes

All user-editable paths, ROI presets, algorithm parameters, and output switches
are in `src/config.py`. The detection logic is in
`src/brush_rotation_activity_index.py`; input/timestamp helper functions are in
`src/io_utils.py`.

## Paths

Pass `--input` and `--output` on the command line. Otherwise the `CBAI_INPUT`
and `CBAI_OUTPUT` environment variables are used, then the repository's
`examples/input` and `examples/output` folders. Results go to
`<output>/<experiment name>/`. Nothing is created until a run starts.

## ROI

Select a study preset with `--roi-preset` or `ROI_KEY` in `src/config.py`.
Each preset has a reference frame size in `ROI_REFERENCE_SIZES` (the resolution
the polygon was drawn on); the polygon is scaled from that size to the processed
frame and to the annotated video.

For a new camera position, add a polygon to `ROI_ALL` and its drawing resolution
to `ROI_REFERENCE_SIZES`. If `ROI_REFERENCE_SIZE` is `None`, the polygon is taken
to be in each video's native resolution. Always check the saved ROI preview.

## Clip joins

Clips in the same hour bucket are analysed as one continuous signal. The
previous-frame state is carried from one clip to the next only when the next clip
starts within `CLIP_JOIN_MAX_GAP_SECONDS` (default 2 s) of the previous clip's
last frame; larger gaps reset it. Frames without a real motion score (the first
frame after a reset and frames that failed to process) are excluded from the
normalisation percentiles. `hour_bucket_summary.xlsx` reports both counts.

## Clustered short-bout rule

`evaluate_window` removes bouts when every bout starting within
`SHORT_BOUT_WINDOW_SECONDS` is shorter than `SHORT_BOUT_SECONDS`. Because
detection already requires bouts longer than `MIN_BOUT_SECONDS`, the rule only
has an effect when `SHORT_BOUT_SECONDS > MIN_BOUT_SECONDS`. With the study
defaults (4 s and 4 s) it removes nothing, and the run prints a config warning.

## Timing priority

The software attempts timing in this order:

1. DVR metadata (`timestamp_source = dvr`)
2. recognised filename start time (`timestamp_source = filename`)
3. standalone relative time (`timestamp_source = relative`)

Filename timing accepts the patterns documented in the main README. Filename and
DVR absolute times are not automatically timezone-converted.

When relative fallback is used, the internal anchor is
`1970-01-01 00:00:00 UTC`, but reports and annotated video display elapsed time
from `00:00:00`. Each relative video is analysed independently and receives its
own output reports.

Set `REQUIRE_DVR_TIMESTAMPS = True` (or pass `--require-dvr-timestamps`) when only supported DVR metadata should be
accepted. Set `ALLOW_STANDALONE_RELATIVE_TIME = False` to disable the relative
fallback.
