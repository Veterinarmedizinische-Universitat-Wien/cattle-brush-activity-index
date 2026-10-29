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

Set `ROI` in `src/config.py` to a preset (`ROI = ROI_ALL["44"]`) or to your own
`np.int32` polygon. `--roi-preset NAME` overrides it for a single run.

Each preset has a reference frame size in `ROI_REFERENCE_SIZES` (the resolution
the polygon was drawn on); the polygon is scaled from that size to the processed
frame and to the annotated video. With `ROI_REFERENCE_SIZE = None` (default) the
size is chosen automatically: a preset uses its entry, and a custom polygon is
taken to be in each video's native resolution. Set `ROI_REFERENCE_SIZE` to a
`(width, height)` tuple only if a custom polygon was drawn on a frame of a
different resolution. Always check the saved ROI preview.

## Clip joins

Clips in the same hour bucket are analysed as one continuous signal. The
previous-frame state is carried from one clip to the next only when the next clip
starts within `CLIP_JOIN_MAX_GAP_SECONDS` (default 2 s) of the previous clip's
last frame; larger gaps reset it. Frames without a real motion score (the first
frame after a reset and frames that failed to process) are excluded from the
normalisation percentiles. `hour_bucket_summary.csv` reports both counts.

## Processing resolution

By default (`PROCESSING_HEIGHT = None`) frames are analysed at the recording's
native resolution, as for the main evaluation results. Setting
`PROCESSING_HEIGHT` to 720, 480 or 360 resizes each decoded frame inside the
pipeline before motion extraction (in-pipeline downscaling, as in the resolution
benchmark); this increases throughput, but check detection performance on
representative day and night footage first.

The ROI polygon is scaled from its reference size to the analysed frame, to the
native frame for the ROI preview, and to `DOWNSCALE` (default 0.5) for annotated
videos. If a video's aspect ratio differs from the ROI's reference size, the run
prints a warning because the polygon would be stretched.
