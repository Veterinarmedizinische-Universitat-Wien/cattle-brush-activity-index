# Timebase and 360p implementation (draft)

This branch introduces configurable 360p resizing before frame-based brush-motion extraction. ROI polygons are scaled from their original camera reference coordinates; new camera viewpoints require manual ROI definition and preview verification.

DVR-derived median positive timestamp intervals are now used to estimate recording FPS for per-hour frame-count-based filters and the annotated-video writer. Filename-only recordings continue to assume the container rate; that assumption is not independent FPS validation.

The original ROI preprocessing, frame blending, absolute-difference sum, normalization, moving average, threshold multiplier formula and postfilters are retained.

## Known limitations requiring follow-up before release

- Detection remains partitioned by clip-start hour; clips that cross an hour boundary and bouts crossing buckets still require a frame-level chronological coordinator.
- Ten-minute windows remain frame-count based, using the estimated FPS, not strictly clock-aligned 600-second windows.
- The 360p resizing changes motion score distributions even when the formula is unchanged; compare detections against original-resolution baseline.
- Existing standalone relative-time fallback remains supported; it is not evidence that the container FPS is verified.
- Output still uses OpenCV mp4v; H.264 compressed export and corresponding runtime/quality tests are not yet integrated.
- Per-frame DVR timestamps may require an alignment check when packet count differs from decoded frame count.

## Basic tests

Run `python -m unittest discover -s tests` after installing dependencies. Tests cover ROI scaling, normalisation references, the segment threshold, the short-bout rule, clip joins, the CLI, and an end-to-end run on synthetic videos. Regression testing against the manuscript results must still be run with local sample recordings.
