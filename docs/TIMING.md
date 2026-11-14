# Timing

Rotation intervals are reported as clock times, and the detector's time windows
(10-min threshold segments, 4-s minimum bout, 2-s merge gap, 4-min context
window, spike rules) are converted to frames with a frame rate. Both depend on
where the frame times come from.

## Sources of frame times

| Source | Frame times | Frame rate used for windows |
|---|---|---|
| DVR timestamp stream in the video | read per frame | median positive interval between DVR timestamps |
| Timestamp in the file name | start time + frame index / container frame rate | container frame rate |
| Neither (relative time) | 00:00:00 + frame index / container frame rate | container frame rate |

`file_timestamp_report.csv` lists the source, the container frame rate and the
clock frame rate of every video; `clip_timing_diagnostics.csv` lists dropped
and duplicate frames per clip.

DVR timestamps that run backwards are repaired by stepping forward with the
last DVR interval. If the DVR stream is shorter than the decoded frame count,
the remaining frames are extended at the median DVR interval; a shortfall of a
single frame is common and is not reported.

## Container frame rate is not verified

The frame rate written into a video file can differ from the recording clock.
In the manuscript's Farm A recordings the container stated 30 fps while the DVR
timestamps advanced at 25 fps. When a video has no DVR stream, the container
rate is assumed. Check it before analysis, for example by comparing the frame
count divided by the container rate with the start and end times in the file
names of consecutive clips.

## Re-encoded videos

Resizing, trimming or converting a DVR recording with a video tool usually
removes the DVR timestamp stream. The copy then falls back to the file name and
the container frame rate. Analyse the original file instead and, for a lower
analysis resolution, use `--processing-height` (in-pipeline resizing), which
keeps the original timestamps.

If re-encoded copies must be analysed, their frame count is normally unchanged,
so frame *i* of the copy is frame *i* of the original: detected times can be
mapped back onto the original DVR timestamps by frame index.

## Hour buckets and clip joins

Clips are grouped by the hour in which they start and analysed per hour as one
continuous signal. The previous-frame state is carried into the next clip only
when it starts within `CLIP_JOIN_MAX_GAP_SECONDS` (2 s) of the previous clip's
last frame. Windows are counted in frames, so recording gaps inside a clip make
a window span slightly more clock time than its nominal length; use
`Bout End − Bout Start` for bout durations.
