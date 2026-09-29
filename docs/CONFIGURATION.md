# Configuration and timing modes

All user-editable paths, ROI presets, algorithm parameters, and output switches
are in `src/config.py`. The detection logic is in
`src/brush_rotation_activity_index.py`; input/timestamp helper functions are in
`src/io_utils.py`.

## Paths

Edit `input_folder` and `base_output_folder` in `src/config.py`. The supplied
configuration retains the local example-folder paths used for testing.

## ROI

Select one of the study presets through `ROI = ROI_ALL[...]`, or define and
validate a new polygon for a new camera position. The supplied coordinates are
camera-specific.

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

Set `REQUIRE_DVR_TIMESTAMPS = True` when only supported DVR metadata should be
accepted. Set `ALLOW_STANDALONE_RELATIVE_TIME = False` to disable the relative
fallback.

## Large example files

Video extensions are ignored by Git. Place a local test recording under
`examples/input/` if desired. Generated annotated videos should remain local or
be supplied separately as manuscript supplementary material.
