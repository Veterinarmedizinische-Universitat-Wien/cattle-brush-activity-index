# Methods-to-code map

Where each step of the method is implemented. Parameters are in
`src/config.py`; timing behaviour is described in [`TIMING.md`](TIMING.md).

| Manuscript concept | Code implementation | File |
|---|---|---|
| Paths, ROI presets, study parameter defaults | configuration constants | `src/config.py` |
| DVR/filename/relative timestamp acquisition | `get_video_timestamps_and_ids`, `get_clip_start_info` | `src/io_utils.py` |
| DVR metadata extraction | `ffprobe_json`, `extract_dvr_metadata_from_stream`, `extract_dvr_metadata` | `src/io_utils.py` |
| Filename timestamp parsing | `parse_timestamp_from_filename` | `src/io_utils.py` |
| Relative-time formatting | `format_relative_timestamp` | `src/io_utils.py` |
| Hour-level grouping / standalone separation | `build_hour_buckets` | `src/brush_rotation_activity_index.py` |
| Timestamp QC | `diagnose_timestamps`, `detect_timestamp_gaps` | `src/brush_rotation_activity_index.py` |
| Low-light enhancement | `enhance_low_light` | `src/brush_rotation_activity_index.py` |
| ROI scaling to processed frame | `scale_polygon`, `prepare_analysis_frame` | `src/brush_rotation_activity_index.py` |
| Clip-join contiguity check | `clips_are_contiguous` | `src/brush_rotation_activity_index.py` |
| ROI masking / intensity preprocessing | `get_roi_from_polygon`, `preprocess_frame_for_white` | `src/brush_rotation_activity_index.py` |
| Previous-frame blending | `blend_with_previous` | `src/brush_rotation_activity_index.py` |
| Raw frame-difference motion score | `compute_motion_score_sum_abs_diff` | `src/brush_rotation_activity_index.py` |
| Log + percentile normalization | `robust_log_normalize` | `src/brush_rotation_activity_index.py` |
| 5-frame smoothing | `moving_average_with_tail` | `src/brush_rotation_activity_index.py` |
| Threshold-only spike suppression | `suppress_isolated_short_spikes` | `src/brush_rotation_activity_index.py` |
| 10-min segment threshold | `compute_segment_threshold_curve` | `src/brush_rotation_activity_index.py` |
| Candidate threshold crossing / duration criterion | `detect_periods_from_threshold_frame_based` | `src/brush_rotation_activity_index.py` |
| Merge short gaps | `merge_close_periods` | `src/brush_rotation_activity_index.py` |
| Low-amplitude bout filter | `filter_low_movement_bouts` | `src/brush_rotation_activity_index.py` |
| Contextual neighbouring-bout filter | `filter_contextual_false_positives_with_edges` | `src/brush_rotation_activity_index.py` |
| Surrounding-window movement filter | `filter_low_movement_within_window` | `src/brush_rotation_activity_index.py` |
| Final detector orchestration | `detect_rotation_periods_for_hour` | `src/brush_rotation_activity_index.py` |
| Continuous hourly signal extraction | `build_continuous_hour_signal` | `src/brush_rotation_activity_index.py` |
| Absolute/relative output orchestration | `run`, `process_hour_bucket`, `process_standalone_relative_video`, `write_reports` | `src/brush_rotation_activity_index.py` |
| Command-line interface | `build_arg_parser`, `context_from_args`, `main` | `src/brush_rotation_activity_index.py` |

## Implementation details

### Frame preprocessing

Before the motion score is calculated, the implementation can apply CLAHE to dark
frames, mask the predefined ROI, convert the ROI to LAB space and normalize the
L channel, apply an inverse binary threshold at 220, convert the processed frame
to grayscale, apply the ROI mask again, and blend the current processed ROI with
the previous ROI using alpha = 0.6. The motion score is the sum of absolute pixel
differences between the current blended ROI and the preceding ROI.

### Normalisation references

The 5th/95th percentile references are computed over frames with a real motion
score only. The first frame of a continuous run (no previous frame to compare)
and frames that failed to process are placeholders: they are kept in the signal
(normalised to 0) but excluded from the percentiles.

### Threshold estimation versus final detection

The spike-suppressed signal is used only for threshold estimation. Final threshold
crossing is evaluated on the original smoothed normalized signal.

### Post-processing

The final detector includes the 2-s merge rule, normalized-amplitude/context
filters, a 4-min surrounding-window rule, and a
maximum-duration rule.

### Timing

See [`TIMING.md`](TIMING.md).
