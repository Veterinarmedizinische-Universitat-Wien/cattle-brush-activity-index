# Methods-to-code map

This document maps the manuscript-level processing stages to the functions in `src/brush_rotation_activity_index.py`.

| Manuscript concept | Code implementation |
|---|---|
| Video metadata / timestamp acquisition | `ffprobe_json`, `extract_dvr_metadata_from_stream`, `extract_dvr_metadata`, `get_video_timestamps_and_ids` |
| Timestamp QC | `diagnose_timestamps`, `detect_timestamp_gaps` |
| Hour-level grouping | `build_hour_buckets`, `build_continuous_hour_signal` |
| Low-light enhancement | `enhance_low_light` |
| ROI masking / intensity preprocessing | `get_roi_from_polygon_legacy`, `preprocess_frame_for_white_legacy` |
| Previous-frame blending | `blend_with_previous` |
| Raw frame-difference motion score | `compute_motion_score_sum_abs_diff` |
| Log + percentile normalization | `robust_log_normalize` |
| 5-frame smoothing | `moving_average_with_tail` |
| Threshold-only spike suppression | `suppress_isolated_short_spikes` |
| 10-min segment threshold | `compute_segment_threshold_curve` |
| Candidate threshold crossing / duration criterion | `detect_periods_from_threshold_frame_based` |
| Merge short gaps | `merge_close_periods` |
| Low-amplitude bout filter | `filter_low_movement_bouts` |
| Contextual neighbouring-bout filter | `filter_contextual_false_positives_with_edges` |
| Clustered short-bout filter | `evaluate_window`, `is_within_threshold` |
| Surrounding-window movement filter | `filter_low_movement_within_window` |
| Final detector orchestration | `detect_rotation_periods_for_hour_legacy` |
| Hour processing / exports | `process_hour_bucket` |

## Important implementation details to match in the manuscript

### Frame preprocessing

The supplied implementation is not raw frame differencing alone. Before the motion score is calculated it can:

1. apply CLAHE to dark frames;
2. mask the predefined ROI;
3. convert the ROI to LAB space and normalize the L channel;
4. apply an inverse binary threshold at 220;
5. convert the resulting processed frame to grayscale;
6. apply the ROI mask again; and
7. blend the current processed ROI with the previous ROI using alpha = 0.6.

The motion score is then the sum of absolute pixel differences between the current blended ROI and the preceding ROI.

### Threshold estimation versus final detection

The spike-suppressed signal is used **only for threshold estimation**. Final threshold crossing is evaluated on the original smoothed normalized signal.

### Post-processing

The final code applies more than the 2-s merge rule. It also includes normalized-amplitude/context filters, a short-bout window rule, a 4-min surrounding-window rule, and a maximum-duration rule. If this repository is the version associated with the paper, the manuscript should describe these operations at a reproducible level.

### Timing

The script may use DVR timestamps or filename-derived timestamps, but time-based filters are converted to frames using the FPS value used for filtering. This should be described accurately when documenting variable-frame-rate recordings.
