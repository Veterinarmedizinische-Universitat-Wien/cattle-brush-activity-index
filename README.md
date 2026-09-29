# Cattle-brush mechanical-rotation activity index

Research code for detecting **mechanical rotation of cattle grooming brushes** from fixed-camera video using a lightweight CPU-based activity index.

The code measures **brush motion**, not direct cow–brush contact and not individual cow identity. Used alone, its output is therefore suitable for brush/resource-level rotation monitoring. Individual-level brush-use monitoring would require a separate animal-identification/contact layer.

## What this release contains

The repository contains the cross-farm analysis version selected for manuscript release. The public script preserves the supplied detection logic and study parameter defaults. The only structural change is that machine-specific input/output paths were replaced with command-line arguments.

Main script:

```text
src/brush_rotation_activity_index.py
```

## Processing sequence

1. Read video and reconstruct timestamps from DVR metadata when available, with filename timestamps as a fallback.
2. Apply the predefined brush ROI and image preprocessing.
3. Calculate frame-to-frame absolute pixel differences and sum them within the ROI.
4. Apply log transformation and 5th–95th percentile normalization.
5. Smooth the normalized signal with a 5-frame moving average.
6. Build a separate threshold-only signal by suppressing isolated short high-amplitude spikes.
7. Divide the signal into 10-min segments and use the 99.5th percentile as the segment reference.
8. Apply the study-specific dynamic multiplier (0.50 for reference values >= 0.90; 0.62 otherwise).
9. Detect candidate rotation intervals on the **original smoothed normalized signal**.
10. Apply temporal and contextual post-processing and export the final mechanical-rotation intervals.

See [`docs/METHODS_TO_CODE_MAP.md`](docs/METHODS_TO_CODE_MAP.md) for the function-level mapping.

## Requirements

The manuscript implementation used Python 3.12.4. The supplied code requires:

- OpenCV (`opencv-python`)
- NumPy
- pandas
- Matplotlib
- tqdm
- openpyxl (for Excel output)
- FFmpeg/ffprobe as a system dependency for DVR metadata extraction

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Install FFmpeg separately using the package manager appropriate for your operating system, then confirm that both `ffmpeg` and `ffprobe` are available on the command line.

## Running the analysis

Example:

```bash
python src/brush_rotation_activity_index.py \
  --input /path/to/input_videos \
  --output /path/to/results \
  --roi-preset E
```

On Windows PowerShell, the same command can be entered on one line:

```powershell
python src/brush_rotation_activity_index.py --input "C:\path\to\videos" --output "C:\path\to\results" --roi-preset E
```

Show all options:

```bash
python src/brush_rotation_activity_index.py --help
```

### Important command-line options

- `--roi-preset`: selects one of the study ROI presets defined in the script.
- `--require-dvr-timestamps`: excludes files for which DVR timestamps cannot be extracted.
- `--no-annotated-video`: disables annotated video export.
- `--no-hour-plots`: disables hourly signal plots.
- `--show-roi-preview`: displays the ROI preview during processing.
- `--no-roi-preview`: disables saved ROI preview images.

## ROI configuration

The included ROI coordinates are **study/camera-specific** and should not be assumed to transfer to a new camera position. For a new deployment, define a polygon that encloses the full brush movement while minimizing unrelated background motion.

The study presets are stored in `ROI_ALL` near the beginning of the main script. For exact reproduction of a study recording, select the preset corresponding to that recording. For a new setup, add a new ROI polygon and validate it using the saved preview image before running a long analysis.

## Timestamp handling

The script first attempts to recover DVR timestamps from non-video streams using FFmpeg/ffprobe. If DVR timestamps are unavailable, it attempts to infer the recording start time from the filename and constructs timestamps from the reported frame rate. Set `--require-dvr-timestamps` when fallback timestamps should not be accepted.

The script also records coarse diagnostics for dropped-frame-like gaps, duplicate-frame-like intervals, clock jumps, and timing jitter. These diagnostics are intended as acquisition-quality checks; they do not replace manual verification of problematic recordings.

## Main study parameter defaults

| Parameter | Default |
|---|---:|
| Low-light CLAHE | enabled when mean frame intensity < 80 |
| CLAHE clip limit | 2.0 |
| CLAHE tile grid | 8 x 8 |
| Previous-frame blend alpha | 0.6 |
| Normalization percentiles | 5th and 95th |
| Log compression | enabled (`log1p`) |
| Smoothing window | 5 frames |
| Threshold segment length | 10 min |
| Threshold reference percentile | 99.5 |
| High-/low-motion switch | 0.90 |
| Multiplier when reference >= 0.90 | 0.50 |
| Multiplier otherwise | 0.62 |
| Threshold-spike high quantile | 99th percentile |
| Threshold-spike duration criterion | <10 s |
| Threshold-spike isolation criterion | >5 s |
| Initial minimum candidate duration | >4 s |
| Merge gap | <=2 s |
| Low movement threshold | 0.26 |
| Contextual high threshold | 0.40 |
| Context window | 4 min |
| Context maximum threshold | 0.45 |
| Short-bout evaluation window | 40 s |
| Short-bout criterion | <4 s |
| Maximum retained bout duration | 2000 s |

These are the defaults implemented in the supplied analysis version. They should not be presented as universally optimal values for other camera/brush configurations without external validation.

## Outputs

For each run, the script can generate:

- `rotation_summary.xlsx` — detected rotation intervals and durations
- `hour_bucket_summary.xlsx` — hourly signal/threshold summary
- `file_timestamp_report.xlsx` — timestamp source and video metadata summary
- `timestamp_gap_report.csv` — timestamp anomalies, when detected
- `plots/` — hourly normalized motion/threshold plots
- `previews/` — ROI preview images
- `annotated_videos/` — optional videos showing ROI, signal/threshold information, time, and detected rotation periods

## Reproducibility notes

The repository intentionally retains the preprocessing and post-processing steps implemented in the supplied study code. These include low-light CLAHE, LAB/intensity preprocessing, frame blending, threshold-only spike suppression, amplitude/context filters, short-bout filtering, gap merging, and a maximum-duration filter.

Before manuscript submission or public release, confirm that the Methods describe all algorithmic steps that materially affect the final rotation intervals. The repository should be tagged at the exact commit used for the final reported results.

## Data

Raw farm video is not included in this repository. If example data can be shared, add a small de-identified sample or a derived signal file that allows users to verify installation and reproduce one example output without redistributing the full study recordings.

## Citation

A `CITATION.cff` file is included. Update it with the final article title, DOI, repository URL, and/or archived software DOI once available.

## License

No software license has been assigned in this draft repository. Choose and add an appropriate license before making the repository public.
