# id_tracking_model

Pose/detection caching, multi-person tracking, and target-child
identification. This is the pipeline that turns raw SAILS videos into
per-child tracks before any action-recognition model runs.

## Layout

```
id_tracking_model/
  pose/
    cache_pose.py                    single-video detection + pose pipeline (DetectionPosePipeline)
    batch_pose.py                    batch-runs cache_pose.py's pipeline over a video CSV
  tracker/
    person_tracker.py                PersonTracker — IoU/motion/ReID-based multi-person tracker
    batch_tracker.py                 batch-runs tracking over a video CSV
    clip/tracker_clip.py             CLIP-ReID-based tracking pipeline variant (MultiPersonTrackingPipeline)
  target_id/
    batch_identify_target.py         TargetIdentifier — batch target-child identification from tracking output
    child_id/
      single_child_identification.py  SingleChildIdentifier — identifies the target child in single-child videos
      single_child_id_api.py          high-level wrapper around single_child_identification (tracking JSON in, annotated video + result dict out)
      single_child_track_selector.py  loads tracks from an H5 tracking export and selects the single relevant track
      batch_child_identification.py   ChildIdentificationProcessor — batch-runs single-child identification over a folder
      analyze_batch_results.py        summarizes batch_child_identification.py's output JSONs into plots/reports
  utils/
    cache_manager.py                  CacheManager — pose/detection cache read/write
    cache_metadata.py                 CacheMetadata / CacheMetadataManager — cache versioning and metadata
    tracking_exporter.py              TrackingDataExporter / TrackingDataCollector — JSON/HDF5 export of tracking results
    utils.py                          soft_nms, oks_iou, oks_nms — detection/keypoint NMS helpers
  visualize_tracking_from_child_id.py rendering tool for visualizing a resolved child-ID track
```

## Pipeline order

1. **Pose caching** (`pose/`) — detect people and estimate whole-body pose
   per frame, cached to disk.
2. **Tracking** (`tracker/`) — link detections into per-person tracks across
   frames.
3. **Target-child identification** (`target_id/`) — from the tracks, select
   the track(s) belonging to the target child.

## Pose caching

```bash
python -m sailsprep.id_tracking_model.pose.batch_pose \
  /path/to/videos.csv \
  --video-dir /path/to/videos \
  --output-dir /path/to/outputs/pose_cache \
  --cache-dir /path/to/cache \
  --exp-id pose_run_001
```

`csv_file` (positional) is a CSV with a `video_path` column. Options:
`--output-dir`, `--video-dir`, `--cache-dir`, `--exp-id`,
`--no-reuse-pipeline` (build a fresh detection/pose pipeline per video
instead of reusing one), `--rmm` (RMM dataset path conversion, on by
default), `--start-row`/`--end-row` (process a CSV slice).

`cache_pose.py` can also be run directly on a single video/CSV via its
`main()` (path constants at the bottom of the file); `batch_pose.py` is the
CLI entry point used for full runs.

## Tracking

```bash
python -m sailsprep.id_tracking_model.tracker.batch_tracker \
  /path/to/videos.csv \
  --video-dir /path/to/standardized_videos \
  --output-dir /path/to/pipeline_outputs \
  --cache-dir /path/to/cache \
  --exp-id tracking_run_001
```

Options: `--ids` (process only selected video IDs), `--start-row`/`--end-row`,
`--no-visualization` (skip rendered tracking videos), `--no-reuse-pipeline`,
`--rmm`.

`tracker/clip/tracker_clip.py` is a separate CLIP-ReID-based tracking
pipeline (`MultiPersonTrackingPipeline`) with its own OKS-NMS, motion, and
ReID threshold configuration; it is a library module, not a CLI script.

## Target-child identification

Batch identification from tracking/embedding output:

```bash
python -m sailsprep.id_tracking_model.target_id.batch_identify_target \
  /path/to/video_metadata.csv \
  --embeddings-dir /path/to/tracking_or_embedding_outputs \
  --video-dir /path/to/standardized_videos \
  --output-dir /path/to/target_identification_results \
  --render
```

`csv_file` (positional) plus `--embeddings-dir` (required), `--output-dir`,
`--ids`, `--render` (render annotated output videos), `--video-dir`, `--rmm`,
`--face-only`, `--min-score`.

Newer single-child identification workflow, for videos expected to contain
one target child:

```bash
python -m sailsprep.id_tracking_model.target_id.child_id.batch_child_identification \
  /path/to/pipeline_output_subfolder \
  --output-dir /path/to/pipeline_outputs \
  --workers 4
```

`target_folder` (positional, a subfolder within the pipeline outputs) plus
`--output-dir`, `--test` (process only the first 3 files), `--max-files`,
`--workers` (parallel workers; sequential if omitted), `--aggressive`
(sparser 5%-of-frames sampling, max 8 frames), `--no-skip` (reprocess files
that already have output).

Summarize a completed batch run:

```bash
python -m sailsprep.id_tracking_model.target_id.child_id.analyze_batch_results
```

Reads JSON analysis files from a `LOG_DIR` constant at the top of the file
and writes `batch_analysis_plots.png` alongside them.

## Visualization

```bash
python -m sailsprep.id_tracking_model.visualize_tracking_from_child_id \
  /path/to/child_id_video_or_log \
  --max-frames 1000 \
  --output /path/to/output.mp4
```
