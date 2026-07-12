# SAILS Action Classification

This repository contains action classification and statistical analysis for
locomotion and repetitive motor movement behaviors in the SAILS video dataset.

The project is organized as a research pipeline:

1. Detect people and estimate whole-body pose from videos.
2. Track people across frames and identify the target child.
3. Build frame-level or window-level action datasets.
4. Train and evaluate action-recognition models.
5. Fuse predictions from complementary models.
6. Extract behavior-specific movement features and run downstream analyses.


## Repository Layout

```text
.
+-- src/
|   +-- sailsprep/
|   |   +-- action_model_testing/        # Action-recognition model experiments
|   |   +-- analysis/                    # Behavior-specific feature/statistical analyses
|   |   +-- annotation/                  # FastAPI annotation tool
|   |   +-- fusion_model/                # Prediction fusion and PySKL/V-JEPA utilities
|   |   +-- id_tracking_model/           # Pose cache, person tracking, child ID pipeline
|   |   +-- tracking_pose_model_testing/ # Pose/tracking model wrappers and tests
|   +-- tests/                           # Unit tests mirroring the package structure
+-- jobs/                                # SLURM job scripts for training/evaluation/analysis
+-- docs_style/                          # pdoc documentation theme files
+-- logs/                                # Runtime log directory placeholder
+-- pyproject.toml                       # Poetry package and dependency configuration
+-- poetry.lock                          # Locked dependency versions
```

## Main Components

### Pose, Tracking, and Target Child Identification

Code: `src/sailsprep/id_tracking_model/`

This part of the repository generates detection and pose caches, tracks people
across frames, and selects the target child track.

Important modules:

- `pose/cache_pose.py` - single-video detection and pose cache pipeline.
- `pose/batch_pose.py` - batch pose-cache processing from a CSV.
- `tracker/person_tracker.py` - person tracking utilities.
- `tracker/batch_tracker.py` - batch tracking over a video CSV.
- `target_id/batch_identify_target.py` - target-child identification from tracking outputs.
- `target_id/child_id/` - single-child and batch child-identification utilities.
- `visualize_tracking_from_child_id.py` - render child-ID tracking visualizations.

### Pose and Tracking Model Experiments

Code: `src/sailsprep/tracking_pose_model_testing/`

This directory contains scripts for testing or wrapping individual tracking and
pose-estimation methods:

- YOLO pose
- HRNet
- ViTPose
- DeepSORT
- DeepSORT + ReID
- ByteTrack
- EntitySAM
- MediaPipe holistic and face pipelines

Several scripts include post-processing utilities for bounding boxes, keypoints,
and visualized outputs.

### Action Model Experiments

Code: `src/sailsprep/action_model_testing/`

This directory contains the model-training and inference scripts used for action
classification experiments.

Included model families:

- `video_swin/` - clip-based, binary sliding-window, full-video sliding-window,
  and two-stage Video Swin pipelines.
- `videomae2/` - VideoMAE v2 finetuning and sliding-window variants.
- `internvideo2/` - InternVideo2 finetuning and inference.
- `slow_fast/` - SlowFast finetuning.
- `motionbert/` - MotionBERT pose/action pipeline.
- `mstcn2/` - MS-TCN++ sequence model over extracted features.
- `open_tad/` - temporal action detection experiments.
- `pyskl/` - PySKL config/logit utilities.
- `vlm_models/` - Qwen/Ovis vision-language model classifiers.
- `feature_extraction/` - I3D/R(2+1)D and V-JEPA feature extraction.
- `vjepa/` - vjepa model variants.
- `dlc_action/` - DLC2Action data preparation workflow.

### Fusion Models

Code: `src/sailsprep/fusion_model/`

Fusion scripts combine predictions from multiple model families.

- `late_fusion/two_model.py` - late fusion between two prediction sources.
- `late_fusion/three_model.py` - late fusion among three prediction sources.
- `pyskl/build_pyskl_sw_pkl.py` - build sliding-window PySKL datasets and configs.
- `pyskl/eval_pyskl_sw.py` - evaluate PySKL sliding-window predictions.
- `vjepa/vjepa_sw.py` - train/evaluate a V-JEPA feature-based window classifier.

### Statistical Analysis

Code: `src/sailsprep/analysis/`

Behavior-specific analysis scripts extract kinematic features from pose tracks
and run statistical analyses. Behaviors include:

- walking
- running
- jumping
- crawling
- cruising
- rocking
- spinning
- hand flapping
- combined locomotion
- combined repetitive motor movements

These scripts compute movement features and run statistical tests such as mixed effects models, GEE,
cluster-robust analyses, permutation tests, bootstrap procedures, and
leave-one-subject-out classification.

### Annotation Tool

Code: `src/sailsprep/annotation/`

The annotation tool is a FastAPI application with a browser UI for reviewing
videos and assigning frame-level behavior labels.

```bash
uvicorn sailsprep.annotation.annotation:app --reload
```

Before running it, update the paths at the top of
`src/sailsprep/annotation/annotation.py`:

- `VIDEO_DIR`
- `CSV_FILE`
- `ANNOTATION_DIR`
- `OUTPUT_DIR`

## Installation

This project uses Poetry and requires Python 3.11 or 3.12.

```bash
git clone https://github.com/sensein/sails_action_classification.git
cd sails_action_classification

poetry install --with dev
poetry run pytest
```

Many model pipelines require optional heavy dependencies. Install only the
groups needed for the experiment you want to reproduce.

Examples:

```bash
# Pose/tracking experiments
poetry install --with dev,pose-estimation,tracking

# Video Swin experiments
poetry install --with dev,video_swin

# VideoMAE / feature extraction style experiments
poetry install --with dev,feature-extraction

# Vision-language model experiments
poetry install --with dev,vlm

# OpenTAD experiments
poetry install --with dev,opentad

# Documentation
poetry install --with docs
```

Some third-party model stacks, especially PySKL, OpenTAD, MMDetection, MMPose,
and CUDA-specific video libraries, may require separate environment setup on an
HPC system. The job scripts in `jobs/` show the environments used for the
original experiments.

## Data Requirements

The SAILS videos and derived data are not included in this repository. To
reproduce the full results, prepare the following inputs:

1. Raw or standardized SAILS videos (this repo used videos from bids preprocessed folder).
2. A split CSV with train/validation/test assignments.
3. Per-video or per-session annotation CSVs with frame-level locomotion and RMM
   labels.
4. Detection/pose model checkpoints for the selected pose pipeline.
5. Optional precomputed features, such as V-JEPA, I3D, or R(2+1)D features.
6. Optional pretrained model checkpoints for Video Swin, VideoMAE2, InternVideo,
   SlowFast, MotionBERT, OpenTAD, or VLM models etc.

Several scripts currently contain absolute paths from the original compute
environment, for example `/orcd/...` and `/home/aparnabg/orcd/...`. Before
reproducing results on a different machine, update those constants or wrap the
scripts with your local paths.

Common paths to update include:

- CSV paths
- raw video directories
- annotation directories
- model checkpoint directories
- output directories
- cache directories
- Hugging Face cache directories

## Expected CSV Format

Different scripts use slightly different subsets of columns, but the common
split CSV is expected to provide:

- a video path or video identifier
- a label CSV path or annotation path
- a `split` column with values such as `train`, `val`, and `test`
- locomotion labels, commonly under `Locomotion`
- repetitive motor movement labels, commonly under
  `Repetitive_Motor_Movements`

Tracking and child-identification scripts may also use participant IDs,
session/timepoint fields, or metadata columns needed to locate videos and
identify target children.

## Reproducing the Pipeline

The exact command sequence depends on which experiment you are reproducing.
The sections below describe the standard order.

### 1. Generate Pose Caches

Run batch pose estimation over a video CSV:

```bash
python -m sailsprep.id_tracking_model.pose.batch_pose \
  /path/to/videos.csv \
  --video-dir /path/to/videos \
  --output-dir /path/to/outputs/pose_cache \
  --cache-dir /path/to/cache \
  --exp-id pose_run_001
```

The output is a cache of detections and pose estimates that downstream tracking
and analysis scripts can reuse.

### 2. Track People

Run batch tracking:

```bash
python -m sailsprep.id_tracking_model.tracker.batch_tracker \
  /path/to/videos.csv \
  --video-dir /path/to/standardized_videos \
  --output-dir /path/to/pipeline_outputs \
  --cache-dir /path/to/cache \
  --exp-id tracking_run_001
```

Useful options:

- `--ids` to process selected IDs.
- `--start-row` and `--end-row` to process a CSV slice.
- `--no-visualization` to skip rendered tracking videos.
- `--no-reuse-pipeline` to force regeneration.

### 3. Identify the Target Child

Run child/target identification from tracking results:

```bash
python -m sailsprep.id_tracking_model.target_id.batch_identify_target \
  /path/to/video_metadata.csv \
  --embeddings-dir /path/to/tracking_or_embedding_outputs \
  --video-dir /path/to/standardized_videos \
  --output-dir /path/to/target_identification_results \
  --render
```

For the newer single-child identification workflow:

```bash
python -m sailsprep.id_tracking_model.target_id.child_id.batch_child_identification \
  /path/to/pipeline_output_subfolder \
  --output-dir /path/to/pipeline_outputs \
  --workers 4
```

### 4. Build Action Datasets

For PySKL sliding-window experiments:

```bash
python -m sailsprep.fusion_model.pyskl.build_pyskl_sw_pkl \
  --task locomotion

python -m sailsprep.fusion_model.pyskl.build_pyskl_sw_pkl \
  --task rmm
```

Use `--configs_only` if the PKL dataset already exists and only PySKL configs
need to be generated.

### 5. Train Action Models

Representative examples:

```bash
# Video Swin full-video sliding-window classifier
python -m sailsprep.action_model_testing.video_swin.sliding_window.video_swin_fullvideo_sliding \
  --task loco \
  --seed 42

# Video Swin binary N/A vs non-N/A classifier
python -m sailsprep.action_model_testing.video_swin.sliding_window.video_swin_binary_sliding \
  --task rmm \
  --seed 42

# Video Swin two-stage classifier
python -m sailsprep.action_model_testing.video_swin.sliding_window.video_swin_twostage_joint \
  --task loco \
  --seed 42

# VideoMAE2 full-video sliding-window classifier
python -m sailsprep.action_model_testing.videomae2.videomae2_fullvideo_sliding \
  --task loco \
  --seed 42

# VideoMAE2 two-stage classifier
python -m sailsprep.action_model_testing.videomae2.videomae2_twostage_sliding \
  --task rmm \
  --seed 42

# MS-TCN++ over extracted features
python -m sailsprep.action_model_testing.mstcn2.mstcn2 \
  --label loco \
  --feature_type i3d \
  --action train \
  --seed 42
```

Most reported model experiments are run over three seeds:

```text
42, 123, 456
```

### 6. Run PySKL Training and Evaluation

The PySKL workflow expects a separate PySKL checkout/environment. The provided
SLURM scripts show the original setup:

```bash
sbatch jobs/fusion_model/pyskl/train_pyskl_sw.sh posec3d locomotion
sbatch jobs/fusion_model/pyskl/train_pyskl_sw.sh ctrgcn_b locomotion
sbatch jobs/fusion_model/pyskl/train_pyskl_sw.sh stgcnpp_b locomotion
sbatch jobs/fusion_model/pyskl/train_pyskl_sw.sh posec3d rmm
sbatch jobs/fusion_model/pyskl/train_pyskl_sw.sh stgcnpp_jm rmm
```

Evaluate trained PySKL models:

```bash
bash jobs/fusion_model/pyskl/test_pyskl_sw.sh
```

### 7. Train V-JEPA Window Classifier

```bash
python -m sailsprep.fusion_model.vjepa.vjepa_sw \
  --task locomotion \
  --seed 42

python -m sailsprep.fusion_model.vjepa.vjepa_sw \
  --task rmm \
  --seed 42
```

### 8. Fuse Model Predictions

Two-model late fusion:

```bash
python -m sailsprep.fusion_model.late_fusion.two_model \
  --task locomotion \
  --seed 42 \
  --alphas 0.25 0.5 0.75
```

Three-model late fusion:

```bash
python -m sailsprep.fusion_model.late_fusion.three_model \
  --task rmm \
  --seed 42 \
  --alphas 0.25 0.5 0.75 \
  --betas 0.25 0.5 0.75
```

Use `--task both` and omit `--seed` to aggregate across tasks/seeds where
supported by the script.

### 9. Run Statistical Analyses

Behavior analysis scripts are currently standalone scripts with constants at the
top of each file for input CSVs and output folders. Update those paths first,
then run the desired behavior analysis.

Examples:

```bash
python -m sailsprep.analysis.walking.walking
python -m sailsprep.analysis.running.running
python -m sailsprep.analysis.jumping.jumping
python -m sailsprep.analysis.handflapping.handflapping
python -m sailsprep.analysis.loco_combined.loco_combined
python -m sailsprep.analysis.rmm_combined.rmm_combined
```

On SLURM, see:

```bash
sbatch jobs/analysis/analysis_job.sh
```

## SLURM Jobs

The `jobs/` directory contains scripts for HPC runs. These
scripts were written for the original cluster environment and should be edited
before reuse.

Common edits:

- `#SBATCH --partition`
- GPU type and count
- conda environment path/name
- project workspace path
- input/output/log directories
- task labels passed to scripts

Representative job scripts:

- `jobs/action_model_testing/video_swin/sliding_window/video_swin_full_video_multi_job.sh`
- `jobs/action_model_testing/video_swin/sliding_window/video_swin_full_video_multi_job_twostage.sh`
- `jobs/action_model_testing/video_swin/sliding_window/full_video_window_job_video_swin.sh`
- `jobs/action_model_testing/videomae2/videomae2_submit_all.sh`
- `jobs/action_model_testing/mstcn2/mstcn2.sh`
- `jobs/action_model_testing/slow_fast/slow_fast.sh`
- `jobs/action_model_testing/vlm_models/clips/submit_qwen_clip.sh`
- `jobs/action_model_testing/vlm_models/clips/submit_ovis_clip.sh`
- `jobs/fusion_model/pyskl/train_pyskl_sw.sh`
- `jobs/fusion_model/pyskl/test_pyskl_sw.sh`
- `jobs/analysis/analysis_job.sh`

## Outputs

Depending on the pipeline stage, scripts write outputs such as:

- pose and detection caches
- person tracking JSON files
- child identification logs and rendered videos
- action model checkpoints
- per-window predictions
- per-video predictions
- classification reports
- top-k accuracy and mean-class accuracy metrics
- fusion metrics
- clip-level and child-level movement feature CSVs
- statistical result CSVs
- figures and summary plots

## Testing

Run the test suite with:

```bash
poetry run pytest
```

The project config uses:

```text
testpaths = src/tests
python_files = test_*.py
```

For faster iteration on a specific area:

```bash
poetry run pytest src/tests/id_tracking_model
poetry run pytest src/tests/action_model_testing
poetry run pytest src/tests/analysis
```

## Code Quality

Ruff, mypy, pytest, and pre-commit are configured in `pyproject.toml`.

```bash
poetry run ruff check src
poetry run mypy src
poetry run pytest
```

Install pre-commit hooks:

```bash
poetry run pre-commit install
```

## Known Reproducibility Notes

- Many scripts currently use absolute paths from the original ORCD/HPC
  environment.
- Some action-model dependencies require CUDA-compatible versions of PyTorch,
  torchvision, MMDetection, MMPose, PySKL, or OpenTAD.
- The package metadata currently declares a `sailsprep-cli` entry point, but the
  repository does not currently include `src/sailsprep/cli.py`. Use the module
  commands shown above or the scripts in `jobs/`.
- For exact reproduction, run multi-seed experiments with seeds `42`, `123`,
  and `456`, then aggregate metrics using the relevant evaluation or fusion
  scripts.
