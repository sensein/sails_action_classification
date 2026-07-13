# dlc_action

Locomotion classification (`Walking`, `Running`, `Crawling`, etc.) from pose
sequences using [DLC2Action](https://github.com/AlexEMG/DLC2Action) with an
MS-TCN model over pre-extracted pose keypoints.

## File

```
run.py   full pipeline: match files -> analyze labels -> prepare data -> split -> create DLC2Action project -> train -> evaluate
```

## Data

Reads a split CSV (`LABEL_MAPPING_CSV` constant near the top of the file)
with columns `video_path`, `hrnet_full_path` (per-frame HRNet pose JSON, 133
keypoints x, y, confidence), `label_path` (frame-level `Locomotion` labels),
and `split`. Frame-level labels are converted into start/end segment CSVs
that DLC2Action's `annotation_type="csv"` project expects; pose keypoints are
flattened to `(F, 133*3)` and saved as `.pt` feature files.

## Usage

```bash
poetry run python src/sailsprep/action_model_testing/dlc_action/run.py
```

There are no CLI flags — all configuration (`LABEL_MAPPING_CSV`,
`PROCESSED_DIR`, `PROJECT_DIR`, `TARGET_COLUMN`, `MODEL_NAME`, epoch/batch/lr
constants) is set at the top of `run.py`. The script deletes and rebuilds
`PROCESSED_DIR` and `PROJECT_DIR` on every run.

## What it does

1. Matches videos, pose JSONs, and label CSVs from the split CSV.
2. Scans all label CSVs to build the set of locomotion classes present.
3. Converts each video's pose JSON + frame-level labels into a flattened
   `.pt` feature file and a segment-format label CSV, discarding files with
   no actual locomotion labels.
4. Splits files into train/val/test using the `split` column.
5. Creates a DLC2Action `Project` (`data_type="features"`,
   `annotation_type="csv"`, model `ms_tcn3`) and runs one training episode.
6. Evaluates the trained episode and prints F1/precision/recall.

Requires the `dlc2action` package (installed via the `dlc2action` Poetry
group, which pulls it from
[amathislab/DLC2action](https://github.com/amathislab/DLC2action) on GitHub).
