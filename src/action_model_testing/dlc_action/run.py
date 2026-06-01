"""DLC2Action pipeline for locomotion behavior classification.

This module implements a full training pipeline for classifying locomotion
behaviors (Walking, Running, Crawling, etc.) from pose estimation data using
the DLC2Action framework with an MS-TCN model.

Example:
    Run the full pipeline on the ORCD cluster::

        $ poetry run python src/sailsprep/action_model_testing/dlc_action/run.py

    Or call individual steps programmatically::

        from sailsprep.action_model_testing.dlc_action.run import match_files
        matched = match_files("/path/to/mapping.csv")
"""

import os
import json
import csv
import shutil
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from dlc2action.project import Project


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

LABEL_MAPPING_CSV = "/home/aparnabg/orcd/scratch/Automatic_Labeling/bids_to_json_mapping.csv"
PROCESSED_DIR     = "/home/aparnabg/orcd/scratch/dlc2action_run/processed_data_full"
PROJECT_DIR       = "/home/aparnabg/orcd/scratch/dlc2action_run/dlc2action_projec_full"

TARGET_COLUMN = "Locomotion"
TRAIN_RATIO   = 0.7
RANDOM_SEED   = 42

MODEL_NAME    = "ms_tcn3"

NUM_EPOCHS    = 100
BATCH_SIZE    = 8
LEARNING_RATE = 0.0001

NUM_KEYPOINTS  = 133
COORDS_PER_KPT = 3
FEATURE_DIM    = NUM_KEYPOINTS * COORDS_PER_KPT  # 399


# ---------------------------------------------------------------------------
# STEP 1: MATCH FILES FROM CSV
# ---------------------------------------------------------------------------

def match_files(csv_path: str) -> list[dict]:
    """Match video, pose, and label files from a mapping CSV.

    Reads a CSV that maps each processed video to its corresponding pose JSON
    and label CSV, and returns a list of matched file dictionaries.

    Args:
        csv_path: Path to the mapping CSV file. Must contain columns
            ``BidsProcessed``, ``JsonPath``, and ``LabelPath``.

    Returns:
        A list of dicts, each with keys:
            - ``name`` (str): Stem of the video filename.
            - ``video`` (str): Full path to the processed video.
            - ``label`` (str): Full path to the label CSV.
            - ``pose`` (str): Full path to the pose JSON.

    Raises:
        ValueError: If required columns are missing from the CSV.

    Example:
        >>> matched = match_files("/data/mapping.csv")
        >>> matched[0].keys()
        dict_keys(['name', 'video', 'label', 'pose'])
    """
    matched = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"BidsProcessed", "JsonPath", "LabelPath"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV must contain columns: {required}")
        for row in reader:
            name = Path(row["BidsProcessed"]).stem
            matched.append({
                "name":  name,
                "video": row["BidsProcessed"],
                "label": row["LabelPath"],
                "pose":  row["JsonPath"],
            })
    print(f"[match_files] Matched {len(matched)} files")
    return matched


# ---------------------------------------------------------------------------
# STEP 2: ANALYZE LABELS
# ---------------------------------------------------------------------------

def analyze_labels(matched_files: list[dict], target_column: str) -> list[str]:
    """Collect all unique behavior class names across label files.

    Args:
        matched_files: List of file dicts as returned by :func:`match_files`.
            Each dict must have a ``label`` key pointing to a CSV file.
        target_column: Name of the column in each label CSV that contains
            behavior annotations (e.g. ``"Locomotion"``).

    Returns:
        Sorted list of unique, non-null behavior class name strings found
        across all label files. Files missing the target column are skipped.

    Example:
        >>> classes = analyze_labels(matched, "Locomotion")
        >>> classes
        ['Crawling', 'Running', 'Walking']
    """
    all_actions: set[str] = set()
    for item in matched_files:
        df = pd.read_csv(item["label"])
        if target_column in df.columns:
            all_actions.update(df[target_column].dropna().unique())
    action_classes = sorted(list(all_actions))
    print(f"[analyze_labels] Classes found: {action_classes}")
    return action_classes


# ---------------------------------------------------------------------------
# STEP 3: LOAD POSE
# ---------------------------------------------------------------------------

def load_pose_from_json(json_path: str) -> np.ndarray:
    """Load per-frame pose keypoints from a JSON file into a NumPy array.

    Reads a pose JSON file where each frame contains a list of detected people
    with keypoints in ``[x, y, confidence]`` format. Person with ``person_id``
    0 is preferred; if absent, the first available person is used. Frames with
    no detected person are filled with zeros.

    Args:
        json_path: Path to the pose JSON file. Expected structure::

            {
                "frames": [
                    {
                        "people": [
                            {
                                "person_id": 0,
                                "keypoints": [[x, y, conf], ...]
                            }
                        ]
                    },
                    ...
                ]
            }

    Returns:
        Float32 NumPy array of shape ``(F, NUM_KEYPOINTS, COORDS_PER_KPT)``
        where ``F`` is the number of frames.

    Raises:
        AssertionError: If the resulting array has unexpected spatial dimensions.

    Example:
        >>> pose = load_pose_from_json("/data/sub-01_pose.json")
        >>> pose.shape
        (300, 133, 3)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data["frames"]
    pose_data = []

    for frame in frames:
        people = frame.get("people", [])
        person = next((p for p in people if p.get("person_id") == 0), None)
        if person is None and people:
            person = people[0]

        if person is None or "keypoints" not in person:
            pose_data.append(np.zeros((NUM_KEYPOINTS, COORDS_PER_KPT), dtype=np.float32))
            continue

        kpts = person["keypoints"]
        arr = []
        for kpt in kpts:
            if isinstance(kpt, (list, tuple)) and len(kpt) >= 2:
                x = float(kpt[0])
                y = float(kpt[1])
                c = float(kpt[2]) if len(kpt) > 2 else 1.0
            else:
                x, y, c = 0.0, 0.0, 0.0
            arr.append([x, y, c])

        arr = np.array(arr, dtype=np.float32)

        if len(arr) < NUM_KEYPOINTS:
            pad = np.zeros((NUM_KEYPOINTS - len(arr), COORDS_PER_KPT), dtype=np.float32)
            arr = np.vstack([arr, pad])
        elif len(arr) > NUM_KEYPOINTS:
            arr = arr[:NUM_KEYPOINTS]

        pose_data.append(arr)

    result = np.array(pose_data, dtype=np.float32)
    assert result.shape[1:] == (NUM_KEYPOINTS, COORDS_PER_KPT), \
        f"Unexpected pose shape {result.shape} in {json_path}"
    return result


# ---------------------------------------------------------------------------
# STEP 4: CONVERT FRAME-LEVEL LABELS TO SEGMENT CSV
# ---------------------------------------------------------------------------

def convert_labels_to_segments(
    label_path: str,
    target_column: str,
    max_frames: int | None = None,
) -> pd.DataFrame:
    """Convert a frame-level label CSV into a run-length encoded segment table.

    Reads frame-level behavior annotations and collapses consecutive identical
    labels into contiguous segments with start/end frame indices. Missing or
    empty values are mapped to ``"unlabeled"``.

    Args:
        label_path: Path to the frame-level label CSV.
        target_column: Column name containing behavior labels.
        max_frames: If provided, only the first ``max_frames`` rows are used.
            Defaults to ``None`` (use all rows).

    Returns:
        DataFrame with columns:
            - ``start`` (int64): First frame index of the segment (inclusive).
            - ``end`` (int64): Last frame index of the segment (inclusive).
            - ``behavior`` (str): Behavior label for the segment.

        Returns an empty DataFrame with these columns if ``target_column`` is
        not present in the label file.

    Example:
        >>> seg = convert_labels_to_segments("/data/labels.csv", "Locomotion")
        >>> seg.head()
           start  end behavior
        0      0    5  Walking
        1      6   12  Running
    """
    df = pd.read_csv(label_path)
    if target_column not in df.columns:
        return pd.DataFrame(columns=["start", "end", "behavior"])

    labels = (
        df[target_column]
        .fillna("unlabeled")
        .astype(str)
        .replace(["", "nan", "NA", "N/A"], "unlabeled")
        .values
    )
    if max_frames is not None:
        labels = labels[:max_frames]

    segments, cur, start = [], labels[0], 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            segments.append({"start": start, "end": i - 1, "behavior": cur})
            cur, start = labels[i], i
    segments.append({"start": start, "end": len(labels) - 1, "behavior": cur})

    seg_df = pd.DataFrame(segments)
    seg_df["start"]    = seg_df["start"].astype("int64")
    seg_df["end"]      = seg_df["end"].astype("int64")
    seg_df["behavior"] = seg_df["behavior"].astype(str)
    return seg_df


# ---------------------------------------------------------------------------
# STEP 5: PREPARE DATA
# ---------------------------------------------------------------------------

def prepare_data(
    matched_files: list[dict],
    target_column: str,
    output_dir: str,
) -> list[str]:
    """Process matched files into flattened pose tensors and segment CSVs.

    For each matched file, loads pose data, truncates to the shorter of pose
    frames or label rows, flattens keypoints to ``(F, FEATURE_DIM)``, saves as
    a ``.pt`` file, and writes the corresponding segment CSV. Failed files are
    skipped with a printed warning.

    Args:
        matched_files: List of file dicts as returned by :func:`match_files`.
        target_column: Behavior column name to extract from label CSVs.
        output_dir: Root directory for output. Creates subdirectories
            ``pose_data/`` and ``labels/`` inside it.

    Returns:
        List of subject name strings for files that were processed successfully.

    Example:
        >>> prepared = prepare_data(matched, "Locomotion", "/scratch/output")
        >>> len(prepared)
        42
    """
    pose_dir  = os.path.join(output_dir, "pose_data")
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(pose_dir,  exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    successful = []
    for item in tqdm(matched_files, desc="Processing"):
        name = item["name"]
        try:
            pose = load_pose_from_json(item["pose"])
            n_label = len(pd.read_csv(item["label"]))
            min_f = min(pose.shape[0], n_label)

            pose_flat = torch.from_numpy(
                pose[:min_f].reshape(min_f, -1).astype(np.float32)
            )
            torch.save({"ind0": pose_flat}, os.path.join(pose_dir, f"{name}_features.pt"))

            seg = convert_labels_to_segments(item["label"], target_column, max_frames=min_f)
            seg.to_csv(
                os.path.join(label_dir, f"{name}.csv"),
                index=False, quoting=csv.QUOTE_NONNUMERIC,
            )
            successful.append(name)

        except Exception as e:
            print(f"  ❌ {name}: {e}")

    print(f"[prepare_data] ✓ {len(successful)}/{len(matched_files)} files prepared")
    return successful


# ---------------------------------------------------------------------------
# DEBUG: verify saved feature files
# ---------------------------------------------------------------------------

def debug_feature_shapes(pose_dir: str) -> None:
    """Validate that all saved feature ``.pt`` files have the expected shape.

    Iterates over all ``*_features.pt`` files in ``pose_dir`` and checks that
    each tensor has shape ``(F, FEATURE_DIM)``. Raises if any file is
    malformed.

    Args:
        pose_dir: Directory containing ``*_features.pt`` files.

    Raises:
        RuntimeError: If any feature file has an incorrect number of feature
            dimensions. The error message lists all bad filenames.

    Example:
        >>> debug_feature_shapes("/scratch/output/pose_data")
        [DEBUG] All feature shapes OK — (F, 399)
    """
    print("\n[DEBUG] Checking saved feature .pt files:")
    bad = []
    for pt_file in sorted(Path(pose_dir).glob("*_features.pt")):
        data = torch.load(pt_file, weights_only=True)
        for clip_id, arr in data.items():
            if hasattr(arr, "shape"):
                shape = arr.shape
                ok = len(shape) == 2 and shape[1] == FEATURE_DIM
                if not ok:
                    print(f"  ❌ {pt_file.name} [{clip_id}]: {shape}")
                    bad.append(pt_file.name)
    if bad:
        raise RuntimeError(
            f"Shape mismatch in {len(bad)} file(s): {bad}\n"
            f"Expected (F, {FEATURE_DIM}). Delete {pose_dir} and rerun."
        )
    print(f"[DEBUG] All feature shapes OK — (F, {FEATURE_DIM})\n")


# ---------------------------------------------------------------------------
# STEP 6: SPLIT
# ---------------------------------------------------------------------------

def split_train_test(
    names: list[str],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split subject names into train and test sets.

    Args:
        names: List of subject/file name strings to split.
        train_ratio: Fraction of names to allocate to the training set.
            Must be between 0 and 1. Defaults to ``0.7``.
        seed: Random seed for reproducibility. Defaults to ``42``.

    Returns:
        Dict with keys ``"train"`` and ``"test"``, each mapping to a sorted
        list of subject name strings.

    Example:
        >>> split = split_train_test(names, train_ratio=0.8, seed=0)
        >>> len(split["train"]), len(split["test"])
        (8, 2)
    """
    train, test = train_test_split(
        names, train_size=train_ratio, random_state=seed, shuffle=True,
    )
    print(f"[split] Train: {len(train)}  |  Test: {len(test)}")
    return {"train": sorted(train), "test": sorted(test)}


# ---------------------------------------------------------------------------
# STEP 7: CREATE PROJECT
# ---------------------------------------------------------------------------

def create_project(
    project_dir: str,
    pose_dir: str,
    label_dir: str,
    action_classes: list[str],
    split_info: dict[str, list[str]],
) -> "Project":
    """Initialise and configure a DLC2Action project for training.

    Removes any existing project directory, creates a fresh
    :class:`dlc2action.project.Project`, and applies all training, data, model,
    and metric parameters.

    Args:
        project_dir: Path where the DLC2Action project will be created.
            Deleted and recreated if it already exists.
        pose_dir: Absolute path to the directory containing
            ``*_features.pt`` files.
        label_dir: Absolute path to the directory containing
            ``*.csv`` segment annotation files.
        action_classes: List of behavior class name strings (including
            ``"unlabeled"``).
        split_info: Dict with ``"train"`` and ``"test"`` keys as returned by
            :func:`split_train_test`. Currently logged only; partitioning is
            handled internally by DLC2Action.

    Returns:
        Configured :class:`dlc2action.project.Project` instance ready for
        :meth:`run_episode`.

    Example:
        >>> project = create_project(PROJECT_DIR, pose_dir, label_dir,
        ...                          action_classes, split_info)
        [create_project] ✓ Project created at: /scratch/dlc2action_project
    """
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)

    project = Project(project_dir, data_type="features", annotation_type="csv")

    project.update_parameters({
        "general": {
            "model_name": MODEL_NAME,
        },
    })

    project.update_parameters({
        "data": {
            "data_path":         os.path.abspath(pose_dir),
            "annotation_path":   os.path.abspath(label_dir),
            "feature_suffix":    "_features.pt",
            "annotation_suffix": ".csv",
            "fps":               1,
            "behaviors":         ["Crawling", "Cruising", "Running", "Vehicle", "Walking"],
        },
        "general": {
            "model_name":        MODEL_NAME,
            "exclusive":         True,
            "metric_functions":  ["f1", "precision", "recall"],
            "ignored_classes":   ["unlabeled"],
        },
        "training": {
            "num_epochs":        NUM_EPOCHS,
            "batch_size":        BATCH_SIZE,
            "lr":                LEARNING_RATE,
            "to_ram":            False,
            "val_frac":          0.2,
            "test_frac":         0.0,
            "partition_method":  "random",
            "augment_train":     0,
            "augment_val":       0,
        },
        "model": {
            "num_f_maps": 64,
            "dims":       "dataset_features",
        },
        "metrics": {
            "f1": {
                "average":         "macro",
                "ignored_classes": None,
                "threshold_value": 0.5,
            },
            "recall": {
                "average":         "macro",
                "ignored_classes": None,
                "threshold_value": 0.5,
            },
            "precision": {
                "average":         "macro",
                "ignored_classes": None,
                "threshold_value": 0.5,
            },
        },
    })

    print(f"[create_project] ✓ Project created at: {project_dir}")
    print(f"  Data type:    features (pre-computed, flattened)")
    print(f"  Classes:      {action_classes}")
    print(f"  FEATURE_DIM:  {FEATURE_DIM}  ({NUM_KEYPOINTS} kpts × {COORDS_PER_KPT})")
    print(f"  Model:        {MODEL_NAME}")
    return project


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Cleanup old runs
    for d in [PROCESSED_DIR, PROJECT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Match files
    matched = match_files(LABEL_MAPPING_CSV)

    # 2. Analyze labels
    action_classes = analyze_labels(matched, TARGET_COLUMN)
    action_classes.append("unlabeled")

    # 3. Prepare data
    prepared = prepare_data(matched, TARGET_COLUMN, PROCESSED_DIR)
    if not prepared:
        raise RuntimeError("No files prepared successfully.")

    pose_dir  = os.path.join(PROCESSED_DIR, "pose_data")
    label_dir = os.path.join(PROCESSED_DIR, "labels")

    # 3b. Filter: keep only files with actual locomotion labels
    filtered = []
    removed = 0
    for name in prepared:
        seg = pd.read_csv(os.path.join(label_dir, f"{name}.csv"))
        if (seg["behavior"] != "unlabeled").any():
            filtered.append(name)
        else:
            os.remove(os.path.join(label_dir, f"{name}.csv"))
            os.remove(os.path.join(pose_dir, f"{name}_features.pt"))
            removed += 1
    print(f"[filter] Kept {len(filtered)} files, removed {removed} unlabeled-only files from disk")
    prepared = filtered

    if not prepared:
        raise RuntimeError("No files have actual locomotion labels!")

    # 4. Debug feature shapes
    debug_feature_shapes(pose_dir)

    # 5. Split
    split_info = split_train_test(prepared, TRAIN_RATIO, RANDOM_SEED)

    # 6. Create project
    project = create_project(PROJECT_DIR, pose_dir, label_dir, action_classes, split_info)

    # 7. Train
    episode = f"trained_{MODEL_NAME}"
    print(f"\n[train] Starting episode: {episode}")
    project.run_episode(episode_name=episode, suppress_name_check=True, force=True)
    print(f"Training done: {episode}")

    # 8. Evaluate
    print(f"\n[evaluate] Evaluating: {episode}")
    results = project.evaluate([episode])