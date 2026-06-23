"""DLC2Action pipeline for locomotion behavior classification.

This module implements a full training pipeline for classifying locomotion
behaviors (Walking, Running, Crawling, etc.) from pose estimation data using
the DLC2Action framework with an MS-TCN model.

Example:
    Run the full pipeline using:

        $ poetry run python src/sailsprep/action_model_testing/dlc_action/run.py

"""
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dlc2action.project import Project
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
LABEL_MAPPING_CSV = "/orcd/data/satra/002/projects/SAILS/action_outputs_features/labels_and_clips/latest_split_csv_new.csv"
PROCESSED_DIR     = "/home/aparnabg/orcd/scratch/dlc2action_run/processed_data_full"
PROJECT_DIR       = "/home/aparnabg/orcd/scratch/dlc2action_run/dlc2action_projec_full"

TARGET_COLUMN = "Locomotion"
RANDOM_SEED   = 42

MODEL_NAME    = "ms_tcn3"

NUM_EPOCHS    = 1
BATCH_SIZE    = 8
LEARNING_RATE = 0.0001

NUM_KEYPOINTS  = 133
COORDS_PER_KPT = 3
FEATURE_DIM    = NUM_KEYPOINTS * COORDS_PER_KPT


# ==============================================================================
# STEP 1: MATCH FILES FROM CSV
# ==============================================================================
def match_files(csv_path: str) -> list[dict[str, str]]:
    matched: list[dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"video_path", "hrnet_full_path", "label_path"}
        fieldnames: list[str] = list(reader.fieldnames or [])
        if not required.issubset(set(fieldnames)):
            raise ValueError(f"CSV must contain columns: {required}")
        for row in reader:
            name = Path(row["video_path"]).stem
            matched.append({
                "name":  name,
                "video": row["video_path"],
                "label": row["label_path"],
                "pose":  row["hrnet_full_path"],
            })
    print(f"[match_files] Matched {len(matched)} files")
    return matched


# ==============================================================================
# STEP 2: ANALYZE LABELS
# ==============================================================================
def analyze_labels(matched_files: list[dict[str, str]], target_column: str) -> list[str]:
    all_actions: set[str] = set()
    for item in matched_files:
        df = pd.read_csv(item["label"])
        if target_column in df.columns:
            all_actions.update(df[target_column].dropna().unique())
    action_classes = sorted(all_actions)
    print(f"[analyze_labels] Classes found: {action_classes}")
    return action_classes


# ==============================================================================
# STEP 3: LOAD POSE (F, 133, 3) array
# ==============================================================================
def load_pose_from_json(json_path: str) -> np.ndarray:
    with open(json_path) as f:
        data = json.load(f)

    frames_dict = data["frames"]

    # Sort frame keys numerically: "1", "2", "10" -> 1, 2, 10
    sorted_keys = sorted(frames_dict.keys(), key=lambda k: int(k))

    pose_data = []
    for key in sorted_keys:
        frame_kps = frames_dict[key]  # dict: {"kp_001": {"x":..,"y":..,"confidence":..}, ...}

        arr = np.zeros((NUM_KEYPOINTS, COORDS_PER_KPT), dtype=np.float32)

        for kp_name, kp_val in frame_kps.items():
            # kp_name is like "kp_001", "kp_023", etc.
            try:
                idx = int(kp_name.split("_")[1])  # "kp_001" -> 1
            except (IndexError, ValueError):
                continue

            if 0 <= idx < NUM_KEYPOINTS:
                arr[idx, 0] = float(kp_val.get("x", 0.0))
                arr[idx, 1] = float(kp_val.get("y", 0.0))
                arr[idx, 2] = float(kp_val.get("confidence", 0.0))

        pose_data.append(arr)

    result = np.array(pose_data, dtype=np.float32)
    assert result.shape[1:] == (NUM_KEYPOINTS, COORDS_PER_KPT), \
        f"Unexpected pose shape {result.shape} in {json_path}"
    return result


# ==============================================================================
# STEP 4: CONVERT FRAME-LEVEL LABELS TO SEGMENT CSV
# ==============================================================================
def convert_labels_to_segments(
    label_path: str,
    target_column: str,
    max_frames: int | None = None,
) -> pd.DataFrame:
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


# ==============================================================================
# STEP 5: PREPARE DATA
# ==============================================================================
def prepare_data(
    matched_files: list[dict[str, str]],
    target_column: str,
    output_dir: str,
) -> list[str]:
    pose_dir  = os.path.join(output_dir, "pose_data")
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(pose_dir,  exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    successful: list[str] = []
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
            print(f" not working{name}: {e}")

    print(f"[prepare_data] working {len(successful)}/{len(matched_files)} files prepared")
    return successful


# ==============================================================================
# DEBUG: verify saved feature files
# ==============================================================================
def debug_feature_shapes(pose_dir: str) -> None:
    print("\n[DEBUG] Checking saved feature .pt files:")
    bad: list[str] = []
    for pt_file in sorted(Path(pose_dir).glob("*_features.pt")):
        data = torch.load(pt_file, weights_only=True)
        for clip_id, arr in data.items():
            if hasattr(arr, "shape"):
                shape = arr.shape
                ok = len(shape) == 2 and shape[1] == FEATURE_DIM
                tag = "working" if ok else "not working"
                if not ok:
                    print(f"  {tag} {pt_file.name} [{clip_id}]: {shape}")
                    bad.append(pt_file.name)
    if bad:
        raise RuntimeError(
            f"Shape mismatch in {len(bad)} file(s): {bad}\n"
            f"Expected (F, {FEATURE_DIM}). Delete {pose_dir} and rerun."
        )
    print(f"[DEBUG] All feature shapes OK — (F, {FEATURE_DIM})\n")


# ==============================================================================
# STEP 6: SPLIT
# ==============================================================================
def split_from_csv(
    csv_path: str,
    matched_files: list[str],
) -> dict[str, list[str]]:
    split_map: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames: list[str] = list(reader.fieldnames or [])
        if "split" not in fieldnames:
            raise ValueError("CSV does not have a 'split' column")
        for row in reader:
            name = Path(row["video_path"]).stem
            split_map[name] = row["split"].strip().lower()  # "train", "val", "test"

    train: list[str] = []
    val:   list[str] = []
    test:  list[str] = []
    for name in matched_files:
        s = split_map.get(name, "train")
        if s == "train":
            train.append(name)
        elif s == "val":
            val.append(name)
        elif s == "test":
            test.append(name)
        else:
            print(f"  ⚠ Unknown split value '{s}' for {name}, defaulting to train")
            train.append(name)

    print(f"[split] Train: {len(train)}  |  Val: {len(val)}  |  Test: {len(test)}")
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


# ==============================================================================
# STEP 7: CREATE PROJECT
# ==============================================================================
def create_project(
    project_dir: str,
    pose_dir: str,
    label_dir: str,
    action_classes: list[str],
    split_info: dict[str, list[str]],
) -> Project:
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
            "fps":               1,  # segments are already in frame indices
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

    print(f"[create_project] working Project created at: {project_dir}")
    print("  Data type:    features (pre-computed, flattened)")
    print(f"  Classes:      {action_classes}")
    print(f"  FEATURE_DIM:  {FEATURE_DIM}  ({NUM_KEYPOINTS} kpts × {COORDS_PER_KPT})")
    print(f"  Model:        {MODEL_NAME}")
    return project


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
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
    #     Remove unlabeled-only files from disk so dlc2action won't load them
    filtered: list[str] = []
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
    split_info = split_from_csv(LABEL_MAPPING_CSV, prepared)

    # 6. Create project
    project = create_project(PROJECT_DIR, pose_dir, label_dir, action_classes, split_info)

    # 7. Train
    episode = f"trained_{MODEL_NAME}"
    print(f"\n[train] Starting episode: {episode}")
    project.run_episode(episode_name=episode, suppress_name_check=True, force=True)
    print(f"working Training done: {episode}")

    # 8. Evaluate
    print(f"\n[evaluate] Evaluating: {episode}")
    results = project.evaluate([episode])
    print("Results:", results)