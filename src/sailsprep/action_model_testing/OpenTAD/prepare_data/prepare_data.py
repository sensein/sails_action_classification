"""
prepare_data.py

Produces:
  <output_dir>/
    annotations/
        locomotion_anno.json           (multi-class annotation)
        locomotion_category_idx.txt    (5 classes)
        rmm_anno.json                  (multi-class annotation)
        rmm_category_idx.txt           (4 classes)
        feature_dims.json
    features/
        vjepa/                         (.npy shape (T, D))
            missing_files.txt
        i3d/                           (.npy shape (T, D))
            missing_files.txt
        r2plus1d/                      (.npy shape (T, D))
            missing_files.txt
        pose/                          (.npy shape (T, D))
            missing_files.txt

Usage:
  # Prepare both tasks (features are shared, annotations are separate)
  python prepare_data.py --split_csv latest_split_csv.csv --output_dir data/locomotion --task both

  # Prepare only one task
  python prepare_data.py --split_csv /home/aparnabg/orcd/scratch/latest_split_csv_new.csv --output_dir /home/aparnabg/orcd/scratch/all_project_files/action_sota_models/opentad/OpenTAD/data/locomotion --task locomotion
  python prepare_data.py --split_csv /home/aparnabg/orcd/scratch/latest_split_csv_new.csv --output_dir /home/aparnabg/orcd/scratch/all_project_files/action_sota_models/opentad/OpenTAD/data/locomotion --task rmm
"""

from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd


# ============================================================
# Constants
# ============================================================

COCO_KEYPOINTS = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
]
POSE_DIM = len(COCO_KEYPOINTS) * 3  # 51

# --- Locomotion: 5 classes ---
LOCOMOTION_CLASSES = ["Walking", "Cruising", "Crawling", "Running", "Vehicle"]
LOCOMOTION_CLASS_TO_ID = {c: i for i, c in enumerate(LOCOMOTION_CLASSES)}

# --- RMM: 4 classes ---
RMM_CLASSES = ["Hands_flapping", "Jumping", "Spinning", "Rocking"]
RMM_CLASS_TO_ID = {c: i for i, c in enumerate(RMM_CLASSES)}

# Task config
TASK_CONFIG = {
    "locomotion": {
        "column": "Locomotion",
        "classes": LOCOMOTION_CLASSES,
        "class_to_id": LOCOMOTION_CLASS_TO_ID,
        "anno_filename": "locomotion_anno.json",
        "category_filename": "locomotion_category_idx.txt",
    },
    "rmm": {
        "column": "Repetitive_Motor_Movements",
        "classes": RMM_CLASSES,
        "class_to_id": RMM_CLASS_TO_ID,
        "anno_filename": "rmm_anno.json",
        "category_filename": "rmm_category_idx.txt",
    },
}

BACKBONE_COLS = {
    "vjepa":    "vjepa_full_path",
    "i3d":      "i3d_full_path",
    "r2plus1d": "r2plus1d_full_path",
}


# ============================================================
# Parse label CSV → action segments (multi-class)
# ============================================================
def parse_label_csv_multiclass(
    label_path: str,
    ann_fps: float,
    column_name: str,
    class_to_id: dict[str, int],
) -> tuple[list[dict], int]:
    """Parse a label CSV and extract per-class action segments."""
    try:
        df = pd.read_csv(label_path)
    except Exception as e:
        print(f"  WARNING: cannot read label CSV {label_path}: {e}")
        return [], 0

    col_map = {c.lower(): c for c in df.columns}
    frame_col = col_map.get("frame", None)
    target_col = col_map.get(column_name.lower(), None)

    if frame_col is None:
        print(f"  WARNING: label CSV missing 'Frame' column: {label_path}")
        return [], 0
    if target_col is None:
        print(f"  WARNING: label CSV missing '{column_name}' column: {label_path}")
        return [], 0

    df = df.rename(columns={frame_col: "Frame", target_col: column_name})
    df = df.sort_values("Frame").reset_index(drop=True)
    total_frames = int(df["Frame"].max()) + 1

    # Build reverse map: id -> name (ensure int keys)
    id_to_class = {int(v): k for k, v in class_to_id.items()}

    # Map each frame to its class_id (None if NA/invalid/not a string)
    def _to_class_id(x):
        # Explicitly catch float NaN (pandas stores missing as float nan in object columns)
        if not isinstance(x, str):
            return None
        x = x.strip()
        if not x:
            return None
        cid = class_to_id.get(x, None)
        return int(cid) if cid is not None else None

    # Use a plain Python list — pandas silently converts None to NaN in object columns
    class_id_list = [_to_class_id(x) for x in df[column_name]]

    # Sanity check
    bad_vals = [(i, v) for i, v in enumerate(class_id_list) if v is not None and not isinstance(v, int)]
    if bad_vals:
        print(f"  WARNING: {len(bad_vals)} unexpected class_id values in {label_path}, forcing to None")
        for i, v in bad_vals[:3]:
            print(f"    row {i}: class_id={repr(v)}, raw={repr(df[column_name].iloc[i])}")
        class_id_list = [v if (v is None or isinstance(v, int)) else None for v in class_id_list]

    # Warn about any label values that are strings but not in class_to_id
    all_labels = df[column_name].dropna().unique()
    unknown = [l for l in all_labels if isinstance(l, str) and l.strip() not in class_to_id]
    if unknown:
        print(f"  WARNING: unknown label values in {label_path}: {unknown}")
        print(f"  Valid classes: {list(class_to_id.keys())}")

    segments = []
    in_action = False
    current_class_id = None
    current_class_name = None
    start_frame = 0

    for i, row in enumerate(df.itertuples()):
        frame = int(row.Frame)
        cid = class_id_list[i]
        cname = id_to_class.get(cid) if cid is not None else None

        # Treat unrecognized classes (cid=None but raw value is a string) as background
        if cid is not None and not in_action:
            start_frame = frame
            current_class_id = cid
            current_class_name = cname
            in_action = True
        elif cid is not None and in_action and cid != current_class_id:
            # Class changed — close current, start new
            start_sec = start_frame / ann_fps
            end_sec = frame / ann_fps
            if end_sec > start_sec:
                segments.append({
                    "segment": [round(start_sec, 4), round(end_sec, 4)],
                    "label": current_class_name,
                    "label_id": int(current_class_id),
                })
            start_frame = frame
            current_class_id = cid
            current_class_name = cname
        elif cid is None and in_action:
            start_sec = start_frame / ann_fps
            end_sec = frame / ann_fps
            if end_sec > start_sec:
                segments.append({
                    "segment": [round(start_sec, 4), round(end_sec, 4)],
                    "label": current_class_name,
                    "label_id": int(current_class_id),
                })
            in_action = False
            current_class_id = None
            current_class_name = None

    # Close last segment if still open
    if in_action:
        start_sec = start_frame / ann_fps
        end_sec = total_frames / ann_fps
        if end_sec > start_sec:
            segments.append({
                "segment": [round(start_sec, 4), round(end_sec, 4)],
                "label": current_class_name,
                "label_id": current_class_id,
            })

    return segments, total_frames

# ============================================================
# Pose keypoints JSON → (T, D)
# ============================================================

def pose_json_to_npy(json_path: str):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  WARNING: cannot read pose JSON {json_path}: {e}")
        return None

    frames_dict = data.get("frames", {})
    if not frames_dict:
        return None

    frame_indices = sorted(int(k) for k in frames_dict.keys())
    max_frame = frame_indices[-1] if frame_indices else 0
    T = max_frame + 1

    features = np.zeros((T, POSE_DIM), dtype=np.float32)
    for fidx_str, kp_dict in frames_dict.items():
        fidx = int(fidx_str)
        if fidx >= T:
            continue
        for ki, kp_name in enumerate(COCO_KEYPOINTS):
            if kp_name in kp_dict:
                kp = kp_dict[kp_name]
                features[fidx, ki * 3 + 0] = kp["x"]
                features[fidx, ki * 3 + 1] = kp["y"]
                features[fidx, ki * 3 + 2] = kp["confidence"]

    # Normalize x,y per-video
    for ki in range(len(COCO_KEYPOINTS)):
        for offset in [0, 1]:
            col = ki * 3 + offset
            vals = features[:, col]
            nonzero_mask = vals != 0
            if nonzero_mask.sum() > 1:
                mean_val = vals[nonzero_mask].mean()
                std_val = vals[nonzero_mask].std()
                if std_val > 1e-6:
                    features[nonzero_mask, col] = (vals[nonzero_mask] - mean_val) / std_val

    return features  # (T, 51)


# ============================================================
# Build annotation JSON (multi-class)
# ============================================================

def build_annotation_json(split_df, ann_fps, video_feature_lengths, task_name):
    tcfg = TASK_CONFIG[task_name]
    column_name = tcfg["column"]
    class_to_id = tcfg["class_to_id"]

    database = {}
    stats = {
        "training": 0, "validation": 0, "test": 0,
        "total_segments": 0, "no_segments": 0,
    }
    class_segment_counts = {c: 0 for c in tcfg["classes"]}

    for _, row in split_df.iterrows():
        video_path = str(row["video_path"]).strip()
        label_path = str(row.get("label_path", "")).strip()
        split = str(row.get("split", "train")).strip().lower()

        video_id = os.path.splitext(os.path.basename(video_path))[0]

        if label_path and os.path.exists(label_path):
            segments, total_frames = parse_label_csv_multiclass(
                label_path, ann_fps, column_name, class_to_id
            )
        else:
            segments, total_frames = [], 0

        feature_length = video_feature_lengths.get(video_id, total_frames)
        if feature_length <= 0:
            feature_length = total_frames

        duration = feature_length / ann_fps

        if split in ("val", "validation"):
            subset = "validation"
        elif split == "test":
            subset = "test"
        else:
            subset = "training"
        stats[subset] += 1

        if segments:
            stats["total_segments"] += len(segments)
            for seg in segments:
                class_segment_counts[seg["label"]] = class_segment_counts.get(seg["label"], 0) + 1
        else:
            stats["no_segments"] += 1

        database[video_id] = {
            "subset": subset,
            "duration": round(duration, 4),
            "fps": ann_fps,
            "frame": feature_length,
            "annotations": segments,
        }

    print(f"\n[{task_name.upper()}] Annotation stats:")
    print(f"  training={stats['training']}  validation={stats['validation']}  test={stats['test']}")
    print(f"  total_segments={stats['total_segments']}  videos_with_no_action={stats['no_segments']}")
    print(f"  Per-class segments:")
    for cls_name in tcfg["classes"]:
        print(f"    {cls_name}: {class_segment_counts.get(cls_name, 0)}")

    return {"database": database}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare locomotion/RMM dataset for OpenTAD")
    parser.add_argument("--split_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ann_fps", type=float, default=15.0)
    parser.add_argument("--task", type=str, default="both",
                        choices=["locomotion", "rmm", "both"],
                        help="Which task to prepare annotations for")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process/overwrite existing feature npy files")
    args = parser.parse_args()

    tasks_to_run = ["locomotion", "rmm"] if args.task == "both" else [args.task]

    print(f"Split CSV: {args.split_csv}")
    print(f"Output dir: {args.output_dir}")
    print(f"Annotation FPS: {args.ann_fps}")
    print(f"Tasks: {tasks_to_run}")

    df = pd.read_csv(args.split_csv)
    print(f"\nLoaded {len(df)} rows from split CSV")
    print(f"Columns: {list(df.columns)}")

    anno_dir = os.path.join(args.output_dir, "annotations")
    os.makedirs(anno_dir, exist_ok=True)
    for bname in ["vjepa", "i3d", "r2plus1d", "pose"]:
        os.makedirs(os.path.join(args.output_dir, "features", bname), exist_ok=True)

    # --------------------------------------------------------
    # 1. Process backbone features (shared across tasks)
    # --------------------------------------------------------
    print("\n--- Processing backbone features (transposing to T,D) ---")
    all_video_ids = set()
    backbone_feature_lengths = {}
    backbone_video_ids = {}
    backbone_detected_dims = {}

    PROBE_N = 20

    def detect_feature_dim(paths, probe_n=PROBE_N):
        shapes = []
        for pth in paths[:probe_n]:
            try:
                arr = np.load(pth, mmap_mode="r")
                if arr.ndim == 2:
                    shapes.append(arr.shape)
            except Exception:
                continue
        if not shapes:
            return None
        axis0_vals = set(s[0] for s in shapes)
        axis1_vals = set(s[1] for s in shapes)
        if len(axis0_vals) == 1 and len(axis1_vals) > 1:
            return axis0_vals.pop()
        if len(axis1_vals) == 1 and len(axis0_vals) > 1:
            return axis1_vals.pop()
        if len(axis0_vals) == 1 and len(axis1_vals) == 1:
            return min(shapes[0])
        if all(s[0] == shapes[0][0] for s in shapes):
            return shapes[0][0]
        if all(s[1] == shapes[0][1] for s in shapes):
            return shapes[0][1]
        return None

    for bname, col_name in BACKBONE_COLS.items():
        if col_name not in df.columns:
            print(f"  SKIP {bname}: column '{col_name}' not in CSV")
            continue

        feat_dir = os.path.join(args.output_dir, "features", bname)

        src_paths = []
        src_to_vid = {}
        for _, row in df.iterrows():
            src = str(row[col_name]).strip()
            if not src or src == "nan" or not os.path.exists(src):
                continue
            video_id = os.path.splitext(os.path.basename(str(row["video_path"]).strip()))[0]
            src_paths.append(src)
            src_to_vid[src] = video_id

        if not src_paths:
            print(f"  SKIP {bname}: no valid feature files found")
            backbone_feature_lengths[bname] = {}
            backbone_video_ids[bname] = set()
            backbone_detected_dims[bname] = None
            continue

        detected_d = detect_feature_dim(src_paths)
        if detected_d is None:
            print(f"  ERROR: could not auto-detect dim for {bname}")
            backbone_feature_lengths[bname] = {}
            backbone_video_ids[bname] = set()
            backbone_detected_dims[bname] = None
            continue
        print(f"  [{bname}] auto-detected feature dim D={detected_d} "
              f"(probed {min(PROBE_N, len(src_paths))} files)")

        count = 0
        feat_lengths = {}
        video_ids_with_feature = set()

        for src in src_paths:
            video_id = src_to_vid[src]
            all_video_ids.add(video_id)
            dst = os.path.join(feat_dir, f"{video_id}.npy")

            if not os.path.exists(dst) or args.overwrite:
                try:
                    arr = np.load(src)
                    if arr.ndim != 2:
                        print(f"  WARNING: bad ndim {arr.ndim}: {src}")
                        continue
                    if arr.shape[0] == detected_d and arr.shape[1] != detected_d:
                        arr = arr.T
                    elif arr.shape[1] == detected_d and arr.shape[0] != detected_d:
                        pass
                    elif arr.shape[0] == detected_d and arr.shape[1] == detected_d:
                        pass
                    else:
                        print(f"  WARNING: neither dim matches detected D={detected_d}: "
                              f"shape={arr.shape}  file={src}")
                        continue
                    arr = arr.astype(np.float32)
                    np.save(dst, arr)
                    feat_lengths[video_id] = arr.shape[0]
                    video_ids_with_feature.add(video_id)
                    count += 1
                except Exception as e:
                    print(f"  ERROR loading {src}: {e}")
                    continue
            else:
                try:
                    arr = np.load(dst, mmap_mode="r")
                    feat_lengths[video_id] = arr.shape[0]
                    video_ids_with_feature.add(video_id)
                    count += 1
                except Exception:
                    pass

        backbone_feature_lengths[bname] = feat_lengths
        backbone_video_ids[bname] = video_ids_with_feature
        backbone_detected_dims[bname] = detected_d
        print(f"  {bname}: {count} features saved  dim={detected_d}  shape=(T, {detected_d})")

    # --------------------------------------------------------
    # 2. Pose keypoints JSON → (T, 51)
    # --------------------------------------------------------
    print("\n--- Converting pose keypoints to numpy ---")
    pose_col = "vitpose_full_path"
    pose_dir = os.path.join(args.output_dir, "features", "pose")
    pose_count = 0
    pose_lengths = {}
    pose_video_ids = set()

    if pose_col in df.columns:
        for _, row in df.iterrows():
            src = str(row[pose_col]).strip()
            if not src or src == "nan" or not os.path.exists(src):
                continue

            video_id = os.path.splitext(os.path.basename(str(row["video_path"]).strip()))[0]
            all_video_ids.add(video_id)
            dst = os.path.join(pose_dir, f"{video_id}.npy")

            if not os.path.exists(dst) or args.overwrite:
                feat = pose_json_to_npy(src)
                if feat is not None:
                    np.save(dst, feat.astype(np.float32))
                    pose_lengths[video_id] = feat.shape[0]
                    pose_video_ids.add(video_id)
                    pose_count += 1
            else:
                try:
                    f = np.load(dst, mmap_mode="r")
                    pose_lengths[video_id] = f.shape[0]
                    pose_video_ids.add(video_id)
                    pose_count += 1
                except Exception:
                    pass

        backbone_feature_lengths["pose"] = pose_lengths
        backbone_video_ids["pose"] = pose_video_ids
        print(f"  pose: {pose_count} features saved (shape=(T, 51))")
    else:
        print(f"  SKIP pose: column '{pose_col}' not in CSV")

    # --------------------------------------------------------
    # 3. Unified feature lengths (prefer vjepa)
    # --------------------------------------------------------
    unified_lengths = {}
    for video_id in all_video_ids:
        for bname in ["vjepa", "pose", "r2plus1d", "i3d"]:
            if bname in backbone_feature_lengths and video_id in backbone_feature_lengths[bname]:
                unified_lengths[video_id] = backbone_feature_lengths[bname][video_id]
                break

    # --------------------------------------------------------
    # 4. Build annotation JSON for each task
    # --------------------------------------------------------
    for task_name in tasks_to_run:
        tcfg = TASK_CONFIG[task_name]
        print(f"\n--- Building {task_name.upper()} annotation JSON ---")

        anno = build_annotation_json(df, args.ann_fps, unified_lengths, task_name)

        anno_path = os.path.join(anno_dir, tcfg["anno_filename"])
        with open(anno_path, "w") as f:
            json.dump(anno, f, indent=2)
        print(f"  Saved: {anno_path}")
        print(f"  Total videos: {len(anno['database'])}")

        # Class map
        class_map_path = os.path.join(anno_dir, tcfg["category_filename"])
        with open(class_map_path, "w") as f:
            for cls_name in tcfg["classes"]:
                f.write(cls_name + "\n")
        print(f"  Saved class_map: {class_map_path}  ({len(tcfg['classes'])} classes)")

    # --------------------------------------------------------
    # 5. missing_files.txt per backbone
    # --------------------------------------------------------
    print("\n--- Writing missing_files.txt per backbone ---")
    # Use union of all annotated videos across tasks
    all_anno_vids = set()
    for task_name in tasks_to_run:
        tcfg = TASK_CONFIG[task_name]
        anno_path = os.path.join(anno_dir, tcfg["anno_filename"])
        if os.path.exists(anno_path):
            with open(anno_path, "r") as f:
                anno_data = json.load(f)
            all_anno_vids.update(anno_data["database"].keys())

    for bname in ["vjepa", "i3d", "r2plus1d", "pose"]:
        feat_dir = os.path.join(args.output_dir, "features", bname)
        if not os.path.isdir(feat_dir):
            continue
        have_videos = backbone_video_ids.get(bname, set())
        missing = sorted(all_anno_vids - have_videos)
        mf_path = os.path.join(feat_dir, "missing_files.txt")
        with open(mf_path, "w") as f:
            for vid in missing:
                f.write(vid + "\n")
        print(f"  {bname}: {len(missing)} videos missing -> {mf_path}")

    # --------------------------------------------------------
    # 6. Save detected feature dims
    # --------------------------------------------------------
    dims_info = {}
    for bname in ["vjepa", "i3d", "r2plus1d"]:
        dims_info[bname] = backbone_detected_dims.get(bname)
    dims_info["pose"] = POSE_DIM
    dims_path = os.path.join(anno_dir, "feature_dims.json")
    with open(dims_path, "w") as f:
        json.dump(dims_info, f, indent=2)
    print(f"  Saved feature dims: {dims_path}")
    print(f"  Detected dims: {dims_info}")

    # --------------------------------------------------------
    # 7. Summary
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("Data preparation complete!")
    print(f"  Tasks prepared: {tasks_to_run}")
    for task_name in tasks_to_run:
        tcfg = TASK_CONFIG[task_name]
        print(f"  [{task_name}] Annotation: {os.path.join(anno_dir, tcfg['anno_filename'])}")
        print(f"  [{task_name}] Class map:  {os.path.join(anno_dir, tcfg['category_filename'])}")
        print(f"  [{task_name}] Num classes: {len(tcfg['classes'])}")
    print(f"  Feature dims: {dims_path}")
    for bname in ["vjepa", "i3d", "r2plus1d", "pose"]:
        feat_dir = os.path.join(args.output_dir, "features", bname)
        n = len([f for f in os.listdir(feat_dir) if f.endswith(".npy")]) if os.path.isdir(feat_dir) else 0
        d = dims_info.get(bname, "?")
        print(f"  {bname:10s}: {n:5d} files  (dim={d}, shape=(T, {d}))")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
