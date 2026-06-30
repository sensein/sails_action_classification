"""
Build PySkl sliding-window PKL files from full untrimmed videos

Outputs pyskl-format pkl files for training/val/test.

Usage:
    python build_pyskl_sw_pkl.py --task locomotion
    python build_pyskl_sw_pkl.py --task rmm
"""
import argparse
import json
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
SPLITS_CSV    = "/home/aparnabg/orcd/scratch/latest_split_csv_new.csv"
OUTPUT_DIR    = "/home/aparnabg/orcd/pool/pyskl_workspace/data/"

WINDOW_FRAMES = 30
STRIDE_FRAMES = 15

TASK_COLUMN = {
    "locomotion": "Locomotion",
    "rmm":        "Repetitive_Motor_Movements",
}

# Class name -> integer label (including None)
# These must be consistent across train/val/test
LABEL_MAPS = {
    "locomotion": {
        "Crawling": 0, "Cruising": 1, "None": 2,
        "Running": 3, "Vehicle": 4, "Walking": 5,
    },
    "rmm": {
        "Hands_flapping": 0, "Jumping": 1, "None": 2,
        "Rocking": 3, "Spinning": 4,
    },
}

COCO_KEYPOINTS = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
]


# ============================================================
# 1. LOAD POSE FROM JSON
# ============================================================
def load_pose_json(json_path):
    """Load vitpose JSON, return keypoints and scores arrays."""
    with open(json_path, "r") as f:
        data = json.load(f)

    frames_dict = data["frames"]
    frame_indices = sorted(int(k) for k in frames_dict.keys())
    T = len(frame_indices)

    keypoints = np.zeros((T, 17, 2), dtype=np.float32)
    scores    = np.zeros((T, 17), dtype=np.float32)

    for t_idx, frame_num in enumerate(frame_indices):
        frame_data = frames_dict[str(frame_num)]
        for kp_idx, kp_name in enumerate(COCO_KEYPOINTS):
            if kp_name in frame_data:
                kp = frame_data[kp_name]
                keypoints[t_idx, kp_idx, 0] = kp["x"]
                keypoints[t_idx, kp_idx, 1] = kp["y"]
                scores[t_idx, kp_idx] = kp.get("confidence", 0.0)

    return keypoints, scores, T


# ============================================================
# 2. LOAD ANNOTATIONS
# ============================================================
def load_annotations(label_path, task_column, T):
    """Load frame-level annotations, return list of label strings length T."""
    try:
        anno = pd.read_csv(label_path)
    except Exception as e:
        print(f"  [WARN] Cannot load {label_path}: {e}")
        return None

    anno.columns = anno.columns.str.strip()
    if task_column not in anno.columns:
        print(f"  [WARN] Column '{task_column}' not in {label_path}")
        return None

    labels = (
        anno[task_column]
        .fillna("None")
        .astype(str)
        .str.strip()
        .replace({"": "None", "nan": "None", "N/A": "None"})
        .tolist()
    )
    if len(labels) < T:
        labels += ["None"] * (T - len(labels))
    labels = labels[:T]

    return labels


# ============================================================
# 3. BUILD SLIDING WINDOWS FOR ONE VIDEO
# ============================================================
def build_windows_one_video(keypoints, kp_scores, labels, video_name,
                             label_map, img_shape):
    """
    Slide windows over one video, return list of pyskl annotation dicts.
    Each annotation has:
        frame_dir, label, img_shape, original_shape, total_frames,
        keypoint (1, W, 17, 2), keypoint_score (1, W, 17)
    """
    T = keypoints.shape[0]
    annotations = []

    for start in range(0, T - WINDOW_FRAMES + 1, STRIDE_FRAMES):
        end = start + WINDOW_FRAMES

        # Extract window
        kp_window = keypoints[start:end]     # (30, 17, 2)
        sc_window = kp_scores[start:end]     # (30, 17)
        lbl_window = labels[start:end]

        # Majority vote label
        label_str = Counter(lbl_window).most_common(1)[0][0]

        if label_str not in label_map:
            label_str = "None"

        label_int = label_map[label_str]

        # Create unique frame_dir ID
        frame_dir = f"{video_name}_{start}_{end-1}_w{start // STRIDE_FRAMES}"

        annotations.append({
            "frame_dir":      frame_dir,
            "label":          label_int,
            "img_shape":      img_shape,
            "original_shape": img_shape,
            "total_frames":   WINDOW_FRAMES,
            "keypoint":       kp_window[np.newaxis, ...].copy(),   # (1, 30, 17, 2)
            "keypoint_score": sc_window[np.newaxis, ...].copy(),   # (1, 30, 17)
        })

    return annotations


# ============================================================
# 4. BUILD FULL DATASET
# ============================================================
def build_dataset(task):
    task_column = TASK_COLUMN[task]
    label_map   = LABEL_MAPS[task]

    splits_df = pd.read_csv(SPLITS_CSV)

    all_annotations = []
    split_indices = {"train": [], "val": [], "test": []}

    total_idx = 0

    for split in ["train", "val", "test"]:
        split_df = splits_df[splits_df["split"] == split]
        split_count = 0
        skip_count = 0

        print(f"\n  Building {split} ({len(split_df)} videos)...")

        for _, row in split_df.iterrows():
            pose_path  = str(row["vitpose_full_path"])
            label_path = str(row["label_path"])
            feat_path  = str(row["vjpe_features_full_video_vit_h_features"])
            video_name = os.path.splitext(os.path.basename(feat_path))[0]

            # Load pose
            if not os.path.exists(pose_path):
                skip_count += 1
                continue

            try:
                keypoints, kp_scores, pose_T = load_pose_json(pose_path)
            except Exception as e:
                print(f"    [ERROR] {video_name}: {e}")
                skip_count += 1
                continue

            # Load annotations
            labels = load_annotations(label_path, task_column, pose_T)
            if labels is None:
                skip_count += 1
                continue

            # Estimate img_shape
            max_y = keypoints[:, :, 1].max()
            max_x = keypoints[:, :, 0].max()
            img_shape = (int(max_y + 50), int(max_x + 50))

            # Build windows
            windows = build_windows_one_video(
                keypoints, kp_scores, labels, video_name,
                label_map, img_shape
            )

            for w in windows:
                split_indices[split].append(w["frame_dir"])
                all_annotations.append(w)
                total_idx += 1
                split_count += 1

        print(f"    {split}: {split_count} windows from {len(split_df) - skip_count} videos "
              f"(skipped {skip_count})")

    # Print class distribution
    for split in ["train", "val", "test"]:
        split_ids = set(split_indices[split])
        split_annots = [a for a in all_annotations if a["frame_dir"] in split_ids]
        counts = Counter(a["label"] for a in split_annots)
        int_to_label = {v: k for k, v in label_map.items()}
        dist = {int_to_label[k]: v for k, v in sorted(counts.items())}
        print(f"    {split} distribution: {dist}")

    return all_annotations, split_indices


# ============================================================
# 5. SAVE PKL
# ============================================================
def save_pkl(annotations, split_indices, task, output_dir):
    """Save in pyskl format: {'split': {...}, 'annotations': [...]}"""
    pkl_data = {
        "split": split_indices,
        "annotations": annotations,
    }

    out_path = os.path.join(output_dir, f"{task}_slidingwindow_pyskl.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n  Saved: {out_path}")
    print(f"  Total annotations: {len(annotations)}")
    for split, indices in split_indices.items():
        print(f"    {split}: {len(indices)}")

    return out_path

def generate_ctrgcn_config(task, ann_file, num_classes, feat, output_dir):
    config = f"""
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='CTRGCN',
        graph_cfg=dict(layout='coco', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes={num_classes}, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = '{ann_file}'

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['{feat}']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['{feat}']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['{feat}']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='test'))

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/ctrgcn_{task}_sw/{feat}'
""".strip()

    config_dir = os.path.join(output_dir, f"ctrgcn_{task}_sw")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"{feat}.py")
    with open(config_path, "w") as f:
        f.write(config)
    print(f"  Config: {config_path}")
    return config_path
# ============================================================
# 6. GENERATE PYSKL CONFIG FILES
# ============================================================
def generate_posec3d_config(task, ann_file, num_classes, output_dir):
    config = f"""
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2)),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes={num_classes},
        dropout=0.5),
    test_cfg=dict(average_clips='prob'))

dataset_type = 'PoseDataset'
ann_file = '{ann_file}'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=30),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=30, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=30, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=64,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(type=dataset_type, ann_file=ann_file, split='train', pipeline=train_pipeline)),
    val=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='test', pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.4, momentum=0.9, weight_decay=0.0003)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/posec3d_{task}_sw/joint'
""".strip()

    config_dir = os.path.join(output_dir, f"posec3d_{task}_sw")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "joint.py")
    with open(config_path, "w") as f:
        f.write(config)
    print(f"  Config: {config_path}")
    return config_path
def generate_stgcnpp_config(task, ann_file, num_classes, feat, output_dir):
    config = f"""
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes={num_classes}, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = '{ann_file}'

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['{feat}']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['{feat}']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['{feat}']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file,
                     pipeline=train_pipeline, split='train')),
    val=dict(type=dataset_type, ann_file=ann_file,
             pipeline=val_pipeline, split='val'),
    test=dict(type=dataset_type, ann_file=ann_file,
              pipeline=test_pipeline, split='test'))

optimizer = dict(type='SGD', lr=0.1, momentum=0.9,
                 weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/stgcnpp_{task}_sw/{feat}'
""".strip()

    config_dir = os.path.join(output_dir, f"stgcnpp_{task}_sw")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"{feat}.py")
    with open(config_path, "w") as f:
        f.write(config)
    print(f"  Config: {config_path}")
    return config_path
# ============================================================
# 7. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=["locomotion", "rmm", "both"])
    parser.add_argument("--configs_only", action="store_true")
    args = parser.parse_args()

    tasks = ["locomotion", "rmm"] if args.task == "both" else [args.task]
    config_dir = "/home/aparnabg/orcd/pool/pyskl_workspace/pyskl/configs/custom/"

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"{'='*60}")

        label_map   = LABEL_MAPS[task]
        num_classes = len(label_map)
        pkl_path    = os.path.join(OUTPUT_DIR, f"{task}_slidingwindow_pyskl.pkl")

        if not args.configs_only:
            annotations, split_indices = build_dataset(task)
            pkl_path = save_pkl(annotations, split_indices, task, OUTPUT_DIR)
        else:
            print(f"  Using existing pkl: {pkl_path}")

        print(f"\n  Generating configs...")

        if task == "locomotion":
            # posec3d/joint already done — generate anyway (no harm)
            generate_posec3d_config(task, pkl_path, num_classes, config_dir)
            # NEW: ctrgcn/b and stgcnpp/b
            generate_ctrgcn_config(task, pkl_path, num_classes, "b", config_dir)
            generate_stgcnpp_config(task, pkl_path, num_classes, "b", config_dir)
        else:  # rmm
            # ctrgcn/jm already done — generate anyway
            generate_ctrgcn_config(task, pkl_path, num_classes, "jm", config_dir)
            # NEW: posec3d/joint and stgcnpp/jm
            generate_posec3d_config(task, pkl_path, num_classes, config_dir)
            generate_stgcnpp_config(task, pkl_path, num_classes, "jm", config_dir)

if __name__ == "__main__":
    main()
