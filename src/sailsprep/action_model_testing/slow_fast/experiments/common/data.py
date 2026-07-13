"""Shared data-loading and batching utilities for the master-CSV-based
fine-tuning scripts (ablation study, class_weight, without_classweight).
"""

import os

import pandas as pd
import torch

from common.labels import ACTION_CLASSES, CLASS_TO_IDX, CSV_CLASS_TO_INTERNAL


def load_splits_from_csv(master_csv, clip_col, split_col):
    """
    Read master CSV and return splits dict with 'train', 'val', 'test' DataFrames.
    """
    print(f"\n[INFO] Loading splits from: {master_csv}")
    df = pd.read_csv(master_csv)
    print(f"  Total rows: {len(df)}")

    # Drop rows with missing clip paths
    df = df.dropna(subset=[clip_col])
    df = df[df[clip_col].str.strip() != ""]
    print(f"  Rows with valid clip paths: {len(df)}")

    # Extract class name from clip path (parent folder name)
    df["csv_label"] = df[clip_col].apply(lambda p: os.path.basename(os.path.dirname(p)))

    # Map CSV class names to internal names
    df["class_name"] = df["csv_label"].map(CSV_CLASS_TO_INTERNAL)
    df = df[df["class_name"].notna()]
    print(f"  Rows matching known classes: {len(df)}")

    # Encode labels
    df["label_encoded"] = df["class_name"].map(CLASS_TO_IDX)
    df["video_path"] = df[clip_col]

    # Split
    df_train = df[df[split_col] == "train"].reset_index(drop=True)
    df_val = df[df[split_col] == "val"].reset_index(drop=True)
    df_test = df[df[split_col] == "test"].reset_index(drop=True)

    # Print summary
    print(f"\n  {'Class':<12} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("  " + "-" * 44)
    for cls in ACTION_CLASSES:
        n_tr = len(df_train[df_train["class_name"] == cls])
        n_va = len(df_val[df_val["class_name"] == cls])
        n_te = len(df_test[df_test["class_name"] == cls])
        print(f"  {cls:<12} {n_tr:>8} {n_va:>8} {n_te:>8} {n_tr+n_va+n_te:>8}")
    print("  " + "-" * 44)
    print(
        f"  {'TOTAL':<12} {len(df_train):>8} {len(df_val):>8} {len(df_test):>8} "
        f"{len(df_train)+len(df_val)+len(df_test):>8}"
    )

    # Verify no overlap
    train_paths = set(df_train["video_path"])
    val_paths = set(df_val["video_path"])
    test_paths = set(df_test["video_path"])
    assert len(train_paths & val_paths) == 0, "Train/Val overlap!"
    assert len(train_paths & test_paths) == 0, "Train/Test overlap!"
    assert len(val_paths & test_paths) == 0, "Val/Test overlap!"
    print("  Overlap check: PASSED")

    return df_train, df_val, df_test


def slowfast_collate(batch):
    """Collate for SlowFast's two-pathway (slow, fast) list input."""
    videos, labels = zip(*batch)
    slow = torch.stack([v[0] for v in videos])
    fast = torch.stack([v[1] for v in videos])
    labels = torch.tensor(labels, dtype=torch.long)
    return [slow, fast], labels


def slow_collate(batch):
    """Collate for single-pathway (Slow-only) input."""
    videos, labels = zip(*batch)
    videos = torch.stack(videos)
    labels = torch.tensor(labels, dtype=torch.long)
    return videos, labels
