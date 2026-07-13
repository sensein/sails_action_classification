"""
common.py
=========
Helpers shared by evaluate.py and evaluate_localization.py.
"""

import glob
from collections import defaultdict

import numpy as np


def iou_1d(pred, gt):
    inter = np.maximum(0,
        np.minimum(pred[1], gt[:, 1]) - np.maximum(pred[0], gt[:, 0]))
    union = (pred[1] - pred[0]) + (gt[:, 1] - gt[:, 0]) - inter
    return inter / np.maximum(union, 1e-6)


def discover_results(pattern="exps/*/*/seed_*/*/result_detection.json"):
    """Group result_detection.json files by (task, model_backbone).

    Returns: {(task, model): [(seed_label, result_path), ...]}
    """
    groups = defaultdict(list)
    for rfile in sorted(glob.glob(pattern)):
        parts = rfile.replace("\\", "/").split("/")
        task, model, seed = parts[1], parts[2], parts[3]
        groups[(task, model)].append((seed, rfile))
    return groups
