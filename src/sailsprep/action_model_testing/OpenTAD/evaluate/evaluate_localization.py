"""
Prints each model's aMAP AND Recall results as mean ± std across seeds,
with per-seed values shown inline.

Usage:
    python evaluate_localization.py
"""

import json, numpy as np

from common import iou_1d, discover_results

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

TASKS = {
    'locomotion': 'data/locomotion/annotations/locomotion_anno.json',
    'rmm':        'data/locomotion/annotations/rmm_anno.json',
}


def load_gt(anno_path):
    with open(anno_path) as f:
        db = json.load(f)['database']
    return {
        vid: np.array([a['segment'] for a in info['annotations']], dtype=np.float32)
        for vid, info in db.items()
        if info['subset'] == 'test' and info['annotations']
    }


def load_pred(result_path):
    with open(result_path) as f:
        data = json.load(f)['results']
    return {
        vid: sorted([(p['score'], p['segment']) for p in plist], key=lambda x: -x[0])
        for vid, plist in data.items()
    }


def compute_recall(pred, gt, tiou):
    """Fraction of GT segments matched by any prediction at >= tiou."""
    found, total = 0, 0
    for vid, gt_segs in gt.items():
        total += len(gt_segs)
        if vid not in pred or not pred[vid]:
            continue
        pred_segs = np.array([s for _, s in pred[vid]], dtype=np.float32)
        for gt_seg in gt_segs:
            if np.max(iou_1d(gt_seg, pred_segs)) >= tiou:
                found += 1
    return found / max(total, 1)


def agnostic_map(pred, gt, tiou):
    """Class-agnostic mAP: rank all predictions by score, label-blind matching."""
    all_preds = [(s, v, seg) for v, entries in pred.items() for s, seg in entries]
    all_preds.sort(key=lambda x: -x[0])

    gt_matched = {v: np.zeros(len(segs), dtype=bool) for v, segs in gt.items()}
    n_gt = sum(len(s) for s in gt.values())
    if n_gt == 0:
        return 0.0

    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))
    for i, (_, vid, seg) in enumerate(all_preds):
        if vid not in gt:
            fp[i] = 1; continue
        ious = iou_1d(np.array(seg), gt[vid])
        best = np.argmax(ious)
        if ious[best] >= tiou and not gt_matched[vid][best]:
            tp[i] = 1; gt_matched[vid][best] = True
        else:
            fp[i] = 1

    tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
    rec  = tp_c / n_gt
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-6)
    return float(sum(
        (np.max(prec[rec >= t]) if np.any(rec >= t) else 0.0)
        for t in np.linspace(0, 1, 11)
    ) / 11)


def fmt_spread(values):
    """Return 'mean ± std%   (v1%  v2%  v3%)'"""
    mean = np.mean(values)
    std  = np.std(values)
    seeds_str = "  ".join(f"{v:.2f}%" for v in values)
    return f"{mean:.2f} ± {std:.2f}%", f"({seeds_str})"


# ── collect per-seed metrics ──────────────────────────────────────────────────

groups = discover_results()

results = []
for (task, model), entries in sorted(groups.items()):
    if task not in TASKS:
        continue
    gt = load_gt(TASKS[task])

    seed_amaps    = []
    seed_r01      = []
    seed_r03      = []
    seed_r05      = []
    seed_r07      = []

    for seed, rfile in entries:
        pred = load_pred(rfile)
        maps = [agnostic_map(pred, gt, t) for t in THRESHOLDS]
        seed_amaps.append(float(np.mean(maps)) * 100)
        seed_r01.append(compute_recall(pred, gt, 0.1) * 100)
        seed_r03.append(compute_recall(pred, gt, 0.3) * 100)
        seed_r05.append(compute_recall(pred, gt, 0.5) * 100)
        seed_r07.append(compute_recall(pred, gt, 0.7) * 100)

    results.append({
        'label':      f"{task} / {model}",
        'amap':       seed_amaps,
        'recall@0.1': seed_r01,
        'recall@0.3': seed_r03,
        'recall@0.5': seed_r05,
        'recall@0.7': seed_r07,
    })


# ── print tables ──────────────────────────────────────────────────────────────

W = 100

def section(title, metric_key):
    print()
    print(title)
    print("─" * W)
    print(f"  {'Task / Model':<33}  {'mean ± std':>14}    per-seed values")
    print("─" * W)
    for r in results:
        ms, ps = fmt_spread(r[metric_key])
        print(f"  {r['label']:<33}  {ms:>14}    {ps}")
    print()

section("aMAP (avg over tIoU 0.1–0.7)  ↑ higher is better", 'amap')
section("Recall @ tIoU 0.1             ↑ higher is better", 'recall@0.1')
section("Recall @ tIoU 0.3             ↑ higher is better", 'recall@0.3')
section("Recall @ tIoU 0.5             ↑ higher is better", 'recall@0.5')
section("Recall @ tIoU 0.7             ↑ higher is better", 'recall@0.7')