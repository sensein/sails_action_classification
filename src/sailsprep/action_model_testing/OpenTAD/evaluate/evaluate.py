"""
evaluate.py
===============
Computes for every model:
  - mAP @ [0.3, 0.4, 0.5, 0.6, 0.7]  (class-aware, ranked by score)
  - Recall @ [0.3, 0.4, 0.5, 0.6, 0.7]

For models with 3 seeds: mean ± std and 95% CI.
For 1 seed: single-seed result.

Saves: evaluation_results.csv  evaluation_results.txt
"""

import json
import csv

import numpy as np
from common import iou_1d, discover_results

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

TASKS = {
    'locomotion': {
        'anno': 'data/locomotion/annotations/locomotion_anno.json',
        'cmap': 'data/locomotion/annotations/locomotion_category_idx.txt',
    },
    'rmm': {
        'anno': 'data/locomotion/annotations/rmm_anno.json',
        'cmap': 'data/locomotion/annotations/rmm_category_idx.txt',
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_class_map(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def load_gt(anno_path, class_map):
    with open(anno_path) as f:
        db = json.load(f)['database']
    gt = {}
    for vid, info in db.items():
        if info['subset'] != 'test':
            continue
        anns = []
        for a in info['annotations']:
            if a['label'] in class_map:
                anns.append({
                    'segment':  a['segment'],
                    'label':    a['label'],
                    'label_id': class_map.index(a['label']),
                })
        gt[vid] = anns
    return gt


def load_pred(result_path, class_map):
    with open(result_path) as f:
        data = json.load(f)['results']
    pred = {}
    for vid, plist in data.items():
        mapped = []
        for p in plist:
            if p['label'] in class_map:
                mapped.append({
                    'segment':  p['segment'],
                    'label':    p['label'],
                    'label_id': class_map.index(p['label']),
                    'score':    p['score'],
                })
        pred[vid] = mapped
    return pred


# ── mAP ───────────────────────────────────────────────────────────────────────

def compute_map_per_thresh(predictions, ground_truth, tiou_thresholds, num_classes):
    """Returns list of mAP values, one per threshold."""
    ap_per_thresh = []
    for tiou in tiou_thresholds:
        ap_per_class = []
        for cls in range(num_classes):
            tp_list, fp_list, scores_list = [], [], []
            n_gt = 0
            for vid, gt_list in ground_truth.items():
                gt_segs = np.array(
                    [g['segment'] for g in gt_list if g['label_id'] == cls],
                    dtype=np.float32)
                n_gt += len(gt_segs)
                preds = sorted(
                    [p for p in predictions.get(vid, []) if p['label_id'] == cls],
                    key=lambda x: -x['score'])
                matched = np.zeros(len(gt_segs), dtype=bool)
                for p in preds:
                    scores_list.append(p['score'])
                    if len(gt_segs) == 0:
                        tp_list.append(0); fp_list.append(1)
                        continue
                    ious = iou_1d(np.array(p['segment']), gt_segs)
                    best = np.argmax(ious)
                    if ious[best] >= tiou and not matched[best]:
                        tp_list.append(1); fp_list.append(0)
                        matched[best] = True
                    else:
                        tp_list.append(0); fp_list.append(1)
            if n_gt == 0:
                continue
            if len(scores_list) == 0:
                ap_per_class.append(0.0)
                continue
            order  = np.argsort(-np.array(scores_list))
            tp_cum = np.cumsum(np.array(tp_list)[order])
            fp_cum = np.cumsum(np.array(fp_list)[order])
            rec  = tp_cum / max(n_gt, 1)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
            ap = sum(
                (np.max(prec[rec >= t]) if np.any(rec >= t) else 0.0)
                for t in np.linspace(0, 1, 11)
            ) / 11
            ap_per_class.append(float(ap))
        ap_per_thresh.append(float(np.mean(ap_per_class)) if ap_per_class else 0.0)
    return ap_per_thresh   # length == len(tiou_thresholds)


# ── Recall ────────────────────────────────────────────────────────────────────

def compute_recall_per_thresh(predictions, ground_truth, tiou_thresholds, num_classes):
    """
    Per-class recall averaged across classes, one value per threshold.
    A GT segment is recalled if any prediction of the same class overlaps it >= tiou.
    """
    recall_per_thresh = []
    for tiou in tiou_thresholds:
        recall_per_class = []
        for cls in range(num_classes):
            found, total = 0, 0
            for vid, gt_list in ground_truth.items():
                gt_segs = [g['segment'] for g in gt_list if g['label_id'] == cls]
                total += len(gt_segs)
                if not gt_segs:
                    continue
                gt_arr  = np.array(gt_segs, dtype=np.float32)
                pred_segs = np.array(
                    [p['segment'] for p in predictions.get(vid, [])
                     if p['label_id'] == cls],
                    dtype=np.float32)
                if len(pred_segs) == 0:
                    continue
                for gt_seg in gt_arr:
                    if np.max(iou_1d(gt_seg, pred_segs)) >= tiou:
                        found += 1
            if total > 0:
                recall_per_class.append(found / total)
        recall_per_thresh.append(
            float(np.mean(recall_per_class)) if recall_per_class else 0.0)
    return recall_per_thresh   # length == len(tiou_thresholds)


# ── discover result files ────────────────────────────────────────────────────

groups = discover_results()


# ── evaluate ──────────────────────────────────────────────────────────────────

rows = []

for (task, model), entries in sorted(groups.items()):
    tcfg = TASKS.get(task)
    if tcfg is None:
        continue
    cmap  = load_class_map(tcfg['cmap'])
    gt    = load_gt(tcfg['anno'], cmap)
    n_cls = len(cmap)

    seed_maps     = {}   # seed -> list of mAP per thresh
    seed_recalls  = {}   # seed -> list of recall per thresh

    for seed_label, rfile in entries:
        pred         = load_pred(rfile, cmap)
        map_vals     = compute_map_per_thresh(pred, gt, THRESHOLDS, n_cls)
        recall_vals  = compute_recall_per_thresh(pred, gt, THRESHOLDS, n_cls)
        seed_maps[seed_label]    = map_vals
        seed_recalls[seed_label] = recall_vals

        avg_map    = float(np.mean(map_vals))
        avg_recall = float(np.mean(recall_vals))
        print(f"  {task}/{model}/{seed_label}"
              f"  mAP@avg={avg_map*100:.2f}%"
              f"  Recall@avg={avg_recall*100:.2f}%"
              f"  per-thresh mAP={[f'{v*100:.1f}' for v in map_vals]}")

    # ── aggregate over seeds ──────────────────────────────────────────────────

    def spread(per_seed_lists):
        """per_seed_lists: {seed: [val_per_thresh]}
           Returns mean_per_thresh, std_per_thresh, avg_mean, avg_std, ci95."""
        mat = np.array(list(per_seed_lists.values()))   # [n_seeds, n_thresh]
        mean_per_thresh = mat.mean(axis=0)
        std_per_thresh  = mat.std(axis=0)
        avgs            = mat.mean(axis=1)              # one avg per seed
        avg_mean        = float(avgs.mean())
        avg_std         = float(avgs.std())
        ci95            = 1.96 * avg_std / np.sqrt(len(avgs)) if len(avgs) > 1 else 0.0
        per_seed_avgs   = {k: float(np.mean(v)) for k, v in per_seed_lists.items()}
        return mean_per_thresh, std_per_thresh, avg_mean, avg_std, ci95, per_seed_avgs

    (map_mean_t, map_std_t,
     map_avg_mean, map_avg_std,
     map_ci95, map_per_seed)       = spread(seed_maps)

    (rec_mean_t, rec_std_t,
     rec_avg_mean, rec_avg_std,
     rec_ci95, rec_per_seed)       = spread(seed_recalls)

    n = len(entries)
    rows.append({
        'task': task, 'model': model, 'n_seeds': n,
        # mAP
        'map_avg_mean':   map_avg_mean * 100,
        'map_avg_std':    map_avg_std  * 100,
        'map_ci95':       map_ci95     * 100,
        'map_per_thresh': (map_mean_t  * 100).tolist(),
        'map_std_thresh': (map_std_t   * 100).tolist(),
        'map_per_seed':   {k: round(v*100, 2) for k, v in map_per_seed.items()},
        # Recall
        'rec_avg_mean':   rec_avg_mean * 100,
        'rec_avg_std':    rec_avg_std  * 100,
        'rec_ci95':       rec_ci95     * 100,
        'rec_per_thresh': (rec_mean_t  * 100).tolist(),
        'rec_std_thresh': (rec_std_t   * 100).tolist(),
        'rec_per_seed':   {k: round(v*100, 2) for k, v in rec_per_seed.items()},
    })


# ── print tables ──────────────────────────────────────────────────────────────

W = 105

def fmt_spread(mean, std, ci95, n, per_seed):
    if n >= 3:
        ci_str = f"[{mean-ci95:.2f}, {mean+ci95:.2f}]"
    else:
        ci_str = "N/A"
    seed_str = "  ".join(f"{v:.2f}%" for v in per_seed.values())
    summary  = f"{mean:.2f} ± {std:.2f}%" if n > 1 else f"{mean:.2f}% (1 seed)"
    return summary, ci_str, seed_str


print("\n\n" + "=" * W)
print("mAP (avg over tIoU thresholds)")
print("=" * W)
print(f"{'Task':<12} {'Model':<25} {'Seeds':>5}  {'mean ± std':>16}  {'95% CI':>22}  per-seed")
print("=" * W)
for r in rows:
    s, ci, ps = fmt_spread(r['map_avg_mean'], r['map_avg_std'],
                           r['map_ci95'], r['n_seeds'], r['map_per_seed'])
    print(f"{r['task']:<12} {r['model']:<25} {r['n_seeds']:>5}  {s:>16}  {ci:>22}  {ps}")

print("\n\nmAP per tIoU threshold (mean ± std over seeds):")
print(f"{'Task':<12} {'Model':<25}  " +
      "  ".join(f"@{t:.1f}" for t in THRESHOLDS))
print("-" * W)
for r in rows:
    vals = "  ".join(
        f"{m:>5.2f}±{s:.2f}" for m, s in zip(r['map_per_thresh'], r['map_std_thresh']))
    print(f"{r['task']:<12} {r['model']:<25}  {vals}")


print("\n\n" + "=" * W)
print("Recall (avg over tIoU thresholds)")
print("=" * W)
print(f"{'Task':<12} {'Model':<25} {'Seeds':>5}  {'mean ± std':>16}  {'95% CI':>22}  per-seed")
print("=" * W)
for r in rows:
    s, ci, ps = fmt_spread(r['rec_avg_mean'], r['rec_avg_std'],
                           r['rec_ci95'], r['n_seeds'], r['rec_per_seed'])
    print(f"{r['task']:<12} {r['model']:<25} {r['n_seeds']:>5}  {s:>16}  {ci:>22}  {ps}")

print("\n\nRecall per tIoU threshold (mean ± std over seeds):")
print(f"{'Task':<12} {'Model':<25}  " +
      "  ".join(f"@{t:.1f}" for t in THRESHOLDS))
print("-" * W)
for r in rows:
    vals = "  ".join(
        f"{m:>5.2f}±{s:.2f}" for m, s in zip(r['rec_per_thresh'], r['rec_std_thresh']))
    print(f"{r['task']:<12} {r['model']:<25}  {vals}")


# ── save CSV ──────────────────────────────────────────────────────────────────

with open('evaluation_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(
        ['task', 'model', 'n_seeds',
         'mean_mAP', 'std_mAP', 'ci95_lower_mAP', 'ci95_upper_mAP'] +
        [f'mAP@{t}' for t in THRESHOLDS] +
        ['mean_Recall', 'std_Recall', 'ci95_lower_Recall', 'ci95_upper_Recall'] +
        [f'Recall@{t}' for t in THRESHOLDS]
    )
    for r in rows:
        writer.writerow([
            r['task'], r['model'], r['n_seeds'],
            round(r['map_avg_mean'], 4), round(r['map_avg_std'], 4),
            round(r['map_avg_mean'] - r['map_ci95'], 4),
            round(r['map_avg_mean'] + r['map_ci95'], 4),
            *[round(v, 4) for v in r['map_per_thresh']],
            round(r['rec_avg_mean'], 4), round(r['rec_avg_std'], 4),
            round(r['rec_avg_mean'] - r['rec_ci95'], 4),
            round(r['rec_avg_mean'] + r['rec_ci95'], 4),
            *[round(v, 4) for v in r['rec_per_thresh']],
        ])


# ── save txt ──────────────────────────────────────────────────────────────────

with open('evaluation_results.txt', 'w') as f:
    for r in rows:
        f.write(f"\n{'='*60}\n{r['task']} / {r['model']}  ({r['n_seeds']} seeds)\n{'='*60}\n")

        f.write("\nmAP:\n")
        f.write(f"  avg  {r['map_avg_mean']:.2f} ± {r['map_avg_std']:.2f}%\n")
        for t, m, s in zip(THRESHOLDS, r['map_per_thresh'], r['map_std_thresh']):
            f.write(f"  @{t}  {m:.2f} ± {s:.2f}%\n")
        for seed, v in r['map_per_seed'].items():
            f.write(f"    {seed}: {v:.2f}%\n")

        f.write("\nRecall:\n")
        f.write(f"  avg  {r['rec_avg_mean']:.2f} ± {r['rec_avg_std']:.2f}%\n")
        for t, m, s in zip(THRESHOLDS, r['rec_per_thresh'], r['rec_std_thresh']):
            f.write(f"  @{t}  {m:.2f} ± {s:.2f}%\n")
        for seed, v in r['rec_per_seed'].items():
            f.write(f"    {seed}: {v:.2f}%\n")

print("\nSaved: evaluation_results.csv  evaluation_results.txt")