import os, json
import numpy as np

SEEDS = [42, 123, 456]

EXPERIMENTS = {
    "clip_loco":      "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips_h5/vmae2/loco",
    "clip_rmm":       "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips_h5/vmae2/rmm",
    "fullvid_loco":   "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips_h5/vmae2/loco_fullvideo",
    "fullvid_rmm":    "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips_h5/vmae2/rmm_fullvideo",
    "twostage_loco":  "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips_h5/vmae2/loco_twostage",
    "twostage_rmm":   "/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips_h5/vmae2/rmm_twostage",
}

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def summarize(values, name):
    v = [x for x in values if x is not None]
    if not v:
        print(f"    {name}: NO DATA")
        return
    arr = np.array(v)
    print(f"    {name}: mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")

print("\n" + "="*70)
print("AGGREGATED RESULTS ACROSS SEEDS")
print("="*70)

for exp_name, base_dir in EXPERIMENTS.items():
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*70}")
    mode = exp_name.split("_")[0]

    if mode == "clip":
        # Reads test_metrics.txt accuracy line — simpler to read the CSV
        import pandas as pd
        accs = []
        for seed in SEEDS:
            csv = os.path.join(base_dir, f"seed_{seed}", "test_predictions.csv")
            if not os.path.exists(csv):
                print(f"  seed {seed}: MISSING"); accs.append(None); continue
            df = pd.read_csv(csv)
            valid = df[df["pred_label"] != "ERROR"]
            acc = valid["correct"].mean() if len(valid) else None
            accs.append(acc)
            print(f"  seed {seed}: acc={acc:.4f}  n={len(valid)}")
        summarize(accs, "Accuracy")

    elif mode == "fullvid":
        # Reads metrics_video_level.json
        for level in ["window_level", "video_level"]:
            print(f"\n  -- {level} --")
            metrics = {"top1_acc": [], "balanced_acc": [], "f1_macro": [],
                       "cohen_kappa": [], "mcc": [], "mAP": []}
            for seed in SEEDS:
                path = os.path.join(base_dir, f"seed_{seed}",
                                    f"metrics_{level}.json")
                d = load_json(path)
                if d is None:
                    print(f"    seed {seed}: MISSING"); continue
                print(f"    seed {seed}: top1={d['top1_acc']:.2f}%  "
                      f"bal_acc={d['balanced_acc']:.2f}%  "
                      f"f1_macro={d['f1_macro']:.4f}")
                for k in metrics:
                    metrics[k].append(d.get(k))
            for k, v in metrics.items():
                summarize(v, k)

    elif mode == "twostage":
        # Binary + fg metrics at window and video level
        for stage, prefix, level in [
            ("binary-window", "binary_metrics", "window_binary"),
            ("binary-video",  "binary_metrics", "video_binary"),
            ("fg-window",     "fg_metrics",     "window_fg"),
            ("fg-video",      "fg_metrics",     "video_fg"),
        ]:
            print(f"\n  -- {stage} --")
            if prefix == "binary_metrics":
                keys = {"accuracy": [], "recall_active": [], "f1_active": [], "roc_auc": []}
            else:
                keys = {"top1_acc": [], "balanced_acc": [], "f1_macro": [],
                        "cohen_kappa": [], "mcc": [], "mAP": []}
            for seed in SEEDS:
                path = os.path.join(base_dir, f"seed_{seed}",
                                    f"{prefix}_{level}.json")
                d = load_json(path)
                if d is None:
                    print(f"    seed {seed}: MISSING"); continue
                summary = {k: d.get(k) for k in keys}
                print(f"    seed {seed}: " +
                      "  ".join(f"{k}={v:.4f}" for k, v in summary.items()
                                if v is not None))
                for k in keys:
                    keys[k].append(d.get(k))
            for k, v in keys.items():
                summarize(v, k)

print("\nDone.")