"""Train feature-based TAD models on the locomotion/RMM dataset.

Backbone: vjepa only.
Seeds: 42, 123, 456 (for confidence intervals).
Models: actionformer, tridet, dyfadet, temporalmaxer (one-stage);
        bmn (two-stage); tadtr (DETR-style).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess

import numpy as np

# ── Task configs ──────────────────────────────────────────────────────────────

TASK_CONFIG: dict[str, dict] = {
    "locomotion": {
        "num_classes": 5,
        "ann_file": "data/locomotion/annotations/locomotion_anno.json",
        "class_map": "data/locomotion/annotations/locomotion_category_idx.txt",
    },
    "rmm": {
        "num_classes": 4,
        "ann_file": "data/locomotion/annotations/rmm_anno.json",
        "class_map": "data/locomotion/annotations/rmm_category_idx.txt",
    },
}

# ── Backbone — vjepa only ─────────────────────────────────────────────────────

BACKBONE_CONFIG: dict[str, dict] = {
    "vjepa": {"dim": 1408, "feat_dir": "data/locomotion/features/vjepa/"},
}

_DIMS_JSON = "data/locomotion/annotations/feature_dims.json"
if os.path.exists(_DIMS_JSON):
    with open(_DIMS_JSON) as _fh:
        _detected: dict = json.load(_fh)
    for _bname, _dim in _detected.items():
        if _bname in BACKBONE_CONFIG and _dim is not None:
            BACKBONE_CONFIG[_bname]["dim"] = int(_dim)
    print(f"[run_locomotion] Loaded feature dims: {_detected}")

# ── Model base paths in OpenTAD ───────────────────────────────────────────────

MODEL_BASE: dict[str, str] = {
    "actionformer": "../../_base_/models/actionformer.py",
    "tridet": "../../_base_/models/tridet.py",
    "dyfadet": "../../_base_/models/dyfadet.py",
    "temporalmaxer": "../../_base_/models/temporalmaxer.py",
    "bmn": "../../_base_/models/bmn.py",
    "tadtr": "../../_base_/models/tadtr.py",
}

ONE_STAGE_MODELS: list[str] = ["actionformer", "tridet", "dyfadet", "temporalmaxer"]
TWO_STAGE_MODELS: list[str] = ["bmn"]
DETR_MODELS: list[str] = ["tadtr"]
ALL_MODELS: list[str] = ONE_STAGE_MODELS + TWO_STAGE_MODELS + DETR_MODELS

SEEDS: list[int] = [42, 123, 456]
NUM_WORKERS: int = 0
BACKBONE: str = "vjepa"

# ── Config templates ──────────────────────────────────────────────────────────

_ONE_STAGE_TEMPLATE = '''\
_base_ = [
    "{dataset_rel}",
    "{model_base}",
]

random_seed = {seed}

model = dict(
    projection=dict(in_channels={feat_dim}),
    rpn_head=dict(num_classes={num_classes}),
)

solver = dict(
    train=dict(batch_size=2, num_workers={num_workers}),
    val=dict(batch_size=1, num_workers={num_workers}),
    test=dict(batch_size=1, num_workers={num_workers}),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay={weight_decay}, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch={warmup}, max_epoch={epochs})

inference = dict(load_from_raw_predictions=False, save_raw_prediction=True)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=2000,
        iou_threshold=0.1,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.7,
    ),
    save_dict=True,
)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch={val_start},
)

work_dir = "exps/{task}/{model}_{backbone}/seed_{seed}"
'''

_ONE_STAGE_HPS: dict[str, dict] = {
    "actionformer": dict(weight_decay=0.05, warmup=5, epochs=35, val_start=25),
    "tridet": dict(weight_decay=0.025, warmup=20, epochs=40, val_start=30),
    "dyfadet": dict(weight_decay=0.05, warmup=5, epochs=35, val_start=25),
    "temporalmaxer": dict(weight_decay=0.05, warmup=5, epochs=35, val_start=25),
}

# BMN predicts a boundary-matching confidence map over a fixed temporal grid.
# num_bins controls how many temporal locations are sampled (like max_seq_len).
_BMN_TEMPLATE = '''\
_base_ = [
    "{dataset_rel}",
    "{model_base}",
]

random_seed = {seed}

model = dict(
    projection=dict(in_channels={feat_dim}),
    proposal_generator=dict(
        num_classes={num_classes},
        num_bins=128,
    ),
)

solver = dict(
    train=dict(batch_size=2, num_workers={num_workers}),
    val=dict(batch_size=1, num_workers={num_workers}),
    test=dict(batch_size=1, num_workers={num_workers}),
    clip_grad_norm=1,
    ema=False,
)

optimizer = dict(type="Adam", lr=1e-3, weight_decay=1e-4)
scheduler = dict(type="MultiStepLR", milestones=[10, 20], gamma=0.1, max_epoch=30)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=True)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.75,
        max_seg_num=2000,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.9,
    ),
    save_dict=True,
)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=15,
)

work_dir = "exps/{task}/bmn_{backbone}/seed_{seed}"
'''

# TadTR uses a set-prediction approach with a transformer decoder.
# num_queries controls how many action instances are predicted per video.
_TADTR_TEMPLATE = '''\
_base_ = [
    "{dataset_rel}",
    "{model_base}",
]

random_seed = {seed}

model = dict(
    projection=dict(in_channels={feat_dim}),
    transformer=dict(
        num_proposals=40,
        num_classes={num_classes},
        loss=dict(
            num_classes={num_classes},
        ),
    ),
)

solver = dict(
    train=dict(batch_size=2, num_workers={num_workers}),
    val=dict(batch_size=1, num_workers={num_workers}),
    test=dict(batch_size=1, num_workers={num_workers}),
    clip_grad_norm=0.1,
    ema=False,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=1e-4, paramwise=True)
scheduler = dict(type="MultiStepLR", milestones=[30], gamma=0.1, max_epoch=40)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=True)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.4,
        max_seg_num=2000,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.95,
    ),
    save_dict=True,
)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=25,
)

work_dir = "exps/{task}/tadtr_{backbone}/seed_{seed}"
'''


# ── Seed setter ───────────────────────────────────────────────────────────────


def set_global_seed(seed: int) -> None:
    """Set Python and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    print(f"[run_locomotion] Seed set to {seed}")


# ── Dataset config ────────────────────────────────────────────────────────────


def _dataset_config(feat_dir: str, task: str) -> str:
    """Return the shared dataset config string for a given task."""
    tcfg = TASK_CONFIG[task]
    ann_file = tcfg["ann_file"]
    class_map = tcfg["class_map"]
    missing_path = feat_dir + "missing_files.txt"
    block_list = f'"{missing_path}"' if os.path.exists(missing_path) else "None"

    return f"""\
# Auto-generated {task} dataset config — vjepa features
dataset_type = "ThumosPaddingDataset"
annotation_path = "{ann_file}"
class_map = "{class_map}"
data_path = "{feat_dir}"

dataset = dict(
    train=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="training",
        block_list={block_list},
        class_map=class_map,
        data_path=data_path,
        filter_gt=True,
        feature_stride=1,
        sample_stride=1,
        offset_frames=0,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=2304, trunc_thresh=0.5, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="validation",
        block_list={block_list},
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        feature_stride=1,
        sample_stride=1,
        offset_frames=0,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="test",
        block_list={block_list},
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        feature_stride=1,
        sample_stride=1,
        offset_frames=0,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)

evaluation = dict(
    type="mAP",
    subset="test",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)
"""


# ── Config generator ──────────────────────────────────────────────────────────


def generate_config(model: str, task: str, seed: int) -> str:
    """Write dataset and model configs to disk; return the model config path."""
    bc = BACKBONE_CONFIG[BACKBONE]
    tcfg = TASK_CONFIG[task]
    feat_dim: int = bc["dim"]
    feat_dir: str = bc["feat_dir"]
    num_classes: int = tcfg["num_classes"]
    model_base: str = MODEL_BASE[model]

    ds_dir = os.path.join("configs", "_base_", "datasets", task)
    os.makedirs(ds_dir, exist_ok=True)
    ds_name = "features_vjepa_pad.py"
    ds_path = os.path.join(ds_dir, ds_name)
    with open(ds_path, "w") as fh:
        fh.write(_dataset_config(feat_dir, task))

    cfg_dir = os.path.join("configs", task, f"seed_{seed}")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, f"{model}_vjepa.py")
    dataset_rel = f"../../_base_/datasets/{task}/{ds_name}"

    if model in ONE_STAGE_MODELS:
        hps = _ONE_STAGE_HPS[model]
        content = _ONE_STAGE_TEMPLATE.format(
            dataset_rel=dataset_rel,
            model_base=model_base,
            feat_dim=feat_dim,
            num_classes=num_classes,
            seed=seed,
            num_workers=NUM_WORKERS,
            backbone=BACKBONE,
            task=task,
            model=model,
            **hps,
        )
    elif model == "bmn":
        content = _BMN_TEMPLATE.format(
            dataset_rel=dataset_rel,
            model_base=model_base,
            feat_dim=feat_dim,
            num_classes=num_classes,
            seed=seed,
            num_workers=NUM_WORKERS,
            backbone=BACKBONE,
            task=task,
        )
    elif model == "tadtr":
        content = _TADTR_TEMPLATE.format(
            dataset_rel=dataset_rel,
            model_base=model_base,
            feat_dim=feat_dim,
            num_classes=num_classes,
            seed=seed,
            num_workers=NUM_WORKERS,
            backbone=BACKBONE,
            task=task,
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    with open(cfg_path, "w") as fh:
        fh.write(content)

    print(
        f"  Config: {cfg_path}"
        f"  (model={model}, seed={seed}, dim={feat_dim}, classes={num_classes})"
    )
    return cfg_path


# ── torchrun helpers ──────────────────────────────────────────────────────────


def _torchrun(
    gpus: int,
    script: str,
    config_path: str,
    port: int,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={gpus}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint=localhost:{port}",
        script,
        config_path,
    ]
    if extra:
        cmd += extra
    return cmd


def run_train(
    config_path: str,
    gpus: int = 1,
    extra_args: list[str] | None = None,
    port: int = 29500,
) -> None:
    """Launch a distributed training job via torchrun."""
    cmd = _torchrun(gpus, "tools/train.py", config_path, port, extra_args)
    print(f"\nTRAIN: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def run_test(
    config_path: str,
    checkpoint: str,
    gpus: int = 1,
    extra_args: list[str] | None = None,
    port: int = 29500,
) -> None:
    """Launch a distributed test job via torchrun."""
    cmd = _torchrun(gpus, "tools/test.py", config_path, port)
    cmd += ["--checkpoint", checkpoint]
    if extra_args:
        cmd += extra_args
    print(f"\nTEST: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def find_best_checkpoint(model: str, task: str, seed: int) -> str | None:
    """Return the most recently modified best.pth for the given run, or None."""
    work_dir = f"exps/{task}/{model}_{BACKBONE}/seed_{seed}"
    if not os.path.isdir(work_dir):
        return None
    candidates: list[tuple[float, str]] = []
    for sub in os.listdir(work_dir):
        ckpt = os.path.join(work_dir, sub, "checkpoint", "best.pth")
        if os.path.exists(ckpt):
            candidates.append((os.path.getmtime(ckpt), ckpt))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ── Aggregate results across seeds ───────────────────────────────────────────


def aggregate_seed_results(model: str, task: str, seeds: list[int]) -> None:
    """Print and save mean +/- std and 95% CI across seeds."""
    print(f"\n{'=' * 60}")
    print(f"Aggregating: {task} / {model}_vjepa   seeds={seeds}")
    print(f"{'=' * 60}")

    all_maps: list[float] = []
    for seed in seeds:
        pattern = f"exps/{task}/{model}_{BACKBONE}/seed_{seed}/**/test_results.json"
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            print(f"  seed {seed}: no test_results.json found")
            continue
        matches.sort(key=os.path.getmtime, reverse=True)
        with open(matches[0]) as fh:
            results: dict = json.load(fh)

        map_val: float | None = None
        for key in ["mAP", "map", "average_mAP", "mAP@0.5"]:
            if key in results:
                map_val = results[key]
                break
        if map_val is None:
            tiou_keys = [k for k in results if "0.5" in str(k)]
            if tiou_keys:
                map_val = results[tiou_keys[0]]

        if map_val is not None:
            all_maps.append(float(map_val))
            print(f"  seed {seed}: mAP@0.5 = {map_val:.4f}")
        else:
            print(f"  seed {seed}: could not parse mAP. Keys: {list(results.keys())}")

    if len(all_maps) >= 2:
        mean = float(np.mean(all_maps))
        std = float(np.std(all_maps))
        ci95 = 1.96 * std / np.sqrt(len(all_maps))
        print(f"\n  mAP@0.5  = {mean:.4f} \u00b1 {std:.4f} (std)")
        print(f"  95% CI   = [{mean - ci95:.4f},  {mean + ci95:.4f}]")
        print(f"  Per seed : {[round(m, 4) for m in all_maps]}")

        summary = {
            "task": task,
            "model": model,
            "backbone": BACKBONE,
            "seeds": seeds,
            "per_seed_mAP": all_maps,
            "mean_mAP": mean,
            "std_mAP": std,
            "ci95_lower": mean - ci95,
            "ci95_upper": mean + ci95,
        }
        out = f"exps/{task}/{model}_{BACKBONE}/seed_summary.json"
        with open(out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"  Saved \u2192 {out}")
    elif len(all_maps) == 1:
        print("  Only 1 seed done — need >= 2 for CI")
    else:
        print("  No results found yet")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments and dispatch training, testing, or aggregation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["locomotion", "rmm"])
    parser.add_argument("--model", choices=ALL_MODELS)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "test", "train_test", "train_all", "generate_config", "aggregate"],
    )
    parser.add_argument("--seed", type=int, default=None, help="Single seed override.")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--extra_args", nargs="*", default=[])
    args, unknown = parser.parse_known_args()
    if unknown:
        args.extra_args = (args.extra_args or []) + unknown

    seeds: list[int] = [args.seed] if args.seed is not None else args.seeds

    if args.mode == "aggregate":
        if not args.model:
            parser.error("--model required for aggregate")
        aggregate_seed_results(args.model, args.task, seeds)
        return

    if args.mode == "train_all":
        for model in ALL_MODELS:
            for seed in seeds:
                set_global_seed(seed)
                port = args.port + ALL_MODELS.index(model) * 10 + seeds.index(seed)
                print(f"\n{'=' * 60}\n[{args.task}] {model}_vjepa  seed={seed}\n{'=' * 60}")
                try:
                    cfg = generate_config(model, args.task, seed)
                    run_train(cfg, args.gpus, args.extra_args, port)
                    ckpt = find_best_checkpoint(model, args.task, seed)
                    if ckpt:
                        run_test(cfg, ckpt, args.gpus, args.extra_args, port)
                    else:
                        print("  WARNING: best.pth not found")
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAILED {model} seed={seed}: {exc}")
            aggregate_seed_results(model, args.task, seeds)
        return

    if not args.model:
        parser.error("--model required")

    if args.mode == "generate_config":
        for seed in seeds:
            generate_config(args.model, args.task, seed)
        return

    if args.mode in ("train", "train_test"):
        for seed in seeds:
            set_global_seed(seed)
            port = args.port + seed % 100
            cfg = generate_config(args.model, args.task, seed)
            run_train(cfg, args.gpus, args.extra_args, port)
            if args.mode == "train_test":
                ckpt = args.checkpoint or find_best_checkpoint(args.model, args.task, seed)
                if not ckpt:
                    print(f"  WARNING: best.pth not found for seed={seed}")
                else:
                    run_test(cfg, ckpt, args.gpus, args.extra_args, port)
        aggregate_seed_results(args.model, args.task, seeds)
        return

    if args.mode == "test":
        for seed in seeds:
            port = args.port + seed % 100
            cfg = generate_config(args.model, args.task, seed)
            ckpt = args.checkpoint or find_best_checkpoint(args.model, args.task, seed)
            if not ckpt:
                print(f"  WARNING: best.pth not found for seed={seed}")
                continue
            run_test(cfg, ckpt, args.gpus, args.extra_args, port)
        aggregate_seed_results(args.model, args.task, seeds)


if __name__ == "__main__":
    main()