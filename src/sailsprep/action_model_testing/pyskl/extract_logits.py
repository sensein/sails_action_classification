# save as: extract_logits.py
"""
Extract logits from all trained models for MLP fusion.
Usage: python extract_logits.py --dataset rmm
"""
import argparse
import os
import pickle
import numpy as np
import torch
from mmcv import Config
from pyskl.datasets import build_dataset, build_dataloader
from pyskl.models import build_model
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel


def extract_logits_from_model(config_path, checkpoint_path, split='test'):
    cfg = Config.fromfile(config_path)

    # Build dataset for the specified split
    if split == 'test':
        dataset = build_dataset(cfg.data.test)
    elif split == 'val':
        dataset = build_dataset(cfg.data.val)
    else:
        raise ValueError(f'Unknown split: {split}')

    dataloader = build_dataloader(
        dataset,
        videos_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False)

    # Build model
    model = build_model(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            result = model(return_loss=False, **data)
            all_logits.append(result)
            all_labels.append(data['label'].item())

    logits = np.vstack(all_logits)
    labels = np.array(all_labels)
    return logits, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['rmm', 'loco'])
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    ds = args.dataset

    # Define all model configs and their best checkpoints
    models = {
        # STGCN++ 4 streams
        'stgcnpp_j': (f'configs/custom/stgcnpp_{ds}/j.py', f'work_dirs/stgcnpp_{ds}/j/best_top1_acc_epoch_*.pth'),
        'stgcnpp_b': (f'configs/custom/stgcnpp_{ds}/b.py', f'work_dirs/stgcnpp_{ds}/b/best_top1_acc_epoch_*.pth'),
        'stgcnpp_jm': (f'configs/custom/stgcnpp_{ds}/jm.py', f'work_dirs/stgcnpp_{ds}/jm/best_top1_acc_epoch_*.pth'),
        'stgcnpp_bm': (f'configs/custom/stgcnpp_{ds}/bm.py', f'work_dirs/stgcnpp_{ds}/bm/best_top1_acc_epoch_*.pth'),
        # CTR-GCN 4 streams
        'ctrgcn_j': (f'configs/custom/ctrgcn_{ds}/j.py', f'work_dirs/ctrgcn_{ds}/j/best_top1_acc_epoch_*.pth'),
        'ctrgcn_b': (f'configs/custom/ctrgcn_{ds}/b.py', f'work_dirs/ctrgcn_{ds}/b/best_top1_acc_epoch_*.pth'),
        'ctrgcn_jm': (f'configs/custom/ctrgcn_{ds}/jm.py', f'work_dirs/ctrgcn_{ds}/jm/best_top1_acc_epoch_*.pth'),
        'ctrgcn_bm': (f'configs/custom/ctrgcn_{ds}/bm.py', f'work_dirs/ctrgcn_{ds}/bm/best_top1_acc_epoch_*.pth'),
        # PoseC3D
        'posec3d_joint': (f'configs/custom/posec3d_{ds}/joint.py', f'work_dirs/posec3d_{ds}/joint/best_top1_acc_epoch_*.pth'),
    }

    import glob
    all_logits = {}
    labels = None

    for model_name, (config_path, ckpt_pattern) in models.items():
        ckpt_files = sorted(glob.glob(ckpt_pattern))
        if not ckpt_files:
            # Try latest checkpoint
            ckpt_dir = os.path.dirname(ckpt_pattern)
            ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, 'latest.pth')))
        if not ckpt_files:
            print(f'WARNING: No checkpoint found for {model_name}, skipping.')
            continue

        ckpt = ckpt_files[-1]
        print(f'Extracting logits: {model_name} from {ckpt}')
        logits, lbls = extract_logits_from_model(config_path, ckpt, split=args.split)
        all_logits[model_name] = logits

        if labels is None:
            labels = lbls
        else:
            assert np.array_equal(labels, lbls), f'Label mismatch for {model_name}!'

    # Save
    out_path = f'work_dirs/fusion_{ds}_{args.split}_logits.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump({'logits': all_logits, 'labels': labels}, f)
    print(f'Saved logits to {out_path}')
    print(f'Models: {list(all_logits.keys())}')
    for k, v in all_logits.items():
        print(f'  {k}: shape={v.shape}')


if __name__ == '__main__':
    main()