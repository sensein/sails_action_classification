import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from decord import VideoReader, cpu
except ImportError as e:
    raise ImportError("Please install decord: pip install eva-decord") from e


def build_dataset_from_folders(clips_dir):
    data = []
    classes = sorted([d for d in os.listdir(clips_dir)
                      if os.path.isdir(os.path.join(clips_dir, d))])
    print(f"Found classes: {classes}")
    for cls in classes:
        cls_dir = os.path.join(clips_dir, cls)
        clips   = glob.glob(os.path.join(cls_dir, "*.mp4"))
        for clip_path in clips:
            data.append({"video_path": clip_path, "label": cls})
        print(f"  {cls}: {len(clips)} clips")
    df = pd.DataFrame(data)
    print(f"Total clips: {len(df)}")
    return df


class VJEPA2VideoDataset(Dataset):
    def __init__(self, video_paths, labels, processor, num_frames=64, crop_size=256, flush=False):
        self.video_paths = video_paths
        self.labels      = labels
        self.processor   = processor
        self.num_frames  = num_frames
        self.crop_size   = crop_size
        self.flush       = flush

    def __len__(self):
        return len(self.video_paths)

    def _sample_frames(self, video_path):
        vr           = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.arange(total_frames)
            indices = np.pad(indices, (0, self.num_frames - total_frames), mode='wrap')
        frames = vr.get_batch(indices).asnumpy()           # [T, H, W, C]
        return torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]

    def __getitem__(self, idx):
        try:
            frames      = self._sample_frames(self.video_paths[idx])
            inputs      = self.processor(frames, return_tensors="pt")
            pixel_values = inputs["pixel_values_videos"].squeeze(0)
            return pixel_values, self.labels[idx]
        except Exception as e:
            print(f"  Error loading {self.video_paths[idx]}: {e}", flush=self.flush)
            dummy = torch.zeros(self.num_frames, 3, self.crop_size, self.crop_size)
            return dummy, self.labels[idx]


@torch.no_grad()
def extract_all_features(model, processor, video_paths, labels, device,
                          num_frames=64, crop_size=256, batch_size=2, num_workers=8, flush=False):
    model.eval()
    model = model.to(device)

    dataset = VJEPA2VideoDataset(video_paths, labels, processor,
                                  num_frames=num_frames, crop_size=crop_size, flush=flush)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

    all_features, all_labels, errors = [], [], []

    for batch_idx, (pixel_values, batch_labels) in enumerate(loader):
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}", flush=flush)
        try:
            pixel_values = pixel_values.to(device)
            outputs      = model(pixel_values_videos=pixel_values, skip_predictor=True)
            features     = outputs.last_hidden_state          # [B, N_tokens, 1408]
            all_features.append(features.cpu().float())
            all_labels.append(batch_labels)
        except Exception as e:
            print(f"  Error at batch {batch_idx}: {e}", flush=flush)
            raise

    all_features = torch.cat(all_features, dim=0)
    all_labels   = torch.cat(
        [l if isinstance(l, torch.Tensor) else torch.tensor(l) for l in all_labels], dim=0
    )
    print(f"  Features shape : {all_features.shape}", flush=flush)
    print(f"  Labels shape   : {all_labels.shape}", flush=flush)
    if errors:
        print(f"  Failed batches : {len(errors)}", flush=flush)
    return all_features, all_labels
