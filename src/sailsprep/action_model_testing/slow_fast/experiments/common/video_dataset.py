"""Shared video-clip Dataset for the master-CSV-based fine-tuning scripts.

On a decode failure, tries up to 10 other clips before falling back to a
dummy zero-tensor, so an isolated corrupt/missing clip doesn't kill an epoch.
"""

import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(
        self,
        video_paths,
        labels,
        clip_duration,
        num_frames,
        crop_size,
        alpha,
        transform=None,
        model_name="slowfast_r50",
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.clip_duration = clip_duration
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.alpha = alpha
        self.transform = transform
        self.model_name = model_name

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        try:
            video = EncodedVideo.from_path(video_path, decode_audio=False, decoder="decord")
            video_data = video.get_clip(start_sec=0, end_sec=self.clip_duration)

            if video_data["video"] is None:
                raise ValueError("Decoder returned None")

            if self.transform:
                video_data = self.transform(video_data)

            return video_data["video"], label

        except Exception:
            # Try next valid video instead of dummy tensor
            for attempt in range(10):
                fallback_idx = (idx + attempt + 1) % len(self.video_paths)
                try:
                    fb_video = EncodedVideo.from_path(
                        self.video_paths[fallback_idx], decode_audio=False, decoder="decord"
                    )
                    fb_data = fb_video.get_clip(start_sec=0, end_sec=self.clip_duration)
                    if fb_data["video"] is None:
                        continue
                    if self.transform:
                        fb_data = self.transform(fb_data)
                    return fb_data["video"], self.labels[fallback_idx]
                except Exception:
                    continue
            # Should never reach here, but just in case
            if self.model_name == "slow_r50":
                dummy = torch.zeros(3, self.num_frames, self.crop_size, self.crop_size)
            else:
                dummy = [
                    torch.zeros(3, self.num_frames // self.alpha, self.crop_size, self.crop_size),
                    torch.zeros(3, self.num_frames, self.crop_size, self.crop_size),
                ]
            return dummy, label
