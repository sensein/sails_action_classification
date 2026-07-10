import torch
import json
import csv
import pandas as pd
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)


# config
CSV_PATH = "/home/aparnabg/orcd/scratch/Automatic_Labeling/child_1_other_0.csv"       # Path to csv file
VIDEO_COL = "BidsProcessed"              # Column name containing video paths
OUTPUT_CSV = "/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/slowfast/output/infernce_predictions.csv"        # output csv path
TOP_K = 5                             # Number of top predictions per video
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print(f"Loading SlowFast model on {DEVICE}")
model = torch.hub.load("facebookresearch/pytorchvideo", model="slowfast_r50", pretrained=True)
model = model.to(DEVICE).eval()


with open("/home/aparnabg/kinetics_classnames.json", "r") as f:
    kinetics_classnames = json.load(f)

id_to_label = {v: str(k).replace('"', "") for k, v in kinetics_classnames.items()}


side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
alpha = 4

class PackPathway(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames, 1,
            torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // alpha).long(),
        )
        return [slow_pathway, fast_pathway]

transform = ApplyTransformToKey(
    key="video",
    transform=Compose([
        UniformTemporalSubsample(num_frames),
        Lambda(lambda x: x / 255.0),
        NormalizeVideo(mean, std),
        ShortSideScale(size=side_size),
        CenterCropVideo(crop_size),
        PackPathway(),
    ]),
)

clip_duration = (num_frames * sampling_rate) / frames_per_second


df = pd.read_csv(CSV_PATH)
video_paths = df[VIDEO_COL].tolist()

results = []
softmax = torch.nn.Softmax(dim=1)

for idx, video_path in enumerate(video_paths):
    print(f"[{idx+1}/{len(video_paths)}] Processing: {video_path}")

    try:
        # Load video and extract clip
        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=0, end_sec=clip_duration)

        # Apply transforms
        video_data = transform(video_data)

        # Move to device and add batch dim
        inputs = video_data["video"]
        inputs = [i.to(DEVICE)[None, ...] for i in inputs]

        # Predict
        with torch.no_grad():
            preds = model(inputs)

        preds = softmax(preds)
        top_preds = preds.topk(k=TOP_K)
        top_labels = [id_to_label[int(i)] for i in top_preds.indices[0]]
        top_scores = [round(float(s), 4) for s in top_preds.values[0]]

        results.append({
            "video_path": video_path,
            "top1_label": top_labels[0],
            "top1_score": top_scores[0],
            "top5_labels": ", ".join(top_labels),
            "top5_scores": ", ".join(map(str, top_scores)),
        })

    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({
            "video_path": video_path,
            "top1_label": "ERROR",
            "top1_score": 0.0,
            "top5_labels": str(e),
            "top5_scores": "",
        })


results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone! Results saved to {OUTPUT_CSV}")
print(results_df.head(10))