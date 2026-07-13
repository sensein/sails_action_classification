import gc
import os
import shutil
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"


predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def extract_frames_to_jpegs(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = f"{frame_count:05d}.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_filename), frame)
        frame_count += 1
    cap.release()
    return frame_count

def get_frame_paths(frame_dir):
    return sorted([
        os.path.join(frame_dir, fname)
        for fname in os.listdir(frame_dir)
        if fname.endswith(".jpg")
    ])

def run_yolo_on_frame(img, yolo_model):
    results = yolo_model(img)
    boxes = []
    for det in results[0].boxes:
        xyxy = det.xyxy[0].cpu().numpy()
        boxes.append(xyxy.astype(np.float32))
    return boxes

def render_segmented_video(video_segments, frame_paths, output_path, frame_size=None, fps=24):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_dir = "./segmented_frames_tmp"
    os.makedirs(temp_dir, exist_ok=True)

    for idx, frame_path in enumerate(frame_paths):
        fig, ax = plt.subplots(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        img = np.array(Image.open(frame_path).convert("RGB"))
        ax.imshow(img)
        ax.axis("off")

        if idx in video_segments:
            for obj_id, mask in video_segments[idx].items():
                show_mask(mask, ax, obj_id=obj_id)

        canvas.draw()
        segmented_img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(canvas.get_width_height()[::-1] + (4,))
        segmented_img = segmented_img[..., :3]

        if frame_size is None:
            frame_size = (segmented_img.shape[1], segmented_img.shape[0])

        out_path = os.path.join(temp_dir, f"frame_{idx:05d}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))
        plt.close(fig)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    for fname in sorted(os.listdir(temp_dir)):
        writer.write(cv2.imread(os.path.join(temp_dir, fname)))
    writer.release()


def process_video(video_path, output_dir, yolo_model, predictor):
    print(f"\n Processing: {os.path.basename(video_path)}")

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = os.path.join(output_dir, f"frames_{video_id}")
    os.makedirs(frame_dir, exist_ok=True)

    extract_frames_to_jpegs(video_path, frame_dir)
    frame_paths = get_frame_paths(frame_dir)


    ann_frame_idx = 40
    img = Image.open(frame_paths[ann_frame_idx]).convert("RGB")


    boxes = run_yolo_on_frame(img, yolo_model)

    inference_state = predictor.init_state(video_path=frame_dir)

    for obj_id, box in enumerate(boxes):
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=obj_id,
            box=box,
        )

    # --- Propagate segmentation ---
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # --- Save segmented video ---
    output_video_path = os.path.join(output_dir, f"segmented_{video_id}.mp4")
    render_segmented_video(video_segments, frame_paths, output_video_path)
    print(f" Done: {output_video_path}")

    shutil.rmtree(frame_dir)
    predictor.reset_state(inference_state)
    del inference_state, boxes, video_segments, img
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def process_all_videos(video_folder, output_root, predictor):
    yolo_model = YOLO("yolov8s.pt")
    video_files = [
        os.path.join(video_folder, fname)
        for fname in os.listdir(video_folder)
        if fname.endswith((".mp4", ".avi", ".mkv"))
    ]
    for video_path in video_files:
        process_video(video_path, output_root, yolo_model, predictor)


process_all_videos("/video", "/output_videos", predictor)
