#!/usr/bin/env python3
"""
EntitySAM Video Inference Script
Simplified script to run EntitySAM on any video file or directory of frames.

Usage:
    python entitysam_inference.py --video /path/to/video.mp4
    python entitysam_inference.py --video /path/to/frames_dir/ --model_size vit-s
"""

import sys
import os

# Add EntitySAM to Python path
ENTITYSAM_ROOT = "/orcd/data/satra/001/users/brukew/entitysam"
if ENTITYSAM_ROOT not in sys.path:
    sys.path.insert(0, ENTITYSAM_ROOT)

import json
import time
import argparse
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm
from pathlib import Path
try:
    import psutil
except ImportError:
    raise ImportError(
        "psutil is required for CPU memory monitoring. "
        "Install it with: pip install psutil"
    )

from sam2.build_sam import build_sam2_video_query_iou_predictor
from panopticapi.utils import IdGenerator, rgb2id


def extract_frames_from_video(video_path, output_dir):
    """Extract frames from video file using OpenCV."""
    # Check if frames already exist
    if os.path.exists(output_dir):
        existing_frames = sorted([
            f for f in os.listdir(output_dir)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])
        if existing_frames:
            print(f"Found {len(existing_frames)} existing frames in {output_dir}")
            print("Skipping frame extraction. Delete the directory to re-extract.")
            return existing_frames

    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV (cv2) is required to extract frames from video files. "
            "Install it with: pip install opencv-python"
        )

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = 0
    frame_names = []

    print(f"Extracting frames from {video_path}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = f"frame_{frame_count:06d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(frame_rgb).save(frame_path)

        frame_names.append(frame_name)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_dir}")

    return frame_names


def get_cpu_memory_info():
    """Get current CPU memory usage information."""
    process = psutil.Process()
    memory_info = process.memory_info()
    virtual_memory = psutil.virtual_memory()

    return {
        'process_rss_gb': memory_info.rss / 1024**3,
        'process_vms_gb': memory_info.vms / 1024**3,
        'system_available_gb': virtual_memory.available / 1024**3,
        'system_used_percent': virtual_memory.percent
    }

def print_memory_status(label=""):
    """Print both CPU and GPU memory status."""
    cpu_mem = get_cpu_memory_info()
    print(f"[MEM{' ' + label if label else ''}] CPU RSS: {cpu_mem['process_rss_gb']:.2f} GB, VMS: {cpu_mem['process_vms_gb']:.2f} GB")
    print(f"[MEM{' ' + label if label else ''}] System available: {cpu_mem['system_available_gb']:.2f} GB ({100-cpu_mem['system_used_percent']:.1f}% free)")

    if torch.cuda.is_available():
        print(f"[MEM{' ' + label if label else ''}] GPU allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"[MEM{' ' + label if label else ''}] GPU reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def inference_video_vps_save_results(
    pred_ious,
    pred_stability_scores,
    pred_masks,
    out_size,
    overlap_threshold=0.5,
    mask_binary_threshold=0.5,
    object_mask_threshold=0.03,
    test_topk_per_image=100,
    chunk_size=32,
):
    """
    Post process of multi-mask into panoptic segmentation.
    Uses GPU acceleration for per-chunk processing when available.
    """
    print(f"[MEM] Starting inference_video_vps_save_results")
    print(f"[MEM] Input shapes - pred_masks: {pred_masks.shape}, out_size: {out_size}")
    print_memory_status("Start")

    scores = pred_ious
    labels = torch.randint(0, 123, (len(scores),), device=scores.device)
    pred_id = torch.arange(len(scores), device=scores.device)

    keep = scores >= max(object_mask_threshold, scores.topk(k=min(len(scores), test_topk_per_image))[0][-1])
    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_masks = pred_masks[keep].contiguous()
    cur_ids = pred_id[keep]

    if pred_stability_scores is not None:
        mask_quality_scores = pred_stability_scores[keep]
    else:
        from train.utils.comm import calculate_mask_quality_scores
        mask_quality_scores = calculate_mask_quality_scores(cur_masks[:, ::5].float())

    del pred_masks
    print(f"[MEM] After filtering - num_instances: {cur_masks.shape[0]}, num_frames: {cur_masks.shape[1]}")
    print_memory_status("After filtering")

    num_instances, num_frames = cur_masks.shape[0], cur_masks.shape[1]
    estimated_size = num_frames * out_size[0] * out_size[1] * 4 / 1024**3
    print(f"[MEM] Need panoptic_seg: {num_frames} x {out_size[0]} x {out_size[1]} = {estimated_size:.2f} GB")

    # Check if we have enough CPU memory for the allocation
    cpu_mem = get_cpu_memory_info()
    if estimated_size > cpu_mem['system_available_gb'] * 0.8:  # Leave 20% buffer
        print(f"[MEM] WARNING: Estimated size ({estimated_size:.2f} GB) may exceed available memory ({cpu_mem['system_available_gb']:.2f} GB)")

    # Use memory-mapped file for large panoptic segmentation to avoid OOM
    import tempfile
    import atexit
    if estimated_size > 10.0:  # If larger than 10GB, use memmap
        print(f"[MEM] Using memory-mapped file to avoid OOM (size > 10GB)")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat')
        temp_file.close()
        panoptic_seg = np.memmap(temp_file.name, dtype='int32', mode='w+',
                                 shape=(num_frames, out_size[0], out_size[1]))
        panoptic_seg[:] = 0  # Initialize to zero
        use_memmap = True
        memmap_path = temp_file.name
        print(f"[MEM] Memory-mapped panoptic_seg created at {memmap_path}")
        print_memory_status("After memmap creation")

        # Register cleanup function in case of crash
        def cleanup_memmap():
            try:
                if os.path.exists(memmap_path):
                    os.unlink(memmap_path)
                    print(f"[MEM] Emergency cleanup: deleted {memmap_path}")
            except:
                pass
        atexit.register(cleanup_memmap)
    else:
        panoptic_seg = torch.zeros((num_frames, out_size[0], out_size[1]), dtype=torch.int32, device="cpu")
        use_memmap = False
        print(f"[MEM] Standard panoptic_seg allocated successfully")
        print_memory_status("After panoptic_seg allocation")

    segments_infos = []
    out_ids = []
    current_segment_id = 0
    chunk_size = max(1, min(chunk_size, num_frames))

    if num_instances == 0:
        return {
            "image_size": out_size,
            "pred_masks": panoptic_seg,
            "segments_infos": segments_infos,
            "pred_ids": out_ids,
            "task": "vps",
        }

    cur_scores = cur_scores + 0.5 * mask_quality_scores

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    mask_dtype = torch.float16 if use_cuda else torch.float32

    print(f"[MEM] Moving cur_scores to {device} (size: {cur_scores.numel() * 2 / 1024**2:.2f} MB)")
    cur_scores = cur_scores.to(device=device, dtype=mask_dtype)
    print(f"[MEM] Converting cur_masks to {mask_dtype} on CPU (size: {cur_masks.numel() * 2 / 1024**3:.2f} GB)")
    cur_masks = cur_masks.to(torch.float16 if use_cuda else torch.float32).contiguous()
    print(f"[MEM] cur_masks converted successfully")
    print_memory_status("After mask conversion")

    mask_area_totals = torch.zeros(num_instances, dtype=torch.float64, device="cpu")
    original_area_totals = torch.zeros(num_instances, dtype=torch.float64, device="cpu")

    num_chunks = (num_frames + chunk_size - 1) // chunk_size
    print(f"[MEM] Starting Pass 1: {num_chunks} chunks of size {chunk_size}")

    # Pass 1: accumulate per-instance statistics
    for chunk_idx, start in enumerate(tqdm(
        range(0, num_frames, chunk_size),
        total=num_chunks,
        desc="Accumulating mask stats",
        leave=False,
    )):
        end = min(start + chunk_size, num_frames)
        if chunk_idx % 10 == 0:  # Print every 10th chunk
            print(f"[MEM] Pass 1 - Chunk {chunk_idx}/{num_chunks} (frames {start}-{end})")
            print_memory_status(f"Pass1-Chunk{chunk_idx}")

        chunk_cpu = cur_masks[:, start:end]
        chunk = chunk_cpu.to(device, dtype=mask_dtype) if use_cuda else chunk_cpu
        chunk_probs = F.interpolate(chunk, size=out_size, mode="bilinear", align_corners=False).sigmoid()
        chunk_binary = chunk_probs >= mask_binary_threshold

        cur_prob_masks = cur_scores.view(-1, 1, 1, 1) * chunk_probs
        chunk_mask_ids = cur_prob_masks.argmax(0).to(torch.int64)
        bg_mask = ~chunk_binary.any(dim=0)
        chunk_mask_ids[bg_mask] = -1

        chunk_binary_cpu = chunk_binary.cpu()
        chunk_mask_ids_cpu = chunk_mask_ids.cpu()

        for inst_idx in range(num_instances):
            inst_binary = chunk_binary_cpu[inst_idx]
            original_area_totals[inst_idx] += inst_binary.sum(dtype=torch.float64)
            mask = (chunk_mask_ids_cpu == inst_idx) & inst_binary
            mask_area_totals[inst_idx] += mask.sum(dtype=torch.float64)

        del chunk, chunk_probs, chunk_binary, cur_prob_masks, chunk_mask_ids, chunk_binary_cpu, chunk_mask_ids_cpu
        if use_cuda:
            torch.cuda.empty_cache()

    print(f"[MEM] Pass 1 complete. Checking mask areas...")
    if mask_area_totals.sum().item() == 0:
        if use_cuda:
            torch.cuda.empty_cache()
        del cur_masks
        return {
            "image_size": out_size,
            "pred_masks": panoptic_seg,
            "segments_infos": segments_infos,
            "pred_ids": out_ids,
            "task": "vps",
        }

    keep_entries = []
    stuff_memory_list = {}
    for k in range(num_instances):
        mask_area = mask_area_totals[k].item()
        original_area = original_area_totals[k].item()
        if mask_area <= 0 or original_area <= 0:
            continue
        if mask_area / max(original_area, 1.0) < overlap_threshold:
            continue

        pred_class = int(cur_classes[k]) + 1
        isthing = True

        if not isthing:
            if pred_class in stuff_memory_list:
                keep_entries.append((k, stuff_memory_list[pred_class], pred_class))
                continue
            else:
                stuff_memory_list[pred_class] = current_segment_id + 1

        current_segment_id += 1
        keep_entries.append((k, current_segment_id, pred_class))
        segments_infos.append({"id": current_segment_id, "isthing": bool(isthing), "category_id": pred_class})
        out_ids.append(cur_ids[k])

    if not keep_entries:
        if use_cuda:
            torch.cuda.empty_cache()
        del cur_masks
        return {
            "image_size": out_size,
            "pred_masks": panoptic_seg,
            "segments_infos": segments_infos,
            "pred_ids": out_ids,
            "task": "vps",
        }

    keep_indices = [entry[0] for entry in keep_entries]
    keep_index_to_segment = {entry[0]: entry[1] for entry in keep_entries}

    # Pass 2: paint panoptic segmentation
    for start in tqdm(
        range(0, num_frames, chunk_size),
        total=num_chunks,
        desc="Painting panoptic masks",
        leave=False,
    ):
        end = min(start + chunk_size, num_frames)
        chunk_cpu = cur_masks[:, start:end]
        chunk = chunk_cpu.to(device, dtype=mask_dtype) if use_cuda else chunk_cpu
        chunk_probs = F.interpolate(chunk, size=out_size, mode="bilinear", align_corners=False).sigmoid()
        chunk_binary = chunk_probs >= mask_binary_threshold

        cur_prob_masks = cur_scores.view(-1, 1, 1, 1) * chunk_probs
        chunk_mask_ids = cur_prob_masks.argmax(0).to(torch.int64)
        chunk_mask_ids[~chunk_binary.any(dim=0)] = -1

        chunk_mask_ids_cpu = chunk_mask_ids.cpu()
        chunk_binary_cpu = chunk_binary.cpu()

        for inst_idx in keep_indices:
            seg_id = keep_index_to_segment[inst_idx]
            mask_cpu = (chunk_mask_ids_cpu == inst_idx) & chunk_binary_cpu[inst_idx]
            if mask_cpu.any():
                if use_memmap:
                    # Convert torch mask to numpy for memmap compatibility
                    panoptic_seg[start:end][mask_cpu.numpy()] = seg_id
                else:
                    panoptic_seg[start:end][mask_cpu] = seg_id

        del chunk, chunk_probs, chunk_binary, cur_prob_masks, chunk_mask_ids, chunk_mask_ids_cpu, chunk_binary_cpu
        if use_cuda:
            torch.cuda.empty_cache()

    del cur_masks
    if use_cuda:
        torch.cuda.empty_cache()

    # If using memmap, keep as numpy array (compatible with downstream code)
    # The process() function works with both torch tensors and numpy arrays
    if use_memmap:
        print(f"[MEM] Returning memmap as numpy array (will be cleaned up after processing)")
        # Store the path for cleanup later
        result = {
            "image_size": out_size,
            "pred_masks": panoptic_seg,
            "segments_infos": segments_infos,
            "pred_ids": out_ids,
            "task": "vps",
            "_memmap_path": memmap_path,  # Store for cleanup
        }
    else:
        result = {
            "image_size": out_size,
            "pred_masks": panoptic_seg,
            "segments_infos": segments_infos,
            "pred_ids": out_ids,
            "task": "vps",
        }

    return result

def process(video_id, frame_names, outputs, categories_dict, output_dir, video_dir):
    """
    Save panoptic segmentation result as an image (frame-by-frame to save memory)
    """
    print("[MEM] Processing frames one at a time to avoid OOM...")
    print_memory_status("Start process")
    color_generator = IdGenerator(categories_dict)

    pan_seg_result = outputs['pred_masks']
    segments_infos = outputs['segments_infos']

    # Pre-compute colors for each segment
    segment_colors = {}
    for segments_info in segments_infos:
        seg_id = segments_info['id']
        sem = segments_info['category_id']
        segment_colors[seg_id] = {
            'color': color_generator.get_color(sem),
            'category_id': sem,
            'isthing': segments_info['isthing']
        }

    # Setup output directory
    pan_pred_dir = os.path.join(output_dir, video_id, 'pan_pred')
    os.makedirs(pan_pred_dir, exist_ok=True)

    # Build a color lookup table for vectorized painting (much faster)
    max_seg_id = max(seg['id'] for seg in segments_infos) if segments_infos else 0
    color_map = np.zeros((max_seg_id + 1, 3), dtype=np.uint8)
    for seg_id, seg_data in segment_colors.items():
        color_map[seg_id] = seg_data['color']

    # Build lookup for segment info by ID
    seg_id_to_idx = {seg['id']: idx for idx, seg in enumerate(segments_infos)}

    def process_and_save_frame(args):
        """Worker function to process and save a single frame."""
        frame_idx, image_name = args

        # Get this frame's segmentation
        frame_seg = pan_seg_result[frame_idx]

        # Handle both torch tensors and numpy arrays
        if hasattr(frame_seg, 'numpy'):
            frame_seg_np = frame_seg.numpy()
        else:
            frame_seg_np = frame_seg

        # Vectorized painting: map segment IDs to colors in one operation
        pan_format_frame = color_map[frame_seg_np]

        # OPTIMIZATION: Only process segments present in this frame
        unique_seg_ids = np.unique(frame_seg_np)
        unique_seg_ids = unique_seg_ids[unique_seg_ids > 0]  # Remove background (0)

        frame_annotations = []
        segments_results = {}  # Map seg_idx to annotation or None

        for seg_id in unique_seg_ids:
            if seg_id not in segment_colors:
                continue

            seg_idx = seg_id_to_idx.get(seg_id)
            if seg_idx is None:
                continue

            sem = segment_colors[seg_id]['category_id']
            color = segment_colors[seg_id]['color']

            # Get mask for this segment
            mask = frame_seg_np == seg_id
            area = int(mask.sum())

            if area > 0:
                # Optimized bbox computation
                rows, cols = np.where(mask)
                y, x = int(rows.min()), int(cols.min())
                height, width = int(rows.max() - y), int(cols.max() - x)

                dt = {
                    "bbox": [x, y, width, height],
                    "area": area,
                    "category_id": int(sem) - 1,
                    "iscrowd": 0,
                    "id": int(rgb2id(color))
                }
                frame_annotations.append(dt)
                segments_results[seg_idx] = dt

        # Fill in None for segments not present
        for seg_idx in range(len(segments_infos)):
            if seg_idx not in segments_results:
                segments_results[seg_idx] = None

        # Save as JPEG (much faster than PNG)
        image_ = Image.fromarray(pan_format_frame)
        output_name = Path(image_name).stem + '.jpg'
        image_.save(os.path.join(pan_pred_dir, output_name), quality=95)

        return frame_idx, frame_annotations, segments_results, image_name

    # Process frames in parallel with ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor

    annotations = []
    segments_infos_ = [[] for _ in range(len(segments_infos))]  # One list per segment

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Prepare arguments for all frames
        frame_args = [(frame_idx, image_name) for frame_idx, image_name in enumerate(frame_names)]

        # Process frames in parallel
        results = list(tqdm(
            executor.map(process_and_save_frame, frame_args),
            total=len(frame_names),
            desc="Saving frames",
            leave=False
        ))

    # Collect results in order
    for frame_idx, frame_annotations, segments_results, image_name in results:
        annotations.append({
            "segments_info": frame_annotations,
            "file_name": image_name
        })

        # Update segments_infos_
        for seg_idx, annotation in segments_results.items():
            segments_infos_[seg_idx].append(annotation)

    del outputs
    print(f"[MEM] Saved {len(frame_names)} frames to {pan_pred_dir}")
    print_memory_status("End process")
    return {'annotations': annotations, 'video_id': video_id}


def create_minimal_categories():
    """Create minimal category dictionary for visualization."""
    categories_dict = {}
    for i in range(1, 124):  # Support up to 123 random category IDs
        # Generate a unique color for each category (use hash-based approach for consistency)
        np.random.seed(i)  # Deterministic colors based on ID
        color = [int(c) for c in np.random.randint(0, 256, size=3)]

        categories_dict[i] = {
            'id': i,
            'name': f'entity_{i}',
            'isthing': 1,
            'color': color
        }
    return categories_dict


def auto_detect_checkpoint(model_size='vit-l'):
    """Auto-detect checkpoint and config paths."""
    base_ckpt_dir = os.path.join(ENTITYSAM_ROOT, "entitysam_checkpoints", "checkpoints")

    if model_size == 'vit-l':
        checkpoint = os.path.join(base_ckpt_dir, "vit-l", "model_0009999.pth")
        # Hydra expects config path relative to sam2 package (configs/filename)
        config = "configs/sam2.1_hiera_l.yaml"
        decoder_depth = 8
    elif model_size == 'vit-s':
        checkpoint = os.path.join(base_ckpt_dir, "vit-s", "model_0009999.pth")
        # Hydra expects config path relative to sam2 package (configs/filename)
        config = "configs/sam2.1_hiera_s.yaml"
        decoder_depth = 4
    else:
        raise ValueError(f"Invalid model_size: {model_size}. Choose 'vit-l' or 'vit-s'")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # Verify config exists (for informational purposes)
    # config is in format "configs/sam2.1_hiera_l.yaml"
    config_filename = config.split('/')[-1] if '/' in config else config
    config_full_path = os.path.join(ENTITYSAM_ROOT, "sam2", "configs", config_filename)
    if not os.path.exists(config_full_path):
        raise FileNotFoundError(f"Config not found: {config_full_path}")

    return checkpoint, config, decoder_depth


def main():
    parser = argparse.ArgumentParser(
        description="Run EntitySAM inference on a video file or directory of frames."
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help="Path to video file (.mp4, .avi, .mov) or directory containing frames"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="/orcd/data/satra/001/users/brukew/entitysam_out",
        help="Output directory for results (default: /orcd/data/satra/001/users/brukew/entitysam_out)"
    )
    parser.add_argument(
        '--model_size',
        type=str,
        default='vit-l',
        choices=['vit-l', 'vit-s'],
        help="Model size: 'vit-l' (large) or 'vit-s' (small) (default: vit-l)"
    )
    parser.add_argument(
        '--video_id',
        type=str,
        default=None,
        help="Video identifier (default: auto-detect from filename/directory name)"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help="Path to EntitySAM checkpoint (default: auto-detect based on model_size)"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help="SAM2 config filename (e.g., 'sam2.1_hiera_l.yaml') - NOT full path (default: auto-detect based on model_size)"
    )

    args = parser.parse_args()

    # Auto-detect checkpoint and config if not provided
    if args.checkpoint is None or args.config is None:
        checkpoint, config, decoder_depth = auto_detect_checkpoint(args.model_size)
        if args.checkpoint is None:
            args.checkpoint = checkpoint
        if args.config is None:
            args.config = config
    else:
        # Set decoder depth based on model size
        decoder_depth = 8 if args.model_size == 'vit-l' else 4

        # If user provided a full path for config, convert to configs/filename format
        # because Hydra expects path relative to sam2 package
        if args.config and os.path.sep in args.config:
            config_filename = os.path.basename(args.config)
            args.config = f"configs/{config_filename}"
            print(f"Note: Using config: {args.config}")

    print(f"Using checkpoint: {args.checkpoint}")
    print(f"Using config: {args.config}")
    print(f"Decoder depth: {decoder_depth}")

    # Determine if input is video file or directory
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video path not found: {args.video}")

    # Handle video file vs directory
    if video_path.is_file():
        # Extract frames from video
        video_id = args.video_id or video_path.stem
        temp_frames_dir = os.path.join(args.output_dir, video_id, "temp_frames")
        frame_names = extract_frames_from_video(str(video_path), temp_frames_dir)
        video_dir = temp_frames_dir
        cleanup_frames = True
    elif video_path.is_dir():
        # Use frames directly
        video_id = args.video_id or video_path.name
        video_dir = str(video_path)
        frame_names = sorted([
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
        ])
        cleanup_frames = False

        if len(frame_names) == 0:
            raise ValueError(f"No image frames found in directory: {video_dir}")

        print(f"Found {len(frame_names)} frames in {video_dir}")
    else:
        raise ValueError(f"Video path must be a file or directory: {args.video}")

    # Initialize model
    print(f"\nInitializing EntitySAM model ({args.model_size})...")

    # EntitySAM code uses relative paths, so we need to change to EntitySAM directory
    original_cwd = os.getcwd()
    os.chdir(ENTITYSAM_ROOT)

    try:
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print("Loading checkpoint (this may take 1-2 minutes for the 2.2GB file)...")
        predictor = build_sam2_video_query_iou_predictor(
            args.config,
            args.checkpoint,
            device="cuda",  # Explicitly set to ensure GPU usage
            mode="eval",
            mask_decoder_depth=decoder_depth
        )
        print("Model loaded successfully and moved to GPU!")
    finally:
        # Change back to original directory
        os.chdir(original_cwd)

    # Create minimal categories for visualization
    categories_dict = create_minimal_categories()

    # Check for cached propagation results
    cache_dir = os.path.join(args.output_dir, video_id, "cache")
    cache_path = os.path.join(cache_dir, "raw_predictions.pt")
    if os.path.exists(cache_path):
        print(f"\nFound cached propagation outputs at: {cache_path}")
        print("Loading cached tensors to skip temporal propagation...")
        cache = torch.load(cache_path, map_location="cpu")
        pred_masks = cache["pred_masks"].to(torch.float16).contiguous()
        pred_eiou_stack = cache["pred_eiou_stack"]
        padding_mask = cache["padding_mask"]
        out_size = tuple(cache["out_size"])
        frame_names = cache["frame_names"]
        pred_stability_scores = None
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
    else:
        # Run inference
        print(f"\nRunning inference on video: {video_id}")
        print(f"Number of frames: {len(frame_names)}")

        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()

        # Add padding frames for temporal stability
        progress_order = [0, 0, 0] + list(range(0, len(frame_names)))
        padding_mask = [True] * 3 + [False] * len(frame_names)

        # Initialize inference state
        inference_state = predictor.init_state(video_path=video_dir, progress_order=progress_order)
        predictor.reset_state(inference_state)

        # Get output size from first frame
        frame_idx = 0
        init_frame = Image.open(os.path.join(video_dir, frame_names[frame_idx]))
        init_frame = np.array(init_frame.convert("RGB"))
        out_size = (init_frame.shape[0], init_frame.shape[1])

        # Run propagation throughout the video
        pred_masks_list = []
        pred_eiou_list = []
        print("Running temporal propagation...")
        for out_frame_idx, _, out_mask_logits, pred_eiou in tqdm(
            predictor.propagate_in_video(inference_state, start_frame_idx=0),
            total=len(progress_order),
            desc="Processing frames"
        ):
            # Move to CPU immediately to save GPU memory
            pred_masks_list.append(out_mask_logits.cpu())
            pred_eiou_list.append(pred_eiou.cpu())

            # Clear GPU cache periodically
            if len(pred_masks_list) % 50 == 0:
                torch.cuda.empty_cache()

        # Aggregate results on CPU first to save GPU memory
        print("Aggregating results...")
        print_memory_status("Before aggregation")
        print(f"Concatenating {len(pred_masks_list)} mask tensors...")
        pred_masks = torch.cat(pred_masks_list, dim=1)
        print(f"Stacking {len(pred_eiou_list)} IoU tensors...")
        pred_eiou_stack = torch.stack(pred_eiou_list)
        pred_stability_scores = None
        print_memory_status("After tensor operations")

        print("Removing padding frames...")
        padding_tensor = torch.tensor(padding_mask, dtype=torch.bool, device=pred_masks.device)
        pred_masks = pred_masks[:, ~padding_tensor].to(torch.float16).contiguous()
        valid_steps = ~torch.tensor(padding_mask, dtype=torch.bool)
        pred_eiou_stack = pred_eiou_stack[valid_steps]
        print_memory_status("After padding removal")

        # Persist raw predictions so post-processing can be rerun without propagation
        print("Saving propagation outputs to cache...")
        print_memory_status("Before cache save")
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(
            {
                "pred_masks": pred_masks,
                "pred_eiou_stack": pred_eiou_stack,
                "padding_mask": padding_mask,
                "out_size": out_size,
                "frame_names": frame_names,
            },
            cache_path,
        )
        print(f"Saved propagation outputs for reuse at: {cache_path}")
        print_memory_status("After cache save")
        del pred_masks_list
        del pred_eiou_list

    # Derive IoU mean on demand
    pred_eious = pred_eiou_stack.mean(0)

    device_label = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Post-processing masks in {device_label} chunks...")
    result = inference_video_vps_save_results(
        pred_eious,
        pred_stability_scores,
        pred_masks,
        out_size,
        chunk_size=16
    )

    # Save results
    print("Saving results...")
    anno = process(video_id, frame_names, result, categories_dict, args.output_dir, video_dir)

    # Clean up memory-mapped file if it was used
    if '_memmap_path' in result:
        print(f"[MEM] Cleaning up memory-mapped file: {result['_memmap_path']}")
        try:
            os.unlink(result['_memmap_path'])
            print(f"[MEM] Memory-mapped file deleted successfully")
        except Exception as e:
            print(f"[MEM] Warning: Could not delete memmap file: {e}")

    # Save metadata JSON
    json_path = os.path.join(args.output_dir, video_id, 'pred.json')
    with open(json_path, 'w') as f:
        json.dump({'annotations': [anno]}, f, indent=2)

    # Calculate and save statistics
    end_time = time.time()
    processing_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    final_gpu_mem = torch.cuda.memory_allocated() / 1024**3

    stats = {
        "video_id": video_id,
        "num_frames": len(frame_names),
        "processing_time_sec": processing_time,
        "fps": len(frame_names) / processing_time,
        "peak_gpu_memory_gb": peak_memory,
        "final_gpu_memory_gb": final_gpu_mem,
        "num_segments_detected": len(result['segments_infos'])
    }

    stats_path = os.path.join(args.output_dir, video_id, 'stats.txt')
    with open(stats_path, 'w') as f:
        f.write("EntitySAM Inference Statistics\n")
        f.write("=" * 50 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

    # Print summary
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Video ID: {video_id}")
    print(f"Frames processed: {len(frame_names)}")
    print(f"Entities detected: {len(result['segments_infos'])}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"FPS: {len(frame_names) / processing_time:.2f}")
    print(f"Peak GPU memory: {peak_memory:.2f} GB")
    print(f"\nResults saved to:")
    print(f"  Panoptic masks: {os.path.join(args.output_dir, video_id, 'pan_pred/')}")
    print(f"  Metadata: {json_path}")
    print(f"  Statistics: {stats_path}")
    print("=" * 60)

    # Cleanup temporary frames if needed
    if cleanup_frames:
        import shutil
        print(f"\nCleaning up temporary frames: {video_dir}")
        shutil.rmtree(video_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Script failed: {e}")
        print_memory_status("Error state")
        raise
