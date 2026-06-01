import argparse
import json
import os
import sys
import time
import tempfile
import shutil
import gc
import traceback
import subprocess
import numpy as np
import torch
import cv2
import pandas as pd
from natsort import natsorted
from panopticapi.utils import IdGenerator, rgb2id
from PIL import Image
from torch.nn import functional as F

from sam2.build_sam import build_sam2_video_query_iou_predictor

import psutil


# =============================================================================
# MEMORY UTILITIES
# =============================================================================

def get_memory_info():
    """Get detailed memory information."""
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        gpu_max = torch.cuda.max_memory_allocated() / 1024**3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_free = gpu_total - gpu_allocated
    else:
        gpu_allocated = gpu_reserved = gpu_max = gpu_total = gpu_free = 0

    process = psutil.Process()
    mem_info = process.memory_info()
    cpu_rss = mem_info.rss / 1024**3

    vm = psutil.virtual_memory()
    total_ram = vm.total / 1024**3
    available_ram = vm.available / 1024**3
    used_ram = vm.used / 1024**3
    percent_ram = vm.percent

    return {
        'gpu_allocated': gpu_allocated,
        'gpu_reserved': gpu_reserved,
        'gpu_max': gpu_max,
        'gpu_total': gpu_total,
        'gpu_free': gpu_free,
        'cpu_process': cpu_rss,
        'total_ram': total_ram,
        'available_ram': available_ram,
        'used_ram': used_ram,
        'percent_ram': percent_ram
    }


def get_cpu_memory_info():
    """Get current CPU memory usage (lightweight version for post-processing)."""
    process = psutil.Process()
    memory_info = process.memory_info()
    virtual_memory = psutil.virtual_memory()
    return {
        'process_rss_gb': memory_info.rss / 1024**3,
        'process_vms_gb': memory_info.vms / 1024**3,
        'system_available_gb': virtual_memory.available / 1024**3,
        'system_used_percent': virtual_memory.percent
    }


def print_memory_debug(stage="", verbose=True):
    """Print memory debug information."""
    mem = get_memory_info()

    print(f"\n{'='*70}")
    print(f"MEMORY DEBUG - {stage}")
    print(f"{'='*70}")
    print(f"GPU Memory:")
    print(f"  Allocated:  {mem['gpu_allocated']:>8.2f} GB / {mem['gpu_total']:.2f} GB ({mem['gpu_allocated']/max(mem['gpu_total'],1)*100:.1f}%)")
    print(f"  Reserved:   {mem['gpu_reserved']:>8.2f} GB")
    print(f"  Free:       {mem['gpu_free']:>8.2f} GB")
    print(f"  Peak:       {mem['gpu_max']:>8.2f} GB")
    print(f"\nCPU/RAM Memory:")
    print(f"  Process:    {mem['cpu_process']:>8.2f} GB")
    print(f"  System:     {mem['used_ram']:>8.2f} GB / {mem['total_ram']:.2f} GB ({mem['percent_ram']:.1f}%)")
    print(f"  Available:  {mem['available_ram']:>8.2f} GB")

    if mem['available_ram'] < 5.0:
        print(f"\nWARNING: Low available RAM ({mem['available_ram']:.2f} GB). Risk of OOM.")
    if mem['gpu_free'] < 2.0:
        print(f"\nWARNING: Low GPU memory ({mem['gpu_free']:.2f} GB). Risk of OOM.")

    print(f"{'='*70}\n")
    return mem


def print_memory_status(label=""):
    """Lightweight memory status print (used inside post-processing loops)."""
    cpu_mem = get_cpu_memory_info()
    print(f"[MEM{' ' + label if label else ''}] CPU RSS: {cpu_mem['process_rss_gb']:.2f} GB, "
          f"VMS: {cpu_mem['process_vms_gb']:.2f} GB")
    print(f"[MEM{' ' + label if label else ''}] System available: {cpu_mem['system_available_gb']:.2f} GB "
          f"({100 - cpu_mem['system_used_percent']:.1f}% free)")
    if torch.cuda.is_available():
        print(f"[MEM{' ' + label if label else ''}] GPU allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"[MEM{' ' + label if label else ''}] GPU reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


# =============================================================================
# FRAME EXTRACTION
# =============================================================================

def extract_frames_from_video(video_path, output_dir, target_fps=None):
    """
    Extract frames from video file to a temporary directory.

    Args:
        video_path:  Path to video file
        output_dir:  Directory to save frames
        target_fps:  Target FPS for extraction (None = original FPS)

    Returns:
        Tuple of (list[str] frame filenames, float effective_fps)
    """
    print(f"Extracting frames from video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Original video: {original_fps:.2f} FPS, {total_frames} total frames")

    if target_fps is None or target_fps >= original_fps:
        frame_interval = 1
        effective_fps = original_fps
        print(f"Using original FPS: {original_fps:.2f}")
    else:
        frame_interval = int(original_fps / target_fps)
        effective_fps = original_fps / frame_interval
        print(f"Target FPS: {target_fps}, Frame interval: {frame_interval}, "
              f"Effective FPS: {effective_fps:.2f}")

    frame_count = 0
    extracted_count = 0
    frame_names = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{extracted_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(frame_rgb).save(frame_path)
            frame_names.append(frame_filename)
            extracted_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {extracted_count} frames out of {total_frames} total frames")
    print(f"Memory reduction: {(1 - extracted_count / total_frames) * 100:.1f}% fewer frames")
    return frame_names, effective_fps


# =============================================================================
# POST-PROCESSING  (Script 2's chunked + memmap version)
# =============================================================================

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
    Post-process multi-mask predictions into panoptic segmentation.

    Uses chunked GPU/CPU processing with an optional memory-mapped fallback
    for very large videos (>10 GB panoptic buffer), taken from Script 2.
    """
    print(f"[Post-processing] Starting inference_video_vps_save_results")
    print(f"[Post-processing] Input shapes — pred_masks: {pred_masks.shape}, out_size: {out_size}")
    print_memory_status("Start")

    scores = pred_ious
    labels = torch.randint(0, 123, (len(scores),), device=scores.device)
    pred_id = torch.arange(len(scores), device=scores.device)

    keep = scores >= max(
        object_mask_threshold,
        scores.topk(k=min(len(scores), test_topk_per_image))[0][-1]
    )
    cur_scores  = scores[keep]
    cur_classes = labels[keep]
    cur_masks   = pred_masks[keep].contiguous()
    cur_ids     = pred_id[keep]

    if pred_stability_scores is not None:
        mask_quality_scores = pred_stability_scores[keep]
    else:
        from train.utils.comm import calculate_mask_quality_scores
        mask_quality_scores = calculate_mask_quality_scores(cur_masks[:, ::5].float())

    del pred_masks
    print(f"[Post-processing] After filtering — instances: {cur_masks.shape[0]}, "
          f"frames: {cur_masks.shape[1]}")
    print_memory_status("After filtering")

    num_instances, num_frames = cur_masks.shape[0], cur_masks.shape[1]
    estimated_size = num_frames * out_size[0] * out_size[1] * 4 / 1024**3
    print(f"[Post-processing] Panoptic buffer estimate: "
          f"{num_frames} x {out_size[0]} x {out_size[1]} = {estimated_size:.2f} GB")

    cpu_mem = get_cpu_memory_info()
    if estimated_size > cpu_mem['system_available_gb'] * 0.8:
        print(f"[Post-processing] WARNING: estimated size ({estimated_size:.2f} GB) may exceed "
              f"available RAM ({cpu_mem['system_available_gb']:.2f} GB)")

    # ---- allocate panoptic buffer (memmap if >10 GB) -------------------------
    import atexit
    use_memmap = False
    memmap_path = None

    if estimated_size > 10.0:
        print(f"[Post-processing] Using memory-mapped file (size > 10 GB)")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.dat')
        tmp.close()
        memmap_path = tmp.name
        panoptic_seg = np.memmap(memmap_path, dtype='int32', mode='w+',
                                 shape=(num_frames, out_size[0], out_size[1]))
        panoptic_seg[:] = 0
        use_memmap = True

        def _cleanup_memmap():
            try:
                if os.path.exists(memmap_path):
                    os.unlink(memmap_path)
            except Exception:
                pass
        atexit.register(_cleanup_memmap)
        print_memory_status("After memmap creation")
    else:
        panoptic_seg = torch.zeros(
            (num_frames, out_size[0], out_size[1]), dtype=torch.int32, device="cpu"
        )
        print_memory_status("After panoptic_seg allocation")

    segments_infos    = []
    out_ids           = []
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

    use_cuda   = torch.cuda.is_available()
    device     = torch.device("cuda") if use_cuda else torch.device("cpu")
    mask_dtype = torch.float16 if use_cuda else torch.float32

    cur_scores = cur_scores.to(device=device, dtype=mask_dtype)
    cur_masks  = cur_masks.to(mask_dtype).contiguous()
    print_memory_status("After mask conversion")

    mask_area_totals     = torch.zeros(num_instances, dtype=torch.float64, device="cpu")
    original_area_totals = torch.zeros(num_instances, dtype=torch.float64, device="cpu")

    num_chunks = (num_frames + chunk_size - 1) // chunk_size
    print(f"[Post-processing] Pass 1: {num_chunks} chunks of size {chunk_size}")

    # ---- Pass 1: accumulate per-instance statistics --------------------------
    from tqdm import tqdm
    for chunk_idx, start in enumerate(tqdm(
        range(0, num_frames, chunk_size),
        total=num_chunks,
        desc="Accumulating mask stats",
        leave=False,
    )):
        end = min(start + chunk_size, num_frames)
        if chunk_idx % 10 == 0:
            print(f"[Post-processing] Pass 1 — chunk {chunk_idx}/{num_chunks} "
                  f"(frames {start}-{end})")
            print_memory_status(f"Pass1-Chunk{chunk_idx}")

        chunk_cpu  = cur_masks[:, start:end]
        chunk      = chunk_cpu.to(device, dtype=mask_dtype) if use_cuda else chunk_cpu
        chunk_probs  = F.interpolate(chunk, size=out_size, mode="bilinear",
                                     align_corners=False).sigmoid()
        chunk_binary = chunk_probs >= mask_binary_threshold

        cur_prob_masks  = cur_scores.view(-1, 1, 1, 1) * chunk_probs
        chunk_mask_ids  = cur_prob_masks.argmax(0).to(torch.int64)
        bg_mask         = ~chunk_binary.any(dim=0)
        chunk_mask_ids[bg_mask] = -1

        chunk_binary_cpu   = chunk_binary.cpu()
        chunk_mask_ids_cpu = chunk_mask_ids.cpu()

        for inst_idx in range(num_instances):
            inst_binary = chunk_binary_cpu[inst_idx]
            original_area_totals[inst_idx] += inst_binary.sum(dtype=torch.float64)
            mask = (chunk_mask_ids_cpu == inst_idx) & inst_binary
            mask_area_totals[inst_idx] += mask.sum(dtype=torch.float64)

        del chunk, chunk_probs, chunk_binary, cur_prob_masks, \
            chunk_mask_ids, chunk_binary_cpu, chunk_mask_ids_cpu
        if use_cuda:
            torch.cuda.empty_cache()

    print(f"[Post-processing] Pass 1 complete.")
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

    # ---- Build keep list -----------------------------------------------------
    keep_entries      = []
    stuff_memory_list = {}

    for k in range(num_instances):
        mask_area     = mask_area_totals[k].item()
        original_area = original_area_totals[k].item()
        if mask_area <= 0 or original_area <= 0:
            continue
        if mask_area / max(original_area, 1.0) < overlap_threshold:
            continue

        pred_class = int(cur_classes[k]) + 1
        isthing    = True

        if not isthing:
            if pred_class in stuff_memory_list:
                keep_entries.append((k, stuff_memory_list[pred_class], pred_class))
                continue
            else:
                stuff_memory_list[pred_class] = current_segment_id + 1

        current_segment_id += 1
        keep_entries.append((k, current_segment_id, pred_class))
        segments_infos.append({
            "id": current_segment_id,
            "isthing": bool(isthing),
            "category_id": pred_class
        })
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

    keep_indices          = [e[0] for e in keep_entries]
    keep_index_to_segment = {e[0]: e[1] for e in keep_entries}

    # ---- Pass 2: paint panoptic segmentation ---------------------------------
    print(f"[Post-processing] Pass 2: painting panoptic masks ({num_chunks} chunks)")
    for start in tqdm(
        range(0, num_frames, chunk_size),
        total=num_chunks,
        desc="Painting panoptic masks",
        leave=False,
    ):
        end = min(start + chunk_size, num_frames)
        chunk_cpu  = cur_masks[:, start:end]
        chunk      = chunk_cpu.to(device, dtype=mask_dtype) if use_cuda else chunk_cpu
        chunk_probs  = F.interpolate(chunk, size=out_size, mode="bilinear",
                                     align_corners=False).sigmoid()
        chunk_binary = chunk_probs >= mask_binary_threshold

        cur_prob_masks = cur_scores.view(-1, 1, 1, 1) * chunk_probs
        chunk_mask_ids = cur_prob_masks.argmax(0).to(torch.int64)
        chunk_mask_ids[~chunk_binary.any(dim=0)] = -1

        chunk_mask_ids_cpu = chunk_mask_ids.cpu()
        chunk_binary_cpu   = chunk_binary.cpu()

        for inst_idx in keep_indices:
            seg_id   = keep_index_to_segment[inst_idx]
            mask_cpu = (chunk_mask_ids_cpu == inst_idx) & chunk_binary_cpu[inst_idx]
            if mask_cpu.any():
                if use_memmap:
                    panoptic_seg[start:end][mask_cpu.numpy()] = seg_id
                else:
                    panoptic_seg[start:end][mask_cpu] = seg_id

        del chunk, chunk_probs, chunk_binary, cur_prob_masks, \
            chunk_mask_ids, chunk_mask_ids_cpu, chunk_binary_cpu
        if use_cuda:
            torch.cuda.empty_cache()

    del cur_masks
    if use_cuda:
        torch.cuda.empty_cache()

    result = {
        "image_size": out_size,
        "pred_masks": panoptic_seg,
        "segments_infos": segments_infos,
        "pred_ids": out_ids,
        "task": "vps",
    }
    if use_memmap:
        result["_memmap_path"] = memmap_path

    return result


# =============================================================================
# VIDEO OUTPUT
# =============================================================================

def create_video_from_masks(pan_format, output_video_path, fps=30):
    """
    Encode segmentation-coloured frames directly to MP4 via ffmpeg pipe.

    Args:
        pan_format:         numpy array (num_frames, H, W, 3) RGB uint8
        output_video_path:  output MP4 path
        fps:                frames per second
    """
    num_frames, height, width, _ = pan_format.shape
    print(f"Creating segmentation video: {output_video_path}")
    print(f"  Resolution: {width}x{height}, Frames: {num_frames}, FPS: {fps}")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        output_video_path
    ]

    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        for i in range(num_frames):
            process.stdin.write(pan_format[i].tobytes())
            if (i + 1) % 50 == 0:
                print(f"  Encoded {i + 1}/{num_frames} frames...")
        process.stdin.close()
        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {process.stderr.read().decode()}")

        print(f"  Video created successfully: {output_video_path}")
    except Exception:
        process.kill()
        process.wait()
        raise


# =============================================================================
# SAVE RESULTS
# =============================================================================

def process(video_id, frame_names, outputs, categories_dict, output_dir, video_dir, fps=30):
    """Save panoptic segmentation as a single MP4 (no per-frame PNGs)."""
    color_generator = IdGenerator(categories_dict)

    img_shape       = outputs["image_size"]
    pan_seg_result  = outputs["pred_masks"]
    segments_infos  = outputs["segments_infos"]
    segments_infos_ = []

    pan_format = np.zeros(
        (pan_seg_result.shape[0], img_shape[0], img_shape[1], 3), dtype=np.uint8
    )

    for seg_info in segments_infos:
        seg_id   = seg_info["id"]
        sem      = seg_info["category_id"]

        # pan_seg_result may be a torch tensor or numpy memmap
        if hasattr(pan_seg_result, 'numpy'):
            mask = (pan_seg_result == seg_id).numpy()
        else:
            mask = pan_seg_result == seg_id

        color = color_generator.get_color(sem)
        pan_format[mask] = color

        dt_base = {"category_id": int(sem) - 1, "iscrowd": 0, "id": int(rgb2id(color))}
        dts = []
        for i in range(pan_format.shape[0]):
            area = int(mask[i].sum())
            if area == 0:
                dts.append(None)
            else:
                rows, cols = np.where(mask[i])
                x, y = int(cols.min()), int(rows.min())
                w, h = int(cols.max() - x), int(rows.max() - y)
                dt = {"bbox": [x, y, w, h], "area": area}
                dt.update(dt_base)
                dts.append(dt)

        segments_infos_.append(dts)

    print(f"  Creating segmentation video without saving individual frames...")
    video_output_path = os.path.join(output_dir, f"{video_id}_segmentation.mp4")
    create_video_from_masks(pan_format, video_output_path, fps=fps)

    annotations = []
    for i, image_name in enumerate(frame_names):
        annotations.append({
            "segments_info": [
                item[i] for item in segments_infos_ if item[i] is not None
            ],
            "file_name": image_name.split("/")[-1],
        })
        if (i + 1) % 50 == 0:
            print(f"  Annotated {i + 1}/{len(frame_names)} frames...")

    del outputs
    del pan_format
    return {"annotations": annotations, "video_id": video_id, "video_path": video_output_path}


# =============================================================================
# SINGLE VIDEO PIPELINE
# =============================================================================

def process_single_video(video_path, output_dir, predictor, categories_dict,
                         temp_base_dir, target_fps=None):
    """
    Full pipeline for one video:
      1. Extract frames
      2. Propagate (or load from cache)
      3. Post-process (chunked + memmap)
      4. Save segmentation video + pred.json
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    temp_dir = tempfile.mkdtemp(prefix=f"video_frames_{video_id}_", dir=temp_base_dir)

    print(f"\nProcessing video: {video_id}")
    print(f"Temporary directory: {temp_dir}")

    try:
        frame_names, effective_fps = extract_frames_from_video(
            video_path, temp_dir, target_fps=target_fps
        )

        # ------------------------------------------------------------------ #
        #  CACHING  (from Script 2)                                           #
        #  Cache lives in <output_dir>/cache/raw_predictions.pt               #
        # ------------------------------------------------------------------ #
        cache_dir  = os.path.join(output_dir, "cache")
        cache_path = os.path.join(cache_dir, "raw_predictions.pt")

        if os.path.exists(cache_path):
            # ---- load from cache, skip propagation -------------------------
            print(f"\nFound cached propagation outputs at: {cache_path}")
            print("Loading cached tensors to skip temporal propagation...")
            cache       = torch.load(cache_path, map_location="cpu")
            pred_masks  = cache["pred_masks"].to(torch.float16).contiguous()
            pred_eiou_stack = cache["pred_eiou_stack"]
            padding_mask    = cache["padding_mask"]
            out_size        = tuple(cache["out_size"])
            frame_names     = cache["frame_names"]
            pred_stability_scores = None
            start_time = time.time()
            torch.cuda.reset_peak_memory_stats()
            print(f"Cache loaded — skipping propagation for {video_id}")

        else:
            # ---- run propagation and then cache results --------------------
            start_time = time.time()
            torch.cuda.reset_peak_memory_stats()

            # Padding frames for temporal stability
            progress_order = [0, 0, 0] + list(range(0, len(frame_names)))
            padding_mask   = [True] * 3 + [False] * len(frame_names)

            inference_state = predictor.init_state(
                video_path=temp_dir,
                progress_order=progress_order,
                offload_state_to_cpu=True
            )
            predictor.reset_state(inference_state)

            # Derive output size from first frame
            init_frame = Image.open(os.path.join(temp_dir, frame_names[0]))
            init_frame = np.array(init_frame.convert("RGB"))
            out_size   = (init_frame.shape[0], init_frame.shape[1])

            pred_masks_list  = []
            pred_eiou_list   = []

            print(f"Starting temporal propagation for {len(frame_names)} frames...")
            for frame_idx, (_, _, out_mask_logits, pred_eiou) in enumerate(
                predictor.propagate_in_video(inference_state, start_frame_idx=0)
            ):
                pred_masks_list.append(out_mask_logits.half().cpu())
                pred_eiou_list.append(pred_eiou.half().cpu())

                if frame_idx % 50 == 0:
                    print(f"  Propagated frame {frame_idx}/{len(frame_names) + 3}")
                    torch.cuda.empty_cache()
                    gc.collect()

            print("Propagation complete")
            print_memory_debug("After Propagation")

            # Clean up predictor state before heavy tensor ops
            del inference_state
            torch.cuda.empty_cache()
            gc.collect()

            # ---- concatenate masks ----------------------------------------
            print(f"\nConcatenating {len(pred_masks_list)} mask tensors...")
            total_mask_size = sum(
                m.element_size() * m.nelement() for m in pred_masks_list
            ) / 1024**3
            print(f"Total mask tensor size: {total_mask_size:.2f} GB")

            try:
                pred_masks = torch.cat(pred_masks_list, dim=1)
            except RuntimeError:
                print("Direct concat failed — using chunk-wise concatenation...")
                chunk_size_cat = 50
                chunks = []
                for i in range(0, len(pred_masks_list), chunk_size_cat):
                    chunks.append(torch.cat(pred_masks_list[i:i + chunk_size_cat], dim=1))
                    torch.cuda.empty_cache()
                    gc.collect()
                pred_masks = torch.cat(chunks, dim=1)
                del chunks
                gc.collect()

            del pred_masks_list
            gc.collect()

            # ---- stack eious ----------------------------------------------
            pred_eiou_stack = torch.stack(pred_eiou_list)
            del pred_eiou_list
            gc.collect()
            print_memory_debug("After Stacking eIoUs")

            # ---- remove padding -------------------------------------------
            padding_tensor = torch.tensor(padding_mask, dtype=torch.bool)
            pred_masks     = pred_masks[:, ~padding_tensor].to(torch.float16).contiguous()
            pred_eiou_stack = pred_eiou_stack[~padding_tensor]
            pred_stability_scores = None
            print_memory_debug("After Padding Removal")

            # ---- save cache -----------------------------------------------
            print("Saving propagation outputs to cache...")
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(
                {
                    "pred_masks":      pred_masks,
                    "pred_eiou_stack": pred_eiou_stack,
                    "padding_mask":    padding_mask,
                    "out_size":        out_size,
                    "frame_names":     frame_names,
                },
                cache_path,
            )
            print(f"Cache saved: {cache_path}")
            print_memory_debug("After Cache Save")

        # ------------------------------------------------------------------ #
        #  POST-PROCESSING  (Script 2's chunked + memmap version)             #
        # ------------------------------------------------------------------ #
        pred_eious = pred_eiou_stack.mean(0)

        device_label = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"\nPost-processing masks in {device_label} chunks...")

        result = inference_video_vps_save_results(
            pred_eious,
            pred_stability_scores,
            pred_masks,
            out_size,
            chunk_size=16,
        )

        # ------------------------------------------------------------------ #
        #  SAVE RESULTS                                                        #
        # ------------------------------------------------------------------ #
        print(f"\nSaving {len(frame_names)} frame annotations...")
        anno = process(
            video_id, frame_names, result, categories_dict,
            output_dir, temp_dir, fps=effective_fps
        )

        # Clean up memmap file if it was used
        if "_memmap_path" in result:
            print(f"[MEM] Cleaning up memory-mapped file: {result['_memmap_path']}")
            try:
                os.unlink(result["_memmap_path"])
                print("[MEM] Memory-mapped file deleted.")
            except Exception as e:
                print(f"[MEM] Warning: could not delete memmap file: {e}")

        # ---- write pred.json ----------------------------------------------
        json_path = os.path.join(output_dir, "pred.json")
        with open(json_path, "w") as f:
            json.dump({"annotations": [anno]}, f)

        end_time        = time.time()
        processing_time = end_time - start_time
        peak_memory     = torch.cuda.max_memory_allocated() / 1024**3
        final_gpu_mem   = torch.cuda.memory_allocated() / 1024**3

        print(f"\n{'='*60}")
        print(f"Video {video_id} Performance Metrics:")
        print(f"{'='*60}")
        print(f"Total frames processed : {len(frame_names)}")
        print(f"Processing time        : {processing_time:.2f} seconds")
        print(f"FPS                    : {len(frame_names) / processing_time:.2f}")
        print(f"Peak GPU Memory        : {peak_memory:.2f} GB")
        print(f"Final GPU Memory       : {final_gpu_mem:.2f} GB")
        print(f"{'='*60}\n")

        print(f"Results saved to: {output_dir}")
        print(f"  - Segmentation video : {anno['video_path']}")
        print(f"  - Annotations JSON   : {json_path}")

        return {
            "video_id":        video_id,
            "video_path":      video_path,
            "output_dir":      output_dir,
            "output_video":    anno["video_path"],
            "frames_processed": len(frame_names),
            "processing_time": processing_time,
            "peak_memory_gb":  peak_memory,
            "status":          "success",
        }

    except MemoryError as e:
        print(f"\n{'!'*70}")
        print("MEMORY ERROR DETECTED")
        print(f"{'!'*70}")
        print(f"Error: {e}")
        print_memory_debug("At Memory Error")
        traceback.print_exc()
        return {"video_id": video_id, "video_path": video_path,
                "status": "failed", "error": str(e)}

    except Exception as e:
        print(f"\n{'!'*70}")
        print("CRITICAL ERROR")
        print(f"{'!'*70}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {str(e)}")
        print_memory_debug("At Error Point")
        traceback.print_exc()
        return {"video_id": video_id, "video_path": video_path,
                "status": "failed", "error": str(e)}

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")


# =============================================================================
# MAIN — SLURM batch entrypoint
# =============================================================================

def main():
    NUM_CATEGORIES = 256

    parser = argparse.ArgumentParser(
        description="Batch process videos with EntitySAM using SLURM array jobs"
    )
    parser.add_argument("--csv_file",           type=str,   required=True,
                        help="CSV with 'input_path' and 'output_path' columns")
    parser.add_argument("--ckpt_dir",           type=str,   required=True,
                        help="Checkpoint directory")
    parser.add_argument("--model_cfg",          type=str,
                        default="configs/sam2.1_hiera_l.yaml",
                        help="SAM2 model config file")
    parser.add_argument("--mask_decoder_depth", type=int,   default=8,
                        help="Mask decoder depth")
    parser.add_argument("--target_fps",         type=float, default=None,
                        help="Target FPS for frame extraction (None = original FPS)")
    parser.add_argument("--task_id",            type=int,   required=True,
                        help="SLURM array task ID")
    parser.add_argument("--num_tasks",          type=int,   required=True,
                        help="Total number of SLURM array tasks")
    parser.add_argument("--temp_dir",           type=str,   default=None,
                        help="Base directory for temporary files")

    args = parser.parse_args()

    print("=" * 70)
    print("EntitySAM Batch Processing")
    print("=" * 70)
    print(f"Task ID      : {args.task_id}")
    print(f"Total tasks  : {args.num_tasks}")
    print(f"CSV file     : {args.csv_file}")
    print(f"Checkpoint   : {args.ckpt_dir}")
    print(f"Model config : {args.model_cfg}")
    print(f"Target FPS   : {args.target_fps}")
    print("=" * 70 + "\n")

    for path, label in [(args.csv_file, "CSV file"), (args.ckpt_dir, "Checkpoint dir")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    print("Loading video list from CSV...")
    try:
        df = pd.read_csv(args.csv_file).iloc[:100]
        required = ['input_path', 'output_path']
        if not all(c in df.columns for c in required):
            print(f"ERROR: CSV must have columns: {required}")
            sys.exit(1)
        df = df.dropna(subset=required)
        print(f"Loaded {len(df)} video entries from CSV")
    except Exception as e:
        print(f"ERROR: Failed to load CSV: {e}")
        sys.exit(1)

    # Split work across SLURM tasks
    video_chunks = [
        {'input_path': df.iloc[idx]['input_path'],
         'output_path': df.iloc[idx]['output_path']}
        for idx in range(args.task_id, len(df), args.num_tasks)
    ]
    print(f"Task {args.task_id}: processing {len(video_chunks)} videos\n")

    if not video_chunks:
        print("No videos assigned to this task. Exiting.")
        sys.exit(0)

    # Task-specific temp directory
    task_temp_dir = os.path.join(args.temp_dir, f"entitysam_task_{args.task_id}")
    os.makedirs(task_temp_dir, exist_ok=True)
    print(f"Task temp directory: {task_temp_dir}\n")

    # PyTorch optimisations
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load model
    sam2_checkpoint = os.path.join(args.ckpt_dir, "model_0009999.pth")
    if not os.path.exists(sam2_checkpoint):
        print(f"ERROR: Checkpoint not found: {sam2_checkpoint}")
        sys.exit(1)

    print("Loading SAM2 model...")
    predictor = build_sam2_video_query_iou_predictor(
        args.model_cfg, sam2_checkpoint,
        mask_decoder_depth=args.mask_decoder_depth
    )
    print("Model loaded successfully\n")

    # Categories
    rng = np.random.default_rng(42)
    categories_dict = {
        cat_id: {
            "id": cat_id,
            "isthing": 1,
            "color": rng.integers(0, 256, size=3).tolist(),
        }
        for cat_id in range(1, NUM_CATEGORIES + 1)
    }

    all_results = []
    all_failed  = []
    start_time  = time.time()

    for i, video_info in enumerate(video_chunks, 1):
        input_path  = video_info['input_path']
        output_path = video_info['output_path']

        print(f"\n{'='*70}")
        print(f"Processing video {i}/{len(video_chunks)}")
        print(f"{'='*70}")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")

        if not os.path.exists(input_path):
            print(f"WARNING: Video file not found: {input_path}")
            all_failed.append({
                "input_path": input_path, "output_path": output_path,
                "status": "failed", "error": "File not found"
            })
            continue

        try:
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)

            result = process_single_video(
                input_path, output_dir, predictor, categories_dict,
                task_temp_dir, target_fps=args.target_fps
            )
            result['expected_output_path'] = output_path

            if result["status"] == "success":
                all_results.append(result)
                print(f"\n✓ Successfully processed: {input_path}")
            else:
                all_failed.append(result)
                print(f"\n✗ Failed: {input_path}")

        except Exception as e:
            print(f"ERROR processing {input_path}: {e}")
            traceback.print_exc()
            all_failed.append({
                "input_path": input_path, "output_path": output_path,
                "status": "failed", "error": str(e)
            })

        torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - start_time

    # Save logs
    if video_chunks:
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(video_chunks[0]['output_path'])), "logs"
        )
    else:
        log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, f"task_{args.task_id}_success.json"), "w") as f:
        json.dump(all_results, f, indent=4)
    with open(os.path.join(log_dir, f"task_{args.task_id}_failed.json"), "w") as f:
        json.dump(all_failed, f, indent=4)

    print(f"\n{'='*70}")
    print(f"TASK {args.task_id} SUMMARY")
    print(f"{'='*70}")
    print(f"Total assigned  : {len(video_chunks)}")
    print(f"Succeeded       : {len(all_results)}")
    print(f"Failed          : {len(all_failed)}")
    print(f"Total time      : {total_time / 3600:.2f} h ({total_time / 60:.1f} min)")

    if all_results:
        total_frames = sum(r["frames_processed"] for r in all_results)
        print(f"Total frames    : {total_frames}")
        print(f"Overall FPS     : {total_frames / total_time:.2f}")

    print(f"Logs saved to   : {log_dir}")
    print(f"{'='*70}\n")

    if os.path.exists(task_temp_dir):
        shutil.rmtree(task_temp_dir)
        print(f"Cleaned up task temp directory: {task_temp_dir}")

    print("Task complete.")


if __name__ == "__main__":
    main()