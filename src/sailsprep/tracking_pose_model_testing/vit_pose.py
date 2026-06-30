"""ViTPose pose estimation pipeline.

Usage (SLURM array):
    python vit_pose.py --array_index $SLURM_ARRAY_TASK_ID --num_jobs $SLURM_ARRAY_TASK_COUNT
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, VitPoseForPoseEstimation

# ── Type aliases ──────────────────────────────────────────────────────────────
KpDict    = dict[str, tuple[float, float, float]]       # kp_name → (x, y, conf)
PoseStore = dict[int, KpDict]                            # frame_idx → KpDict
BboxMap   = dict[int, tuple[int, int, int, int]]         # frame_idx → (x1,y1,x2,y2)

# ── Fixed hyper-params ────────────────────────────────────────────────────────
SCORE_THRESHOLD = 0.3
ANN_FPS         = 15.0

WINDOW          = 7
SIZE_THRESH     = 0.25
CENTER_THRESH   = 0.20
NEIGHBOR_AGREE  = 0.25
N_CLEAN_PASSES  = 2
BBOX_AR_THRESH  = 0.30

KP_JUMP_THRESH   = 0.25
KP_MEMORY_FRAMES = 10

POST_JUMP_THRESH    = 0.30
POST_WINDOW         = 7
POST_NEIGHBOR_AGREE = 0.25
POST_N_PASSES       = 2

SUSPECT_SCORE_THRESHOLD = 0.55
KP_SCORE_W      = 0.35
KPC_SCORE_W     = 0.25
KPAR_SCORE_W    = 0.25
BBOX_AR_SCORE_W = 0.15


# ═══════════════════════════════════════════════════════════════════════════════
#  H5 bbox loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_bbox_map(h5_path: str) -> BboxMap:
    with h5py.File(h5_path, "r") as f:
        table = f["bboxes/table"][()]
    vb1 = table["values_block_1"]
    return {int(r[0]): (int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in vb1}


# ═══════════════════════════════════════════════════════════════════════════════
#  Bbox cleanup
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_one_pass(
    arr: np.ndarray,
    suspect: np.ndarray,
    window: int,
    edge_thresh: float,
    center_thresh: float,
    neighbor_agree: float,
    ar_thresh: float,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    N = len(arr)
    cleaned = arr.copy()
    new_suspect = suspect.copy()
    n_edges = n_centers = n_ar = n_frames = 0

    for i in range(N):
        bi = [k for k in range(max(0, i - window), i)          if not suspect[k]]
        ai = [k for k in range(i + 1, min(N, i + 1 + window)) if not suspect[k]]
        if len(bi) < 2 or len(ai) < 2:
            continue

        before   = arr[bi]
        after    = arr[ai]
        combined = np.vstack([before, after])

        med_before = np.median(before,   axis=0)
        med_after  = np.median(after,    axis=0)
        med_all    = np.median(combined, axis=0)

        bw_m = med_all[2] - med_all[0]
        bh_m = med_all[3] - med_all[1]
        if bw_m <= 0 or bh_m <= 0:
            continue

        edge_tol = np.array([bw_m, bh_m, bw_m, bh_m]) * neighbor_agree
        if (np.abs(med_after - med_before) > edge_tol).any():
            continue

        current = cleaned[i].copy()
        touched = False

        edge_norm     = np.array([bw_m, bh_m, bw_m, bh_m])
        deviations    = np.abs(current - med_all) / edge_norm
        outlier_edges = deviations > edge_thresh
        if outlier_edges.any():
            current[outlier_edges] = med_all[outlier_edges]
            n_edges += int(outlier_edges.sum())
            touched = True

        cx_cur  = (current[0] + current[2]) / 2
        cy_cur  = (current[1] + current[3]) / 2
        cx_med  = (med_all[0] + med_all[2]) / 2
        cy_med  = (med_all[1] + med_all[3]) / 2
        diag_m  = np.sqrt(bw_m**2 + bh_m**2)
        c_shift = np.sqrt((cx_cur - cx_med)**2 + (cy_cur - cy_med)**2) / max(diag_m, 1)
        if c_shift > center_thresh:
            w, h = current[2] - current[0], current[3] - current[1]
            current[0] = cx_med - w / 2
            current[1] = cy_med - h / 2
            current[2] = cx_med + w / 2
            current[3] = cy_med + h / 2
            n_centers += 1
            touched = True

        cw     = max(current[2] - current[0], 1)
        ch     = max(current[3] - current[1], 1)
        ar_cur = cw / ch
        ar_med = bw_m / bh_m
        ar_dev = abs(ar_cur - ar_med) / max(ar_med, 1e-6)
        if ar_dev > ar_thresh:
            current[:] = med_all
            n_ar += 1
            touched = True

        if current[2] <= current[0]:
            current[2] = current[0] + 1
        if current[3] <= current[1]:
            current[3] = current[1] + 1

        if touched:
            cleaned[i] = current
            new_suspect[i] = True
            n_frames += 1

    return cleaned, new_suspect, n_edges, n_centers, n_ar, n_frames


def clean_bbox_map(
    bbox_map: BboxMap,
    window: int = 7,
    edge_thresh: float = 0.25,
    center_thresh: float = 0.20,
    neighbor_agree: float = 0.25,
    ar_thresh: float = BBOX_AR_THRESH,
    n_passes: int = 2,
) -> tuple[BboxMap, int, int, int, int, list[tuple[int, int, int, int]]]:
    if not bbox_map:
        return bbox_map, 0, 0, 0, 0, []

    frames  = sorted(bbox_map.keys())
    arr     = np.array([bbox_map[f] for f in frames], dtype=float)
    suspect = np.zeros(len(frames), dtype=bool)

    tot_edges = tot_centers = tot_ar = tot_frames = 0
    per_pass: list[tuple[int, int, int, int]] = []

    for _ in range(n_passes):
        arr, suspect, ne, nc, na, nf = _clean_one_pass(
            arr, suspect, window, edge_thresh, center_thresh, neighbor_agree, ar_thresh,
        )
        per_pass.append((ne, nc, na, nf))
        tot_edges   += ne
        tot_centers += nc
        tot_ar      += na
        tot_frames  += nf
        if nf == 0:
            break

    cleaned_map: BboxMap = {}
    for k, f in enumerate(frames):
        c = arr[k].astype(int)
        if c[2] <= c[0]:
            c[2] = c[0] + 1
        if c[3] <= c[1]:
            c[3] = c[1] + 1
        cleaned_map[f] = (int(c[0]), int(c[1]), int(c[2]), int(c[3]))
    return cleaned_map, tot_edges, tot_centers, tot_ar, tot_frames, per_pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Keypoint feature helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _kp_features(kmap: KpDict) -> dict[str, Any] | None:
    if not kmap:
        return None
    pts = np.array([[v[0], v[1]] for v in kmap.values()])
    cx, cy = pts.mean(axis=0)
    if len(pts) < 2:
        spread_ar: float = float(np.nan)
    else:
        kw = pts[:, 0].max() - pts[:, 0].min()
        kh = pts[:, 1].max() - pts[:, 1].min()
        spread_ar = kw / max(kh, 1.0)
    return {"centroid": np.array([cx, cy]), "spread_ar": spread_ar, "pts": pts}


# ═══════════════════════════════════════════════════════════════════════════════
#  Post-hoc compound suspicion filter
# ═══════════════════════════════════════════════════════════════════════════════

def _compound_score_one_pass(
    pose_store: PoseStore,
    bbox_map: BboxMap,
    frames: list[int],
    suspect: np.ndarray,
    window: int,
    jump_thresh: float,
    neighbor_agree: float,
    score_thresh: float,
    kp_w: float,
    kpc_w: float,
    kpar_w: float,
    bboxar_w: float,
) -> tuple[np.ndarray, int]:
    KPAR_THRESH = 0.40
    N = len(frames)
    frame_to_idx = {f: i for i, f in enumerate(frames)}
    new_suspect = suspect.copy()
    n_flagged = 0

    feats: dict[int, dict[str, Any] | None] = {}
    for f in frames:
        i = frame_to_idx[f]
        feats[f] = None if suspect[i] else _kp_features(pose_store.get(f, {}))

    kp_names: set[str] = set()
    for f in frames:
        kp_names.update(pose_store.get(f, {}).keys())

    kp_pos: dict[str, np.ndarray] = {}
    for kp in kp_names:
        arr = np.full((N, 2), np.nan)
        for i, f in enumerate(frames):
            if not suspect[i]:
                kd = pose_store.get(f, {})
                if kp in kd:
                    arr[i] = kd[kp][:2]
        kp_pos[kp] = arr

    centroid_arr  = np.full((N, 2), np.nan)
    spread_ar_arr = np.full(N, np.nan)
    bbox_ar_arr   = np.full(N, np.nan)
    for i, f in enumerate(frames):
        if suspect[i]:
            continue
        ft = feats.get(f)
        if ft is not None:
            centroid_arr[i]  = ft["centroid"]
            spread_ar_arr[i] = ft["spread_ar"]
        bb = bbox_map.get(f)
        if bb is not None:
            bw = max(bb[2] - bb[0], 1)
            bh = max(bb[3] - bb[1], 1)
            bbox_ar_arr[i] = bw / bh

    for i, f in enumerate(frames):
        if suspect[i] or new_suspect[i]:
            continue
        if not pose_store.get(f):
            continue

        bi = [k for k in range(max(0, i - window), i)          if not suspect[k]]
        ai = [k for k in range(i + 1, min(N, i + 1 + window)) if not suspect[k]]
        if len(bi) < 2 or len(ai) < 2:
            continue

        bb = bbox_map.get(f)
        if bb is None:
            continue
        bw   = max(bb[2] - bb[0], 1)
        bh   = max(bb[3] - bb[1], 1)
        diag = np.sqrt(bw**2 + bh**2)

        # Signal A — per-keypoint jump fraction
        n_kp_total = n_kp_outlier = 0
        for kp in kp_names:
            if kp not in pose_store.get(f, {}):
                continue
            pos    = kp_pos[kp]
            nb_all = [k for k in bi + ai if not np.isnan(pos[k, 0])]
            if len(nb_all) < 3:
                continue
            b_v = [k for k in bi if not np.isnan(pos[k, 0])]
            a_v = [k for k in ai if not np.isnan(pos[k, 0])]
            if (
                len(b_v) >= 2
                and len(a_v) >= 2
                and np.linalg.norm(np.median(pos[b_v], 0) - np.median(pos[a_v], 0))
                > diag * neighbor_agree
            ):
                continue
            med_nb = np.median(pos[nb_all], axis=0)
            n_kp_total += 1
            if np.linalg.norm(pos[i] - med_nb) > jump_thresh * diag:
                n_kp_outlier += 1
        score_A = (n_kp_outlier / n_kp_total) if n_kp_total > 0 else 0.0

        # Signal B — centroid jump
        score_B = 0.0
        if not np.isnan(centroid_arr[i, 0]):
            nb_c = [k for k in bi + ai if not np.isnan(centroid_arr[k, 0])]
            if len(nb_c) >= 3:
                bc = [k for k in bi if not np.isnan(centroid_arr[k, 0])]
                ac = [k for k in ai if not np.isnan(centroid_arr[k, 0])]
                stable = not (
                    len(bc) >= 2
                    and len(ac) >= 2
                    and np.linalg.norm(
                        np.median(centroid_arr[bc], 0) - np.median(centroid_arr[ac], 0)
                    ) > diag * neighbor_agree
                )
                if stable:
                    med_c  = np.median(centroid_arr[nb_c], axis=0)
                    shift  = np.linalg.norm(centroid_arr[i] - med_c)
                    score_B = min(shift / max(jump_thresh * diag, 1e-6), 1.0)

        # Signal C — keypoint spread AR deviation
        score_C = 0.0
        if not np.isnan(spread_ar_arr[i]):
            nb_ar = [k for k in bi + ai if not np.isnan(spread_ar_arr[k])]
            if len(nb_ar) >= 3:
                bar = [k for k in bi if not np.isnan(spread_ar_arr[k])]
                aar = [k for k in ai if not np.isnan(spread_ar_arr[k])]
                stable = not (
                    len(bar) >= 2
                    and len(aar) >= 2
                    and abs(np.median(spread_ar_arr[bar]) - np.median(spread_ar_arr[aar]))
                    / max(np.median(spread_ar_arr[nb_ar]), 1e-6) > neighbor_agree
                )
                if stable:
                    med_ar  = np.median(spread_ar_arr[nb_ar])
                    ar_dev  = abs(spread_ar_arr[i] - med_ar) / max(med_ar, 1e-6)
                    score_C = min(ar_dev / KPAR_THRESH, 1.0)

        # Signal D — bbox AR deviation
        score_D = 0.0
        if not np.isnan(bbox_ar_arr[i]):
            nb_bar = [k for k in bi + ai if not np.isnan(bbox_ar_arr[k])]
            if len(nb_bar) >= 3:
                bbar = [k for k in bi if not np.isnan(bbox_ar_arr[k])]
                abar = [k for k in ai if not np.isnan(bbox_ar_arr[k])]
                stable = not (
                    len(bbar) >= 2
                    and len(abar) >= 2
                    and abs(np.median(bbox_ar_arr[bbar]) - np.median(bbox_ar_arr[abar]))
                    / max(np.median(bbox_ar_arr[nb_bar]), 1e-6) > neighbor_agree
                )
                if stable:
                    med_bar = np.median(bbox_ar_arr[nb_bar])
                    bar_dev = abs(bbox_ar_arr[i] - med_bar) / max(med_bar, 1e-6)
                    score_D = min(bar_dev / BBOX_AR_THRESH, 1.0)

        total = kp_w * score_A + kpc_w * score_B + kpar_w * score_C + bboxar_w * score_D
        if total >= score_thresh:
            new_suspect[i] = True
            n_flagged += 1

    return new_suspect, n_flagged


def post_filter_keypoints(
    pose_store: PoseStore,
    bbox_map: BboxMap,
    window: int = POST_WINDOW,
    jump_thresh: float = POST_JUMP_THRESH,
    neighbor_agree: float = POST_NEIGHBOR_AGREE,
    n_passes: int = POST_N_PASSES,
    score_thresh: float = SUSPECT_SCORE_THRESHOLD,
    kp_w: float = KP_SCORE_W,
    kpc_w: float = KPC_SCORE_W,
    kpar_w: float = KPAR_SCORE_W,
    bboxar_w: float = BBOX_AR_SCORE_W,
) -> tuple[PoseStore, set[int], int]:
    if not pose_store:
        return pose_store, set(), 0

    frames  = sorted(pose_store.keys())
    suspect = np.zeros(len(frames), dtype=bool)

    for p in range(n_passes):
        suspect, n_flagged = _compound_score_one_pass(
            pose_store, bbox_map, frames, suspect,
            window, jump_thresh, neighbor_agree, score_thresh,
            kp_w, kpc_w, kpar_w, bboxar_w,
        )
        print(f"    post-filter pass {p + 1}: flagged {n_flagged} frames")
        if n_flagged == 0:
            break

    flagged = {f for i, f in enumerate(frames) if suspect[i]}
    cleaned: PoseStore = {f: ({} if f in flagged else pose_store[f]) for f in frames}
    return cleaned, flagged, int(suspect.sum())


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point  (all side-effects live here; module is safely importable)
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_csv", type=str,
        default="/home/aparnabg/orcd/scratch/all_project_files/latest_split_csv.csv",
    )
    parser.add_argument(
        "--h5_dir", type=str,
        default="/orcd/data/satra/002/projects/SAILS/vjepa_features/interpolate_full_video/h5folders/",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/home/aparnabg/orcd/scratch/pose_vitpose_h5guided_json/",
    )
    parser.add_argument("--model_name", type=str, default="usyd-community/vitpose-plus-huge")
    parser.add_argument("--array_index", type=int, default=0,
                        help="SLURM_ARRAY_TASK_ID  (0-based index of this job)")
    parser.add_argument("--num_jobs", type=int, default=1,
                        help="Total number of array jobs")
    args = parser.parse_args()

    split_csv  = args.split_csv
    h5_dir     = args.h5_dir
    output_dir = args.output_dir
    model_name = args.model_name
    array_idx  = args.array_index
    num_jobs   = args.num_jobs

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  |  array_index={array_idx}/{num_jobs}")

    processor = AutoProcessor.from_pretrained(model_name)  # type: ignore[no-untyped-call]
    model = VitPoseForPoseEstimation.from_pretrained(model_name, device_map=device)
    model.eval()
    dataset_index = torch.zeros(1, dtype=torch.long, device=device)

    # ── Build video list ──────────────────────────────────────────────────────
    split_df = pd.read_csv(split_csv)
    all_videos: list[tuple[str, str]] = []
    for _, row in split_df.iterrows():
        vid_path = str(row["video_path"]).strip()
        h5_orig  = str(row["h5_file_path"]).strip()
        h5_base  = os.path.basename(h5_orig).replace(".h5", "_interpolated_action_rmm.h5")
        h5_path  = os.path.join(h5_dir, h5_base)
        if os.path.exists(vid_path) and os.path.exists(h5_path):
            all_videos.append((vid_path, h5_path))

    videos_to_run = all_videos[array_idx::num_jobs]
    print(f"Total valid videos : {len(all_videos)}")
    print(f"This job's slice   : {len(videos_to_run)} videos "
          f"(index {array_idx}, every {num_jobs})\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    for vid_idx, (video_path, h5_path) in enumerate(videos_to_run):
        base     = os.path.splitext(os.path.basename(video_path))[0]
        out_json = os.path.join(output_dir, f"{base}_keypoints.json")

        if os.path.exists(out_json):
            print(f"[{vid_idx + 1}/{len(videos_to_run)}] {base}: already exists, skipping")
            continue

        raw_bbox_map = load_bbox_map(h5_path)
        if not raw_bbox_map:
            print(f"[{vid_idx + 1}] {base}: empty H5, skip")
            continue

        bbox_map, n_edges, n_centers, n_ar, n_frames_fixed, _per_pass = clean_bbox_map(
            raw_bbox_map,
            window=WINDOW, edge_thresh=SIZE_THRESH,
            center_thresh=CENTER_THRESH, neighbor_agree=NEIGHBOR_AGREE,
            ar_thresh=BBOX_AR_THRESH, n_passes=N_CLEAN_PASSES,
        )
        print(f"[{vid_idx + 1}/{len(videos_to_run)}] {base}")
        print(f"  bbox cleanup: {n_edges} edges, {n_centers} center-shifts, "
              f"{n_ar} AR-fixes, {n_frames_fixed}/{len(raw_bbox_map)} frames touched")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  cannot open {video_path}\n")
            continue
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step  = max(1, int(round(fps / ANN_FPS)))
        print(f"  video: {W}x{H}, {total} frames, fps={fps:.2f}, step={step}")

        # ── Inference pass ────────────────────────────────────────────────────
        pose_store: PoseStore = {}
        last_kp: dict[str, tuple[float, float, float, int]] = {}
        pose_count = error_count = no_bbox_count = kp_dropped = 0

        print("  Running ViTPose inference...")
        with tqdm(total=total, desc="  inference") as pbar:
            vidx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                ann_f = vidx // step
                if ann_f not in bbox_map:
                    no_bbox_count += 1
                    vidx += 1
                    pbar.update(1)
                    continue

                x1, y1, x2, y2 = bbox_map[ann_f]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)
                if x2 - x1 < 5 or y2 - y1 < 5:
                    vidx += 1
                    pbar.update(1)
                    continue

                diag     = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                max_jump = diag * KP_JUMP_THRESH

                crop = frame[y1:y2, x1:x2]
                crop_h, crop_w = crop.shape[:2]
                boxes_xywh = [np.array([[0.0, 0.0, float(crop_w), float(crop_h)]],
                                       dtype=np.float32)]

                try:
                    image  = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    inputs = processor(image, boxes=boxes_xywh, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs, dataset_index=dataset_index)
                    results = processor.post_process_pose_estimation(
                        outputs, boxes=boxes_xywh, threshold=SCORE_THRESHOLD,
                    )

                    frame_kps: KpDict = {}
                    for person in results[0]:
                        kps    = person["keypoints"].cpu().numpy()
                        scores = person["scores"].cpu().numpy()
                        names  = [
                            model.config.id2label[label_id]
                            for label_id in person["labels"].cpu().numpy().tolist()
                        ]

                        for ii, n in enumerate(names):
                            kx = float(kps[ii, 0]) + x1
                            ky = float(kps[ii, 1]) + y1
                            s  = float(scores[ii])
                            accepted = True
                            if s < SCORE_THRESHOLD:
                                accepted = False
                            else:
                                prev = last_kp.get(n)
                                if prev is not None:
                                    px, py, _ps, age = prev
                                    if age <= KP_MEMORY_FRAMES:
                                        allowed = max_jump * max(1, age)
                                        if np.sqrt((kx - px)**2 + (ky - py)**2) > allowed:
                                            accepted = False
                                            kp_dropped += 1
                            if accepted:
                                last_kp[n]   = (kx, ky, s, 0)
                                frame_kps[n] = (kx, ky, s)

                        for n in list(last_kp.keys()):
                            if n not in frame_kps:
                                px, py, ps, age = last_kp[n]
                                if age + 1 > KP_MEMORY_FRAMES:
                                    del last_kp[n]
                                else:
                                    last_kp[n] = (px, py, ps, age + 1)

                    if frame_kps:
                        pose_store[ann_f] = frame_kps
                        pose_count += 1

                except Exception as e:  # noqa: BLE001
                    error_count += 1
                    if error_count == 1:
                        print(f"\n  Error: {e}")

                vidx += 1
                pbar.update(1)

        cap.release()
        print(f"  Inference done: {pose_count} poses, {kp_dropped} kp dropped online, "
              f"{error_count} errors")

        # ── Post-hoc filter ───────────────────────────────────────────────────
        print("  Post-hoc compound suspicion filtering...")
        pose_store, _flagged_frames, n_post_flagged = post_filter_keypoints(
            pose_store, bbox_map,
        )
        print(f"  Post-filter removed {n_post_flagged} frames total")

        # ── Save JSON ─────────────────────────────────────────────────────────
        json_frames: dict[str, dict[str, dict[str, float]]] = {}
        for frame_idx, kmap in pose_store.items():
            if not kmap:
                continue
            json_frames[str(frame_idx)] = {
                kp_name: {
                    "x":          round(float(vals[0]), 3),
                    "y":          round(float(vals[1]), 3),
                    "confidence": round(float(vals[2]), 4),
                }
                for kp_name, vals in kmap.items()
            }

        output: dict[str, Any] = {
            "video":   base,
            "ann_fps": ANN_FPS,
            "frames":  json_frames,
        }

        with open(out_json, "w") as fh:
            json.dump(output, fh)

        print(f"  Saved {len(json_frames)} frames -> {out_json}")
        print(f"  Summary: poses={pose_count}  post-removed={n_post_flagged}  "
              f"written={len(json_frames)}\n")

    print("Done!")


if __name__ == "__main__":
    main()