# Motion + Appearance Tracking Pipeline

This document explains the notebook runner and tracking scaffold added to this repo.

## Files
- `notebooks/scripts/motion_appearance_reid.py`: End‑to‑end pipeline (detection → pose → features → motion/appearance matching → re‑ID → visualization).
- `src/sailsprep/tracking/person_tracker.py`: Reusable tracking utilities (camera motion, Kalman, adaptive thresholds, similarities).

## Dependencies
- OpenMMLab: `mmdet`, `mmpose`, `mmcv`, `mmengine`
- Vision/ML: `opencv-python`, `facenet-pytorch`, `deepface`, `tqdm`, `scipy`, `scikit-learn`
- Optional GPU; CPU is supported but slow.

## Running
- Single video (CPU):
  - `python notebooks/scripts/motion_appearance_reid.py --in <video.mp4> --out <out.mp4> --device cpu --det-config <rtmdet_person.py> --pose-config <hrnet_wholebody.py>`
- Folder:
  - `python notebooks/scripts/motion_appearance_reid.py --in-dir <in_dir> --out-dir <out_dir> --device cpu --det-config <...> --pose-config <...>`
- Checkpoints: use defaults (downloaded) or pass `--det-ckpt` and `--pose-ckpt` paths.

## Core Logic
- Camera motion compensation: Lucas–Kanade flow on non‑central features; smoothed (5 frames).
- Kalman tracking: 8‑state bbox filter, high noise on size for depth changes; prediction confidence decays with missed updates.
- Adaptive thresholds per frame:
  - `crowding = calculate_scene_crowding(bboxes)` → distances in bbox units
  - `iou_thresh, center_w, motion_conf_th = get_adaptive_thresholds(cfg, crowding)`
  - Matching uses combined similarity: `sim = (1-center_w)*IoU + center_w*CenterSim`.
- Spatial plausibility: reject pairs with center jump > `2.5×` avg bbox size.
- Matching order: motion‑based (Hungarian) → appearance‑based (face/upper/lower) → re‑ID from lost tracks → new tracks.

## Scenario Coverage
- Single kid: crowding=0.0 → IoU=0.20, Center=0.75 → lenient, robust to depth/camera motion.
- Multiple kids close: crowding≈0.7 → IoU≈0.34, Center≈0.50 → stricter, fewer ID swaps.
- Multiple far apart: crowding=0.0 → lenient per subject.
- Occlusion/overlap: crowding=1.0 → IoU=0.40, Center=0.40 + plausibility gate → stable IDs.

## Notes & Extensibility
- The runner keeps imports lazy where possible; errors include install hints.
- `person_tracker.py` exposes small, testable units (e.g., `create_kalman_filter`, `predict_motion_with_camera_compensation`).
- Add appearance models or trackers by swapping `compute_person_similarity`/feature extractors.
