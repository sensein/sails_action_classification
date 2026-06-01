"""
reconstruct_detections_csv.py
==============================
Rebuilds missing detections.csv for folders that have results.json
but no detections.csv.

The detections.csv schema matches what the original SAM3 pipeline writes:
    frame_idx, obj_id, x1, y1, x2, y2, score

All data is read directly from results.json → per_frame.

Usage:
    # Reconstruct for ALL folders missing detections.csv:
    python reconstruct_detections_csv.py \
        --results_dir /orcd/data/satra/002/projects/SAILS/vjepa_features/sam3_outputs_job2

    # Reconstruct only for a specific list of video folders:
    python reconstruct_detections_csv.py \
        --results_dir /orcd/data/satra/002/projects/SAILS/vjepa_features/sam3_outputs_job2 \
        --only sub-A4E8K1L5Y2_ses-01_task-other_run-01_desc-processed_beh \
               sub-D4Y7P4G2V4_ses-02_task-toyplay_run-10_desc-processed_beh
"""

import argparse
import csv
import json
from pathlib import Path


# ── Optionally restrict to specific folders ───────────────────────────────────
TARGET_VIDEOS = {
    "sub-A4E8K1L5Y2_ses-01_task-other_run-01_desc-processed_beh",
    "sub-A4E8K1L5Y2_ses-01_task-other_run-02_desc-processed_beh",
    "sub-A4E8K1L5Y2_ses-01_task-other_run-03_desc-processed_beh",
    "sub-D4Y7P4G2V4_ses-02_task-generalsocialcommunicationinteraction_run-01_desc-processed_beh",
    "sub-D4Y7P4G2V4_ses-02_task-toyplay_run-10_desc-processed_beh",
    "sub-D4Y7P4G2V4_ses-02_task-toyplay_run-11_desc-processed_beh",
    "sub-D4Y7P4G2V4_ses-02_task-toyplay_run-12_desc-processed_beh",
    "sub-L0B0Q5O3Q3_ses-02_task-toyplay_run-08_desc-processed_beh",
    "sub-N3L7A1I2B9_ses-01_task-generalsocialcommunicationinteraction_run-05_desc-processed_beh",
    "sub-N3L7A1I2B9_ses-01_task-generalsocialcommunicationinteraction_run-34_desc-processed_beh",
    "sub-O7X6W5O8E0_ses-02_task-toyplay_run-02_desc-processed_beh",
}
# ─────────────────────────────────────────────────────────────────────────────

FIELDNAMES = ["frame_idx", "obj_id", "x1", "y1", "x2", "y2", "score"]


def reconstruct_csv_from_json(video_dir: Path) -> bool:
    """
    Read results.json in video_dir and write detections.csv beside it.
    Returns True on success, False on failure.
    """
    json_path = video_dir / "results.json"
    csv_path  = video_dir / "detections.csv"

    if not json_path.exists():
        print(f"  ⚠ SKIP — no results.json: {video_dir.name}")
        return False

    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ✗ Could not read results.json for {video_dir.name}: {e}")
        return False

    per_frame = data.get("per_frame", {})
    rows = []

    for frame_str, frame_info in sorted(per_frame.items(), key=lambda kv: int(kv[0])):
        frame_idx  = int(frame_str)
        obj_ids    = frame_info.get("object_ids",  frame_info.get("obj_ids",  []))
        scores     = frame_info.get("scores",      [])
        bboxes     = frame_info.get("bboxes_xyxy", [])

        for k, obj_id in enumerate(obj_ids):
            # bbox
            if k < len(bboxes):
                bbox = bboxes[k]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            else:
                x1 = y1 = x2 = y2 = -1

            # score (may be None if reconstructed via reconstruct_results_json)
            raw_score = scores[k] if k < len(scores) else None
            if raw_score is None:
                score_str = ""
            else:
                try:
                    score_str = str(round(float(raw_score), 4))
                except (TypeError, ValueError):
                    score_str = ""

            rows.append({
                "frame_idx": frame_idx,
                "obj_id":    obj_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "score": score_str,
            })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✓ Written: {csv_path}  ({len(rows)} detection rows across "
          f"{len(per_frame)} frames with detections)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                    help="Root directory containing one sub-folder per video")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Optional list of specific folder names to process "
                         "(overrides the hardcoded TARGET_VIDEOS set). "
                         "If omitted, uses TARGET_VIDEOS; pass --only ALL to "
                         "process every folder.")
    ap.add_argument("--overwrite", action="store_true", default=False,
                    help="Overwrite existing detections.csv files")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)

    # Decide which folders to target
    if args.only is not None:
        if len(args.only) == 1 and args.only[0].upper() == "ALL":
            target_set = None          # None → process all sub-dirs
        else:
            target_set = set(args.only)
    else:
        target_set = TARGET_VIDEOS     # default: the hardcoded list above

    # Collect candidate directories
    if target_set is None:
        candidates = sorted(
            d for d in results_dir.iterdir()
            if d.is_dir() and (d / "results.json").exists()
        )
    else:
        candidates = []
        not_found  = []
        for name in sorted(target_set):
            d = results_dir / name
            if not d.exists():
                not_found.append(name)
            elif not (d / "results.json").exists():
                print(f"  ⚠ SKIP — folder exists but has no results.json: {name}")
            else:
                candidates.append(d)
        if not_found:
            print(f"\n⚠ These folders were NOT found in {results_dir}:")
            for n in not_found:
                print(f"    {n}")

    # Filter out already-done unless --overwrite
    if not args.overwrite:
        already_done = [d for d in candidates if (d / "detections.csv").exists()]
        if already_done:
            print(f"\nSkipping {len(already_done)} folder(s) that already have "
                  f"detections.csv (use --overwrite to force):")
            for d in already_done:
                print(f"    {d.name}")
        candidates = [d for d in candidates if not (d / "detections.csv").exists()]

    print(f"\nReconstructing detections.csv for {len(candidates)} folder(s)...\n")

    ok = fail = 0
    for d in candidates:
        print(f"  [{ok + fail + 1}/{len(candidates)}] {d.name}")
        if reconstruct_csv_from_json(d):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} succeeded  |  {fail} failed")


if __name__ == "__main__":
    main()