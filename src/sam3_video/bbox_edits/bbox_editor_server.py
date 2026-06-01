"""
SAM3 Bbox Detection Editor — FastAPI Backend
=============================================
- UI uses native video fps (read from masked_video.mp4, default 30).
- Saved H5/CSV output is downsampled to OUTPUT_FPS (15 fps).
- Inputs (detections.csv, results.json, masked_video.mp4, original video) are READ-ONLY.

Usage:
    python bbox_editor_server.py \
        --sam3_dir /orcd/data/satra/002/projects/SAILS/vjepa_features/sam3_outputs_job2 \
        --output_dir /orcd/data/satra/002/projects/SAILS/vjepa_features/corrected_bboxes_h5 \
        --port 9020
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn


# ──────────────────────────────────────────────
# Globals (set from CLI args)
# ──────────────────────────────────────────────
SAM3_DIR = None
OUTPUT_DIR = None
VIDEO_LIST = []  # ordered list of video folder names

# FPS configuration
DEFAULT_SOURCE_FPS = 30.0  # fallback if cv2 unavailable / video unreadable
OUTPUT_FPS = 15            # all saved H5/CSV files are downsampled to this

# Per-video fps cache so we don't re-probe on every request
_FPS_CACHE: Dict[str, float] = {}


# ──────────────────────────────────────────────
# Pydantic
# ──────────────────────────────────────────────
class SaveRequest(BaseModel):
    video_name: str
    keep_ids: List[int]
    remove_ids: List[int]
    excluded_ranges: Optional[Dict[str, List[List[int]]]] = None  # {"id": [[start,end],...]} in source-fps frames
    remap_to: Optional[int] = 1
    child_id: Optional[str] = ""
    timepoint_label: Optional[str] = ""


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(title="SAM3 Bbox Editor")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def scan_videos():
    """Build ordered list of video folders."""
    global VIDEO_LIST
    VIDEO_LIST = []
    if not SAM3_DIR or not os.path.isdir(SAM3_DIR):
        return
    for name in sorted(os.listdir(SAM3_DIR)):
        sub = os.path.join(SAM3_DIR, name)
        if os.path.isdir(sub):
            det = os.path.join(sub, "detections.csv")
            vid = os.path.join(sub, "masked_video.mp4")
            if os.path.exists(det) or os.path.exists(vid):
                VIDEO_LIST.append(name)


def is_saved(video_name: str) -> bool:
    if not OUTPUT_DIR:
        return False
    return os.path.exists(os.path.join(OUTPUT_DIR, f"{video_name}_bboxes.h5"))


def load_detections(video_name: str) -> pd.DataFrame:
    csv_path = os.path.join(SAM3_DIR, video_name, "detections.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["frame_idx", "obj_id", "x1", "y1", "x2", "y2", "score"])
    df = pd.read_csv(csv_path)
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("frame_idx", "frame_index"):
            col_map[c] = "frame_idx"
        elif cl in ("obj_id", "track_id", "object_id"):
            col_map[c] = "obj_id"
    df = df.rename(columns=col_map)
    for col in ["frame_idx", "obj_id", "x1", "y1", "x2", "y2"]:
        if col not in df.columns:
            df[col] = 0
    if "score" not in df.columns:
        df["score"] = 0.0
    return df


def load_results_json(video_name: str) -> dict:
    p = os.path.join(SAM3_DIR, video_name, "results.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def get_video_fps(video_name: str) -> float:
    """Read fps from masked_video.mp4 (read-only). Fallback to DEFAULT_SOURCE_FPS."""
    if video_name in _FPS_CACHE:
        return _FPS_CACHE[video_name]

    fps = DEFAULT_SOURCE_FPS
    try:
        import cv2
        vid_path = os.path.join(SAM3_DIR, video_name, "masked_video.mp4")
        if os.path.exists(vid_path):
            cap = cv2.VideoCapture(vid_path)
            v_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if v_fps and v_fps > 0:
                fps = float(v_fps)
    except Exception as e:
        print(f"[fps] could not read fps for {video_name}: {e}")

    _FPS_CACHE[video_name] = fps
    return fps


def save_corrected_h5(video_name, df_filtered, child_id="", timepoint_label="", input_path=""):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    h5_path = os.path.join(OUTPUT_DIR, f"{video_name}_bboxes.h5")

    out = pd.DataFrame()
    out["child_id"] = [child_id or video_name] * len(df_filtered)
    out["timepoint_label"] = [timepoint_label] * len(df_filtered)
    out["input_path"] = [input_path] * len(df_filtered)
    out["frame_index"] = df_filtered["frame_idx"].values
    out["track_id"] = df_filtered["obj_id"].values
    out["x0"] = df_filtered["x1"].values
    out["y0"] = df_filtered["y1"].values
    out["x1"] = df_filtered["x2"].values
    out["y1"] = df_filtered["y2"].values
    out["score"] = df_filtered["score"].values if "score" in df_filtered.columns else 0.0
    out["area"] = (out["x1"] - out["x0"]).abs() * (out["y1"] - out["y0"]).abs()

    out.to_hdf(h5_path, key="bboxes", mode="w", format="table")
    return h5_path


def downsample_to_output_fps(df: pd.DataFrame, source_fps: float) -> pd.DataFrame:
    """
    Downsample detections from source_fps to OUTPUT_FPS by keeping every Nth frame
    and renumbering frame_idx so the saved file is a clean OUTPUT_FPS sequence.

    Example: source 30 -> output 15, step=2.
        keep frames 0,2,4,6,...  -> renumber to 0,1,2,3,...
    """
    if source_fps <= 0 or OUTPUT_FPS <= 0:
        return df

    # Use rounded ratio so 29.97 -> 2, 60 -> 4, etc.
    step = max(1, int(round(source_fps / OUTPUT_FPS)))

    if step == 1:
        return df.copy()

    out = df[df["frame_idx"] % step == 0].copy()
    out["frame_idx"] = (out["frame_idx"] // step).astype(int)
    return out


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = Path(__file__).parent / "bbox_editor.html"
    if html_file.exists():
        return FileResponse(html_file)
    return HTMLResponse("<h1>bbox_editor.html not found — place it next to this script</h1>")


@app.get("/api/videos")
async def list_videos():
    result = []
    for i, name in enumerate(VIDEO_LIST):
        df = load_detections(name)
        unique_ids = sorted(df["obj_id"].unique().tolist()) if len(df) > 0 else []
        num_frames = int(df["frame_idx"].max()) + 1 if len(df) > 0 else 0
        vid_exists = os.path.exists(os.path.join(SAM3_DIR, name, "masked_video.mp4"))

        per_id = {}
        for oid in unique_ids:
            sub = df[df["obj_id"] == oid]
            per_id[str(oid)] = {
                "frame_count": len(sub),
                "first_frame": int(sub["frame_idx"].min()),
                "last_frame": int(sub["frame_idx"].max()),
                "avg_score": round(float(sub["score"].mean()), 3),
            }

        result.append({
            "index": i,
            "name": name,
            "has_video": vid_exists,
            "status": "saved" if is_saved(name) else "unprocessed",
            "num_frames": num_frames,
            "unique_ids": unique_ids,
            "per_id_stats": per_id,
        })
    return {"videos": result, "total": len(result)}


@app.get("/api/video_file/{video_name}")
async def serve_video(video_name: str):
    vid = os.path.join(SAM3_DIR, video_name, "masked_video.mp4")
    if not os.path.exists(vid):
        raise HTTPException(404, "Video not found")
    return FileResponse(vid, media_type="video/mp4")


@app.get("/api/detections/{video_name}")
async def get_detections(video_name: str):
    df = load_detections(video_name)
    fps = get_video_fps(video_name)

    if len(df) == 0:
        return {
            "unique_ids": [], "per_id_stats": {}, "per_frame": {},
            "total_detections": 0, "fps": fps, "output_fps": OUTPUT_FPS,
        }

    unique_ids = sorted(df["obj_id"].unique().tolist())
    per_id = {}
    for oid in unique_ids:
        sub = df[df["obj_id"] == oid]
        per_id[str(oid)] = {
            "frame_count": len(sub),
            "first_frame": int(sub["frame_idx"].min()),
            "last_frame": int(sub["frame_idx"].max()),
            "avg_score": round(float(sub["score"].mean()), 3),
        }

    per_frame = {}
    for _, row in df.iterrows():
        fidx = int(row["frame_idx"])
        if fidx not in per_frame:
            per_frame[fidx] = []
        per_frame[fidx].append({
            "obj_id": int(row["obj_id"]),
            "x1": int(row["x1"]), "y1": int(row["y1"]),
            "x2": int(row["x2"]), "y2": int(row["y2"]),
            "score": round(float(row["score"]), 3),
        })

    return {
        "unique_ids": unique_ids,
        "per_id_stats": per_id,
        "per_frame": per_frame,
        "total_detections": len(df),
        "fps": fps,
        "output_fps": OUTPUT_FPS,
    }


@app.post("/api/save")
async def save_corrections(req: SaveRequest):
    df = load_detections(req.video_name)
    if len(df) == 0:
        raise HTTPException(400, "No detections")

    source_fps = get_video_fps(req.video_name)
    original_len = len(df)

    # 1. Apply ID-level keep/remove (entire-track)
    if req.keep_ids:
        df_f = df[df["obj_id"].isin(req.keep_ids)].copy()
    elif req.remove_ids:
        df_f = df[~df["obj_id"].isin(req.remove_ids)].copy()
    else:
        df_f = df.copy()

    # 2. Apply per-ID excluded frame ranges (in SOURCE fps frame space, what the UI shows)
    if req.excluded_ranges:
        for id_str, ranges in req.excluded_ranges.items():
            try:
                oid = int(id_str)
            except ValueError:
                continue
            for rng in ranges:
                if len(rng) != 2:
                    continue
                start, end = int(rng[0]), int(rng[1])
                mask = ~((df_f["obj_id"] == oid) &
                         (df_f["frame_idx"] >= start) &
                         (df_f["frame_idx"] <= end))
                df_f = df_f[mask]

    # 3. Downsample source_fps -> OUTPUT_FPS (15) and renumber frame_idx
    df_f = downsample_to_output_fps(df_f, source_fps)

    # 4. Remap obj_id (e.g. all kept IDs collapse to id=1)
    if req.remap_to is not None:
        df_f["obj_id"] = req.remap_to

    # 5. Dedup per frame (keep highest-score detection per frame)
    if "score" in df_f.columns:
        df_f = df_f.sort_values("score", ascending=False)
    df_f = df_f.drop_duplicates(subset=["frame_idx"], keep="first")
    df_f = df_f.sort_values("frame_idx").reset_index(drop=True)

    # 6. Write outputs
    rjson = load_results_json(req.video_name)
    input_path = rjson.get("video_path", "")

    h5 = save_corrected_h5(
        req.video_name, df_f,
        child_id=req.child_id or req.video_name,
        timepoint_label=req.timepoint_label or "",
        input_path=input_path,
    )

    csv_out = os.path.join(OUTPUT_DIR, f"{req.video_name}_detections_corrected.csv")
    df_f.to_csv(csv_out, index=False)

    return {
        "success": True,
        "h5_path": h5,
        "kept_frames": len(df_f),
        "removed_frames": original_len - len(df_f),
        "source_fps": source_fps,
        "output_fps": OUTPUT_FPS,
    }


@app.post("/api/auto_save_single_id")
async def auto_save_single_id():
    """Auto-save all videos that have only 1 unique ID (no manual review needed)."""
    auto_saved = []
    for name in VIDEO_LIST:
        if is_saved(name):
            continue
        df = load_detections(name)
        if len(df) == 0:
            continue
        unique_ids = df["obj_id"].unique().tolist()
        if len(unique_ids) == 1:
            source_fps = get_video_fps(name)
            rjson = load_results_json(name)
            input_path = rjson.get("video_path", "")

            df_copy = df.copy()
            # Downsample source -> 15 fps
            df_copy = downsample_to_output_fps(df_copy, source_fps)
            df_copy["obj_id"] = 1  # remap to 1

            if "score" in df_copy.columns:
                df_copy = df_copy.sort_values("score", ascending=False)
            df_copy = df_copy.drop_duplicates(subset=["frame_idx"], keep="first")
            df_copy = df_copy.sort_values("frame_idx").reset_index(drop=True)

            save_corrected_h5(name, df_copy, child_id=name, input_path=input_path)
            csv_out = os.path.join(OUTPUT_DIR, f"{name}_detections_corrected.csv")
            df_copy.to_csv(csv_out, index=False)
            auto_saved.append(name)
    return {"auto_saved": len(auto_saved), "names": auto_saved}


@app.get("/api/first_unprocessed")
async def first_unprocessed():
    """Return first unsaved video that has MORE than 1 unique ID (needs review)."""
    for i, name in enumerate(VIDEO_LIST):
        if is_saved(name):
            continue
        df = load_detections(name)
        unique_ids = df["obj_id"].unique().tolist() if len(df) > 0 else []
        if len(unique_ids) > 1:
            return {"index": i, "found": True}
    return {"index": -1, "found": False}


@app.get("/api/stats")
async def get_stats():
    """Get counts for the UI."""
    total = len(VIDEO_LIST)
    saved = sum(1 for n in VIDEO_LIST if is_saved(n))
    single_id_unsaved = 0
    multi_id_unsaved = 0
    for name in VIDEO_LIST:
        if is_saved(name):
            continue
        df = load_detections(name)
        uids = df["obj_id"].unique().tolist() if len(df) > 0 else []
        if len(uids) <= 1:
            single_id_unsaved += 1
        else:
            multi_id_unsaved += 1
    return {
        "total": total, "saved": saved,
        "single_id_unsaved": single_id_unsaved,
        "multi_id_unsaved": multi_id_unsaved,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sam3_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--port", type=int, default=9020)
    p.add_argument("--host", type=str, default="0.0.0.0")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    SAM3_DIR = args.sam3_dir
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scan_videos()

    print(f"\n{'='*60}")
    print(f"  SAM3 Bbox Editor")
    print(f"{'='*60}")
    print(f"  SAM3 dir   : {SAM3_DIR}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Videos     : {len(VIDEO_LIST)}")
    print(f"  Output FPS : {OUTPUT_FPS} (saved files)")
    print(f"  URL        : http://localhost:{args.port}")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=args.host, port=args.port)