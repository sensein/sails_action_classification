"""
SAM3 Bbox Detection Editor — FastAPI Backend
=============================================
Simple loop-through interface: prev/next/save/skip.

Usage:
    python bbox_editor_server.py \
        --sam3_dir /path/to/sam3_outputs_job2 \
        --output_dir /path/to/corrected_h5 \
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


# ──────────────────────────────────────────────
# Pydantic
# ──────────────────────────────────────────────
class SaveRequest(BaseModel):
    video_name: str
    keep_ids: List[int]
    remove_ids: List[int]
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

        # Per-ID stats
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
    if len(df) == 0:
        return {"unique_ids": [], "per_id_stats": {}, "per_frame": {}, "total_detections": 0}

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
    }


@app.post("/api/save")
async def save_corrections(req: SaveRequest):
    df = load_detections(req.video_name)
    if len(df) == 0:
        raise HTTPException(400, "No detections")

    if req.keep_ids:
        df_f = df[df["obj_id"].isin(req.keep_ids)].copy()
    elif req.remove_ids:
        df_f = df[~df["obj_id"].isin(req.remove_ids)].copy()
    else:
        df_f = df.copy()

    if req.remap_to is not None:
        df_f["obj_id"] = req.remap_to

    if "score" in df_f.columns:
        df_f = df_f.sort_values("score", ascending=False)
    df_f = df_f.drop_duplicates(subset=["frame_idx"], keep="first")
    df_f = df_f.sort_values("frame_idx").reset_index(drop=True)

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
        "removed_frames": len(df) - len(df_f),
    }


@app.get("/api/first_unprocessed")
async def first_unprocessed():
    for i, name in enumerate(VIDEO_LIST):
        if not is_saved(name):
            return {"index": i, "found": True}
    return {"index": -1, "found": False}


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
    print(f"  SAM3 dir  : {SAM3_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Videos    : {len(VIDEO_LIST)}")
    print(f"  URL       : http://localhost:{args.port}")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=args.host, port=args.port)