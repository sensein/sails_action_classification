"""
Combined Video Annotation + BBox Editor — FastAPI Backend
==========================================================
Video playback: source fps (e.g. 30)
BBox: stored/edited in source fps, downsampled to OUTPUT_FPS on save
Annotations: ALWAYS stored + loaded at ANN_FPS (15) — input & output

Usage:
    python combined_server.py --port 9025
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
CSV_FILE            = Path("/home/aparnabg/orcd/scratch/all_project_files/rmm_folder/missing_with_sam3.csv")
SAM3_OUTPUTS_DIR    = Path("/orcd/data/satra/002/projects/SAILS/vjepa_features/sam3_outputs_job2/")
ANNOTATION_INPUT_DIR  = Path("/home/aparnabg/orcd/scratch/all_project_files/all_annoation_rmm")
ANNOTATION_OUTPUT_DIR = Path("/home/aparnabg/orcd/scratch/all_project_files/rmm_folder/annioation_rmm_multi")
BBOX_OUTPUT_DIR     = Path("/orcd/data/satra/002/projects/SAILS/vjepa_features/corrected_bboxes_h5")

DEFAULT_SOURCE_FPS = 30.0
OUTPUT_FPS         = 15      # bbox h5 output fps
ANN_FPS            = 15      # annotation csv fps (input & output)

# ──────────────────────────────────────────────────────────────
# Label Categories
# ──────────────────────────────────────────────────────────────
ALL_CATEGORIES = {
    "Locomotion": ["Crawling", "Cruising", "Walking", "Running", "Vehicle"],
    "Repetitive_Motor_Movements": ["Hands flapping", "One hand flap", "Jumping", "Spinning", "Rocking", "Body tensing"],
    "Gestures_Functional_Actions": ["Reach", "Show", "Point", "Take", "Head shake", "Head nod", "Wave", "Clap", "Sign"],
    "Visual_Attention": ["Eye contact", "Looking at caregiver", "Looking at camera", "Looking at object", "Looking at food", "Staring"],
    "Play_Object_Use": ["Playing with toy", "Lining up objects", "Manipulating object", "Stacking objects",
                        "Holding object", "Throwing object", "Dropping object", "Spinning/rotating objects"],
    "Posture_Positions": ["Standing up", "Sitting down", "Bending", "Reaching overhead", "Freezing", "Crawling", "knee standing", "Lying down"],
    "Self_Regulatory_Behaviors": ["Hand-to-mouth", "Hand-to-face touching", "Covering ears", "Covering eyes/face", "Self-hugging"],
}

ACTION_CATEGORIES = {
    "Locomotion": ALL_CATEGORIES["Locomotion"],
    "Repetitive_Motor_Movements": ALL_CATEGORIES["Repetitive_Motor_Movements"],
}

CATEGORIES_WITH_REFERENCE = ["Locomotion", "Repetitive_Motor_Movements"]

# ──────────────────────────────────────────────────────────────
# Caches
# ──────────────────────────────────────────────────────────────
_csv_cache: Dict = {"data": None, "mtime": 0}
_fps_cache: Dict[str, float] = {}

# ──────────────────────────────────────────────────────────────
# Pydantic
# ──────────────────────────────────────────────────────────────
class AnnotationSegment(BaseModel):
    start_frame: int
    end_frame: int
    label: str

class CategoryData(BaseModel):
    breakpoints: List[int]
    segments: List[AnnotationSegment]

class SaveRequest(BaseModel):
    video_index: int
    total_frames: int           # SOURCE-fps frames (for bbox filtering)
    ann_total_frames: int       # 15-fps frames (for annotation CSV rows)
    keep_ids: List[int]
    remove_ids: List[int]
    excluded_ranges: Optional[Dict[str, List[List[int]]]] = None
    remap_to: Optional[int] = 1
    categories: Dict[str, CategoryData]   # segments in 15-fps frame space

class UpdateStatusRequest(BaseModel):
    video_index: int
    status: int

# ──────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────
app = FastAPI(title="Combined Annotation + BBox Editor")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
# CSV helpers
# ──────────────────────────────────────────────────────────────
def load_csv_data():
    global _csv_cache
    if not CSV_FILE.exists():
        return [], [], {}

    mtime = CSV_FILE.stat().st_mtime
    if _csv_cache["data"] is not None and _csv_cache["mtime"] == mtime:
        return _csv_cache["data"]

    df = pd.read_csv(CSV_FILE)
    bids_list = df.iloc[:, 0].astype(str).tolist()
    status_list = df["status"].fillna(0).astype(int).tolist() if "status" in df.columns else [0] * len(bids_list)

    ref_vals = {}
    for _, row in df.iterrows():
        vp = str(row.iloc[0])
        vals = {}
        for cat in CATEGORIES_WITH_REFERENCE:
            v = row[cat] if cat in df.columns else ""
            vals[cat] = str(v) if pd.notna(v) and str(v).strip() else ""
        ref_vals[vp] = vals

    result = (bids_list, status_list, ref_vals)
    _csv_cache["data"] = result
    _csv_cache["mtime"] = mtime
    return result


def invalidate_cache():
    global _csv_cache
    _csv_cache = {"data": None, "mtime": 0}


def update_status_in_csv(bids_list, status_list):
    try:
        df = pd.read_csv(CSV_FILE)
        df["status"] = status_list
        df.to_csv(CSV_FILE, index=False)
    except Exception as e:
        print(f"[status] {e}")


def video_stem(bids_path: str) -> str:
    return Path(bids_path).stem


def is_empty_val(val):
    if pd.isna(val):
        return True
    # NOTE: "multiple" means multiple labels exist — it is NOT empty.
    return str(val).strip().lower() in ("", "nill", "n/a", "na", "none", "nan")

# ──────────────────────────────────────────────────────────────
# BBox helpers
# ──────────────────────────────────────────────────────────────
def sam3_dir(stem: str) -> Path:
    return SAM3_OUTPUTS_DIR / stem


def get_fps(stem: str) -> float:
    if stem in _fps_cache:
        return _fps_cache[stem]
    fps = DEFAULT_SOURCE_FPS
    try:
        import cv2
        vp = sam3_dir(stem) / "masked_video.mp4"
        if vp.exists():
            cap = cv2.VideoCapture(str(vp))
            v = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if v and v > 0:
                fps = float(v)
    except Exception as e:
        print(f"[fps] {stem}: {e}")
    _fps_cache[stem] = fps
    return fps


def load_detections(stem: str) -> pd.DataFrame:
    p = sam3_dir(stem) / "detections.csv"
    if not p.exists():
        return pd.DataFrame(columns=["frame_idx", "obj_id", "x1", "y1", "x2", "y2", "score"])
    df = pd.read_csv(p)
    remap = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("frame_idx", "frame_index"):
            remap[c] = "frame_idx"
        elif cl in ("obj_id", "track_id", "object_id"):
            remap[c] = "obj_id"
    df = df.rename(columns=remap)
    for col in ["frame_idx", "obj_id", "x1", "y1", "x2", "y2"]:
        if col not in df.columns:
            df[col] = 0
    if "score" not in df.columns:
        df["score"] = 0.0
    return df


def downsample(df: pd.DataFrame, source_fps: float) -> pd.DataFrame:
    step = max(1, int(round(source_fps / OUTPUT_FPS)))
    if step == 1:
        return df.copy()
    out = df[df["frame_idx"] % step == 0].copy()
    out["frame_idx"] = (out["frame_idx"] // step).astype(int)
    return out


def save_bbox_h5(stem: str, df_f: pd.DataFrame, input_path: str = "") -> Path:
    BBOX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    h5 = BBOX_OUTPUT_DIR / f"{stem}_bboxes.h5"
    out = pd.DataFrame({
        "child_id":        [stem] * len(df_f),
        "timepoint_label": [""] * len(df_f),
        "input_path":      [input_path] * len(df_f),
        "frame_index":     df_f["frame_idx"].values,
        "track_id":        df_f["obj_id"].values,
        "x0":              df_f["x1"].values,
        "y0":              df_f["y1"].values,
        "x1":              df_f["x2"].values,
        "y1":              df_f["y2"].values,
        "score":           df_f["score"].values if "score" in df_f.columns else 0.0,
    })
    out["area"] = (out["x1"] - out["x0"]).abs() * (out["y1"] - out["y0"]).abs()
    out.to_hdf(str(h5), key="bboxes", mode="w", format="table")
    return h5


def bbox_saved(stem: str) -> bool:
    return (BBOX_OUTPUT_DIR / f"{stem}_bboxes.h5").exists()

# ──────────────────────────────────────────────────────────────
# Annotation helpers  (ALL frame indices here are in ANN_FPS = 15)
# ──────────────────────────────────────────────────────────────
def load_annotations(stem: str, ref_vals: dict):
    """
    Load existing annotation CSV. Rows are assumed to be at ANN_FPS (15).
    Returns category → {breakpoints, segments} in 15-fps frame space.
    """
    out_path = ANNOTATION_OUTPUT_DIR / f"{stem}_actions_corrected.csv"
    in_path  = ANNOTATION_INPUT_DIR  / f"{stem}_actions_corrected.csv"
    load_path = out_path if out_path.exists() else (in_path if in_path.exists() else None)

    if load_path is None:
        return {}, False, 0

    try:
        df = pd.read_csv(load_path)
        n = len(df)
        if n == 0:
            return {}, False, 0

        result = {}
        for cat in ALL_CATEGORIES:
            if cat not in df.columns:
                continue
            # Force-N/A rule applies ONLY to Locomotion when its reference is empty.
            # RMM and all other categories always load actual CSV segments.
            if cat == "Locomotion" and is_empty_val(ref_vals.get(cat, "")):
                result[cat] = {
                    "breakpoints": [],
                    "segments": [{"start_frame": 0, "end_frame": n, "label": "N/A"}]
                }
                continue
            labels = df[cat].fillna("N/A")
            changes = labels.ne(labels.shift()).cumsum()
            segs = []
            for _, grp in labels.groupby(changes):
                segs.append({
                    "start_frame": int(grp.index[0]),
                    "end_frame":   int(grp.index[-1] + 1),
                    "label":       str(grp.iloc[0])
                })
            result[cat] = {"breakpoints": [], "segments": segs}
        return result, True, n
    except Exception as e:
        print(f"[ann] load {stem}: {e}")
        return {}, False, 0


def save_annotations_csv(stem: str, ann_total_frames: int, categories: dict) -> Path:
    """Save annotations at 15 fps. `ann_total_frames` is row count (15-fps)."""
    ANNOTATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = ANNOTATION_OUTPUT_DIR / f"{stem}_actions_corrected.csv"
    data = {"Frame": list(range(ann_total_frames))}
    for cat in ALL_CATEGORIES:
        labels = ["N/A"] * ann_total_frames
        if cat in categories:
            for seg in categories[cat].segments:
                for f in range(seg.start_frame, min(seg.end_frame, ann_total_frames)):
                    labels[f] = seg.label
        data[cat] = labels
    pd.DataFrame(data).to_csv(str(out), index=False)
    return out

# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html = Path(__file__).parent / "combined_editor.html"
    if html.exists():
        return FileResponse(str(html))
    return HTMLResponse("<h1>combined_editor.html not found</h1>")


@app.get("/api/config")
async def get_config():
    return {
        "all_categories": ALL_CATEGORIES,
        "action_categories": ACTION_CATEGORIES,
        "categories_with_reference": CATEGORIES_WITH_REFERENCE,
        "output_fps": OUTPUT_FPS,
        "ann_fps":    ANN_FPS,
    }


@app.get("/api/videos")
async def get_videos():
    bids_list, status_list, ref_vals = load_csv_data()
    videos = []
    for i, bp in enumerate(bids_list):
        stem = video_stem(bp)
        d = sam3_dir(stem)
        det_path = d / "detections.csv"

        videos.append({
            "index": i,
            "bids_processed": bp,
            "stem": stem,
            "status": status_list[i],
            "category_values": ref_vals.get(bp, {}),
            "has_video": (d / "masked_video.mp4").exists(),
            "has_detections": det_path.exists(),
            "bbox_saved": bbox_saved(stem),
            "unique_ids": [],   # populated on demand via /api/detections
        })

    saved = sum(1 for v in videos if v["status"] == 1)
    return {"videos": videos, "total": len(videos), "saved": saved,
            "multi_id_unsaved": 0, "single_id_unsaved": 0}


@app.get("/api/video_file/{video_index}")
async def serve_video(video_index: int):
    bids_list, _, _ = load_csv_data()
    if video_index < 0 or video_index >= len(bids_list):
        raise HTTPException(404, "Video not found")
    vp = sam3_dir(video_stem(bids_list[video_index])) / "masked_video.mp4"
    if not vp.exists():
        raise HTTPException(404, "masked_video.mp4 not found")
    return FileResponse(str(vp), media_type="video/mp4")


@app.get("/api/detections/{video_index}")
async def get_detections(video_index: int):
    bids_list, _, _ = load_csv_data()
    if video_index < 0 or video_index >= len(bids_list):
        raise HTTPException(404, "Video not found")
    stem = video_stem(bids_list[video_index])
    df   = load_detections(stem)
    fps  = get_fps(stem)

    if len(df) == 0:
        return {"unique_ids": [], "per_id_stats": {}, "per_frame": {},
                "total_detections": 0, "fps": fps, "output_fps": OUTPUT_FPS}

    uids = sorted(df["obj_id"].unique().tolist())
    per_id = {}
    for oid in uids:
        sub = df[df["obj_id"] == oid]
        per_id[str(oid)] = {
            "frame_count": len(sub),
            "first_frame": int(sub["frame_idx"].min()),
            "last_frame":  int(sub["frame_idx"].max()),
            "avg_score":   round(float(sub["score"].mean()), 3),
        }

    per_frame: Dict = {}
    # Vectorized: group once, much faster than iterrows on big detection CSVs
    for fi, grp in df.groupby("frame_idx", sort=False):
        per_frame[int(fi)] = [
            {
                "obj_id": int(r.obj_id),
                "x1": int(r.x1), "y1": int(r.y1),
                "x2": int(r.x2), "y2": int(r.y2),
                "score": round(float(r.score), 3),
            }
            for r in grp.itertuples(index=False)
        ]

    return {"unique_ids": uids, "per_id_stats": per_id, "per_frame": per_frame,
            "total_detections": len(df), "fps": fps, "output_fps": OUTPUT_FPS}


@app.get("/api/annotations/{video_index}")
async def get_annotations(video_index: int):
    bids_list, _, ref_vals = load_csv_data()
    if video_index < 0 or video_index >= len(bids_list):
        raise HTTPException(404, "Video not found")
    bp   = bids_list[video_index]
    stem = video_stem(bp)
    ann, found, n_rows = load_annotations(stem, ref_vals.get(bp, {}))
    return {"categories": ann, "found": found, "ann_total_frames": n_rows, "ann_fps": ANN_FPS}


@app.post("/api/save")
async def save_all(req: SaveRequest):
    bids_list, status_list, _ = load_csv_data()
    if req.video_index < 0 or req.video_index >= len(bids_list):
        raise HTTPException(404, "Video not found")

    bp   = bids_list[req.video_index]
    stem = video_stem(bp)
    fps  = get_fps(stem)

    # ── BBox (source-fps space) ───────────────────────────────
    df = load_detections(stem)
    bbox_result = {"skipped": True}
    if len(df) > 0:
        df_f = df[df["obj_id"].isin(req.keep_ids)].copy() if req.keep_ids else df.copy()
        if req.remove_ids:
            df_f = df_f[~df_f["obj_id"].isin(req.remove_ids)]

        if req.excluded_ranges:
            for id_str, ranges in req.excluded_ranges.items():
                try:
                    oid = int(id_str)
                except ValueError:
                    continue
                for rng in ranges:
                    if len(rng) == 2:
                        s, e = int(rng[0]), int(rng[1])
                        mask = ~((df_f["obj_id"] == oid) & (df_f["frame_idx"] >= s) & (df_f["frame_idx"] <= e))
                        df_f = df_f[mask]

        df_f = downsample(df_f, fps)
        if req.remap_to is not None:
            df_f["obj_id"] = req.remap_to
        if "score" in df_f.columns:
            df_f = df_f.sort_values("score", ascending=False)
        df_f = df_f.drop_duplicates(subset=["frame_idx"], keep="first").sort_values("frame_idx").reset_index(drop=True)

        h5 = save_bbox_h5(stem, df_f, input_path=bp)
        BBOX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (BBOX_OUTPUT_DIR / f"{stem}_detections_corrected.csv").write_text(df_f.to_csv(index=False))
        bbox_result = {"h5_path": str(h5), "kept_frames": len(df_f),
                       "source_fps": fps, "output_fps": OUTPUT_FPS}

    # ── Annotations (15-fps space) ────────────────────────────
    ann_n = req.ann_total_frames if req.ann_total_frames > 0 else req.total_frames
    ann_path = save_annotations_csv(stem, ann_n, req.categories)

    # ── Status ────────────────────────────────────────────────
    status_list[req.video_index] = 1
    update_status_in_csv(bids_list, status_list)
    invalidate_cache()

    return {"success": True, "stem": stem, "bbox": bbox_result,
            "annotation_file": str(ann_path), "ann_rows": ann_n, "ann_fps": ANN_FPS}


@app.post("/api/update-status")
async def update_status(req: UpdateStatusRequest):
    bids_list, status_list, _ = load_csv_data()
    if req.video_index < 0 or req.video_index >= len(bids_list):
        raise HTTPException(404, "Video not found")
    status_list[req.video_index] = req.status
    update_status_in_csv(bids_list, status_list)
    invalidate_cache()
    return {"success": True}


@app.get("/api/first-unprocessed")
async def first_unprocessed():
    bids_list, status_list, _ = load_csv_data()
    for i in range(len(bids_list)):
        if status_list[i] == 0:
            return {"index": i, "found": True}
    return {"index": -1, "found": False}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9025)
    p.add_argument("--host", type=str, default="0.0.0.0")
    args = p.parse_args()
    print(f"\n{'='*60}\n  Combined BBox + Annotation Editor\n{'='*60}")
    print(f"  CSV       : {CSV_FILE}")
    print(f"  SAM3 dir  : {SAM3_OUTPUTS_DIR}")
    print(f"  Ann in    : {ANNOTATION_INPUT_DIR}")
    print(f"  Ann out   : {ANNOTATION_OUTPUT_DIR}")
    print(f"  BBox out  : {BBOX_OUTPUT_DIR}")
    print(f"  BBox FPS  : {OUTPUT_FPS}   Ann FPS: {ANN_FPS}")
    print(f"  URL       : http://localhost:{args.port}\n{'='*60}\n")
    uvicorn.run(app, host=args.host, port=args.port)