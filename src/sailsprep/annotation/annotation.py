from pathlib import Path
from typing import Any, cast

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ----------------------------
# Configuration
# ----------------------------
VIDEO_DIR = Path('/orcd/scratch/bcs/001/sensein/sails/BIDS_data/final_bids-dataset/derivatives/preprocessed/')
CSV_FILE = Path('/home/aparnabg/orcd/scratch/all_project_files/annotation/video.csv')
ANNOTATION_DIR = Path('/home/aparnabg/orcd/scratch/all_project_files/annotation/')
OUTPUT_DIR = Path('/home/aparnabg/orcd/scratch/all_project_files/annotation/')

# ----------------------------
# Label Categories
# ----------------------------
ACTION_CATEGORIES = {
    "Repetitive_Motor_Movements": ["Hands flapping", "Jumping", "Spinning" ,"Rocking"],
    "Locomotion": ["Crawling", "Cruising", "Walking", "Running", "Vehicle"],
}

CATEGORIES_WITH_REFERENCE = ["Locomotion", "Repetitive_Motor_Movements"]

ALL_CATEGORIES = {
    "Locomotion": ["Crawling", "Cruising", "Walking", "Running", "Vehicle"],
    "Repetitive_Motor_Movements": ["Hands flapping", "One hand flap", "Jumping", "Spinning", "Rocking", "Body tensing"],
}

# ----------------------------
# Pydantic Models
# ----------------------------
class VideoInfo(BaseModel):
    index: int
    path: str
    status: int
    category_values: dict[str, str]

class AnnotationSegment(BaseModel):
    start_frame: int
    end_frame: int
    label: str

class CategoryData(BaseModel):
    breakpoints: list[int]
    segments: list[AnnotationSegment]

class SaveRequest(BaseModel):
    video_index: int
    total_frames: int
    categories: dict[str, CategoryData]

class UpdateStatusRequest(BaseModel):
    video_index: int
    status: int

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Video Annotation Tool")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")

# Cache for CSV data to avoid repeated reads
_csv_cache: dict[str, Any] = {"data": None, "mtime": 0}

# ----------------------------
# Helper Functions
# ----------------------------
def is_empty_value(val: Any) -> bool:
    """Check if a value is considered empty/null"""
    if pd.isna(val):
        return True
    val_str = str(val).strip().lower()
    return val_str in ['', 'nill', 'n/a', 'na', 'none']

def get_video_filename(video_relative_path: str) -> str:
    return Path(video_relative_path).stem

def load_csv_data() -> tuple[list[str], list[int], dict[str, dict[str, str]]]:
    """Load CSV data with caching to improve performance"""
    global _csv_cache

    if not CSV_FILE.exists():
        return [], [], {}

    # Check if file has been modified
    current_mtime = CSV_FILE.stat().st_mtime
    if _csv_cache["data"] is not None and _csv_cache["mtime"] == current_mtime:
        return cast(tuple[list[str], list[int], dict[str, dict[str, str]]], _csv_cache["data"])

    df = pd.read_csv(CSV_FILE)
    video_list: list[str] = df.iloc[:, 0].astype(str).tolist()

    if 'status' in df.columns:
        video_status: list[int] = df['status'].fillna(0).astype(int).tolist()
    else:
        video_status = [0] * len(video_list)

    category_values: dict[str, dict[str, str]] = {}
    for _idx, row in df.iterrows():
        video_path = row.iloc[0]
        values: dict[str, str] = {}
        for category in CATEGORIES_WITH_REFERENCE:
            if category in df.columns:
                val = row[category]
                values[category] = str(val) if pd.notna(val) and str(val).strip() else ''
            else:
                values[category] = ''
        category_values[video_path] = values

    result: tuple[list[str], list[int], dict[str, dict[str, str]]] = (video_list, video_status, category_values)
    _csv_cache["data"] = result
    _csv_cache["mtime"] = current_mtime

    return result

# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/", response_class=HTMLResponse)
async def read_root() -> HTMLResponse | FileResponse:
    """Serve the main HTML page"""
    html_file = Path(__file__).parent / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    return HTMLResponse(content=get_html_content())

@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    """Get application configuration"""
    return {
        "action_categories": ACTION_CATEGORIES,
        "all_categories": ALL_CATEGORIES,
        "categories_with_reference": CATEGORIES_WITH_REFERENCE,
        "video_fps": 15
    }

@app.get("/api/videos")
async def get_videos() -> dict[str, Any]:
    """Get list of all videos with their status"""
    video_list, video_status, category_values = load_csv_data()

    videos = []
    for i, path in enumerate(video_list):
        videos.append({
            "index": i,
            "path": path,
            "status": video_status[i],
            "category_values": category_values.get(path, {})
        })

    return {"videos": videos, "total": len(videos)}

@app.get("/api/video/{video_index}")
async def get_video_info(video_index: int) -> dict[str, Any]:
    """Get information about a specific video"""
    video_list, video_status, category_values = load_csv_data()

    if video_index < 0 or video_index >= len(video_list):
        raise HTTPException(status_code=404, detail="Video not found")

    path = video_list[video_index]
    return {
        "index": video_index,
        "path": path,
        "status": video_status[video_index],
        "category_values": category_values.get(path, {})
    }

@app.get("/api/annotations/{video_index}")
async def get_annotations(video_index: int) -> dict[str, Any]:
    """Load annotations for a specific video - OPTIMIZED"""
    video_list, video_status, category_values = load_csv_data()

    if video_index < 0 or video_index >= len(video_list):
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = video_list[video_index]
    is_processed = video_status[video_index] == 1
    video_filename = get_video_filename(video_path)

    if is_processed:
        csv_file_path = OUTPUT_DIR / f"{video_filename}_actions_corrected.csv"
    else:
        csv_file_path = ANNOTATION_DIR / f"{video_filename}_actions_corrected.csv"

    if not csv_file_path.exists():
        return {"categories": {}, "found": False}

    try:
        # Use only necessary columns for faster loading
        df = pd.read_csv(csv_file_path, usecols=lambda x: x in list(ALL_CATEGORIES.keys()) or x == 'Frame')

        if len(df) == 0:
            return {"categories": {}, "found": False}

        ref_values = category_values.get(video_path, {})
        result: dict[str, Any] = {}

        for category in ALL_CATEGORIES:
            if category not in df.columns:
                continue

            should_set_na = False
            if category in CATEGORIES_WITH_REFERENCE:
                ref_val = ref_values.get(category, '')
                if is_empty_value(ref_val):
                    should_set_na = True

            if should_set_na:
                segments: list[dict[str, Any]] = [{"start_frame": 0, "end_frame": len(df), "label": "N/A"}]
            else:
                # Optimized segment creation using pandas
                labels = df[category].fillna('N/A')

                # Find where labels change
                changes = labels.ne(labels.shift()).cumsum()

                # Group by change points
                segments = []
                for _, group in labels.groupby(changes):
                    start_idx = group.index[0]
                    end_idx = group.index[-1] + 1
                    label = group.iloc[0]
                    segments.append({
                        "start_frame": int(start_idx),
                        "end_frame": int(end_idx),
                        "label": str(label)
                    })

            result[category] = {
                "breakpoints": [],
                "segments": segments
            }

        return {"categories": result, "found": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/save")
async def save_annotations(request: SaveRequest) -> dict[str, Any]:
    """Save annotations to CSV file"""
    global _csv_cache

    video_list, video_status, _ = load_csv_data()

    if request.video_index < 0 or request.video_index >= len(video_list):
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = video_list[request.video_index]
    video_filename = get_video_filename(video_path)
    output_csv_path = OUTPUT_DIR / f"{video_filename}_actions_corrected.csv"

    try:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, list[Any]] = {'Frame': list(range(request.total_frames))}

        for category in ALL_CATEGORIES:
            frame_labels: list[str] = ['N/A'] * request.total_frames

            if category in request.categories:
                cat_data = request.categories[category]
                for segment in cat_data.segments:
                    for frame_num in range(segment.start_frame, min(segment.end_frame, request.total_frames)):
                        frame_labels[frame_num] = segment.label

            data[category] = frame_labels

        df = pd.DataFrame(data)
        df.to_csv(output_csv_path, index=False)

        # Update status
        video_status[request.video_index] = 1
        update_status_in_csv(video_list, video_status)

        # Invalidate cache
        _csv_cache = {"data": None, "mtime": 0}

        return {
            "success": True,
            "message": f"Saved to {output_csv_path.name}",
            "filename": output_csv_path.name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/update-status")
async def update_status(request: UpdateStatusRequest) -> dict[str, Any]:
    """Update the status of a video"""
    global _csv_cache

    video_list, video_status, _ = load_csv_data()

    if request.video_index < 0 or request.video_index >= len(video_list):
        raise HTTPException(status_code=404, detail="Video not found")

    video_status[request.video_index] = request.status
    update_status_in_csv(video_list, video_status)

    # Invalidate cache
    _csv_cache = {"data": None, "mtime": 0}

    return {"success": True, "status": request.status}

@app.get("/api/first-unprocessed")
async def get_first_unprocessed() -> dict[str, Any]:
    """Get the index of the first unprocessed video"""
    video_list, video_status, _ = load_csv_data()

    for i in range(len(video_list)):
        if video_status[i] == 0:
            return {"index": i, "found": True}

    return {"index": -1, "found": False}

def update_status_in_csv(video_list: list[str], video_status: list[int]) -> None:
    """Update the status column in the CSV file"""
    try:
        df = pd.read_csv(CSV_FILE)

        if 'status' in df.columns:
            df['status'] = video_status
        else:
            df.insert(1, 'status', video_status)

        df.to_csv(CSV_FILE, index=False)
    except Exception as e:
        print(f"Error updating CSV status: {e}")

def get_html_content() -> str:
    """Return the HTML content if index.html doesn't exist"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Video Annotation Tool</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100">
        <div id="app" class="container mx-auto p-4">
            <h1 class="text-2xl font-bold mb-4">Video Annotation Tool</h1>
            <p class="text-gray-600">Loading application...</p>
            <p class="mt-2 text-sm text-red-600">Note: Please create an index.html file in the same directory as this script.</p>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9018)