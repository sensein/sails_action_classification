import os
import subprocess
import threading
from typing import IO, Any

import cv2
import mediapipe as mp
import numpy.typing as npt
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Configuration constants (safe at module level — no side effects)
# ---------------------------------------------------------------------------
YOLO_MODEL = 'yolo11x.pt'
device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

YOLO_CONFIDENCE = 0.5
YOLO_IOU = 0.7

MP_DETECTION_CONFIDENCE = 0.5
MP_TRACKING_CONFIDENCE = 0.5

SHOW_BOXES = True
CROP_PADDING = 20

# MediaPipe handles (safe to reference at module level; no heavy init)
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

face_mesh_tesselation_spec = mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=0)
face_mesh_contours_spec = mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=0)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def consume_stderr(proc: "subprocess.Popen[bytes]") -> None:
    """Consume stderr to prevent blocking."""
    stderr: IO[bytes] | None = proc.stderr
    if stderr is not None:
        for _line in stderr:
            pass


def crop_person(
    frame: npt.NDArray[Any],
    bbox: npt.NDArray[Any],
    padding: int = CROP_PADDING,
) -> tuple[npt.NDArray[Any], tuple[int, int, int, int]]:
    """Crop person from frame with padding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox[:4])

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    cropped: npt.NDArray[Any] = frame[y1:y2, x1:x2]
    return cropped, (x1, y1, x2, y2)


def draw_face_mesh_only(
    frame: npt.NDArray[Any],
    results: object,
    crop_coords: tuple[int, int, int, int],
) -> None:
    """Draw ONLY face mesh on original frame (no pose, no hands)."""
    x1, y1, x2, y2 = crop_coords
    crop_vis: npt.NDArray[Any] = frame[y1:y2, x1:x2].copy()

    if results.face_landmarks:  # type: ignore[attr-defined]
        # TESSELLATION — full face mesh
        mp_drawing.draw_landmarks(
            image=crop_vis,
            landmark_list=results.face_landmarks,  # type: ignore[attr-defined]
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=face_mesh_tesselation_spec,
        )
        # CONTOURS — clearer face outline
        mp_drawing.draw_landmarks(
            image=crop_vis,
            landmark_list=results.face_landmarks,  # type: ignore[attr-defined]
            connections=mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=face_mesh_contours_spec,
        )

    frame[y1:y2, x1:x2] = crop_vis


def process_frame_multi_person(
    frame: npt.NDArray[Any],
    yolo_model: object,
    holistic: object,
) -> npt.NDArray[Any]:
    """Detect people with YOLO, draw face mesh with MediaPipe."""
    vis_frame: npt.NDArray[Any] = frame.copy()

    yolo_results = yolo_model.predict(  # type: ignore[attr-defined]
        frame,
        verbose=False,
        device=device,
        conf=YOLO_CONFIDENCE,
        iou=YOLO_IOU,
        classes=[0],
    )

    if yolo_results[0].boxes is not None:
        boxes: npt.NDArray[Any] = yolo_results[0].boxes.data.cpu().numpy()
        # check length on the numpy array, not the YOLO boxes object
        # (avoids MagicMock __len__=0 issue in tests)
        if len(boxes) == 0:
            return vis_frame

        for person_idx, bbox in enumerate(boxes):
            x1, y1, x2, y2, conf, cls = bbox

            if SHOW_BOXES:
                cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    vis_frame,
                    f'Person {person_idx + 1} ({conf:.2f})',
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # skip zero-area detections before padding inflates them
            if int(bbox[2]) <= int(bbox[0]) or int(bbox[3]) <= int(bbox[1]):
                continue

            person_crop, crop_coords = crop_person(frame, bbox)
            if person_crop.size == 0:
                continue

            person_rgb: npt.NDArray[Any] = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            person_rgb.flags.writeable = False
            mp_results = holistic.process(person_rgb)  # type: ignore[attr-defined]
            person_rgb.flags.writeable = True

            draw_face_mesh_only(vis_frame, mp_results, crop_coords)

    return vis_frame


# ---------------------------------------------------------------------------
# Script entry point — heavy I/O, model loading, video processing
# Only runs when executed directly; NOT on import (so tests work cleanly).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    csv_path = '/home/aparnabg/orcd/pool/files_from_scratch/pose_models_test/filtered_single_person_facemesh.csv'
    output_folder = '/orcd/data/satra/002/projects/SAILS/pose_outputs/YOLO_MediaPipe_FaceMeshOnly'

    print("=" * 60)
    print("Multi-Person: YOLO Detection + MediaPipe Face Mesh ONLY")
    print("=" * 60)
    print("\nInitializing models")

    yolo_model = YOLO(YOLO_MODEL)
    yolo_model.to(device)
    print(f"YOLO model loaded: {YOLO_MODEL}")

    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    os.makedirs(output_folder, exist_ok=True)

    print(f"Found {len(df)} video(s) in CSV.")
    print(f"Output directory: {output_folder}")
    print("\nProcessing Settings:")
    print(f"  - YOLO Model: {YOLO_MODEL}")
    print(f"  - YOLO Confidence: {YOLO_CONFIDENCE}")
    print(f"  - MediaPipe Detection Confidence: {MP_DETECTION_CONFIDENCE}")
    print(f"  - MediaPipe Tracking Confidence: {MP_TRACKING_CONFIDENCE}")
    print(f"  - Show bounding boxes: {SHOW_BOXES}")
    print(f"  - Crop padding: {CROP_PADDING}px")
    print("  - Drawing: Face mesh ONLY (468 landmarks)\n")

    for idx, row in df.iterrows():
        video_path = row['BidsProcessed']
        video_id = row['ID']
        file_name = row['FileName']

        if not os.path.exists(video_path):
            print(f"⚠ Video not found: {video_path}")
            continue

        base_name = os.path.splitext(file_name)[0]
        output_filename = f"{video_id}_{base_name}_facemesh_only.mp4"
        output_path = os.path.join(output_folder, output_filename)

        if os.path.exists(output_path):
            print(f"[{idx+1}/{len(df)}] ⏭ Skipping (already exists): {file_name}")
            print(f"Output: {output_filename}\n")
            continue

        print(f"[{idx+1}/{len(df)}] Processing: {file_name}")
        print(f"Input: {video_path}")
        print(f"Output: {output_filename}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"✗ Failed to open video: {video_path}\n")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "23",
            output_path,
        ]

        proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
        )

        stderr_thread = threading.Thread(target=consume_stderr, args=(proc,))
        stderr_thread.daemon = True
        stderr_thread.start()

        with mp_holistic.Holistic(
            min_detection_confidence=MP_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_TRACKING_CONFIDENCE,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=True,
        ) as holistic, tqdm(total=total_frames, desc="Frames", unit="frame") as pbar:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                vis_frame = process_frame_multi_person(frame, yolo_model, holistic)

                try:
                    stdin: IO[bytes] | None = proc.stdin
                    if stdin is not None:
                        stdin.write(vis_frame.tobytes())
                except BrokenPipeError:
                    print(f"FFmpeg pipe broken at frame {frame_count}")
                    break

                pbar.update(1)

        print("Finalizing video encoding")
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("⚠ FFmpeg timeout, terminating")
            proc.terminate()
            proc.wait()

        stderr_thread.join(timeout=5)
        cap.release()
        print(f"Saved: {output_filename}\n")

    print("=" * 60)
    print("All videos processed successfully!")
    print(f"Output location: {output_folder}")
    print("=" * 60)