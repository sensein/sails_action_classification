import os
import cv2
import numpy as np
import pandas as pd
import subprocess
import threading
import mediapipe as mp
from tqdm import tqdm
from ultralytics import YOLO

# Configuration
YOLO_MODEL = 'yolo11x.pt'
device = 'cuda:0'

YOLO_CONFIDENCE = 0.5
YOLO_IOU = 0.7

MP_DETECTION_CONFIDENCE = 0.5
MP_TRACKING_CONFIDENCE = 0.5

SHOW_BOXES = True
CROP_PADDING = 20

print("=" * 60)
print("Multi-Person: YOLO Detection + MediaPipe Face Mesh ONLY")
print("=" * 60)
print(f"\nInitializing models")

# Load YOLO
yolo_model = YOLO(YOLO_MODEL)
yolo_model.to(device)
print(f"YOLO model loaded: {YOLO_MODEL}")

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
#print("MediaPipe Holistic initialized\n")

# Face mesh drawing specification
face_mesh_tesselation_spec = mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=0)
face_mesh_contours_spec = mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=0)

def consume_stderr(proc):
    """Consume stderr to prevent blocking"""
    for line in proc.stderr:
        pass

def crop_person(frame, bbox, padding=CROP_PADDING):
    """Crop person from frame with padding"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox[:4])
    
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    cropped = frame[y1:y2, x1:x2]
    return cropped, (x1, y1, x2, y2)

def draw_face_mesh_only(frame, results, crop_coords):
    """
    Draw ONLY face mesh on original frame.
    No pose, no hands, just the face mesh.
    """
    x1, y1, x2, y2 = crop_coords
    
    # Create temporary frame for the crop region
    crop_vis = frame[y1:y2, x1:x2].copy()
    
    # Draw face mesh ONLY
    if results.face_landmarks:
        # Draw TESSELATION for full face mesh
        mp_drawing.draw_landmarks(
            image=crop_vis,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=face_mesh_tesselation_spec
        )
        
        # Draw CONTOURS for clearer face outline
        mp_drawing.draw_landmarks(
            image=crop_vis,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=face_mesh_contours_spec
        )
    
    # Copy annotated crop back to original frame
    frame[y1:y2, x1:x2] = crop_vis

def process_frame_multi_person(frame, yolo_model, holistic):
    """
    Process frame: detect people with YOLO, draw face mesh with MediaPipe.
    """
    vis_frame = frame.copy()
    
    # Step 1: Detect all people with YOLO
    yolo_results = yolo_model.predict(
        frame,
        verbose=False,
        device=device,
        conf=YOLO_CONFIDENCE,
        iou=YOLO_IOU,
        classes=[0]
    )
    
    # Step 2: Process each detected person
    if yolo_results[0].boxes is not None and len(yolo_results[0].boxes) > 0:
        boxes = yolo_results[0].boxes.data.cpu().numpy()
        
        for person_idx, bbox in enumerate(boxes):
            x1, y1, x2, y2, conf, cls = bbox
            
            # Draw bounding box
            if SHOW_BOXES:
                cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                cv2.putText(vis_frame, f'Person {person_idx+1} ({conf:.2f})', 
                          (int(x1), int(y1) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Crop person region
            person_crop, crop_coords = crop_person(frame, bbox)
            
            if person_crop.size == 0:
                continue
            
            # Convert to RGB for MediaPipe
            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            person_rgb.flags.writeable = False
            
            # Run MediaPipe on this person
            mp_results = holistic.process(person_rgb)
            person_rgb.flags.writeable = True
            
            # Draw ONLY face mesh
            draw_face_mesh_only(vis_frame, mp_results, crop_coords)
    
    return vis_frame

# Read CSV
csv_path = '/home/aparnabg/orcd/pool/files_from_scratch/pose_models_test/filtered_single_person_facemesh.csv'
output_folder = '/orcd/data/satra/002/projects/SAILS/pose_outputs/YOLO_MediaPipe_FaceMeshOnly'

print(f"Reading CSV from: {csv_path}")
df = pd.read_csv(csv_path)
os.makedirs(output_folder, exist_ok=True)

print(f"Found {len(df)} video(s) in CSV.")
print(f"Output directory: {output_folder}")
print(f"\nProcessing Settings:")
print(f"  - YOLO Model: {YOLO_MODEL}")
print(f"  - YOLO Confidence: {YOLO_CONFIDENCE}")
print(f"  - MediaPipe Detection Confidence: {MP_DETECTION_CONFIDENCE}")
print(f"  - MediaPipe Tracking Confidence: {MP_TRACKING_CONFIDENCE}")
print(f"  - Show bounding boxes: {SHOW_BOXES}")
print(f"  - Crop padding: {CROP_PADDING}px")
print(f"  - Drawing: Face mesh ONLY (468 landmarks)\n")

# Process each video
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
    
    # Setup ffmpeg
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
        output_path
    ]
    
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, 
                           stderr=subprocess.PIPE, bufsize=10**8)
    
    stderr_thread = threading.Thread(target=consume_stderr, args=(proc,))
    stderr_thread.daemon = True
    stderr_thread.start()
    
    # Initialize MediaPipe for this video
    with mp_holistic.Holistic(
        min_detection_confidence=MP_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_TRACKING_CONFIDENCE,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=True
    ) as holistic:
        
        with tqdm(total=total_frames, desc=f"Frames", unit="frame") as pbar:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                vis_frame = process_frame_multi_person(frame, yolo_model, holistic)
                
                # Write to ffmpeg
                try:
                    proc.stdin.write(vis_frame.tobytes())
                except BrokenPipeError:
                    print(f"FFmpeg pipe broken at frame {frame_count}")
                    break
                
                pbar.update(1)
    
    # Finalize
    print("Finalizing video encoding")
    try:
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
print("\nVisualization Details:")