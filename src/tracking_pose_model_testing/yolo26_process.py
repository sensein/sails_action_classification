import os
import cv2
import numpy as np
import pandas as pd
import subprocess
import threading
from tqdm import tqdm
from ultralytics import YOLO

# Configuration
MODEL_NAME = 'yolo11x-pose.pt'  # YOLOv11 pose model
device = 'cuda:0'  # Use 'cpu' if no GPU available

# Detection and filtering thresholds
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.7
KEYPOINT_CONFIDENCE = 0.3

# Visualization settings
KEYPOINT_RADIUS = 3
LINE_THICKNESS = 2
SHOW_BOXES = False

print("=" * 60)
print("YOLOv11-Pose Video Processor (Single Person)")
print("=" * 60)
print(f"\nInitializing YOLOv11-Pose model")
print(f"Model: {MODEL_NAME}")
print(f"Device: {device}")

# Load YOLO model
model = YOLO(MODEL_NAME)
model.to(device)

print("Model loaded successfully!\n")

# COCO 17 keypoint connections including face (0-indexed)
SKELETON = [
    # Face connections
    (0, 1), (0, 2),           # Nose to eyes
    (1, 3), (2, 4),           # Eyes to ears
    (1, 2),                   # Between eyes
    
    # Torso connections
    (5, 6),                   # Between shoulders
    (5, 11), (6, 12),         # Shoulders to hips
    (11, 12),                 # Between hips
    
    # Left arm
    (5, 7), (7, 9),           # Shoulder -> Elbow -> Wrist
    
    # Right arm
    (6, 8), (8, 10),          # Shoulder -> Elbow -> Wrist
    
    # Left leg
    (11, 13), (13, 15),       # Hip -> Knee -> Ankle
    
    # Right leg
    (12, 14), (14, 16)        # Hip -> Knee -> Ankle
]

# Keypoint colors (17 distinct colors for each keypoint)
KEYPOINT_COLORS = [
    (255, 0, 0),      # 0: Nose - Red
    (255, 85, 0),     # 1: Left Eye - Orange
    (255, 170, 0),    # 2: Right Eye - Light Orange
    (255, 255, 0),    # 3: Left Ear - Yellow
    (170, 255, 0),    # 4: Right Ear - Yellow-Green
    (85, 255, 0),     # 5: Left Shoulder - Light Green
    (0, 255, 0),      # 6: Right Shoulder - Green
    (0, 255, 85),     # 7: Left Elbow - Green-Cyan
    (0, 255, 170),    # 8: Right Elbow - Cyan-Green
    (0, 255, 255),    # 9: Left Wrist - Cyan
    (0, 170, 255),    # 10: Right Wrist - Light Blue
    (0, 85, 255),     # 11: Left Hip - Blue
    (0, 0, 255),      # 12: Right Hip - Deep Blue
    (85, 0, 255),     # 13: Left Knee - Blue-Purple
    (170, 0, 255),    # 14: Right Knee - Purple
    (255, 0, 255),    # 15: Left Ankle - Magenta
    (255, 0, 170)     # 16: Right Ankle - Pink
]

# Skeleton line colors (matching connections)
SKELETON_COLORS = [
    (255, 0, 0),      # Face connections (red tones)
    (255, 0, 0),
    (255, 100, 100),
    (255, 100, 100),
    (255, 150, 150),
    
    (0, 255, 0),      # Torso (green tones)
    (0, 200, 0),
    (0, 200, 0),
    (0, 150, 0),
    
    (0, 255, 255),    # Left arm (cyan)
    (0, 255, 255),
    
    (255, 255, 0),    # Right arm (yellow)
    (255, 255, 0),
    
    (255, 0, 255),    # Left leg (magenta)
    (255, 0, 255),
    
    (0, 0, 255),      # Right leg (blue)
    (0, 0, 255)
]

def consume_stderr(proc):
    """Consume stderr to prevent blocking"""
    for line in proc.stderr:
        pass

def visualize_pose(frame, results):
    """
    Visualize pose keypoints and skeleton on the frame.
    
    Args:
        frame: Input frame (BGR format)
        results: YOLO results object
    
    Returns:
        frame: Frame with visualized poses (BGR format)
    """
    vis_frame = frame.copy()
    
    if results[0].keypoints is None or len(results[0].keypoints) == 0:
        return vis_frame
    
    # Get keypoints data
    keypoints = results[0].keypoints.data.cpu().numpy()  # Shape: (num_persons, 17, 3)
    boxes = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else None
    
    # Draw for each detected person
    for person_idx, person_kpts in enumerate(keypoints):
        # Draw bounding box if enabled
        if SHOW_BOXES and boxes is not None and person_idx < len(boxes):
            x1, y1, x2, y2, conf, cls = boxes[person_idx]
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(vis_frame, f'{conf:.2f}', (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw skeleton connections first (so keypoints appear on top)
        for idx, (pt1_idx, pt2_idx) in enumerate(SKELETON):
            if pt1_idx >= len(person_kpts) or pt2_idx >= len(person_kpts):
                continue
            
            x1, y1, conf1 = person_kpts[pt1_idx]
            x2, y2, conf2 = person_kpts[pt2_idx]
            
            # Only draw if both keypoints are visible and confident
            if conf1 > KEYPOINT_CONFIDENCE and conf2 > KEYPOINT_CONFIDENCE:
                color = SKELETON_COLORS[idx % len(SKELETON_COLORS)]
                cv2.line(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                        color, LINE_THICKNESS, cv2.LINE_AA)
        
        # Draw keypoints on top of skeleton
        for kpt_idx, (x, y, conf) in enumerate(person_kpts):
            if conf > KEYPOINT_CONFIDENCE:
                color = KEYPOINT_COLORS[kpt_idx % len(KEYPOINT_COLORS)]
                cv2.circle(vis_frame, (int(x), int(y)), KEYPOINT_RADIUS, color, -1, cv2.LINE_AA)
    
    return vis_frame

# Read CSV file with video paths - SAME AS MEDIAPIPE
csv_path = '/home/aparnabg/orcd/pool/files_from_scratch/pose_models_test/csv2_filtered_multiple_people.csv'
output_folder = '/orcd/data/satra/002/projects/SAILS/pose_outputs/mutiperson_YOLOx11_Pose'

print(f"Reading CSV from: {csv_path}")
df = pd.read_csv(csv_path)
os.makedirs(output_folder, exist_ok=True)

print(f"Found {len(df)} video(s) in CSV.")
print(f"Output directory: {output_folder}")
print(f"\nProcessing Settings:")
print(f"- Model: {MODEL_NAME}")
print(f"- Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"- IoU threshold (NMS): {IOU_THRESHOLD}")
print(f"- Keypoint confidence: {KEYPOINT_CONFIDENCE}")
print(f"- Show bounding boxes: {SHOW_BOXES}\n")

# Process each video from the CSV
for idx, row in df.iterrows():
    video_path = row['BidsProcessed']
    video_id = row['ID']
    file_name = row['FileName']
    
    # Check if video_path is NaN or empty
    if pd.isna(video_path) or not isinstance(video_path, str):
        print(f"[{idx+1}/{len(df)}]   Skipping: Missing video path for {file_name}")
        continue
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        continue
    
    base_name = os.path.splitext(file_name)[0]
    output_filename = f"{video_id}_{base_name}_yolo26xpose.mp4"
    output_path = os.path.join(output_folder, output_filename)
    
    # Check if output file already exists
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
    
    print(f"Video info: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
    
    # Setup ffmpeg for h264 encoding
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
    
    # Start thread to consume stderr
    stderr_thread = threading.Thread(target=consume_stderr, args=(proc,))
    stderr_thread.daemon = True
    stderr_thread.start()
    
    with tqdm(total=total_frames, desc=f"Frames", unit="frame") as pbar:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run YOLO pose estimation
            results = model.predict(
                frame,
                verbose=False,
                device=device,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD
            )
            
            # Visualize poses on the frame
            vis_frame = visualize_pose(frame, results)
            
            # Write frame to ffmpeg
            try:
                proc.stdin.write(vis_frame.tobytes())
            except BrokenPipeError:
                print(f"\nFFmpeg pipe broken at frame {frame_count}")
                break
            
            pbar.update(1)
    
    # Close ffmpeg properly
    print("Finalizing video encoding")
    try:
        proc.stdin.close()
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        print("FFmpeg timeout, terminating")
        proc.terminate()
        proc.wait()
    
    stderr_thread.join(timeout=5)
    
    cap.release()
    print(f"Saved: {output_filename}\n")