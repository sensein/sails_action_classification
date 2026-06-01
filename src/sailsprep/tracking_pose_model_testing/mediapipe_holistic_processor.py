import os
import cv2
import numpy as np
import pandas as pd
import subprocess
import threading
import mediapipe as mp
from tqdm import tqdm

# Configuration
CONFIDENCE_DETECTION = 0.5  # Minimum detection confidence
CONFIDENCE_TRACKING = 0.5   # Minimum tracking confidence

# Visualization settings
SHOW_BOXES = False  # Set to True to show bounding boxes around detected persons

print("=" * 60)
print("MediaPipe Holistic Video Processor (543 Landmarks)")
print("=" * 60)
print(f"\nInitializing MediaPipe Holistic model")

# Initialize MediaPipe Holistic (Face + Hands + Pose = 543 landmarks)
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

print("MediaPipe Holistic loaded successfully!\n")

# Custom drawing specifications for better visualization
# Face landmarks
face_landmark_spec = mp_drawing.DrawingSpec(
    color=(80, 110, 10),  # Green for face
    thickness=1,
    circle_radius=1
)
face_connection_spec = mp_drawing.DrawingSpec(
    color=(80, 256, 121),  # Light green for face connections
    thickness=1
)

# Pose landmarks
pose_landmark_spec = mp_drawing.DrawingSpec(
    color=(245, 117, 66),  # Orange for pose keypoints
    thickness=3,
    circle_radius=4
)
pose_connection_spec = mp_drawing.DrawingSpec(
    color=(245, 66, 230),  # Pink for pose connections
    thickness=3
)

# Hand landmarks
hand_landmark_spec = mp_drawing.DrawingSpec(
    color=(121, 22, 76),  # Purple for hands
    thickness=2,
    circle_radius=2
)
hand_connection_spec = mp_drawing.DrawingSpec(
    color=(250, 44, 250),  # Magenta for hand connections
    thickness=2
)

def consume_stderr(proc):
    """Consume stderr to prevent blocking"""
    for line in proc.stderr:
        pass

def visualize_holistic(frame, results):
    """
    Visualize all holistic landmarks on the frame.
    Includes: Face (468), Pose (33), Left Hand (21), Right Hand (21) = 543 total
    
    Args:
        frame: Input frame (BGR format)
        results: MediaPipe holistic results object
    
    Returns:
        frame: Frame with visualized landmarks (BGR format)
    """
    vis_frame = frame.copy()
    
    # Draw face landmarks (468 landmarks)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image=vis_frame,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=face_landmark_spec,
            connection_drawing_spec=face_connection_spec
        )
        mp_drawing.draw_landmarks(
            image=vis_frame,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=face_landmark_spec,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1)
        )
    
    # Draw pose landmarks (33 landmarks)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=vis_frame,
            landmark_list=results.pose_landmarks,
            connections=mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=pose_landmark_spec,
            connection_drawing_spec=pose_connection_spec
        )
    
    # Draw left hand landmarks (21 landmarks)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=vis_frame,
            landmark_list=results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=hand_landmark_spec,
            connection_drawing_spec=hand_connection_spec
        )
    
    # Draw right hand landmarks (21 landmarks)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=vis_frame,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=hand_landmark_spec,
            connection_drawing_spec=hand_connection_spec
        )
    
    # Optional: Draw bounding box
    if SHOW_BOXES and results.pose_landmarks:
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark
        
        # Get x, y coordinates of all visible landmarks
        x_coords = [lm.x * w for lm in landmarks if lm.visibility > 0.5]
        y_coords = [lm.y * h for lm in landmarks if lm.visibility > 0.5]
        
        if x_coords and y_coords:
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            cv2.rectangle(vis_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(vis_frame, "Person", (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return vis_frame

# Read CSV file with video paths
csv_path = '/home/aparnabg/orcd/pool/files_from_scratch/pose_models_test/filtered_single_person_facemesh.csv'
output_folder = '/orcd/data/satra/002/projects/SAILS/pose_outputs/MediaPipe_holistic'

print(f"Reading CSV from: {csv_path}")
df = pd.read_csv(csv_path)
os.makedirs(output_folder, exist_ok=True)

print(f"Found {len(df)} video(s) in CSV.")
print(f"Output directory: {output_folder}")
print(f"\nProcessing Settings:")
print(f"  - Detection confidence: {CONFIDENCE_DETECTION}")
print(f"  - Tracking confidence: {CONFIDENCE_TRACKING}")
print(f"  - Show bounding boxes: {SHOW_BOXES}")
print(f"  - Total landmarks tracked: 543 (Face: 468, Pose: 33, Hands: 42)\n")

# Process each video from the CSV
for idx, row in df.iterrows():
    video_path = row['BidsProcessed']
    video_id = row['ID']
    file_name = row['FileName']
    
    if not os.path.exists(video_path):
        print(f" Video not found: {video_path}")
        continue
    
    base_name = os.path.splitext(file_name)[0]
    output_filename = f"{video_id}_{base_name}_holistic.mp4"
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
    
    # Initialize MediaPipe Holistic for this video
    with mp_holistic.Holistic(
        min_detection_confidence=CONFIDENCE_DETECTION,
        min_tracking_confidence=CONFIDENCE_TRACKING,
        model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
        smooth_landmarks=True,  # Smooth landmarks across frames
        enable_segmentation=False,  # Set to True if you want segmentation mask
        smooth_segmentation=True,
        refine_face_landmarks=True  # More accurate face mesh
    ) as holistic:
        
        with tqdm(total=total_frames, desc=f"Frames", unit="frame") as pbar:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Convert BGR to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # To improve performance, mark the image as not writeable
                frame_rgb.flags.writeable = False
                
                # Process the frame with MediaPipe Holistic
                results = holistic.process(frame_rgb)
                
                # Re-enable writability
                frame_rgb.flags.writeable = True
                
                # Visualize all landmarks on the frame
                vis_frame = visualize_holistic(frame, results)
                
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
        print(" FFmpeg timeout, terminating")
        proc.terminate()
        proc.wait()
    
    stderr_thread.join(timeout=5)
    
    cap.release()
    print(f"Saved: {output_filename}\n")

print("=" * 60)
print("All videos processed successfully!")
print(f"Output location: {output_folder}")
