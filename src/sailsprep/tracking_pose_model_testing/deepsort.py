import os

import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO  # type: ignore[attr-defined]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def draw_pose(
    img: np.ndarray,
    keypoints: list[list[float]],
    track_id: int | None = None,
) -> None:

    """Draw pose keypoints and connections with optional track ID"""
    connections = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), 
                   (6,8), (8,10), (5,11), (6,12), (11,12), (11,13), 
                   (13,15), (12,14), (14,16)]
    
    # Draw keypoints
    for x, y, conf in keypoints:
        if conf > 0.5:
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
    
    # Draw connections
    for start, end in connections:
        if (keypoints[start][2] > 0.5 and keypoints[end][2] > 0.5):
            pt1 = (int(keypoints[start][0]), int(keypoints[start][1]))
            pt2 = (int(keypoints[end][0]), int(keypoints[end][1]))
            cv2.line(img, pt1, pt2, (255, 0, 0), 2)
    
    # Draw track ID 
    if track_id is not None and keypoints[0][2] > 0.5:
        head_x, head_y = int(keypoints[0][0]), int(keypoints[0][1])
        cv2.putText(img, f'ID:{track_id}', (head_x, head_y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


video_path = "video.mp4"
output_folder = "output_folder"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = os.path.join(output_folder, f"deepsort_{os.path.basename(video_path)}")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
model = YOLO('yolov8n-pose.pt').to(device)
deepsort_tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0) # embedder=''

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    results = model(frame)
    detections = []
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for _i, (box, conf) in enumerate(zip(boxes, confs, strict=False)):
                if conf > 0.5: 
                    x1, y1, x2, y2 = box
                    detections.append(([x1, y1, x2-x1, y2-y1], conf, None))
    
    tracks = deepsort_tracker.update_tracks(detections, frame=frame)
    active_tracks = []
    for track in tracks:
        if track.is_confirmed():
            active_tracks.append({
                'id': track.track_id,
                'bbox': track.to_ltrb()
            })
    
    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()
            confidences = result.keypoints.conf.cpu().numpy()
            
            for i, kpts in enumerate(keypoints):
                pose_data = [[kpts[j][0], kpts[j][1], confidences[i][j]] 
                           for j in range(len(kpts))]

                track_id = None
                pose_center = (kpts[5][0] + kpts[6][0])/2, (kpts[5][1] + kpts[6][1])/2  
                
                for track in active_tracks:
                    x1, y1, x2, y2 = track['bbox']
                    if (x1 <= pose_center[0] <= x2 and y1 <= pose_center[1] <= y2):
                        track_id = track['id']
                        break
                
                draw_pose(frame, pose_data, track_id)
    
    out.write(frame)

cap.release()
out.release()
print(f"Saved {output_path}")