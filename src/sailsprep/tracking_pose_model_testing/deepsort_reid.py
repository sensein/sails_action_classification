import warnings
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import torchreid
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import transforms
from ultralytics import YOLO  # type: ignore[attr-defined]

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

pose_model = YOLO('yolov8n-pose.pt')
osnet_model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=0,
    pretrained=True
)
osnet_model.to(device).eval()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
tracker = DeepSort(max_age=70, n_init=3, max_cosine_distance=0.2,
                   nn_budget=100, embedder=None, polygon=False)

PoseLandmarks = list[tuple[int, int, float]]
PoseData = dict[int, PoseLandmarks]


def detect_humans_and_poses(
    frame: npt.NDArray[np.uint8],
    confidence_threshold: float = 0.5,
) -> tuple[npt.NDArray[np.float32], PoseData]:
    results = pose_model(frame, verbose=False)
    detections: list[list[float]] = []
    pose_data: PoseData = {}
    for result in results:
        boxes, keypoints = result.boxes, result.keypoints
        if boxes is not None and keypoints is not None:
            for box, kpts in zip(boxes, keypoints, strict=False):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf, cls = box.conf[0].cpu().numpy(), box.cls[0].cpu().numpy()
                if int(cls) == 0 and conf > confidence_threshold:
                    detections.append([x1, y1, x2, y2, conf])
                    keypoints_array = kpts.xy[0].cpu().numpy()
                    confidence_array = kpts.conf[0].cpu().numpy()
                    pose_landmarks: PoseLandmarks = [
                        (int(x), int(y), float(c))
                        for (x, y), c in zip(keypoints_array, confidence_array, strict=False)
                    ]
                    pose_data[len(detections) - 1] = pose_landmarks
    return np.array(detections), pose_data


def extract_osnet_features(
    img_crops: list[npt.NDArray[np.uint8]],
) -> npt.NDArray[np.float32]:
    valid_crops = [transform(img).to(device) for img in img_crops if img.size != 0]
    if not valid_crops:
        return np.zeros((0, 512), dtype=np.float32)
    batch = torch.stack(valid_crops)
    with torch.no_grad():
        features = osnet_model(batch)
        features = F.normalize(features, p=2, dim=1)
    return features.cpu().numpy()


def crop_person_regions(
    frame: npt.NDArray[np.uint8],
    detections: npt.NDArray[np.float32],
) -> tuple[list[npt.NDArray[np.uint8]], list[Any]]:
    crops: list[npt.NDArray[np.uint8]] = []
    valid_detections: list[Any] = []
    for detection in detections:
        x1, y1, x2, y2 = detection[:4].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 > x1 and y2 > y1:
            crops.append(frame[y1:y2, x1:x2])
            valid_detections.append(detection)
    return crops, valid_detections


def _iou(boxA: npt.NDArray[np.float32], boxB: npt.NDArray[np.float32]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return float(interArea / float(boxAArea + boxBArea - interArea + 1e-6))


def assign_poses_to_tracks(
    tracks: list[Any],
    detections: npt.NDArray[np.float32],
    detection_poses: PoseData,
) -> PoseData:
    track_pose_data: PoseData = {}
    for track in tracks:
        if not track.is_confirmed():
            continue
        best_iou, best_idx = 0.0, -1
        for i, det in enumerate(detections):
            val = _iou(np.array(track.to_ltrb()), det[:4])
            if val > best_iou:
                best_iou, best_idx = val, i
        if best_idx != -1 and best_idx in detection_poses:
            track_pose_data[track.track_id] = detection_poses[best_idx]
    return track_pose_data


def draw_pose_keypoints(
    frame: npt.NDArray[np.uint8],
    keypoints: PoseLandmarks,
) -> npt.NDArray[np.uint8]:
    POSE_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    for _i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.3:
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), 2, (0, 0, 0), -1)
    for c in POSE_CONNECTIONS:
        if c[0] < len(keypoints) and c[1] < len(keypoints):
            pt1, pt2 = keypoints[c[0]], keypoints[c[1]]
            if pt1[2] > 0.3 and pt2[2] > 0.3:
                cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 255, 0), 2)
    return frame


def process_frame(
    frame: npt.NDArray[np.uint8],
) -> tuple[list[Any], npt.NDArray[np.uint8], PoseData]:
    detections, detection_poses = detect_humans_and_poses(frame)
    if len(detections) == 0:
        return [], frame, {}
    crops, valid_detections = crop_person_regions(frame, detections)
    det_list = [([x1, y1, x2, y2], conf, 'person') for x1, y1, x2, y2, conf in valid_detections]
    features = extract_osnet_features(crops)
    tracks: list[Any] = tracker.update_tracks(det_list, embeds=features, frame=frame) if det_list else []
    track_pose_data = assign_poses_to_tracks(tracks, detections, detection_poses)
    return tracks, frame, track_pose_data


def process_video(
    video_path: str,
    output_path: str | None = None,
    display: bool = False,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) if output_path else None
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_u8: npt.NDArray[np.uint8] = frame.astype(np.uint8)
        tracks, processed_frame, pose_data = process_frame(frame_u8)
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(processed_frame, f'ID: {track.track_id}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if track.track_id in pose_data:
                processed_frame = draw_pose_keypoints(processed_frame, pose_data[track.track_id])
        cv2.putText(
            processed_frame,
            f'Frame: {frame_count} | Tracks: {len(tracks)} | Poses: {len(pose_data)}',
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        if writer:
            writer.write(processed_frame)
        if display:
            cv2.imshow('Tracking', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "video.mp4"
    output_path = "output_yolo_pose_tracked.mp4"
    process_video(video_path, output_path, display=False)