import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO  # type: ignore[attr-defined]

device = "cuda" if torch.cuda.is_available() else "cpu"

CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
]


def draw_pose(
    img: np.ndarray,
    keypoints: list[list[float]],
    track_id: int | None = None,
) -> None:
    """Draw pose keypoints and connections with optional track ID."""
    if not keypoints:
        return

    # Draw keypoints
    for x, y, conf in keypoints:
        if conf > 0.5:
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

    # Draw connections
    n = len(keypoints)
    for start, end in CONNECTIONS:
        if start >= n or end >= n:
            continue
        if keypoints[start][2] > 0.5 and keypoints[end][2] > 0.5:
            pt1 = (int(keypoints[start][0]), int(keypoints[start][1]))
            pt2 = (int(keypoints[end][0]), int(keypoints[end][1]))
            cv2.line(img, pt1, pt2, (255, 0, 0), 2)

    # Draw track ID above head keypoint (index 0)
    if track_id is not None and keypoints and keypoints[0][2] > 0.5:
        head_x, head_y = int(keypoints[0][0]), int(keypoints[0][1])
        cv2.putText(
            img, f"ID:{track_id}", (head_x, head_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )


def main() -> None:
    video_path = "video.mp4"
    output_folder = "output_folder"

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(output_folder, f"pose_bytetrack_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter.fourcc(*"mp4v") 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    model: YOLO = YOLO("yolov8n-pose.pt").to(device)  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        for result in results:
            if result.keypoints is None:
                continue

            kp_xy = result.keypoints.xy
            kp_conf = result.keypoints.conf

            keypoints_np: np.ndarray = (
                kp_xy.cpu().numpy() if isinstance(kp_xy, torch.Tensor) else np.asarray(kp_xy)
            )
            confidences_np: np.ndarray = (
                kp_conf.cpu().numpy()
                if isinstance(kp_conf, torch.Tensor)
                else np.asarray(kp_conf)
            ) if kp_conf is not None else np.zeros((len(keypoints_np), keypoints_np.shape[1]))

            track_ids = None
            if result.boxes is not None and result.boxes.id is not None:
                box_ids = result.boxes.id
                track_ids = (
                    box_ids.cpu().numpy()
                    if isinstance(box_ids, torch.Tensor)
                    else np.asarray(box_ids)
                )

            for i, kpts in enumerate(keypoints_np):
                pose_data = [
                    [float(kpts[j][0]), float(kpts[j][1]), float(confidences_np[i][j])]
                    for j in range(len(kpts))
                ]
                current_track_id = (
                    int(track_ids[i]) if track_ids is not None and i < len(track_ids) else None
                )
                draw_pose(frame, pose_data, current_track_id)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()