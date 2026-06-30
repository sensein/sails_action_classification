import os
import subprocess
import threading
from typing import IO, Any

import cv2
import mediapipe as mp
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

# Configuration
CONFIDENCE_DETECTION = 0.5
CONFIDENCE_TRACKING = 0.5
SHOW_BOXES = False

# MediaPipe handles (safe at module level)
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

# Drawing specs
face_landmark_spec = mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
face_connection_spec = mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1)
pose_landmark_spec = mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=3, circle_radius=4)
pose_connection_spec = mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=3)
hand_landmark_spec = mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2)
hand_connection_spec = mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2)


def consume_stderr(proc: "subprocess.Popen[bytes]") -> None:
    """Consume stderr to prevent blocking."""
    stderr: IO[bytes] | None = proc.stderr
    if stderr is not None:
        for _line in stderr:
            pass


def visualize_holistic(frame: npt.NDArray[Any], results: object) -> npt.NDArray[Any]:
    """
    Visualize all holistic landmarks on the frame.
    Includes: Face (468), Pose (33), Left Hand (21), Right Hand (21) = 543 total.
    """
    vis_frame: npt.NDArray[Any] = frame.copy()

    if results.face_landmarks:  # type: ignore[attr-defined]
        mp_drawing.draw_landmarks(
            image=vis_frame,
            landmark_list=results.face_landmarks,  # type: ignore[attr-defined]
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=face_landmark_spec,
            connection_drawing_spec=face_connection_spec,
        )
        mp_drawing.draw_landmarks(
            image=vis_frame,
            landmark_list=results.face_landmarks,  # type: ignore[attr-defined]
            connections=mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=face_landmark_spec,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1),
        )

    if results.pose_landmarks:  # type: ignore[attr-defined]
        mp_drawing.draw_landmarks(
            image=vis_frame,
            landmark_list=results.pose_landmarks,  # type: ignore[attr-defined]
            connections=mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=pose_landmark_spec,
            connection_drawing_spec=pose_connection_spec,
        )

    if results.left_hand_landmarks:  # type: ignore[attr-defined]
        mp_drawing.draw_landmarks(
            image=vis_frame,
            landmark_list=results.left_hand_landmarks,  # type: ignore[attr-defined]
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=hand_landmark_spec,
            connection_drawing_spec=hand_connection_spec,
        )

    if results.right_hand_landmarks:  # type: ignore[attr-defined]
        mp_drawing.draw_landmarks(
            image=vis_frame,
            landmark_list=results.right_hand_landmarks,  # type: ignore[attr-defined]
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=hand_landmark_spec,
            connection_drawing_spec=hand_connection_spec,
        )

    if SHOW_BOXES and results.pose_landmarks:  # type: ignore[attr-defined]
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark  # type: ignore[attr-defined]
        x_coords = [lm.x * w for lm in landmarks if lm.visibility > 0.5]
        y_coords = [lm.y * h for lm in landmarks if lm.visibility > 0.5]

        if x_coords and y_coords:
            padding = 20
            x_min = max(0, int(min(x_coords)) - padding)
            y_min = max(0, int(min(y_coords)) - padding)
            x_max = min(w, int(max(x_coords)) + padding)
            y_max = min(h, int(max(y_coords)) + padding)
            cv2.rectangle(vis_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(vis_frame, "Person", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return vis_frame


if __name__ == "__main__":
    csv_path = '/home/aparnabg/orcd/pool/files_from_scratch/pose_models_test/filtered_single_person_facemesh.csv'
    output_folder = '/orcd/data/satra/002/projects/SAILS/pose_outputs/MediaPipe_holistic'

    print("=" * 60)
    print("MediaPipe Holistic Video Processor (543 Landmarks)")
    print("=" * 60)
    print("\nInitializing MediaPipe Holistic model")
    print("MediaPipe Holistic loaded successfully!\n")

    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    os.makedirs(output_folder, exist_ok=True)

    print(f"Found {len(df)} video(s) in CSV.")
    print(f"Output directory: {output_folder}")
    print("\nProcessing Settings:")
    print(f"  - Detection confidence: {CONFIDENCE_DETECTION}")
    print(f"  - Tracking confidence: {CONFIDENCE_TRACKING}")
    print(f"  - Show bounding boxes: {SHOW_BOXES}")
    print("  - Total landmarks tracked: 543 (Face: 468, Pose: 33, Hands: 42)\n")

    for idx, row in df.iterrows():
        video_path = row['BidsProcessed']
        video_id = row['ID']
        file_name = row['FileName']

        if not os.path.exists(video_path):
            print(f"⚠ Video not found: {video_path}")
            continue

        base_name = os.path.splitext(file_name)[0]
        output_filename = f"{video_id}_{base_name}_holistic.mp4"
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
            min_detection_confidence=CONFIDENCE_DETECTION,
            min_tracking_confidence=CONFIDENCE_TRACKING,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            refine_face_landmarks=True,
        ) as holistic, tqdm(total=total_frames, desc="Frames", unit="frame") as pbar:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                frame_rgb: npt.NDArray[Any] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = holistic.process(frame_rgb)
                frame_rgb.flags.writeable = True

                vis_frame = visualize_holistic(frame, results)

                try:
                    stdin: IO[bytes] | None = proc.stdin
                    if stdin is not None:
                        stdin.write(vis_frame.tobytes())
                except BrokenPipeError:
                    print(f"\nFFmpeg pipe broken at frame {frame_count}")
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