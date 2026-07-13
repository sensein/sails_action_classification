import os

import cv2
import numpy as np
from rtmlib import Wholebody, draw_skeleton


device = 'cuda' # options: 'cpu', 'cuda', 'mps'
backend = 'onnxruntime' # options: 'opencv', 'onnxruntime', 'openvino'
openpose_skeleton = False  # True = OpenPose style, False = MMPose style


input_folder = '/video'
output_folder = '/video_output'

os.makedirs(output_folder, exist_ok=True)

video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.mkv', '.avi'))]
'''# By mode
wholebody = Wholebody(mode='performance',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend,
                      device=device)

# By det and pose
body = Body(det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
            det_input_size=(640, 640),
            pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip',
            pose_input_size=(288, 384),
            backend=backend,
            device=device)

# By det and pose with custom classes
custom = Custom(det_class='RTMDet',
                det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip',
                det_input_size=(320,320),
                pose_class='RTMPose',
                pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip',
                pose_input_size=(256, 256),
                backend=backend,
                device=device)'''

wholebody = Wholebody(
    to_openpose=openpose_skeleton,
    mode='balanced',
    backend=backend,
    device=device
)

for video_file in video_files:
    input_path = os.path.join(input_folder, video_file)
    output_filename = f"{os.path.splitext(video_file)[0]}.mp4"
    output_path = os.path.join(output_folder, output_filename)

    print(f"\n Processing: {video_file}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f" Failed to open {video_file}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, scores = wholebody(frame)
        frame_pose = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)
        out_video.write(frame_pose)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...", end='\r')

    cap.release()
    out_video.release()
    print(f" Saved to: {output_path}")

print("\n All videos processed.")
