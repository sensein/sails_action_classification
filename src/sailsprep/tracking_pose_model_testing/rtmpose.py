import os

import cv2
import mmcv
import mmengine
import numpy as np
from mmcv import imread
from mmdet.apis import inference_detector, init_detector
from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from tqdm import tqdm


det_config = 'projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py'
det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
pose_config = 'projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmw-x_8xb320-270e_cocktail14-384x288.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth'

device = 'cuda:0'
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))


detector = init_detector(
    det_config,
    det_checkpoint,
    device=device
)


# build pose estimator
pose_estimator = init_pose_estimator(
    pose_config,
    pose_checkpoint,
    device=device,
    cfg_options=cfg_options
)

# init visualizer
pose_estimator.cfg.visualizer.radius = 3
pose_estimator.cfg.visualizer.line_width = 1
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
visualizer.set_dataset_meta(pose_estimator.dataset_meta)


def visualize_img(img_path, detector, pose_estimator, visualizer,
                  show_interval, out_file):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    scope = detector.cfg.get('default_scope', 'mmdet')
    if scope is not None:
        init_default_scope(scope)
    detect_result = inference_detector(detector, img_path)
    pred_instance = detect_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > 0.3)]
    bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img_path, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    img = mmcv.imread(img_path, channel_order='rgb')

    visualizer.add_datasample(
        'result',
        img,
        data_sample=data_samples,
        draw_gt=False,
        draw_heatmap=False, # make True if you want heatmap
        draw_bbox=False, # make True if you want bounding box
        show=False,
        wait_time=show_interval,
        out_file=out_file,
        kpt_thr=0.3)


input_folder = '/videos'
output_folder = '/video_output'


os.makedirs(output_folder, exist_ok=True)
video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]

print(f"Found {len(video_files)} video(s).")

for video_file in video_files:
    video_path = os.path.join(input_folder, video_file)
    output_path = os.path.join(output_folder, video_file.replace('.mp4', '_pose.mp4'))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    print(f"\n Processing: {video_file}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        frame_path = 'temp_frame.jpg'
        cv2.imwrite(frame_path, frame)

        visualize_img(
            frame_path,
            detector,
            pose_estimator,
            visualizer,
            show_interval=0,
            out_file=None
        )


        vis_result = visualizer.get_image()

        writer.write(vis_result[..., ::-1])

        # preview
        # if frame_idx % 200 == 0:
        #     cv2.imshow("frame", vis_result[..., ::-1])
        #     cv2.waitKey(1)

    cap.release()
    writer.release()
    print(f"Saved: {output_path}")

print("All videos processed and saved to output folder.")
