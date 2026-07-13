# tracking_pose_model_testing

Standalone scripts for trying out individual pose-estimation, tracking, and
segmentation methods, independent of the `id_tracking_model/` pipeline.


## Files

```
yolo_pose.py           YOLOv11-Pose (single person) 
hrnet.py                HRNet wholebody 133-keypoint pose estimation 
vit_pose.py              ViTPose (usyd-community/vitpose-plus-huge) 
deepsort.py              YOLO-Pose + DeepSORT tracking 
deepsort_reid.py         YOLO-Pose + DeepSORT + a ReID (torchreid) embedding model —
bytetrack.py             YOLO-Pose + Ultralytics ByteTrack 
entitysam.py            EntitySAM-based video segmentation 
mediapipe_holistic.py    MediaPipe Holistic (pose + face + hands) 
face_mediapipe.py        YOLO person detection + MediaPipe Face Mesh 
movenet.py               MoveNet (TF-Hub) single-person pose w/ adaptive cropping
poseformer.py            PoseFormer 3D pose (delegates to an external poseformer_demo repo)
openpose_video.py        OpenPose (pyopenpose) body + face + hands, CSV-batch
openpifpaf.py            OpenPifPaf multi-person pose estimation
rtmlib.py                RTMLib Wholebody pose (ONNX/OpenVINO, no mm-stack)
rtmpose.py               RTMPose wholebody pose (RTMDet + mmpose)
sam2_yolov8.py           YOLOv8 detection + SAM2 video segmentation/tracking
deva.py                  DEVA + Grounding DINO text-prompted segmentation/tracking
efficientpose.py         EfficientPose (Keras/TF/TFLite/PyTorch) pose estimation
```

## Single-video demo scripts

`deepsort.py`, `deepsort_reid.py`, `bytetrack.py`, `movenet.py`, and
`poseformer.py` hardcode a single input video path (`video_path` /
`video_name`, and `output_folder`/`output_path`) near the top of the file.
Edit that path and run directly:

```bash
python deepsort.py
python deepsort_reid.py
python bytetrack.py
python movenet.py
python poseformer.py
```

`poseformer.py` only trims the input to 10s with ffmpeg and copies the
result out of an external `poseformer_demo` checkout (`/content/poseformer_demo`
in the script) — that repo's own demo pipeline does the actual 3D pose
inference and must be run separately.

## CSV-batch scripts

`yolo_pose.py`, `mediapipe_holistic.py`, and `face_mediapipe.py` read a
`csv_path` constant near the top of the file (a CSV with a `BidsProcessed`
column of video paths) and process every row:

```bash
python yolo_pose.py
python mediapipe_holistic.py
python face_mediapipe.py
```

`openpose_video.py` follows the same CSV/`BidsProcessed` pattern but also
accepts optional positional overrides for `csv_path`, `output_dir`, and
`model_folder`:

```bash
python openpose_video.py [csv_path] [output_dir] [model_folder]
```

## Folder-batch scripts

`openpifpaf.py`, `rtmlib.py`, `rtmpose.py`, `sam2_yolov8.py`, `deva.py`, and
`efficientpose.py` hardcode an input folder and output folder near the top
of the file and process every video (or, for `efficientpose.py`, every
`.mp4`) found there:

```bash
python openpifpaf.py
python rtmlib.py
python rtmpose.py
python sam2_yolov8.py
python deva.py
python efficientpose.py
```

`sam2_yolov8.py` uses YOLOv8 to detect people on one anchor frame per video,
then propagates SAM2 video segmentation from those boxes across the clip.
`deva.py` uses Grounding DINO with a fixed `'person'` text prompt to drive
DEVA's text-conditioned segmentation/tracking.

## SLURM array batch scripts

`hrnet.py` and `vit_pose.py` share the same interface — a split CSV plus an
H5 directory used to guide processing, sharded across a SLURM array:

```bash
python hrnet.py \
  --split_csv /path/to/latest_split_csv.csv \
  --h5_dir /path/to/h5folders \
  --output_dir /path/to/pose_hrnet_h5guided_json \
  --array_index $SLURM_ARRAY_TASK_ID \
  --num_jobs $SLURM_ARRAY_TASK_COUNT

python vit_pose.py \
  --split_csv /path/to/latest_split_csv.csv \
  --h5_dir /path/to/h5folders \
  --output_dir /path/to/pose_vitpose_h5guided_json \
  --model_name usyd-community/vitpose-plus-huge \
  --array_index $SLURM_ARRAY_TASK_ID \
  --num_jobs $SLURM_ARRAY_TASK_COUNT
```

`entitysam.py` runs EntitySAM segmentation over a CSV with `input_path` and
`output_path` columns, also sharded by SLURM array index:

```bash
python entitysam.py \
  --csv_file /path/to/videos.csv \
  --ckpt_dir /path/to/sam2_checkpoints \
  --model_cfg configs/sam2.1_hiera_l.yaml \
  --task_id $SLURM_ARRAY_TASK_ID \
  --num_tasks $SLURM_ARRAY_TASK_COUNT
```

Corresponding SLURM job scripts for all nine methods live under
`jobs/tracking_pose_model_testing/`
(`yolo_pose.sh`, `hrnet.sh`, `vit_pose.sh`, `deepsort.sh`, `deepsort_reid.sh`,
`bytetrack.sh`, `entitysam.sh`, `mediapipe_holistic.sh`, `face_mediapipe.sh`).

## Requirements

Covered by a mix of Poetry groups depending on the method: `tracking`
(ultralytics, deep-sort-realtime, mediapipe), `pose-estimation` /
`clip_tracker` (mmcv/mmdet/mmpose, used by `hrnet.py` and `rtmpose.py`),
`vitpose` (transformers-based ViTPose), and `entity-sam` (SAM2 +
panopticapi, used by `entitysam.py` and `sam2_yolov8.py`).
`deepsort_reid.py` additionally needs `torchreid`, which is not covered by
any Poetry group and must be installed separately.

None of the remaining new scripts are covered by a Poetry group; each
depends on a separate external install:

- `movenet.py` — `tensorflow`, `tensorflow_hub`, `tensorflow_docs`, `imageio`
- `openpifpaf.py` — `openpifpaf`
- `rtmlib.py` — `rtmlib` (ONNX/OpenVINO runtime, no mm-stack needed)
- `deva.py` — a `Tracking-Anything-with-DEVA` checkout plus Grounding DINO
  (or `GroundingDINO`) and its SAM checkpoints
- `openpose_video.py` — a built `openpose` install exposing `pyopenpose`,
  plus `ffmpeg` on `PATH` for H.264 encoding (falls back to OpenCV X264/mp4v)
- `poseformer.py` — an external `poseformer_demo` checkout and `ffmpeg`
- `efficientpose.py` — `pymediainfo`, `scikit-video` (`skvideo`), and
  whichever of `tensorflow`/`torch` matches the chosen framework, plus the
  corresponding EfficientPose model weights under `models/`
