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
```

## Single-video demo scripts

`deepsort.py`, `deepsort_reid.py`, and `bytetrack.py` hardcode
`video_path = "video.mp4"` (and `output_folder`/`output_path`) at the top of
their `main()`. Edit that path and run directly:

```bash
python deepsort.py
python deepsort_reid.py
python bytetrack.py
```

## CSV-batch scripts

`yolo_pose.py`, `mediapipe_holistic.py`, and `face_mediapipe.py` read a
`csv_path` constant near the top of the file (a CSV with a `BidsProcessed`
column of video paths) and process every row:

```bash
python yolo_pose.py
python mediapipe_holistic.py
python face_mediapipe.py
```

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
`clip_tracker` (mmcv/mmdet/mmpose, used by `hrnet.py`), `vitpose`
(transformers-based ViTPose), and `entity-sam` (SAM2 + panopticapi).
`deepsort_reid.py` additionally needs `torchreid`, which is not covered by
any Poetry group and must be installed separately.
