# sails_action_classification

sails_action_classification: action classification and analysis of locomotion and Repetitive Motor Movements actions
from video data in the SAILS dataset, using pose estimation and tracking as
the underlying pipeline.

## Directory Overview

- `src/sailsprep/analysis/` — behavior-specific analysis modules (running, jumping, walking, crawling, cruising, rocking, spinning, handflapping, rmm_combined, loco_combined).
- `src/sailsprep/tracking_pose_model_testing/` — pose and tracking model wrappers (YOLO, HRNet, ViTPose, DeepSORT, Mediapipe, etc.).
- `src/sailsprep/action_model_testing/` — action classification models (pyskl, Video_Swin, VideoMAEv2, OpenTAD, VLMs, MS-TCN2, InternVideo, etc.).
- `src/sailsprep/id_tracking_model/` — child identity tracking pipeline.
- `src/sailsprep/fusion_model/` — late-fusion and combined-model pipelines.
- `src/sailsprep/annotation/` — annotation tools.
- `src/tests/` — unit tests mirroring the `src/sailsprep/` structure.
- `jobs/` — cluster/batch job scripts for running models and pipelines.
- `docs/` — project documentation.