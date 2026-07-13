# Action Model Testing

Training and inference scripts for the action-recognition experiments run on
the SAILS locomotion (`loco`) and repetitive motor movement (`rmm`) tasks.
Each subfolder below is a separate model family with its own inputs, and most
have their own README with setup and run instructions.

## Model folders

| Folder | Model | README |
|---|---|---|
| `video_swin/` | Video Swin Transformer (clip-based, binary/full-video/two-stage sliding window) | [video_swin/README.md](video_swin/README.md) |
| `videomae2/` | VideoMAE V2 finetuning (per-clip, full-video sliding window, two-stage) | [videomae2/README.md](videomae2/README.md) |
| `internvideo2/` | InternVideo2-6B finetuning | [internvideo2/README.md](internvideo2/README.md) |
| `slow_fast/` | SlowFast finetuning (single clip pipeline + ablation experiments) | [slow_fast/README.md](slow_fast/README.md) |
| `motionbert/` | MotionBERT 2D->3D pose lifting + skeleton action recognition | [motionbert/README.md](motionbert/README.md) |
| `mstcn2/` | MS-TCN++ frame-level action segmentation over pre-extracted features | [mstcn2/README.md](mstcn2/README.md) |
| `feature_extraction/` | I3D, R(2+1)D, and V-JEPA2 per-frame feature extraction | [feature_extraction/README.md](feature_extraction/README.md) |
| `vjepa/` | V-JEPA2 feature extraction, attentive-probe training, and fine-tuning variants | [vjepa/README.md](vjepa/README.md) |
| `open_tad/` | Feature-based temporal action detection with OpenTAD | [open_tad/README.md](open_tad/README.md) |
| `pyskl/` | PySKL skeleton-based action recognition configs, training helpers, and logit fusion | [pyskl/README.md](pyskl/README.md) |
| `vlm_models/` | Qwen2.5-VL / Ovis2 vision-language model classifiers | [vlm_models/README.md](vlm_models/README.md) |
| `dlc_action/` | DLC2Action pipeline for pose-based locomotion classification | [dlc_action/README.md](dlc_action/README.md) |

## Tasks

Nearly every script here trains for one of two label sets from the SAILS
annotation CSVs:

| Task flag | Annotation column | Classes |
|---|---|---|
| `loco` | `Locomotion` | Crawling, Cruising, Running, Vehicle, Walking |
| `rmm` | `Repetitive_Motor_Movements` | Hands_flapping, Jumping, Rocking, Spinning |

## Conventions across these scripts

- Most scripts read a single split CSV (with `video_path`, `label_path`, and a
  `split` column of `train`/`val`/`test`) and take `--task`/`--label` plus
  `--seed` as CLI flags. Paths to that CSV, along with checkpoint/output
  directories, are set as constants near the top of each script rather than
  passed as flags — edit those constants for your environment.
- Reported experiments are generally run over three seeds: `42`, `123`, `456`.
- Several folders import shared helpers with bare imports (e.g.
  `from common.utils import ...`, `from utils.bbox import ...`). Those scripts
  must be run with their own folder as the working directory, not with
  `python -m`. Each model's README states which convention it uses.
- `pyskl/` and `open_tad/` are standalone scripts that operate inside a
  separate upstream repo checkout (PySKL, OpenTAD) rather than importing
  `sailsprep` directly — see their READMEs for the copy-in setup.

Corresponding SLURM job scripts for these models live under
`jobs/action_model_testing/`.
