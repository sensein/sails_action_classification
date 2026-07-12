# vlm_models

Vision-language model (VLM) classifiers for child movement analysis, using Qwen2.5-VL and Ovis2. Two classification tasks are supported:

- **loco**: Locomotion (Crawling, Cruising, Walking, Running, Vehicle)
- **rmm**: Repetitive Motor Movements (Jumping, Hands_flapping, Rocking, Spinning)

## Structure

```
common/
  clip_metrics.py     # Ground-truth extraction + evaluation metrics for clip-level classifiers
  window_parsers.py   # Response parsers (multiclass / binary / finegrained) for window classifiers
  shared_utils.py      # Task config, frame sampling, label loading, metrics for window classifiers

clips/
  qwen_clip_classifier.py   # Clip-level classifier using Qwen2.5-VL
  ovis_clip_classifier.py   # Clip-level classifier using Ovis2

window_classification/
  window_classifier_qwen.py  # 2-second window classifier using Qwen2.5-VL (per-frame vote)
  window_classifier_ovis.py  # 2-second window classifier using Ovis2 (temporal frame grid)
```

## Requirements

- Python 3.10+
- `torch`, `transformers`
- `opencv-python` (`cv2`)
- `numpy`, `pandas`
- `scikit-learn`
- `Pillow`
- `qwen_vl_utils` (for the Qwen classifiers)
- A CUDA-capable GPU is expected for model inference

## Clip classifiers

Classify whole video clips by sampling frames, running per-frame inference, and aggregating predictions via majority vote.

```bash
python clips/qwen_clip_classifier.py --task loco --csv splits_loco.csv --output-dir out/
python clips/ovis_clip_classifier.py --task loco --csv splits_loco.csv --output-dir out/
```

Common arguments: `--task {loco,rmm}`, `--csv`, `--clip-column`, `--output-dir`, `--model`, `--num-frames`, `--random-frames`, `--seed`.

Model-specific arguments:
- Qwen: `--dtype {bfloat16,float16}`, `--cache-dir`
- Ovis: `--max-partition`, `--no-flash-attn`

The input CSV must contain a column with paths to clip files, where the ground-truth label is inferred from the class-name folder in each path.

Outputs (written to `--output-dir`): `evaluation_metrics.json`, `confusion_matrix.csv`, `classification_report.csv`.

## Window classifiers

Classify individual 2-second windows within a clip, using one of three approaches:

- **a** — direct multi-class
- **b** — 2-stage (binary → fine-grained)
- **c** — binary only

```bash
python window_classification/window_classifier_qwen.py --task loco --approach a --csv split.csv --output-dir out/
python window_classification/window_classifier_ovis.py --task loco --approach a --csv split.csv --output-dir out/
```

Common arguments: `--task {loco,rmm}`, `--approach {a,b,c}`, `--csv`, `--video-col`, `--label-col`, `--output-dir`, `--model`, `--num-frames`, `--random-frames`, `--seed`.

Model-specific arguments:
- Qwen: `--dtype {bfloat16,float16}`, `--cache-dir`
- Ovis: `--max-partition`, `--no-flash-attn`

## Notes

- `--random-frames` derives a per-clip seed as `seed + clip_index`, giving a different but reproducible frame draw per clip.
