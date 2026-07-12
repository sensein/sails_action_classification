"""Clip-level child action classifier using the Qwen2.5-VL model.

Supports two tasks:
  - ``loco``: Locomotion classification (5 classes).
  - ``rmm``: Repetitive Motor Movements classification (4 classes).

Each clip is classified by sampling *N* frames uniformly or randomly,
classifying each frame independently via the Qwen2.5-VL chat interface,
and aggregating predictions via majority vote.

Usage:
    python qwen_clip_classifier.py --task loco --csv splits_loco.csv --output-dir out/
    python qwen_clip_classifier.py --task rmm  --csv splits_rmm.csv  --output-dir out/

    # Random frame sampling with a fixed seed (for reproducible variance):
    python qwen_clip_classifier.py --task loco --csv splits_loco.csv \
        --output-dir out/ --random-frames --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.clip_metrics import (
    compute_metrics,
    compute_top2_accuracy,
    extract_label_from_path,
)

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
TASK_CLASSES: dict[str, list[str]] = {
    "loco": ["Crawling", "Cruising", "Walking", "Running", "Vehicle"],
    "rmm": ["Jumping", "Hands_flapping", "Rocking", "Spinning"],
}

_TASK_DESCRIPTIONS: dict[str, str] = {
    "loco": "locomotion action",
    "rmm": "repetitive motor movement",
}

_RMM_DEFINITIONS: str = (
    "\nDefinitions:\n"
    "- Jumping: repetitive jumping or bouncing movements\n"
    "- Hands_flapping: repetitive flapping or waving of hands/arms\n"
    "- Rocking: repetitive back-and-forth or side-to-side rocking of the "
    "body\n"
    "- Spinning: repetitive spinning or turning of the body\n"
)

_LOCO_DEFINITIONS: str = (
    "\nDefinitions:\n"
    "- Crawling: any instance of motion in any direction while the child "
    "is on all fours — can be hands and knees or hands and feet\n"
    "- Cruising: any instance of motion in any direction while the child "
    "is holding onto an inanimate object (e.g., table, chair, toy, wall, "
    "etc.) for support\n"
    "- Walking: any instance of motion in any direction while the child "
    "is in a bipedal stance, moving at a slow pace, and one foot is "
    "always on the ground\n"
    "- Running: any instance of motion in any direction while the child "
    "is in a bipedal stance, moving quickly, and at times both feet are "
    "off the ground\n"
    "- Vehicle: any instance of a child using a toy vehicle (e.g., "
    "scooter, walking bike, etc.) to move in any direction\n"
)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class ClipActionClassifier:
    """Classify short video clips via per-frame Qwen2.5-VL inference + vote.

    Args:
        task: Either ``"loco"`` or ``"rmm"``.
        model_name: HuggingFace model identifier for Qwen2.5-VL.
        num_sample_frames: Number of frames to uniformly sample per clip.
        dtype: Model precision — ``"bfloat16"`` or ``"float16"``.
        cache_dir: Optional HuggingFace cache directory override.
        random_frames: If ``True``, sample frames randomly instead of
            uniformly via linspace.  Enables variance across seeds.
        seed: Base RNG seed.  The per-clip seed is derived as
            ``seed + clip_index`` so every clip gets a different but
            reproducible draw when ``random_frames=True``.
    """

    def __init__(
        self,
        task: str,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        num_sample_frames: int = 8,
        dtype: str = "bfloat16",
        cache_dir: str | None = None,
        *,
        random_frames: bool = False,
        seed: int = 42,
    ) -> None:
        if task not in TASK_CLASSES:
            raise ValueError(
                f"Unknown task '{task}'. Choose from {list(TASK_CLASSES)}."
            )
        self.task = task
        self.class_names = TASK_CLASSES[task]
        self.num_sample_frames = num_sample_frames
        self.random_frames = random_frames
        self.seed = seed
        self._torch_dtype = (
            torch.bfloat16 if dtype == "bfloat16" else torch.float16
        )

        print(f"Loading model: {model_name} ...")
        print(
            f"Frame sampling: {'random (seed=' + str(seed) + ')' if random_frames else 'uniform (linspace)'}"
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self._torch_dtype,
            device_map="auto",
            cache_dir=cache_dir,
        )
        self.model.eval()

        self._prompt = self._build_prompt()
        print(f"Model loaded on {next(self.model.parameters()).device}.")

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_prompt(self) -> str:
        """Construct the zero-shot classification prompt for the task."""
        classes_str = ", ".join(self.class_names)
        desc = _TASK_DESCRIPTIONS[self.task]
        definitions = (
            _RMM_DEFINITIONS if self.task == "rmm" else _LOCO_DEFINITIONS
        )

        return (
            f"You are analyzing a video frame of a child performing a "
            f"{desc}.\n"
            f"The child is definitely performing one of these actions: "
            f"{classes_str}\n"
            f"{definitions}\n"
            f"Identify which single {desc} the child is performing.\n\n"
            f"Return your answer in this EXACT format (one line only):\n"
            f"ACTION: [action name]\n\n"
            f"Now analyze this frame:"
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def _parse_action(self, response: str) -> str | None:
        """Extract a valid class name from the model's generated text."""
        upper = response.upper()

        if "ACTION:" in upper:
            idx = upper.find("ACTION:")
            after = (
                response[idx + 7:]
                .strip()
                .split("\n")[0]
                .strip()
                .strip("\"'")
            )
            for cls in self.class_names:
                if cls.lower() == after.lower():
                    return cls
            if self.task == "rmm" and "flap" in after.lower():
                return "Hands_flapping"

        lower = response.lower()
        for cls in self.class_names:
            if cls.lower() in lower:
                return cls
        if self.task == "rmm" and "flap" in lower:
            return "Hands_flapping"

        return None

    # ------------------------------------------------------------------
    # Single-frame inference
    # ------------------------------------------------------------------
    def classify_frame(self, image: Image.Image) -> tuple[str | None, str]:
        """Classify one frame."""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self._prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=30, do_sample=False
                )

            generated_ids = [
                out[len(inp):]
                for inp, out in zip(inputs["input_ids"], output_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            del inputs, output_ids, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            predicted = self._parse_action(response)
            return predicted, response

        except Exception:
            traceback.print_exc()
            return None, ""

    # ------------------------------------------------------------------
    # Video frame sampling
    # ------------------------------------------------------------------
    def _sample_frames(
        self, video_path: str, clip_index: int = 0
    ) -> list[Image.Image]:
        """Sample frames from a video clip.

        When ``random_frames=False`` (default), frames are drawn via
        ``np.linspace`` — fully deterministic regardless of seed.

        When ``random_frames=True``, frames are drawn without replacement
        using a per-clip RNG seeded as ``self.seed + clip_index``.  This
        gives a different but reproducible draw for every (seed, clip) pair.

        Args:
            video_path: Path to the video file.
            clip_index: Zero-based position of this clip in the run; used
                only when ``random_frames=True`` to derive a per-clip seed.

        Returns:
            A list of PIL RGB images.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        n = min(self.num_sample_frames, total_frames)

        if self.random_frames:
            # Per-clip seed: different frames per clip, reproducible per seed.
            rng = np.random.default_rng(self.seed + clip_index)
            indices = sorted(
                rng.choice(total_frames, size=n, replace=False).tolist()
            )
        else:
            # Original deterministic uniform sampling.
            if total_frames <= self.num_sample_frames:
                indices = list(range(total_frames))
            else:
                indices = np.linspace(
                    0, total_frames - 1, self.num_sample_frames, dtype=int
                ).tolist()

        images: list[Image.Image] = []
        try:
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    images.append(Image.fromarray(rgb))
        finally:
            cap.release()

        return images

    # ------------------------------------------------------------------
    # Clip-level classification
    # ------------------------------------------------------------------
    def classify_clip(
        self, video_path: str, clip_index: int = 0
    ) -> tuple[str | None, list[str], float]:
        """Classify a clip via majority vote over sampled frames.

        Args:
            video_path: Path to the video clip.
            clip_index: Zero-based position of this clip; forwarded to
                ``_sample_frames`` for per-clip seed derivation.

        Returns:
            A tuple of ``(predicted_label, frame_predictions, confidence)``.
        """
        images = self._sample_frames(video_path, clip_index=clip_index)
        if not images:
            return None, [], 0.0

        frame_preds: list[str] = []
        for img in images:
            pred, raw = self.classify_frame(img)
            if pred is not None:
                frame_preds.append(pred)
            else:
                print(f"  [WARN] Unparsable response: {raw!r}")

        if not frame_preds:
            return None, [], 0.0

        counter = Counter(frame_preds)
        best_label = counter.most_common(1)[0][0]
        confidence = counter[best_label] / len(frame_preds)
        return best_label, frame_preds, confidence


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Parse CLI arguments and run clip-level classification + evaluation."""
    parser = argparse.ArgumentParser(
        description="Clip-level child action classifier using Qwen2.5-VL.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["loco", "rmm"],
        help="Classification task: 'loco' or 'rmm'.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the split CSV with a clip-path column.",
    )
    parser.add_argument(
        "--clip-column",
        type=str,
        default="cut_clip_path",
        help="CSV column containing clip file paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for result files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model identifier.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of frames to sample per clip.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Model precision.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional HuggingFace cache directory.",
    )
    # ---- New: random frame sampling ----
    parser.add_argument(
        "--random-frames",
        action="store_true",
        help=(
            "Sample frames randomly instead of uniformly via linspace. "
            "Enables meaningful metric variance across seeds."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Base RNG seed for random frame sampling. "
            "Per-clip seed is derived as seed + clip_index. "
            "Has no effect when --random-frames is not set."
        ),
    )

    args = parser.parse_args()
    class_names = TASK_CLASSES[args.task]

    print(f"\n{'=' * 60}")
    print("CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"  Task         : {args.task}")
    print(f"  Classes      : {class_names}")
    print(f"  CSV          : {args.csv}")
    print(f"  Clip column  : {args.clip_column}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Model        : {args.model}")
    print(f"  Frames/clip  : {args.num_frames}")
    print(f"  Dtype        : {args.dtype}")
    print(f"  Random frames: {args.random_frames}")
    if args.random_frames:
        print(f"  Seed         : {args.seed}")

    # ---- Load CSV and extract ground-truth labels ----
    df = pd.read_csv(args.csv)
    print(f"\n[INFO] CSV loaded — {df.shape[0]} rows.")

    if args.clip_column not in df.columns:
        print(
            f"[ERROR] Column '{args.clip_column}' not found. "
            f"Available: {list(df.columns)}"
        )
        return

    clip_paths: list[str] = df[args.clip_column].dropna().tolist()
    gt_labels: list[str] = []
    valid_clips: list[str] = []

    for cp in clip_paths:
        label = extract_label_from_path(cp, class_names)
        if label is not None:
            gt_labels.append(label)
            valid_clips.append(cp)
        else:
            print(f"[WARN] Cannot extract label from: {cp}")

    print(f"[INFO] {len(valid_clips)} clips with valid labels.")
    print(f"[INFO] Distribution: {dict(Counter(gt_labels))}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Initialise model ----
    classifier = ClipActionClassifier(
        task=args.task,
        model_name=args.model,
        num_sample_frames=args.num_frames,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
        random_frames=args.random_frames,
        seed=args.seed,
    )

    # ---- Classify every clip ----
    results: list[dict] = []
    all_frame_preds: list[list[str]] = []
    successful = 0
    failed = 0

    for i, (clip_path, true_label) in enumerate(
        zip(valid_clips, gt_labels), 1
    ):
        print(
            f"\n--- [{i}/{len(valid_clips)}] "
            f"{Path(clip_path).name}  (GT: {true_label})"
        )

        try:
            global_clip_index = valid_clips.index(clip_path)
            pred_label, frame_preds, confidence = classifier.classify_clip(
                clip_path, clip_index=global_clip_index
            )

            if pred_label is None:
                pred_label = "Unknown"
                frame_preds = []
                confidence = 0.0
                failed += 1
            else:
                successful += 1

            is_correct = pred_label == true_label
            print(
                f"  Predicted: {pred_label}  |  Confidence: {confidence:.2f}  "
                f"|  Correct: {is_correct}"
            )
            print(f"  Frame votes: {dict(Counter(frame_preds))}")

            results.append(
                {
                    "clip_path": clip_path,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "confidence": confidence,
                    "correct": is_correct,
                    "frame_predictions": str(frame_preds),
                    "num_frames_sampled": len(frame_preds),
                    "seed": args.seed if args.random_frames else "N/A",
                    "random_frames": args.random_frames,
                }
            )
            all_frame_preds.append(frame_preds)

        except Exception:
            traceback.print_exc()
            failed += 1
            results.append(
                {
                    "clip_path": clip_path,
                    "true_label": true_label,
                    "predicted_label": "Error",
                    "confidence": 0.0,
                    "correct": False,
                    "frame_predictions": "[]",
                    "num_frames_sampled": 0,
                    "seed": args.seed if args.random_frames else "N/A",
                    "random_frames": args.random_frames,
                }
            )
            all_frame_preds.append([])

    # ---- Save raw predictions ----
    results_df = pd.DataFrame(results)
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    pred_csv = os.path.join(
        args.output_dir, f"clip_predictions_{array_id}.csv"
    )
    results_df.to_csv(pred_csv, index=False)
    print(f"\n[SAVED] {pred_csv}")

    # ---- Evaluate (this chunk only) ----
    valid_mask = results_df["predicted_label"].isin(class_names)
    eval_df = results_df[valid_mask]

    if eval_df.empty:
        print("[WARN] No valid predictions in this chunk to evaluate.")
        return

    y_true = eval_df["true_label"].tolist()
    y_pred = eval_df["predicted_label"].tolist()
    excluded = len(results_df) - len(eval_df)
    print(
        f"\n[INFO] Evaluating {len(eval_df)} clips "
        f"({excluded} excluded due to errors)."
    )

    chunk_metrics_dir = os.path.join(args.output_dir, f"metrics_{array_id}")
    metrics = compute_metrics(y_true, y_pred, class_names, chunk_metrics_dir)

    valid_fp = [all_frame_preds[i] for i in eval_df.index]
    top2_acc = compute_top2_accuracy(valid_fp, y_true)
    print(f"  Top-2 accuracy: {top2_acc:.4f}")

    json_path = os.path.join(chunk_metrics_dir, "evaluation_metrics.json")
    with open(json_path) as fh:
        metrics_data = json.load(fh)
    metrics_data.update(
        {
            "top2_accuracy": top2_acc,
            "total_clips": len(results_df),
            "valid_clips_evaluated": len(eval_df),
            "failed_clips": excluded,
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "task": args.task,
            "num_frames_per_clip": args.num_frames,
            "slurm_array_task_id": array_id,
            "random_frames": args.random_frames,
            "seed": args.seed if args.random_frames else None,
        }
    )
    with open(json_path, "w") as fh:
        json.dump(metrics_data, fh, indent=2)

    print(f"\n{'=' * 60}")
    print(
        f"DONE — {len(results_df)} total  |  "
        f"{successful} succeeded  |  {failed} failed"
    )
    print(f"Results: {chunk_metrics_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()