"""Clip-level child action classifier using the Ovis2.

Supports two tasks:
  - ``loco``: Locomotion classification (5 classes).
  - ``rmm``: Repetitive Motor Movements classification (4 classes).

Each clip is classified by sampling N frames uniformly or randomly,
classifying each frame independently, and aggregating predictions via
majority vote.

Usage:
    python ovis_clip_classifier.py --task loco --csv splits_loco.csv --output-dir out/
    python ovis_clip_classifier.py --task rmm  --csv splits_rmm.csv  --output-dir out/

    # Random frame sampling with a fixed seed (for reproducible variance):
    python ovis_clip_classifier.py --task loco --csv splits_loco.csv \
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
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
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
from transformers import AutoConfig, AutoModelForCausalLM

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
# Monkey-patch: suppress duplicate aimv2 config registration
# ---------------------------------------------------------------------------
def _patch_aimv2_registration() -> None:
    """Silently skip duplicate ``aimv2`` ``AutoConfig`` registrations."""
    _original_register = AutoConfig.register

    def _safe_register(model_type: str, config: Any, exist_ok: bool = False) -> None:
        try:
            _original_register(model_type, config, exist_ok=exist_ok) 
        except ValueError as exc:
            if "aimv2" in str(exc):
                pass
            else:
                raise

    AutoConfig.register = _safe_register  # type: ignore[method-assign ]

_patch_aimv2_registration()


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class ClipActionClassifier:
    """Classify short video clips via per-frame VLM inference + majority vote.

    Args:
        task: Either ``"loco"`` or ``"rmm"``.
        model_name: HuggingFace model identifier for Ovis2.
        num_sample_frames: Number of frames to uniformly sample per clip.
        max_partition: Maximum visual-token partition for Ovis2.
        use_flash_attn: If ``False``, force eager attention.
        random_frames: If ``True``, sample frames randomly instead of
            uniformly via linspace.  Enables variance across seeds.
        seed: Base RNG seed.  The per-clip seed is derived as
            ``seed + clip_index`` so every clip gets a different but
            reproducible draw when ``random_frames=True``.
    """

    def __init__(
        self,
        task: str,
        model_name: str = "AIDC-AI/Ovis2-8B",
        num_sample_frames: int = 8,
        max_partition: int = 9,
        *,
        use_flash_attn: bool = True,
        random_frames: bool = False,
        seed: int = 42,
    ) -> None:
        if task not in TASK_CLASSES:
            raise ValueError(
                f"Unknown task '{task}'. Choose from {list(TASK_CLASSES)}."
            )
        self.task = task
        self.class_names = TASK_CLASSES[task]
        self.random_frames = random_frames
        self.seed = seed

        print(f"Loading model: {model_name} ...")
        print(
            f"Frame sampling: {'random (seed={seed})' if random_frames else 'uniform (linspace)'}"
        )

        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "multimodal_max_length": 32768,
            "trust_remote_code": True,
        }

        if not use_flash_attn:
            config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=True
            )
            config.llm_attn_implementation = "eager"
            if hasattr(config, "llm_config"):
                config.llm_config.attn_implementation = "eager"
                config.llm_config._attn_implementation = "eager"
            load_kwargs["config"] = config
            load_kwargs["attn_implementation"] = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(  # type: ignore[call-arg]
            model_name, **load_kwargs
        ).cuda()

        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.num_sample_frames = num_sample_frames
        self.max_partition = max_partition
        self._prompt = self._build_prompt()

        print(f"Model loaded on {self.model.device}.")

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
            query = f"<image>\n{self._prompt}"

            _, input_ids, pixel_values = self.model.preprocess_inputs(
                query, [image], max_partition=self.max_partition
            )

            attention_mask = torch.ne(
                input_ids, self.text_tokenizer.pad_token_id
            )
            input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
            attention_mask = attention_mask.unsqueeze(0).to(
                device=self.model.device
            )

            if pixel_values is not None:
                pixel_values = pixel_values.to(
                    dtype=self.visual_tokenizer.dtype,
                    device=self.visual_tokenizer.device,
                )
                pixel_values = [pixel_values]

            

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    max_new_tokens=30,
                    do_sample=False,
                    top_p=None,
                    top_k=None,
                    temperature=None,
                    repetition_penalty=None,
                    eos_token_id=self.model.generation_config.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                    use_cache=True,
                )[0]

            response = self.text_tokenizer.decode(
                output_ids, skip_special_tokens=True
            ).strip()

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
        gives a different but reproducible draw for every (seed, clip) pair,
        which is what you want when measuring metric spread across seeds.

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
# Ground-truth extraction
# ---------------------------------------------------------------------------
def extract_label_from_path(
    clip_path: str, class_names: list[str]
) -> str | None:
    """Extract the ground-truth label from a folder name in *clip_path*."""
    for part in Path(clip_path).parts:
        if part in class_names:
            return part
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    class_names: list[str],
    output_dir: str,
) -> dict[str, Any]:
    """Compute classification metrics and persist results to *output_dir*."""
    metrics: dict[str, Any] = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    for avg in ("micro", "macro", "weighted"):
        metrics[f"precision_{avg}"] = precision_score(
            y_true, y_pred, average=avg, zero_division=0
        )
        metrics[f"recall_{avg}"] = recall_score(
            y_true, y_pred, average=avg, zero_division=0
        )
        metrics[f"f1_{avg}"] = f1_score(
            y_true, y_pred, average=avg, zero_division=0
        )

    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    metrics["classification_report"] = report

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    metrics["confusion_matrix"] = cm.tolist()

    present = sorted(set(y_true) | set(y_pred))
    try:
        y_true_bin = label_binarize(y_true, classes=class_names)
        y_pred_bin = label_binarize(y_pred, classes=class_names)
        if y_true_bin.shape[1] > 1:
            ap: dict[str, float] = {}
            for i, cls in enumerate(class_names):
                if cls in present and y_true_bin[:, i].sum() > 0:
                    ap[cls] = float(
                        average_precision_score(
                            y_true_bin[:, i], y_pred_bin[:, i]
                        )
                    )
            if ap:
                metrics["mAP"] = float(np.mean(list(ap.values())))
            metrics["AP_per_class"] = ap
    except Exception as exc:
        print(f"[WARN] Could not compute mAP: {exc}")

    os.makedirs(output_dir, exist_ok=True)

    serialisable: dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            serialisable[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            serialisable[k] = float(v)
        else:
            serialisable[k] = v

    json_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(json_path, "w") as fh:
        json.dump(serialisable, fh, indent=2)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))

    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Cohen's kappa:     {metrics['cohen_kappa']:.4f}")
    print(f"  MCC:               {metrics['mcc']:.4f}")
    print(f"  Macro F1:          {metrics['f1_macro']:.4f}")
    print(f"  Weighted F1:       {metrics['f1_weighted']:.4f}")
    if "mAP" in metrics:
        print(f"  mAP:               {metrics['mAP']:.4f}")
    print(f"\nConfusion matrix:\n{cm_df}")
    print(f"\nPer-class report:\n{report_df.to_string()}")

    return metrics


def compute_top2_accuracy(
    frame_preds_list: list[list[str]],
    y_true: list[str],
) -> float:
    """Compute top-2 accuracy from per-clip frame vote distributions."""
    correct = 0
    total = 0
    for preds, true_label in zip(frame_preds_list, y_true, strict=False):
        if not preds:
            continue
        top2 = [cls for cls, _ in Counter(preds).most_common(2)]
        if true_label in top2:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Parse CLI arguments and run clip-level classification + evaluation."""
    parser = argparse.ArgumentParser(
        description="Clip-level child action classifier using Ovis2.",
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
        default="AIDC-AI/Ovis2-8B",
        help="HuggingFace model identifier.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of frames to sample per clip.",
    )
    parser.add_argument(
        "--max-partition",
        type=int,
        default=9,
        help="Maximum visual-token partition for Ovis2.",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable flash attention (use eager instead).",
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
            "Per-clip seed is derived as seed + clip_index so every clip "
            "gets a different but reproducible draw. "
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
    print(f"  Flash attn   : {not args.no_flash_attn}")
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
        max_partition=args.max_partition,
        use_flash_attn=not args.no_flash_attn,
        random_frames=args.random_frames,
        seed=args.seed,
    )

    # ---- Classify every clip ----
    results: list[dict[str, Any]] = []
    all_frame_preds: list[list[str]] = []
    successful = 0
    failed = 0

    for i, (clip_path, true_label) in enumerate(
        zip(valid_clips, gt_labels, strict=False), 1
    ):
        print(
            f"\n--- [{i}/{len(valid_clips)}] "
            f"{Path(clip_path).name}  (GT: {true_label})"
        )

        try:
            # Pass clip_index (0-based) so per-clip seeds are consistent
            # regardless of which SLURM chunk this clip lands in.
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
    compute_metrics(y_true, y_pred, class_names, chunk_metrics_dir)

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