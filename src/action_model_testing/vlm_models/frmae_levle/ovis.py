"""Batch child action classifier using the Ovis2 vision-language model.

Classifies two behavioral categories from video frames:
  - Locomotion (5 classes)
  - Repetitive Motor Movements (4 classes)

Randomly samples a fraction of frames per video, controlled by a seed.

Usage:
    python ovis.py --csv /path/to/split.csv --output-dir /path/to/output --seed 123
    python ovis.py --csv /path/to/split.csv --column video_path --sample-rate 0.5 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import traceback
from datetime import timedelta
from pathlib import Path

import cv2
import pandas as pd
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Action category definitions
# ---------------------------------------------------------------------------
ACTION_CATEGORIES: dict[str, list[str]] = {
    "Locomotion": ["Crawling", "Cruising", "Walking", "Running", "Vehicle"],
    "Repetitive_Motor_Movements": [
        "Hands flapping",
        "Jumping",
        "Spinning",
        "Rocking",
    ],
}

_DEFAULT_ACTIONS: dict[str, str] = {
    "Locomotion": "N/A",
    "Repetitive_Motor_Movements": "N/A",
}


# ---------------------------------------------------------------------------
# Monkey-patch helper for the aimv2 registration conflict
# ---------------------------------------------------------------------------
def _patch_aimv2_registration() -> None:
    _original_register = AutoConfig.register

    def _safe_register(model_type: str, config, exist_ok: bool = False) -> None:
        try:
            _original_register(model_type, config, exist_ok=exist_ok)
        except ValueError as exc:
            if "aimv2" in str(exc):
                pass
            else:
                raise

    AutoConfig.register = _safe_register


_patch_aimv2_registration()


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class ChildActionClassifier:
    """Frame-level action classifier backed by the Ovis2 VLM.

    Args:
        model_name: HuggingFace model identifier for Ovis2.
        sample_rate: Fraction of frames to randomly sample (e.g. 0.5 = 50%).
        max_partition: Maximum visual-token partition passed to the model.
        use_flash_attn: If False, force eager attention.
        seed: Random seed controlling which frames are sampled.
    """

    def __init__(
        self,
        model_name: str = "AIDC-AI/Ovis2-8B",
        sample_rate: float = 0.5,
        max_partition: int = 9,
        *,
        use_flash_attn: bool = True,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.sample_rate = sample_rate
        self.max_partition = max_partition

        print(
            f"[INFO] Seed={seed}  |  sampling {sample_rate * 100:.0f}% of frames randomly"
        )

        print(f"Loading model: {model_name} ...")

        load_kwargs: dict = {
            "torch_dtype": torch.bfloat16,
            "multimodal_max_length": 32768,
            "trust_remote_code": True,
        }

        if not use_flash_attn:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config.llm_attn_implementation = "eager"
            if hasattr(config, "llm_config"):
                config.llm_config.attn_implementation = "eager"
                config.llm_config._attn_implementation = "eager"
            load_kwargs["config"] = config
            load_kwargs["attn_implementation"] = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs
        ).cuda()

        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()

        print(f"Model loaded on {self.model.device}.")

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    @staticmethod
    def _build_prompt() -> str:
        locomotion_str = ", ".join(ACTION_CATEGORIES["Locomotion"])
        rmm_str = ", ".join(ACTION_CATEGORIES["Repetitive_Motor_Movements"])

        return (
            "You are analyzing a frame of a child. Identify the child's "
            "actions from the two categories below.\n\n"
            f"Locomotion actions: {locomotion_str}\n"
            f"Repetitive Motor Movement actions: {rmm_str}\n\n"
            "Return your answer in this EXACT format:\n"
            "LOCOMOTION: [action name or N/A]\n"
            "REPETITIVE_MOTOR: [action name or N/A]\n\n"
            "Example (child walking and flapping hands):\n"
            "LOCOMOTION: Walking\n"
            "REPETITIVE_MOTOR: Hands flapping\n\n"
            "Now analyze this frame:"
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_action(response: str, key: str) -> str:
        for line in response.split("\n"):
            if key in line:
                value = line.split(key, maxsplit=1)[1].strip().strip("\"'")
                return value if value and value.lower() != "n/a" else "N/A"
        return "N/A"

    # ------------------------------------------------------------------
    # Single-frame inference
    # ------------------------------------------------------------------
    def analyze_frame(self, image: Image.Image) -> dict[str, str]:
        try:
            prompt = self._build_prompt()
            query = f"<image>\n{prompt}"

            _, input_ids, pixel_values = self.model.preprocess_inputs(
                query, [image], max_partition=self.max_partition
            )

            attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)

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
                    max_new_tokens=50,
                    do_sample=False,
                    top_p=None,
                    top_k=None,
                    temperature=None,
                    repetition_penalty=None,
                    eos_token_id=self.model.generation_config.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                    use_cache=True,
                )[0]

            response = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)

            return {
                "Locomotion": self._extract_action(response, "LOCOMOTION:"),
                "Repetitive_Motor_Movements": self._extract_action(
                    response, "REPETITIVE_MOTOR:"
                ),
            }

        except Exception:
            traceback.print_exc()
            return dict(_DEFAULT_ACTIONS)

    # ------------------------------------------------------------------
    # Full-video inference  (ONE definition only — random sampling)
    # ------------------------------------------------------------------
    def analyze_video(self, video_path: str, output_csv: str) -> pd.DataFrame | None:
        """Randomly sample sample_rate fraction of frames, controlled by seed."""
        if not os.path.exists(video_path):
            print(f"[ERROR] Video not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Isolated RNG — won't disturb any global random state
        n_frames_to_sample = max(1, int(total_frames * self.sample_rate))
        rng = random.Random(self.seed)
        sampled_frames = set(rng.sample(range(total_frames), n_frames_to_sample))

        print(
            f"[INFO] {video_path}  |  {total_frames} total frames @ {fps:.1f} FPS  "
            f"|  seed={self.seed}  |  sampling {n_frames_to_sample} frames "
            f"({self.sample_rate * 100:.0f}%)"
        )

        results: list[dict[str, object]] = []
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx in sampled_frames:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    actions = self.analyze_frame(image)

                    timestamp = str(timedelta(seconds=frame_idx / fps)).split(".")[0]

                    results.append(
                        {
                            "Frame": frame_idx,
                            "Time": timestamp,
                            "Locomotion": actions["Locomotion"],
                            "Repetitive_Motor_Movements": actions[
                                "Repetitive_Motor_Movements"
                            ],
                        }
                    )

                frame_idx += 1
        finally:
            cap.release()

        if not results:
            print("[ERROR] No frames were successfully analysed.")
            return None

        df = pd.DataFrame(results).sort_values("Frame").reset_index(drop=True)
        df.to_csv(output_csv, index=False)
        print(
            f"[SUCCESS] {len(results)} frames → {output_csv}  "
            f"({os.path.getsize(output_csv)} bytes)"
        )
        return df


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def process_csv_videos(
    csv_path: str,
    output_dir: str,
    model_name: str = "AIDC-AI/Ovis2-8B",
    sample_rate: float = 0.5,
    max_partition: int = 9,
    *,
    use_flash_attn: bool = True,
    column_name: str = "video_path",
    seed: int = 42,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"BATCH PROCESSING  (seed={seed})")
    print(f"{'=' * 60}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[ERROR] Failed to load CSV: {exc}")
        return

    if column_name not in df.columns:
        print(
            f"[ERROR] Column '{column_name}' not found. "
            f"Available: {list(df.columns)}"
        )
        return

    video_paths: list[str] = df[column_name].dropna().tolist()
    if not video_paths:
        print(f"[ERROR] No video paths found in column '{column_name}'.")
        return

    print(f"[INFO] {len(video_paths)} videos to process.")
    os.makedirs(output_dir, exist_ok=True)

    classifier = ChildActionClassifier(
        model_name=model_name,
        sample_rate=sample_rate,
        max_partition=max_partition,
        use_flash_attn=use_flash_attn,
        seed=seed,
    )

    successful = 0
    failed = 0

    for i, video_path in enumerate(video_paths, 1):
        print(f"\n--- [{i}/{len(video_paths)}] {video_path}")
        video_name = Path(video_path).stem
        output_csv = os.path.join(
            output_dir, f"{video_name}_actions_seed{seed}.csv"
        )

        if os.path.exists(output_csv):
            print("[SKIP] Already processed.")
            successful += 1
            continue

        try:
            result = classifier.analyze_video(video_path, output_csv)
            if result is not None:
                successful += 1
            else:
                failed += 1
        except Exception:
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(
        f"DONE — {len(video_paths)} total  |  "
        f"{successful} succeeded  |  {failed} failed"
    )
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch video action classifier using the Ovis2 VLM.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="/home/aparnabg/orcd/scratch/latest_split_csv_new.csv",
    )
    parser.add_argument("--column", type=str, default="video_path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "/orcd/scratch/bcs/001/sensein/sails/"
            "action_model_outputs/vlm_models/ovis/frame_level"
        ),
    )
    parser.add_argument("--model", type=str, default="AIDC-AI/Ovis2-8B")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.5,
        help="Fraction of frames to randomly sample, e.g. 0.5 = 50%% (default: 0.5).",
    )
    parser.add_argument("--max-partition", type=int, default=9)
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling which frames are sampled (default: 42).",
    )

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"  CSV          : {args.csv}")
    print(f"  Column       : {args.column}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Model        : {args.model}")
    print(f"  Sample rate  : {args.sample_rate * 100:.0f}% of frames")
    print(f"  Flash attn   : {not args.no_flash_attn}")
    print(f"  Seed         : {args.seed}")

    process_csv_videos(
        csv_path=args.csv,
        output_dir=args.output_dir,
        model_name=args.model,
        sample_rate=args.sample_rate,
        max_partition=args.max_partition,
        use_flash_attn=not args.no_flash_attn,
        column_name=args.column,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()