"""
Batch Child Action Classifier using Qwen2.5-VL Vision-Language Model
Classifies: Locomotion and Repetitive Motor Movements
Randomly samples a fraction of frames per video, controlled by a seed.
"""

import torch
from PIL import Image
import cv2
import os
import random
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
from datetime import timedelta
from pathlib import Path
import traceback

# Action categories
ACTION_CATEGORIES = {
    "Locomotion": ["Crawling", "Cruising", "Walking", "Running", "Vehicle"],
    "Repetitive_Motor_Movements": ["Hands flapping", "Jumping", "Spinning", "Rocking"]
}


class ChildActionClassifier:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        sample_rate: float = 0.5,
        dtype="bfloat16",
        cache_dir=None,
        seed: int = 42,
    ):
        """Initialize the classifier.

        Args:
            model_name: HuggingFace model identifier.
            sample_rate: Fraction of frames to randomly sample (e.g. 0.5 = 50%).
            dtype: Model dtype, 'bfloat16' or 'float16'.
            cache_dir: Optional HuggingFace cache directory.
            seed: Random seed controlling which frames are sampled.
        """
        self.seed = seed
        self.sample_rate = sample_rate

        print(f"Loading model: {model_name}...")
        print(f"[INFO] Seed={seed}  |  sampling {sample_rate * 100:.0f}% of frames randomly")

        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto",
            cache_dir=cache_dir
        )

        self.model.eval()
        print("Model loaded successfully!")
        print(f"Model device: {next(self.model.parameters()).device}")

    def create_prompt(self):
        """Create the prompt for action classification."""
        locomotion_actions = ", ".join(ACTION_CATEGORIES["Locomotion"])
        repetitive_actions = ", ".join(ACTION_CATEGORIES["Repetitive_Motor_Movements"])

        return f"""You are analyzing a frame of a child. Identify the child's actions from the two categories below.

Locomotion actions: {locomotion_actions}
Repetitive Motor Movement actions: {repetitive_actions}

Return your answer in this EXACT format:
LOCOMOTION: [action name or N/A]
REPETITIVE_MOTOR: [action name or N/A]

Example (child walking and flapping hands):
LOCOMOTION: Walking
REPETITIVE_MOTOR: Hands flapping

Now analyze this frame:"""

    def _extract_action(self, response, category_key):
        """Extract action from response for a specific category."""
        try:
            for line in response.split('\n'):
                if category_key in line:
                    action = line.split(category_key)[1].strip().strip('"').strip("'")
                    return action if action and action.lower() != "n/a" else "N/A"
        except Exception as e:
            print(f"[DEBUG] Error extracting {category_key}: {e}")
        return "N/A"

    def analyze_frame(self, image):
        """Analyze a single frame and extract actions."""
        try:
            prompt = self.create_prompt()

            print(f"\n[DEBUG] Processing frame | Image size: {image.size}")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
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

            print(f"[DEBUG] Input IDs shape: {inputs['input_ids'].shape}")

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )

            generated_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs['input_ids'], output_ids)
            ]

            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            del inputs, output_ids, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[DEBUG] Raw response:\n{response}\n")

            actions = {
                "Locomotion": self._extract_action(response, "LOCOMOTION:"),
                "Repetitive_Motor_Movements": self._extract_action(response, "REPETITIVE_MOTOR:"),
            }

            print(f"[DEBUG] Parsed actions: {actions}\n")
            return actions

        except Exception as e:
            print(f"\n[ERROR] Exception in analyze_frame: {e}")
            traceback.print_exc()
            return {"Locomotion": "N/A", "Repetitive_Motor_Movements": "N/A"}

    def analyze_video(self, video_path: str, output_csv: str):
        """Randomly sample sample_rate fraction of frames, controlled by seed."""
        print(f"\n[INFO] Starting video analysis: {video_path}")
        print(f"[INFO] Output CSV: {output_csv}")

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

        print(f"\n[INFO] Video info:")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames / fps:.2f} seconds")
        print(f"  Seed: {self.seed}")
        print(f"  Sampling {n_frames_to_sample} frames ({self.sample_rate * 100:.0f}%)")

        results = []
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"\n[INFO] Reached end of video at frame {frame_idx}")
                    break

                if frame_idx in sampled_frames:
                    print(f"\n[INFO] Analyzing frame {frame_idx}/{total_frames} ...")
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame_rgb)
                        actions = self.analyze_frame(image)

                        timestamp = str(timedelta(seconds=frame_idx / fps)).split('.')[0]

                        results.append({
                            'Frame': frame_idx,
                            'Time': timestamp,
                            'Locomotion': actions["Locomotion"],
                            'Repetitive_Motor_Movements': actions["Repetitive_Motor_Movements"],
                        })
                        print(f"[INFO] Results so far: {len(results)}")

                    except Exception as e:
                        print(f"[ERROR] Failed to analyze frame {frame_idx}: {e}")
                        traceback.print_exc()

                frame_idx += 1
        finally:
            cap.release()

        if not results:
            print("[ERROR] No results generated!")
            return None

        df = pd.DataFrame(results).sort_values("Frame").reset_index(drop=True)
        df.to_csv(output_csv, index=False)

        if os.path.exists(output_csv):
            print(f"[SUCCESS] {len(results)} frames → {output_csv}  ({os.path.getsize(output_csv)} bytes)")
        else:
            print("[ERROR] CSV file was not created!")

        return df


def process_csv_videos(
    csv_path,
    output_dir,
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    sample_rate: float = 0.5,
    dtype="bfloat16",
    cache_dir=None,
    column_name="video_path",
    seed: int = 42,
):
    """Process all videos listed in a CSV file."""
    print(f"\n{'=' * 60}")
    print(f"BATCH PROCESSING  (seed={seed})")
    print(f"{'=' * 60}")
    print(f"CSV path: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model_name}")
    print(f"Sample rate: {sample_rate * 100:.0f}% of frames randomly")
    print(f"Seed: {seed}")

    try:
        df = pd.read_csv(csv_path)
        print(f"\n[INFO] CSV loaded. Shape: {df.shape}")
        print(f"[INFO] Columns: {list(df.columns)}")
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        return

    if column_name not in df.columns:
        print(f"[ERROR] Column '{column_name}' not found. Available: {list(df.columns)}")
        return

    video_paths = df[column_name].dropna().tolist()
    print(f"\n[INFO] Found {len(video_paths)} video paths")

    if not video_paths:
        print(f"[ERROR] No video paths found in column '{column_name}'")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("INITIALIZING MODEL")
    print(f"{'=' * 60}")

    try:
        classifier = ChildActionClassifier(
            model_name=model_name,
            sample_rate=sample_rate,
            dtype=dtype,
            cache_dir=cache_dir,
            seed=seed,
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize classifier: {e}")
        traceback.print_exc()
        return

    successful = 0
    failed = 0

    for i, video_path in enumerate(video_paths, 1):
        print(f"\n{'=' * 60}")
        print(f"PROCESSING VIDEO {i}/{len(video_paths)}: {video_path}")
        print(f"{'=' * 60}")

        try:
            video_name = Path(video_path).stem
            # Output filename includes seed so runs never overwrite each other
            output_csv = os.path.join(output_dir, f"{video_name}_actions_seed{seed}.csv")

            if os.path.exists(output_csv):
                print("[SKIP] Already processed.")
                successful += 1
                continue

            result = classifier.analyze_video(video_path, output_csv)

            if result is not None:
                successful += 1
                print("[SUCCESS] Video processed successfully")
            else:
                failed += 1
                print("[FAILED] Video processing failed")

        except Exception as e:
            print(f"[ERROR] Exception processing video: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total: {len(video_paths)} | Successful: {successful} | Failed: {failed}")
    print(f"Results directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch video classifier using Qwen2.5-VL")
    parser.add_argument("--csv", type=str,
                        default="/home/aparnabg/orcd/scratch/latest_split_csv_new.csv")
    parser.add_argument("--column", type=str, default="video_path")
    parser.add_argument("--output-dir", type=str,
                        default="/orcd/data/satra/002/projects/SAILS/vjepa_features/action_model_outputs/vlm_models/qwen2_5/frame_level")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.5,
        help="Fraction of frames to randomly sample, e.g. 0.5 = 50%% (default: 0.5).",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"])
    parser.add_argument("--cache-dir", type=str, default=None)
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
    print(f"  Dtype        : {args.dtype}")
    print(f"  Cache dir    : {args.cache_dir}")
    print(f"  Seed         : {args.seed}")

    process_csv_videos(
        csv_path=args.csv,
        output_dir=args.output_dir,
        model_name=args.model,
        sample_rate=args.sample_rate,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
        column_name=args.column,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()