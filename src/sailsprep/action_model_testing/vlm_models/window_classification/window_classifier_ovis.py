"""Unified 2-sec window classifier using Ovis2.

Supports three approaches and two tasks:
  - Approach A: Direct multi-class (6-class loco or 5-class RMM).
  - Approach B: 2-stage (binary → fine-grained).
  - Approach C: Binary only.

Uses a temporal grid (all sampled frames tiled into one image) so the
model can observe motion across the clip instead of seeing isolated frames.

Usage:
    python window_classifier_ovis.py --task loco --approach a --csv split.csv
    python window_classifier_ovis.py --task loco --approach a --csv split.csv \
        --random-frames --seed 42
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import traceback
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from PIL import Image

from shared_utils import (
    TASK_CONFIG,
    add_metadata_to_metrics,
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_top2_from_votes,
    frame_labels_to_clip_labels,
    get_processed_videos,
    iterate_videos,
    load_frame_labels,
    sample_frames_from_window,
    save_predictions_csv,
)


def _patch_aimv2_registration() -> None:
    _orig = AutoConfig.register

    def _safe(model_type: str, config, exist_ok: bool = False) -> None:
        try:
            _orig(model_type, config, exist_ok=exist_ok)
        except ValueError as exc:
            if "aimv2" in str(exc):
                pass
            else:
                raise

    AutoConfig.register = _safe


_patch_aimv2_registration()


# ──────────────────────────────────────────────────────────────
# Task-specific prompt definitions
# ──────────────────────────────────────────────────────────────
LOCO_DEFINITIONS = (
    "Category definitions:\n"
    "- Crawling: any motion in any direction while the child is on all "
    "fours, either hands-and-knees or hands-and-feet\n"
    "- Cruising: any motion in any direction while the child is holding "
    "onto an inanimate object (table, chair, toy, wall) for support\n"
    "- Walking: any motion in any direction while the child is in a "
    "bipedal stance, moving at a slow pace, with one foot always on "
    "the ground\n"
    "- Running: any motion in any direction while the child is in a "
    "bipedal stance, moving quickly, with at times both feet off "
    "the ground\n"
    "- Vehicle: any instance of the child using a toy vehicle (scooter, "
    "walking bike, etc.) to move in any direction\n"
    "- No_Locomotion: the child is stationary for the entire clip, "
    "no movement in any direction"
)

RMM_DEFINITIONS = (
    "Category definitions:\n"
    "- Hands_flapping: repetitive movement of the hands or only one "
    "hand at the wrists, either vertically or horizontally\n"
    "- Jumping: repetitive bouncing of the entire body involving "
    "bending of the knees and feet leaving the floor or nearly "
    "leaving the floor\n"
    "- Spinning: repetitive turning of the body in a circular motion\n"
    "- Rocking: repetitive front-to-back or side-to-side movement "
    "of the body\n"
    "- No_RMM: no repetitive motor movements observed"
)

LOCO_BINARY_DESC = (
    "Active movement includes: crawling on all fours in any direction, "
    "cruising while holding onto furniture or a wall, walking or running "
    "on two feet, or riding a toy vehicle. Any motion in any direction "
    "counts as YES. Answer NO only if the child is completely stationary "
    "for the entire clip with no movement in any direction."
)

RMM_BINARY_DESC = (
    "Repetitive motor movements include: hands flapping at the wrists "
    "(one or both hands, vertically or horizontally), jumping or "
    "repetitive bouncing with knees bending and feet leaving or nearly "
    "leaving the floor, spinning the body in a circular motion, or "
    "rocking the body front-to-back or side-to-side. Answer NO only if "
    "no repetitive motor movements are observed."
)

LOCO_FINE_DEFINITIONS = (
    "Category definitions:\n"
    "- Crawling: motion on all fours, hands-and-knees or hands-and-feet\n"
    "- Cruising: moving while holding an inanimate object (table, chair, "
    "toy, wall) for support\n"
    "- Walking: bipedal stance, slow pace, one foot always on the ground\n"
    "- Running: bipedal stance, moving quickly, at times both feet off "
    "the ground\n"
    "- Vehicle: child using a toy vehicle (scooter, walking bike) to move"
)

RMM_FINE_DEFINITIONS = (
    "Category definitions:\n"
    "- Hands_flapping: repetitive movement of the hands or one hand at "
    "the wrists, vertically or horizontally\n"
    "- Jumping: repetitive bouncing, knees bending, feet leaving or "
    "nearly leaving the floor\n"
    "- Spinning: repetitive turning of the body in a circular motion\n"
    "- Rocking: repetitive front-to-back or side-to-side movement of "
    "the body"
)

GRID_PREAMBLE = (
    "This image is a 3×2 grid of 6 video frames sampled from a "
    "2-second clip of a young child. The frames read left-to-right, "
    "top-to-bottom in chronological order. Compare the child's body "
    "position, limb placement, and surroundings across all 6 frames "
    "to determine what movement is happening."
)


# ---------------------------------------------------------------------------
# Ovis2 model wrapper
# ---------------------------------------------------------------------------
class OvisClassifier:
    def __init__(
        self,
        task: str,
        num_frames: int = 6,
        model_name: str = "AIDC-AI/Ovis2-8B",
        max_partition: int = 9,
        *,
        use_flash_attn: bool = True,
        random_frames: bool = False,
        seed: int = 42,
    ) -> None:
        self.task = task
        self.cfg = TASK_CONFIG[task]
        self.active_classes = self.cfg["active_classes"]
        self.all_classes = self.cfg["all_classes"]
        self.no_label = self.cfg["no_action_label"]
        self.binary_pos = self.cfg["binary_positive"]
        self.num_frames = num_frames
        self.max_partition = max_partition
        self.random_frames = random_frames
        self.seed = seed

        print(f"Loading Ovis2: {model_name} ...")
        print(
            f"Frame sampling: "
            f"{'random (seed=' + str(seed) + ')' if random_frames else 'uniform (linspace)'}"
        )

        # Match the working ovis_clip_classifier.py load pattern exactly.
        load_kwargs: dict = {
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

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs
        ).cuda()

        self.text_tok = self.model.get_text_tokenizer()
        self.vis_tok = self.model.get_visual_tokenizer()
        self.max_partition = max_partition

        print(f"Ovis2 loaded on {self.model.device}.")

    # ------------------------------------------------------------------
    # Low-level call — uses the SAME decode pattern as the working
    # ovis_clip_classifier.py: generate()[0] + decode full sequence
    # ------------------------------------------------------------------
    def _call(self, image: Image.Image, prompt: str) -> str:
        """Send a single image + prompt to Ovis2 and return the text."""
        query = f"<image>\n{prompt}"
        _, input_ids, pixel_values = self.model.preprocess_inputs(
            query, [image], max_partition=self.max_partition
        )

        attention_mask = torch.ne(input_ids, self.text_tok.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(
            device=self.model.device
        )

        if pixel_values is not None:
            pixel_values = pixel_values.to(
                dtype=self.vis_tok.dtype,
                device=self.vis_tok.device,
            )
            pixel_values = [pixel_values]

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=64,
                do_sample=False,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tok.pad_token_id,
                use_cache=True,
            )[0]  # [0] to get 1-D tensor — exactly like the working code

        # Decode full sequence; skip_special_tokens strips prompt tokens
        resp = self.text_tok.decode(
            output_ids, skip_special_tokens=True
        ).strip()

        if not resp:
            print(
                f"  [WARN] Empty response. "
                f"output_ids shape: {output_ids.shape}"
            )

        del output_ids, input_ids, attention_mask, pixel_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return resp

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_frames(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        clip_index: int,
    ) -> list[Image.Image]:
        return sample_frames_from_window(
            video_path,
            start_sec,
            end_sec,
            num_frames=self.num_frames,
            random_frames=self.random_frames,
            seed=self.seed,
            clip_index=clip_index,
        )

    def _make_grid(
        self,
        images: list[Image.Image],
        cols: int = 3,
    ) -> Image.Image:
        """Tile a list of PIL images into a single grid image."""
        if not images:
            raise ValueError("No images to grid")
        if len(images) == 1:
            return images[0]

        rows = math.ceil(len(images) / cols)
        w, h = images[0].size
        grid = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
        for i, img in enumerate(images):
            r, c = divmod(i, cols)
            if img.size != (w, h):
                img = img.resize((w, h))
            grid.paste(img, (c * w, r * h))
        return grid

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------
    def _multiclass_prompt(self) -> str:
        definitions = (
            LOCO_DEFINITIONS if self.task == "loco" else RMM_DEFINITIONS
        )
        classes_str = ", ".join(self.all_classes)
        return (
            f"{GRID_PREAMBLE}\n\n"
            f"{definitions}\n\n"
            f"Possible movements: {classes_str}\n\n"
            f"If none of the active categories match, answer "
            f"{self.no_label}.\n\n"
            "Respond with exactly one line:\n"
            "ACTION: <category name>"
        )

    def _binary_prompt(self) -> str:
        description = (
            LOCO_BINARY_DESC if self.task == "loco" else RMM_BINARY_DESC
        )
        return (
            f"{GRID_PREAMBLE}\n\n"
            f"{description}\n\n"
            "Respond with exactly one line:\n"
            "ANSWER: YES or ANSWER: NO"
        )

    def _finegrained_prompt(self) -> str:
        definitions = (
            LOCO_FINE_DEFINITIONS if self.task == "loco"
            else RMM_FINE_DEFINITIONS
        )
        classes_str = ", ".join(self.active_classes)
        return (
            "This image is a 3×2 grid of 6 video frames from a "
            "2-second clip of a young child who IS performing an active "
            "movement. The frames read left-to-right, top-to-bottom in "
            "chronological order.\n\n"
            f"{definitions}\n\n"
            f"Possible movements: {classes_str}\n\n"
            "Respond with exactly one line:\n"
            "ACTION: <category name>"
        )

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------
    def _parse_multiclass(self, resp: str) -> str | None:
        if not resp:
            return None

        upper = resp.upper()
        if "ACTION:" in upper:
            after = (
                resp[upper.find("ACTION:") + 7:]
                .strip().split("\n")[0].strip().strip("\"'")
            )
            for cls in self.all_classes:
                if cls.lower() == after.lower():
                    return cls
            for cls in self.all_classes:
                if cls.replace("_", " ").lower() == after.lower():
                    return cls

        for cls in self.active_classes:
            if cls.lower() in resp.lower():
                return cls

        no_variants = [
            self.no_label.lower(),
            self.no_label.replace("_", " ").lower(),
        ]
        if any(v in resp.lower() for v in no_variants):
            return self.no_label

        return None

    def _parse_binary(self, resp: str) -> bool | None:
        if not resp:
            return None

        upper = resp.upper()
        if "ANSWER:" in upper:
            after = upper.split("ANSWER:")[-1].strip().split()[0]
            if after.startswith("YES"):
                return True
            if after.startswith("NO"):
                return False

        stripped = upper.strip().rstrip(".")
        if stripped == "YES":
            return True
        if stripped == "NO":
            return False

        return None

    def _parse_finegrained(self, resp: str) -> str | None:
        if not resp:
            return None

        upper = resp.upper()
        if "ACTION:" in upper:
            after = (
                resp[upper.find("ACTION:") + 7:]
                .strip().split("\n")[0].strip().strip("\"'")
            )
            for cls in self.active_classes:
                if cls.lower() == after.lower():
                    return cls
            for cls in self.active_classes:
                if cls.replace("_", " ").lower() == after.lower():
                    return cls
            if self.task == "rmm" and "flap" in after.lower():
                return "Hands_flapping"

        for cls in self.active_classes:
            if cls.lower() in resp.lower():
                return cls

        if self.task == "rmm" and "flap" in resp.lower():
            return "Hands_flapping"

        return None

    # ------------------------------------------------------------------
    # Approach A — multi-class via temporal grid
    # ------------------------------------------------------------------
    def classify_multiclass(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        clip_index: int,
    ) -> tuple[str, list[str], float]:
        images = self._get_frames(video_path, start_sec, end_sec, clip_index)
        if not images:
            return self.no_label, [], 0.0

        grid = self._make_grid(images)
        print(
            f"  [DEBUG] clip={clip_index} grid_size={grid.size} "
            f"num_frames={len(images)}"
        )
        prompt = self._multiclass_prompt()
        try:
            raw = self._call(grid, prompt)
            print(f"  [DEBUG] clip={clip_index} raw={raw!r}")
            pred = self._parse_multiclass(raw)
            if pred:
                return pred, [pred], 1.0
        except Exception:
            traceback.print_exc()

        return self.no_label, [], 0.0

    # ------------------------------------------------------------------
    # Approach B — 2-stage via temporal grid
    # ------------------------------------------------------------------
    def classify_2stage(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        clip_index: int,
    ) -> tuple[str, float, str, list[str], str]:
        images = self._get_frames(video_path, start_sec, end_sec, clip_index)
        if not images:
            return self.no_label, 0.0, self.no_label, [], self.no_label

        grid = self._make_grid(images)

        # Stage 1: binary
        try:
            raw_bin = self._call(grid, self._binary_prompt())
            print(f"  [DEBUG] clip={clip_index} binary_raw={raw_bin!r}")
            v = self._parse_binary(raw_bin)
        except Exception:
            traceback.print_exc()
            v = None

        if v is None or v is False:
            return (
                self.no_label, 0.0, self.no_label,
                [self.no_label], self.no_label,
            )

        # Stage 2: fine-grained (reuse same grid)
        try:
            raw_fine = self._call(grid, self._finegrained_prompt())
            print(f"  [DEBUG] clip={clip_index} fine_raw={raw_fine!r}")
            pred = self._parse_finegrained(raw_fine)
        except Exception:
            traceback.print_exc()
            pred = None

        if pred is None:
            fallback = self.active_classes[0]
            return self.binary_pos, 1.0, fallback, [fallback], fallback

        return self.binary_pos, 1.0, pred, [pred], pred

    # ------------------------------------------------------------------
    # Approach C — binary via temporal grid
    # ------------------------------------------------------------------
    def classify_binary(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        clip_index: int,
    ) -> tuple[str, float, list[bool]]:
        images = self._get_frames(video_path, start_sec, end_sec, clip_index)
        if not images:
            return self.no_label, 0.0, []

        grid = self._make_grid(images)
        prompt = self._binary_prompt()
        try:
            raw = self._call(grid, prompt)
            print(f"  [DEBUG] clip={clip_index} raw={raw!r}")
            v = self._parse_binary(raw)
            if v is not None:
                label = self.binary_pos if v else self.no_label
                conf = 1.0 if v else 0.0
                return label, conf, [v]
        except Exception:
            traceback.print_exc()

        return self.no_label, 0.0, []


# ---------------------------------------------------------------------------
# Approach runners
# ---------------------------------------------------------------------------
def _run_approach_a(
    classifier: OvisClassifier,
    video_pairs: list[tuple[str, str]],
    task: str,
    num_frames: int,
    output_dir: str,
    model_name: str,
) -> None:
    cfg = TASK_CONFIG[task]
    processed = get_processed_videos(output_dir, prefix="multiclass_")
    all_results: list[dict] = []
    all_fpreds: list[list[str]] = []
    global_clip_index = 0

    for vid_path, lab_path in video_pairs:
        if vid_path in processed:
            print(f"[SKIP] {Path(vid_path).name}")
            continue
        print(f"\n{'─' * 60}\nVideo: {Path(vid_path).name}")
        try:
            labels = load_frame_labels(lab_path, task)
            clips = frame_labels_to_clip_labels(labels, task)
            print(f"  {len(clips)} clips from {len(labels)} frames")

            for ci, clip in enumerate(clips):
                pred, fpreds, conf = classifier.classify_multiclass(
                    vid_path, clip["start_sec"], clip["end_sec"],
                    clip_index=global_clip_index,
                )
                global_clip_index += 1

                all_results.append({
                    "video_path": vid_path,
                    "clip_index": ci,
                    "start_sec": clip["start_sec"],
                    "end_sec": clip["end_sec"],
                    "true_label": clip["label_full"],
                    "predicted_label": pred,
                    "confidence": conf,
                    "correct": pred == clip["label_full"],
                    "frame_predictions": str(fpreds),
                    "gt_frame_counts": str(clip["frame_label_counts"]),
                    "seed": classifier.seed if classifier.random_frames else "N/A",
                    "random_frames": classifier.random_frames,
                })
                all_fpreds.append(fpreds)

                if (ci + 1) % 20 == 0:
                    print(
                        f"    clip {ci + 1}/{len(clips)}  "
                        f"GT={clip['label_full']}  pred={pred}"
                    )
        except Exception:
            traceback.print_exc()

    if not all_results:
        print("[WARN] No new results to save.")
        return

    os.makedirs(output_dir, exist_ok=True)
    save_predictions_csv(all_results, output_dir, prefix="multiclass_")

    y_true = [r["true_label"] for r in all_results]
    y_pred = [r["predicted_label"] for r in all_results]
    valid = [
        (t, p, fp) for t, p, fp in zip(y_true, y_pred, all_fpreds)
        if p in cfg["all_classes"]
    ]
    if not valid:
        print("[ERROR] No valid predictions.")
        return

    vt, vp, vfp = zip(*valid)
    prefix_full = f"{len(cfg['all_classes'])}class_"
    compute_multiclass_metrics(
        list(vt), list(vp), cfg["all_classes"], output_dir, prefix=prefix_full
    )
    top2 = compute_top2_from_votes(list(vfp), list(vt))
    print(f"  Top-2 accuracy: {top2:.4f}")
    add_metadata_to_metrics(
        os.path.join(output_dir, f"{prefix_full}evaluation_metrics.json"),
        top2_accuracy=top2,
        model=model_name,
        num_frames_per_clip=num_frames,
        total_clips=len(all_results),
        valid_clips=len(valid),
        random_frames=classifier.random_frames,
        seed=classifier.seed if classifier.random_frames else None,
    )

    active = cfg["active_classes"]
    loco_valid = [
        (t, p, fp) for t, p, fp in zip(vt, vp, vfp)
        if t in active and p in active
    ]
    if loco_valid:
        lt, lp, lfp = zip(*loco_valid)
        prefix_fine = f"{len(active)}class_active_only_"
        compute_multiclass_metrics(
            list(lt), list(lp), active, output_dir, prefix=prefix_fine
        )
        top2_fine = compute_top2_from_votes(list(lfp), list(lt))
        add_metadata_to_metrics(
            os.path.join(output_dir, f"{prefix_fine}evaluation_metrics.json"),
            top2_accuracy=top2_fine,
        )


def _run_approach_b(
    classifier: OvisClassifier,
    video_pairs: list[tuple[str, str]],
    task: str,
    num_frames: int,
    output_dir: str,
    model_name: str,
) -> None:
    cfg = TASK_CONFIG[task]
    processed = get_processed_videos(output_dir, prefix="2stage_")
    all_results: list[dict] = []
    all_fpreds: list[list[str]] = []
    global_clip_index = 0

    for vid_path, lab_path in video_pairs:
        if vid_path in processed:
            print(f"[SKIP] {Path(vid_path).name}")
            continue
        print(f"\n{'─' * 60}\nVideo: {Path(vid_path).name}")
        try:
            labels = load_frame_labels(lab_path, task)
            clips = frame_labels_to_clip_labels(labels, task)
            print(f"  {len(clips)} clips")

            for ci, clip in enumerate(clips):
                (bin_pred, conf, fine_pred,
                 fine_fpreds, combined) = classifier.classify_2stage(
                    vid_path, clip["start_sec"], clip["end_sec"],
                    clip_index=global_clip_index,
                )
                global_clip_index += 1

                all_results.append({
                    "video_path": vid_path,
                    "clip_index": ci,
                    "start_sec": clip["start_sec"],
                    "end_sec": clip["end_sec"],
                    "true_full": clip["label_full"],
                    "true_binary": clip["label_binary"],
                    "true_fine": clip["label_fine"],
                    "pred_binary": bin_pred,
                    "confidence": conf,
                    "pred_fine": fine_pred,
                    "pred_combined": combined,
                    "correct_binary": bin_pred == clip["label_binary"],
                    "correct_full": combined == clip["label_full"],
                    "fine_frame_predictions": str(fine_fpreds),
                    "gt_frame_counts": str(clip["frame_label_counts"]),
                    "seed": classifier.seed if classifier.random_frames else "N/A",
                    "random_frames": classifier.random_frames,
                })
                all_fpreds.append(fine_fpreds)

                if (ci + 1) % 20 == 0:
                    print(
                        f"    clip {ci + 1}/{len(clips)}  "
                        f"GT={clip['label_full']}  pred={combined}"
                    )
        except Exception:
            traceback.print_exc()

    if not all_results:
        print("[WARN] No new results to save.")
        return

    os.makedirs(output_dir, exist_ok=True)
    save_predictions_csv(all_results, output_dir, prefix="2stage_")

    y_true_bin = [
        1 if r["true_binary"] == cfg["binary_positive"] else 0
        for r in all_results
    ]
    y_pred_bin = [
        1 if r["pred_binary"] == cfg["binary_positive"] else 0
        for r in all_results
    ]
    y_scores = [r["confidence"] for r in all_results]
    compute_binary_metrics(
        y_true_bin, y_pred_bin, y_scores, output_dir,
        prefix="stage1_binary_",
    )

    y_true_f = [r["true_full"] for r in all_results]
    y_pred_f = [r["pred_combined"] for r in all_results]
    valid_f = [
        (t, p, fp) for t, p, fp in zip(y_true_f, y_pred_f, all_fpreds)
        if p in cfg["all_classes"]
    ]
    if valid_f:
        vt, vp, vfp = zip(*valid_f)
        prefix_comb = f"combined_{len(cfg['all_classes'])}class_"
        compute_multiclass_metrics(
            list(vt), list(vp), cfg["all_classes"], output_dir,
            prefix=prefix_comb,
        )
        top2 = compute_top2_from_votes(list(vfp), list(vt))
        add_metadata_to_metrics(
            os.path.join(
                output_dir, f"{prefix_comb}evaluation_metrics.json"
            ),
            top2_accuracy=top2,
            model=model_name,
            random_frames=classifier.random_frames,
            seed=classifier.seed if classifier.random_frames else None,
        )

    active = cfg["active_classes"]
    fine_results = [
        r for r in all_results
        if r["true_fine"] is not None and r["pred_fine"] in active
    ]
    if fine_results:
        lt = [r["true_fine"] for r in fine_results]
        lp = [r["pred_fine"] for r in fine_results]
        lfp = [
            ast.literal_eval(r["fine_frame_predictions"])
            for r in fine_results
        ]
        prefix_fine = f"stage2_{len(active)}class_"
        compute_multiclass_metrics(
            lt, lp, active, output_dir, prefix=prefix_fine
        )
        top2_fine = compute_top2_from_votes(lfp, lt)
        add_metadata_to_metrics(
            os.path.join(
                output_dir, f"{prefix_fine}evaluation_metrics.json"
            ),
            top2_accuracy=top2_fine,
        )


def _run_approach_c(
    classifier: OvisClassifier,
    video_pairs: list[tuple[str, str]],
    task: str,
    num_frames: int,
    output_dir: str,
    model_name: str,
) -> None:
    cfg = TASK_CONFIG[task]
    processed = get_processed_videos(output_dir, prefix="binary_")
    all_results: list[dict] = []
    global_clip_index = 0

    for vid_path, lab_path in video_pairs:
        if vid_path in processed:
            print(f"[SKIP] {Path(vid_path).name}")
            continue
        print(f"\n{'─' * 60}\nVideo: {Path(vid_path).name}")
        try:
            labels = load_frame_labels(lab_path, task)
            clips = frame_labels_to_clip_labels(labels, task)
            print(f"  {len(clips)} clips")

            for ci, clip in enumerate(clips):
                pred, conf, votes = classifier.classify_binary(
                    vid_path, clip["start_sec"], clip["end_sec"],
                    clip_index=global_clip_index,
                )
                global_clip_index += 1

                all_results.append({
                    "video_path": vid_path,
                    "clip_index": ci,
                    "start_sec": clip["start_sec"],
                    "end_sec": clip["end_sec"],
                    "true_binary": clip["label_binary"],
                    "pred_binary": pred,
                    "confidence": conf,
                    "correct": pred == clip["label_binary"],
                    "frame_votes": str(votes),
                    "num_frames_voted": len(votes),
                    "gt_frame_counts": str(clip["frame_label_counts"]),
                    "seed": classifier.seed if classifier.random_frames else "N/A",
                    "random_frames": classifier.random_frames,
                })

                if (ci + 1) % 20 == 0:
                    print(
                        f"    clip {ci + 1}/{len(clips)}  "
                        f"GT={clip['label_binary']}  pred={pred}  "
                        f"conf={conf:.2f}"
                    )
        except Exception:
            traceback.print_exc()

    if not all_results:
        print("[WARN] No new results to save.")
        return

    os.makedirs(output_dir, exist_ok=True)
    save_predictions_csv(all_results, output_dir, prefix="binary_")

    y_true_bin = [
        1 if r["true_binary"] == cfg["binary_positive"] else 0
        for r in all_results
    ]
    y_pred_bin = [
        1 if r["pred_binary"] == cfg["binary_positive"] else 0
        for r in all_results
    ]
    y_scores = [r["confidence"] for r in all_results]
    compute_binary_metrics(
        y_true_bin, y_pred_bin, y_scores, output_dir, prefix="binary_"
    )
    add_metadata_to_metrics(
        os.path.join(output_dir, "binary_evaluation_metrics.json"),
        model=model_name,
        num_frames_per_clip=num_frames,
        total_clips=len(all_results),
        random_frames=classifier.random_frames,
        seed=classifier.seed if classifier.random_frames else None,
    )


_APPROACH_RUNNERS = {
    "a": _run_approach_a,
    "b": _run_approach_b,
    "c": _run_approach_c,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified 2-sec window classifier (Ovis2).",
    )
    parser.add_argument("--task", required=True, choices=["loco", "rmm"])
    parser.add_argument(
        "--approach", required=True, choices=["a", "b", "c"],
        help="a=multi-class, b=2-stage, c=binary.",
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video-col", default="video_path")
    parser.add_argument("--label-col", default="label_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="AIDC-AI/Ovis2-8B")
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--max-partition", type=int, default=9)
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument(
        "--random-frames", action="store_true",
        help="Sample frames randomly instead of uniformly.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base RNG seed (only used with --random-frames).",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"Ovis2 — task={args.task}  approach={args.approach}")
    print(f"{'=' * 60}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    classifier = OvisClassifier(
        task=args.task,
        num_frames=args.num_frames,
        model_name=args.model,
        max_partition=args.max_partition,
        use_flash_attn=not args.no_flash_attn,
        random_frames=args.random_frames,
        seed=args.seed,
    )

    pairs = iterate_videos(args.csv, args.video_col, args.label_col)
    runner = _APPROACH_RUNNERS[args.approach]
    runner(
        classifier, pairs, args.task, args.num_frames,
        args.output_dir, args.model,
    )

    print(f"\n{'=' * 60}")
    print(f"DONE — results in {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()