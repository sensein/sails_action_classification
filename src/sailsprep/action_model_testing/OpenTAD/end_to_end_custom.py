"""Custom transforms for  dataset"""

from __future__ import annotations

import json
import os
import random
from typing import Optional

import numpy as np
import torch

from ..builder import PIPELINES


@PIPELINES.register_module()
class PrepareVideoInfoAbsPath:
    """Use a pre-built JSON path map instead of constructing paths from data_path.

    The JSON map has the form::

        { "video_id": "/absolute/path/to/video.mp4" }

    so that videos can live anywhere on disk — not just under data_path.
    Videos are opened read-only by the downstream frame decoder.
    Nothing is written, copied, or modified.

    Args:
        video_path_map: Path to the JSON file mapping video_id to absolute path.
        modality: Always ``"RGB"`` for our case.
    """

    def __init__(
        self,
        video_path_map: str,
        modality: str = "RGB",
    ) -> None:
        assert os.path.exists(video_path_map), (
            f"video_path_map not found: {video_path_map}\n"
            "Run: python run_e2e.py --mode build_path_map --split_csv <csv> first."
        )
        with open(video_path_map) as fh:
            self.path_map: dict[str, str] = json.load(fh)
        self.modality = modality

    def __call__(self, results: dict[str, object]) -> dict[str, object]:
        """Inject the absolute video path into the results dict.

        Args:
            results: Pipeline results dict containing ``"video_name"``.

        Returns:
            Updated results dict with ``"filename"`` and ``"modality"`` set.
        """
        video_name = str(results["video_name"])
        assert video_name in self.path_map, (
            f"video_id '{video_name}' not found in video_path_map. "
            "Re-run build_path_map if you added new videos."
        )
        # Set filename to the absolute path — read-only, no modification.
        results["filename"] = self.path_map[video_name]
        results["modality"] = self.modality
        return results


@PIPELINES.register_module()
class LoadFramesAt15fps:
    """Sample frames from a 30fps video at effective 15fps.

    Uses ``frame_interval=2`` (every other frame) to match the
    label/feature FPS. Works with mmaction2's RawFrameDecode pipeline
    downstream. Computes frame indices only — actual pixel loading is
    done by mmaction2's SampleFrames/RawFrameDecode which opens the file
    read-only via decord/cv2.

    Videos are NEVER modified. Frame indices are computed in RAM.

    Args:
        clip_len: Frames per clip fed to backbone (e.g. ``16``).
        method: ``"random_trunc"`` for train, ``"sliding_window"`` for
            val/test.
        trunc_len: Number of snippets to sample during ``random_trunc``.
        trunc_thresh: Minimum overlap ratio for a segment to be kept
            after truncation.
        crop_ratio: ``[min, max]`` fraction for random crop if the video
            is shorter than ``trunc_len``.
        source_fps: Native video FPS (``30.0`` for your videos).
        target_fps: Desired decode FPS to match labels (``15.0``).
    """

    def __init__(
        self,
        clip_len: int,
        method: str = "random_trunc",
        trunc_len: Optional[int] = None,
        trunc_thresh: Optional[float] = None,
        crop_ratio: Optional[list[float]] = None,
        source_fps: float = 30.0,
        target_fps: float = 15.0,
    ) -> None:
        self.clip_len = clip_len
        self.method = method
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.source_fps = source_fps
        self.target_fps = target_fps

        # Every Nth frame from source gives target_fps.
        # e.g. source=30, target=15 → stride=2 (take every 2nd frame).
        assert source_fps % target_fps == 0, (
            f"source_fps ({source_fps}) must be divisible by "
            f"target_fps ({target_fps})"
        )
        self.frame_stride: int = int(source_fps / target_fps)  # = 2 for 30→15fps

    def _random_trunc(
        self,
        frame_idxs: np.ndarray,
        trunc_len: int,
        gt_segments: np.ndarray,
        gt_labels: np.ndarray,
        offset: int = 0,
        max_num_trials: int = 200,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Randomly truncate ``frame_idxs`` to ``trunc_len`` snippets.

        Args:
            frame_idxs: Array of source-frame indices at target FPS.
            trunc_len: Number of snippets in the output window.
            gt_segments: Ground-truth segment boundaries, shape ``(N, 2)``.
            gt_labels: Ground-truth labels, shape ``(N,)``.
            offset: Boundary tolerance when computing segment overlap.
            max_num_trials: Maximum random window attempts before giving up.

        Returns:
            Tuple of ``(frame_idxs, gt_segments, gt_labels)`` after
            truncation and filtering.
        """
        feat_len = len(frame_idxs)
        num_segs = gt_segments.shape[0]

        if feat_len <= trunc_len:
            if self.crop_ratio is None:
                return frame_idxs, gt_segments, gt_labels
            trunc_len = random.randint(
                max(round(self.crop_ratio[0] * feat_len), 1),
                min(round(self.crop_ratio[1] * feat_len), feat_len),
            )
            if feat_len == trunc_len:
                return frame_idxs, gt_segments, gt_labels

        st = 0
        seg_idx = np.zeros(num_segs, dtype=bool)
        left = gt_segments[:, 0].copy()
        right = gt_segments[:, 1].copy()

        for _ in range(max_num_trials):
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            window = np.array([st, ed], dtype=np.float32)
            window = np.repeat(window[None, :], num_segs, axis=0)
            left = np.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = np.minimum(window[:, 1] + offset, gt_segments[:, 1])
            inter = np.clip(right - left, a_min=0, a_max=None)
            area = np.abs(gt_segments[:, 1] - gt_segments[:, 0])
            ratio = inter / (area + 1e-6)
            seg_idx = ratio >= self.trunc_thresh
            if seg_idx.sum() > 0:
                break

        frame_idxs = frame_idxs[st : st + trunc_len]
        gt_segments = np.stack((left[seg_idx], right[seg_idx]), axis=1)
        gt_segments = gt_segments - st
        gt_labels = gt_labels[seg_idx]
        return frame_idxs, gt_segments, gt_labels

    def __call__(self, results: dict[str, object]) -> dict[str, object]:
        """Compute frame indices at target FPS and populate the results dict.

        Args:
            results: Pipeline results dict. Must contain ``"total_frames"``.
                For ``"random_trunc"`` also needs ``"gt_segments"`` and
                ``"gt_labels"``. For ``"sliding_window"`` also needs
                ``"window_size"``, ``"feature_start_idx"``, and
                ``"feature_end_idx"``.

        Returns:
            Updated results dict with ``"frame_inds"``, ``"num_clips"``,
            ``"clip_len"``, ``"masks"``, and ``"effective_fps"`` set.

        Raises:
            AssertionError: If ``"total_frames"`` is missing from results.
            ValueError: If ``method`` is not ``"random_trunc"`` or
                ``"sliding_window"``.
        """
        assert "total_frames" in results, "total_frames must be in results"
        total_frames = int(results["total_frames"])  # type: ignore[arg-type]

        # Build frame index array at target_fps (15fps) by striding source frames.
        # e.g. [0, 2, 4, 6, ...] for 30→15fps.
        # This is pure index arithmetic — no pixel data touched yet.
        frame_idxs_15fps = np.arange(0, total_frames, self.frame_stride)
        num_snippets_15fps = len(frame_idxs_15fps)

        masks: torch.Tensor

        if self.method == "random_trunc":
            assert self.trunc_len is not None
            snippet_num = self.trunc_len

            frame_idxs_15fps, gt_segments, gt_labels = self._random_trunc(
                frame_idxs_15fps,
                trunc_len=snippet_num,
                gt_segments=results["gt_segments"],  # type: ignore[arg-type]
                gt_labels=results["gt_labels"],  # type: ignore[arg-type]
            )
            results["gt_segments"] = gt_segments
            results["gt_labels"] = gt_labels

            # Pad if video is shorter than trunc_len.
            if len(frame_idxs_15fps) < snippet_num:
                valid_len = len(frame_idxs_15fps)
                frame_idxs_15fps = np.pad(
                    frame_idxs_15fps,
                    (0, snippet_num - valid_len),
                    mode="edge",
                )
                masks = torch.cat(
                    [
                        torch.ones(valid_len),
                        torch.zeros(snippet_num - valid_len),
                    ]
                ).bool()
            else:
                masks = torch.ones(snippet_num).bool()

        elif self.method == "sliding_window":
            snippet_num = int(results["window_size"])  # type: ignore[arg-type]
            start_idx = min(
                int(results["feature_start_idx"]), num_snippets_15fps  # type: ignore[arg-type]
            )
            end_idx = min(
                int(results["feature_end_idx"]) + 1, num_snippets_15fps  # type: ignore[arg-type]
            )
            frame_idxs_15fps = frame_idxs_15fps[start_idx:end_idx]

            if len(frame_idxs_15fps) < snippet_num:
                valid_len = len(frame_idxs_15fps)
                frame_idxs_15fps = np.pad(
                    frame_idxs_15fps,
                    (0, snippet_num - valid_len),
                    mode="edge",
                )
                masks = torch.cat(
                    [
                        torch.ones(valid_len),
                        torch.zeros(snippet_num - valid_len),
                    ]
                ).bool()
            else:
                masks = torch.ones(snippet_num).bool()

        else:
            raise ValueError(
                f"LoadFramesAt15fps: unsupported method '{self.method}'. "
                "Use 'random_trunc' (train) or 'sliding_window' (val/test)."
            )

        # Clip to valid frame range and convert to int.
        # These are indices into the SOURCE video at 30fps.
        frame_idxs_15fps = (
            np.clip(frame_idxs_15fps, 0, total_frames - 1).round().astype(int)
        )

        # Results expected by mmaction2 downstream decoders (read-only decode).
        results["frame_inds"] = frame_idxs_15fps  # which frames to decode
        results["num_clips"] = 1
        results["clip_len"] = len(frame_idxs_15fps)
        results["masks"] = masks
        results["effective_fps"] = self.target_fps  # 15fps, for reference
        return results