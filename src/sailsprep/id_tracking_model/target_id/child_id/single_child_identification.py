"""
Single-child identification scaffold.

Goal: Trust per-ID temporal continuity but allow merging across IDs using
post-tracking evidence (age, skeleton ratios, quality) with a simple
graph-based selection. This file provides minimal structures and placeholders
so we can implement features incrementally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Optional dependency for age estimation
try:  # DeepFace is optional; we guard all calls
    from deepface import DeepFace
except Exception:  # pragma: no cover - environment may not have DeepFace
    DeepFace = None

# Optional dependency for SigLIP age estimation
try:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, SiglipForImageClassification
    SIGLIP_AVAILABLE = True
except Exception:  # pragma: no cover - environment may not have SigLIP
    SIGLIP_AVAILABLE = False

# Optional dependency for video frame loading
try:
    import cv2
except Exception:
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore[assignment]


# --------------------------- Config & Inputs ---------------------------


@dataclass
class ChildIdentificationConfig:
    """Configuration knobs for child identification (minimal scaffold)."""

    # Sampling / evidence thresholds
    sampling_percentage: float = 0.25
    sampling_max_frames_per_track: int = 40
    min_track_frames: int = 30
    keypoint_conf_threshold: float = 0.3
    face_conf_threshold: float = 0.8

    # Smart sampling configuration
    sampling_mode: str = "even"  # "even" or "smart"
    min_pose_confidence: float = 0.7  # Minimum pose confidence for smart sampling

    # Age estimation method
    age_estimation_method: str = "siglip"  # "deepface" or "siglip"
    siglip_model_name: str = "prithivMLmods/Age-Classification-SigLIP2"

    # Body visibility filtering
    enable_body_visibility_filter: bool = True  # Enable filtering based on visible keypoints
    min_visible_keypoints: int = 4
    body_keypoint_indices: list[int] | None = None  # Default: [0,1,2,3,4,5,6,7,8,9]
    enable_roi_size_filter: bool = False  # Enable filtering based on ROI size
    min_roi_width: int = 30
    min_roi_height: int = 50

    # Age mapping
    age_tau: float = 2.5
    age_child_years_threshold: float = 10.0

    # Weights
    w_age_default: float = 0.65
    w_skel_default: float = 0.35

    # Skeleton ratio configuration
    enable_skeleton_ratios: bool = True  # Enable skeleton-based child detection
    skeleton_min_confidence: float = 0.3  # Minimum keypoint confidence
    skeleton_min_visible_for_ratio: int = 2  # Min keypoints visible for each ratio

    # Continuity / merging
    continuity_gap_seconds: float = 1.0
    intra_id_gamma: float = 0.2  # base same-ID bonus scale
    intra_id_tau: float = 0.75   # decay for same-ID bonus by gap (seconds)
    switch_epsilon: float = 0.05 # minimal gain to switch near boundaries

    # Edge penalties
    age_inconsistency_penalty_weight: float = 2.0  # Higher = stricter penalty
    age_inconsistency_threshold: float = 0.3  # Min score difference to trigger penalty

    # Rigidity detection (detect static pictures vs real people)
    enable_rigidity_detection: bool = True  # Detect static pictures via keypoint rigidity
    rigidity_weight: float = 0.2  # Weight for rigidity signal in scoring
    min_rigidity_score: float = 0.2  # Hard threshold: reject if below (0=rigid/picture, 1=moving)
    rigidity_min_frames: int = 10  # Minimum frames needed for rigidity analysis


@dataclass
class AnnotationInfo:
    """Minimal annotation info we use as priors."""

    age_in_months: float | None = None
    quality_flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class Track:
    """A completed tracker ID timeline."""

    id: int
    start_frame: int
    end_frame: int
    fps: float
    keypoints: list[Any] | None = None
    bboxes: list[tuple[float, float, float, float]] | None = None
    face_crops: list[Any] | None = None
    video_path: str | None = None
    frame_numbers: list[int] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def duration_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame + 1)

    def duration_seconds(self) -> float:
        return float(self.duration_frames()) / float(self.fps or 1.0)


@dataclass
class Tracklet:
    """Contiguous segment (potentially a sub-interval of a Track)."""

    parent_id: int
    start_frame: int
    end_frame: int
    fps: float
    keypoints: list[Any] | None = None
    bboxes: list[tuple[float, float, float, float]] | None = None
    face_crops: list[Any] | None = None
    video_path: str | None = None
    frame_numbers: list[int] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> int:
        return self.parent_id

    def duration_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame + 1)

    def duration_seconds(self) -> float:
        return float(self.duration_frames()) / float(self.fps or 1.0)


# --------------------------- Evidence & Scores ---------------------------


@dataclass
class Evidence:
    p_age: float | None = None
    p_skeleton: float | None = None
    p_rigidity: float | None = None  # 0=rigid/picture, 1=natural motion
    flags: list[str] = field(default_factory=list)


@dataclass
class NodeScore:
    tracklet: Tracklet
    score: float
    weight: float
    evidence: Evidence


@dataclass
class EdgeScore:
    src_index: int
    dst_index: int
    score: float
    reasons: dict[str, float] = field(default_factory=dict)


@dataclass
class ChildResult:
    child_track_id_sequence: list[int]
    segments: list[Tracklet]
    confidence: float
    uncertainty: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


# --------------------------- Skeleton Utilities ---------------------------


@dataclass
class SkeletonRatios:
    """Scale-invariant skeleton ratios for child detection."""
    head_shoulder: float | None = None
    leg_torso: float | None = None
    shoulder_hip: float | None = None
    arm_torso: float | None = None


def compute_scale_invariant_ratios(
    keypoints: Any,
    min_confidence: float = 0.3
) -> SkeletonRatios:
    """Compute scale-invariant skeleton ratios from COCO pose keypoints.

    COCO pose keypoint indices:
    0: nose, 1-2: eyes, 3-4: ears
    5-6: shoulders (L, R), 7-8: elbows, 9-10: wrists
    11-12: hips (L, R), 13-14: knees, 15-16: ankles
    """
    if not NUMPY_AVAILABLE or keypoints is None:
        return SkeletonRatios()

    try:
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)

        if len(keypoints) < 17 or keypoints.shape[-1] < 3:
            return SkeletonRatios()

        # FIX B023: capture keypoints in default arg to bind at definition time
        def get_kp(idx: int, kp: Any = keypoints) -> Any | None:
            if idx < len(kp) and kp[idx][2] > min_confidence:
                return kp[idx][:2]
            return None

        nose = get_kp(0)
        l_shoulder, r_shoulder = get_kp(5), get_kp(6)
        l_elbow, r_elbow = get_kp(7), get_kp(8)
        l_wrist, r_wrist = get_kp(9), get_kp(10)
        l_hip, r_hip = get_kp(11), get_kp(12)
        l_knee, r_knee = get_kp(13), get_kp(14)
        l_ankle, r_ankle = get_kp(15), get_kp(16)

        ratios = SkeletonRatios()

        # Ratio 1: head_height / shoulder_width
        if nose is not None and l_shoulder is not None and r_shoulder is not None:
            shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)
            shoulder_mid = (l_shoulder + r_shoulder) / 2
            head_height = np.linalg.norm(nose - shoulder_mid)
            if shoulder_width > 1e-6:
                ratios.head_shoulder = float(head_height / shoulder_width)

        # Ratio 2: leg_length / torso_height
        if l_shoulder is not None and r_shoulder is not None and l_hip is not None and r_hip is not None:
            shoulder_mid = (l_shoulder + r_shoulder) / 2
            hip_mid = (l_hip + r_hip) / 2
            torso_height = np.linalg.norm(shoulder_mid - hip_mid)

            left_leg_visible = sum(x is not None for x in [l_hip, l_knee, l_ankle])
            right_leg_visible = sum(x is not None for x in [r_hip, r_knee, r_ankle])

            leg_length = None
            if left_leg_visible >= 2 and l_hip is not None:
                if l_knee is not None and l_ankle is not None:
                    leg_length = (np.linalg.norm(l_hip - l_knee) +
                                 np.linalg.norm(l_knee - l_ankle))
                elif l_knee is not None:
                    leg_length = np.linalg.norm(l_hip - l_knee) * 2
            elif right_leg_visible >= 2 and r_hip is not None:
                if r_knee is not None and r_ankle is not None:
                    leg_length = (np.linalg.norm(r_hip - r_knee) +
                                 np.linalg.norm(r_knee - r_ankle))
                elif r_knee is not None:
                    leg_length = np.linalg.norm(r_hip - r_knee) * 2

            if leg_length is not None and torso_height > 1e-6:
                ratios.leg_torso = float(leg_length / torso_height)

        # Ratio 3: shoulder_width / hip_width
        if l_shoulder is not None and r_shoulder is not None and l_hip is not None and r_hip is not None:
            shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)
            hip_width = np.linalg.norm(r_hip - l_hip)
            if hip_width > 1e-6:
                ratios.shoulder_hip = float(shoulder_width / hip_width)

        # Ratio 4: arm_length / torso_height
        if l_shoulder is not None and r_shoulder is not None and l_hip is not None and r_hip is not None:
            shoulder_mid = (l_shoulder + r_shoulder) / 2
            hip_mid = (l_hip + r_hip) / 2
            torso_height = np.linalg.norm(shoulder_mid - hip_mid)

            left_arm_visible = sum(x is not None for x in [l_shoulder, l_elbow, l_wrist])
            right_arm_visible = sum(x is not None for x in [r_shoulder, r_elbow, r_wrist])

            arm_length = None
            if left_arm_visible >= 2 and l_shoulder is not None:
                if l_elbow is not None and l_wrist is not None:
                    arm_length = (np.linalg.norm(l_shoulder - l_elbow) +
                                 np.linalg.norm(l_elbow - l_wrist))
                elif l_elbow is not None:
                    arm_length = np.linalg.norm(l_shoulder - l_elbow) * 2
            elif right_arm_visible >= 2 and r_shoulder is not None:
                if r_elbow is not None and r_wrist is not None:
                    arm_length = (np.linalg.norm(r_shoulder - r_elbow) +
                                 np.linalg.norm(r_elbow - r_wrist))
                elif r_elbow is not None:
                    arm_length = np.linalg.norm(r_shoulder - r_elbow) * 2

            if arm_length is not None and torso_height > 1e-6:
                ratios.arm_torso = float(arm_length / torso_height)

        return ratios

    except Exception:
        return SkeletonRatios()


def ratio_to_child_score(ratios: SkeletonRatios) -> float | None:
    """Map skeleton ratios to child-likelihood score."""
    scores = []

    if ratios.head_shoulder is not None:
        hs = ratios.head_shoulder
        if hs > 1.0:
            score = 0.9
        elif hs > 0.8:
            score = 0.5 + (hs - 0.8) * 2.0
        elif hs > 0.6:
            score = 0.2 + (hs - 0.6) * 1.5
        else:
            score = 0.1
        scores.append(score)

    if ratios.leg_torso is not None:
        lt = ratios.leg_torso
        if lt < 1.5:
            score = 0.9
        elif lt < 2.0:
            score = 0.9 - (lt - 1.5) * 0.8
        elif lt < 2.5:
            score = 0.5 - (lt - 2.0) * 0.6
        else:
            score = 0.1
        scores.append(score)

    if ratios.shoulder_hip is not None:
        sh = ratios.shoulder_hip
        deviation_from_1 = abs(sh - 1.0)
        if deviation_from_1 < 0.1:
            score = 0.9
        elif deviation_from_1 < 0.2:
            score = 0.7
        elif deviation_from_1 < 0.3:
            score = 0.4
        else:
            score = 0.2
        scores.append(score)

    if ratios.arm_torso is not None:
        at = ratios.arm_torso
        if at < 1.3:
            score = 0.9
        elif at < 1.8:
            score = 0.9 - (at - 1.3) * 0.8
        elif at < 2.2:
            score = 0.5 - (at - 1.8) * 0.75
        else:
            score = 0.1
        scores.append(score)

    if not scores:
        return None

    return float(sum(scores) / len(scores))


def compute_keypoint_rigidity_score(
    keypoints_list: list[Any],
    min_confidence: float = 0.3,
    min_frames: int = 10
) -> float | None:
    """Detect if keypoints move rigidly (picture) or naturally (real person).

    Camera-motion invariant: measures relative inter-keypoint distances, not
    absolute positions.

    Returns:
        0.0 = rigid/static (picture), 1.0 = natural motion (real person)
    """
    if not NUMPY_AVAILABLE or not keypoints_list or len(keypoints_list) < min_frames:
        return None

    try:
        # FIX var-annotated: explicit type annotation for distance_series
        distance_series: dict[str, list[float]] = {
            'shoulder_width': [],
            'hip_width': [],
            'torso_height': [],
            'left_upper_arm': [],
            'right_upper_arm': [],
            'left_forearm': [],
            'right_forearm': [],
            'left_thigh': [],
            'right_thigh': [],
            'left_shank': [],
            'right_shank': [],
        }

        for kp in keypoints_list:
            if kp is None or len(kp) < 17:
                continue

            kp_array = np.array(kp) if not isinstance(kp, np.ndarray) else kp

            # FIX B023: bind kp_array via default argument
            def get_kp(idx: int, _kp: Any = kp_array) -> Any | None:
                if idx < len(_kp) and _kp[idx][2] > min_confidence:
                    return _kp[idx][:2]
                return None

            l_shoulder = get_kp(5)
            r_shoulder = get_kp(6)
            l_elbow = get_kp(7)
            r_elbow = get_kp(8)
            l_wrist = get_kp(9)
            r_wrist = get_kp(10)
            l_hip = get_kp(11)
            r_hip = get_kp(12)
            l_knee = get_kp(13)
            r_knee = get_kp(14)
            l_ankle = get_kp(15)
            r_ankle = get_kp(16)

            if l_shoulder is not None and r_shoulder is not None:
                distance_series['shoulder_width'].append(float(np.linalg.norm(r_shoulder - l_shoulder)))

            if l_hip is not None and r_hip is not None:
                distance_series['hip_width'].append(float(np.linalg.norm(r_hip - l_hip)))

                if l_shoulder is not None and r_shoulder is not None:
                    shoulder_mid = (l_shoulder + r_shoulder) / 2
                    hip_mid = (l_hip + r_hip) / 2
                    distance_series['torso_height'].append(float(np.linalg.norm(shoulder_mid - hip_mid)))

            if l_shoulder is not None and l_elbow is not None:
                distance_series['left_upper_arm'].append(float(np.linalg.norm(l_shoulder - l_elbow)))

            if r_shoulder is not None and r_elbow is not None:
                distance_series['right_upper_arm'].append(float(np.linalg.norm(r_shoulder - r_elbow)))

            if l_elbow is not None and l_wrist is not None:
                distance_series['left_forearm'].append(float(np.linalg.norm(l_elbow - l_wrist)))

            if r_elbow is not None and r_wrist is not None:
                distance_series['right_forearm'].append(float(np.linalg.norm(r_elbow - r_wrist)))

            if l_hip is not None and l_knee is not None:
                distance_series['left_thigh'].append(float(np.linalg.norm(l_hip - l_knee)))

            if r_hip is not None and r_knee is not None:
                distance_series['right_thigh'].append(float(np.linalg.norm(r_hip - r_knee)))

            if l_knee is not None and l_ankle is not None:
                distance_series['left_shank'].append(float(np.linalg.norm(l_knee - l_ankle)))

            if r_knee is not None and r_ankle is not None:
                distance_series['right_shank'].append(float(np.linalg.norm(r_knee - r_ankle)))

        cvs = []
        # FIX B007: rename unused loop variable `name` to `_name`
        for _name, distances in distance_series.items():
            if len(distances) < min_frames:
                continue

            mean_dist = np.mean(distances)
            std_dist = np.std(distances)

            if mean_dist > 1e-6:
                cv = std_dist / mean_dist
                cvs.append(cv)

        if not cvs:
            return None

        avg_cv = float(np.mean(cvs))

        if avg_cv < 0.05:
            return 0.0
        elif avg_cv > 0.15:
            return 1.0
        else:
            return (avg_cv - 0.05) / (0.15 - 0.05)

    except Exception:
        return None


def aggregate_skeleton_ratios_over_track(
    keypoints_list: list[Any],
    min_confidence: float = 0.3
) -> float | None:
    """Aggregate skeleton ratios across all frames in a track using median."""
    if not keypoints_list:
        return None

    all_head_shoulder = []
    all_leg_torso = []
    all_shoulder_hip = []
    all_arm_torso = []

    for kp in keypoints_list:
        if kp is None:
            continue
        ratios = compute_scale_invariant_ratios(kp, min_confidence)
        if ratios.head_shoulder is not None:
            all_head_shoulder.append(ratios.head_shoulder)
        if ratios.leg_torso is not None:
            all_leg_torso.append(ratios.leg_torso)
        if ratios.shoulder_hip is not None:
            all_shoulder_hip.append(ratios.shoulder_hip)
        if ratios.arm_torso is not None:
            all_arm_torso.append(ratios.arm_torso)

    if not NUMPY_AVAILABLE:
        return None

    median_ratios = SkeletonRatios()
    if all_head_shoulder:
        median_ratios.head_shoulder = float(np.median(all_head_shoulder))
    if all_leg_torso:
        median_ratios.leg_torso = float(np.median(all_leg_torso))
    if all_shoulder_hip:
        median_ratios.shoulder_hip = float(np.median(all_shoulder_hip))
    if all_arm_torso:
        median_ratios.arm_torso = float(np.median(all_arm_torso))

    return ratio_to_child_score(median_ratios)


# --------------------------- Main Identifier ---------------------------

class SigLipModel:
    """Mixin class to load SigLIP model for age classification."""

    siglip_model: Any
    siglip_processor: Any

    def load_siglip_model(self) -> None:  # FIX no-untyped-def: added return type
        """Load SigLIP model and processor for age classification."""
        if not SIGLIP_AVAILABLE:
            return

        try:
            self.siglip_model = SiglipForImageClassification.from_pretrained(
                "prithivMLmods/Age-Classification-SigLIP2"
            )
            self.siglip_processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                "prithivMLmods/Age-Classification-SigLIP2"
            )
        except Exception:
            self.siglip_model = None
            self.siglip_processor = None


class SingleChildIdentifier:
    """Identify and merge the single child across fragmented tracker IDs."""

    def __init__(
        self,
        cfg: ChildIdentificationConfig,
        annotations: AnnotationInfo,
        siglip_model: SigLipModel | None = None,
    ):
        self.cfg = cfg
        self.ann = annotations
        self._siglip_model = siglip_model.siglip_model if siglip_model else None
        self._siglip_processor = siglip_model.siglip_processor if siglip_model else None

    # -------- Orchestration --------

    def identify_child(self, tracks: list[Track]) -> ChildResult:
        tracklets = self._split_into_tracklets(tracks)
        nodes = [self._score_node(tl) for tl in tracklets]
        edges = self._build_edges(nodes)
        path_indices = self._select_best_path(nodes, edges)
        path_segments = [nodes[i].tracklet for i in path_indices]
        confidence = self._estimate_confidence([nodes[i] for i in path_indices])
        return ChildResult(
            child_track_id_sequence=[seg.id for seg in path_segments],
            segments=path_segments,
            confidence=confidence,
            uncertainty=None,
            diagnostics={
                "nodes": nodes,
                "edges": edges,
                "path_indices": path_indices,
            },
        )

    # -------- Tracklet handling --------

    def _split_into_tracklets(self, tracks: list[Track]) -> list[Tracklet]:
        """For now, treat each Track as a single Tracklet."""
        tracklets: list[Tracklet] = []
        for tr in tracks:
            if tr.duration_frames() < max(1, self.cfg.min_track_frames):
                continue
            tracklets.append(
                Tracklet(
                    parent_id=tr.id,
                    start_frame=tr.start_frame,
                    end_frame=tr.end_frame,
                    fps=tr.fps,
                    keypoints=tr.keypoints,
                    bboxes=tr.bboxes,
                    face_crops=tr.face_crops,
                    video_path=tr.video_path,
                    frame_numbers=tr.frame_numbers,
                    meta=dict(tr.meta),
                )
            )
        return tracklets

    # -------- Evidence & scoring --------

    def _compute_age_prob(self, tl: Tracklet, flags: list[str]) -> float | None:
        """Compute age-based child probability for a tracklet."""
        if self.cfg.age_estimation_method == "siglip":
            return self._compute_age_prob_siglip(tl, flags)
        elif self.cfg.age_estimation_method == "deepface":
            return self._compute_age_prob_deepface(tl, flags)
        else:
            flags.append(f"unknown_age_method_{self.cfg.age_estimation_method}")
            return None

    def _compute_age_prob_siglip(self, tl: Tracklet, flags: list[str]) -> float | None:
        """Compute age-based child probability using SigLIP."""
        if (getattr(tl, "bboxes", None) and tl.bboxes and
                getattr(tl, "video_path", None) and tl.video_path):

            child_prob, age_flags = median_child_prob_from_bboxes_siglip(
                video_path=tl.video_path,
                frame_numbers=tl.frame_numbers or [],
                bboxes=tl.bboxes,
                keypoints=tl.keypoints,
                model=self._siglip_model,
                processor=self._siglip_processor,
                sampling_percentage=self.cfg.sampling_percentage,
                sampling_max=self.cfg.sampling_max_frames_per_track,
                enable_body_visibility_filter=self.cfg.enable_body_visibility_filter,
                min_visible_keypoints=self.cfg.min_visible_keypoints,
                keypoint_conf_threshold=self.cfg.keypoint_conf_threshold,
                enable_roi_size_filter=self.cfg.enable_roi_size_filter,
                min_roi_width=self.cfg.min_roi_width,
                min_roi_height=self.cfg.min_roi_height,
                body_keypoint_indices=self.cfg.body_keypoint_indices,
                sampling_mode=self.cfg.sampling_mode,
                min_pose_confidence=self.cfg.min_pose_confidence,
            )

            flags.extend(age_flags)
            return child_prob
        else:
            if not getattr(tl, "bboxes", None):
                flags.append("no_bboxes")
            if not getattr(tl, "video_path", None):
                flags.append("no_video_path")
            return None

    def _compute_age_prob_deepface(self, tl: Tracklet, flags: list[str]) -> float | None:
        """Compute age-based child probability using DeepFace."""
        if getattr(tl, "face_crops", None) and tl.face_crops:
            median_age, _ = median_age_from_face_crops(
                tl.face_crops,
                face_conf_threshold=self.cfg.face_conf_threshold,
                sampling_percentage=self.cfg.sampling_percentage,
                sampling_max=self.cfg.sampling_max_frames_per_track,
            )
            if median_age is not None:
                return map_age_to_child_prob(
                    median_age,
                    child_years_threshold=self.cfg.age_child_years_threshold,
                    tau=self.cfg.age_tau,
                )
            else:
                flags.append("age_unavailable")

        elif (getattr(tl, "bboxes", None) and tl.bboxes and
              getattr(tl, "video_path", None) and tl.video_path):

            median_age, age_flags = median_age_from_bboxes(
                video_path=tl.video_path,
                frame_numbers=tl.frame_numbers or [],
                bboxes=tl.bboxes,
                keypoints=tl.keypoints,
                face_conf_threshold=self.cfg.face_conf_threshold,
                sampling_percentage=self.cfg.sampling_percentage,
                sampling_max=self.cfg.sampling_max_frames_per_track,
                enable_body_visibility_filter=self.cfg.enable_body_visibility_filter,
                min_visible_keypoints=self.cfg.min_visible_keypoints,
                keypoint_conf_threshold=self.cfg.keypoint_conf_threshold,
                enable_roi_size_filter=self.cfg.enable_roi_size_filter,
                min_roi_width=self.cfg.min_roi_width,
                min_roi_height=self.cfg.min_roi_height,
                body_keypoint_indices=self.cfg.body_keypoint_indices,
                sampling_mode=self.cfg.sampling_mode,
                min_pose_confidence=self.cfg.min_pose_confidence,
            )

            flags.extend(age_flags)

            if median_age is not None:
                return map_age_to_child_prob(
                    median_age,
                    child_years_threshold=self.cfg.age_child_years_threshold,
                    tau=self.cfg.age_tau,
                )
            else:
                flags.append("age_unavailable_from_bbox")
        else:
            if not getattr(tl, "face_crops", None):
                flags.append("no_face_crop")
            if not getattr(tl, "bboxes", None):
                flags.append("no_bboxes")
            if not getattr(tl, "video_path", None):
                flags.append("no_video_path")

        return None

    def _compute_skeleton_prob(self, tl: Tracklet, flags: list[str]) -> float | None:
        """Compute skeleton-based child probability for a tracklet."""
        if not NUMPY_AVAILABLE:
            flags.append("numpy_not_available")
            return None

        if not getattr(tl, "keypoints", None) or not tl.keypoints:
            flags.append("no_keypoints")
            return None

        try:
            child_prob = aggregate_skeleton_ratios_over_track(
                keypoints_list=tl.keypoints,
                min_confidence=self.cfg.skeleton_min_confidence
            )

            if child_prob is None:
                flags.append("skeleton_insufficient_data")
                return None

            return child_prob

        except Exception as e:
            flags.append(f"skeleton_error_{type(e).__name__}")
            return None

    def _compute_rigidity_prob(self, tl: Tracklet, flags: list[str]) -> float | None:
        """Compute rigidity score to detect static pictures vs real people."""
        if not NUMPY_AVAILABLE:
            flags.append("numpy_not_available")
            return None

        if not getattr(tl, "keypoints", None) or not tl.keypoints:
            flags.append("no_keypoints")
            return None

        try:
            rigidity_score = compute_keypoint_rigidity_score(
                keypoints_list=tl.keypoints,
                min_confidence=self.cfg.keypoint_conf_threshold,
                min_frames=self.cfg.rigidity_min_frames
            )

            if rigidity_score is None:
                flags.append("rigidity_insufficient_data")
                return None

            return rigidity_score

        except Exception as e:
            flags.append(f"rigidity_error_{type(e).__name__}")
            return None

    def _compute_evidence(self, tl: Tracklet) -> Evidence:
        """Compute evidence for child identification."""
        flags: list[str] = []

        p_age: float | None = self._compute_age_prob(tl, flags)

        p_skel: float | None = None
        if self.cfg.enable_skeleton_ratios:
            p_skel = self._compute_skeleton_prob(tl, flags)
        else:
            flags.append("skeleton_disabled")

        p_rigidity: float | None = None
        if self.cfg.enable_rigidity_detection:
            p_rigidity = self._compute_rigidity_prob(tl, flags)
        else:
            flags.append("rigidity_disabled")

        return Evidence(p_age=p_age, p_skeleton=p_skel, p_rigidity=p_rigidity, flags=flags)

    def _score_node(self, tl: Tracklet) -> NodeScore:
        ev = self._compute_evidence(tl)

        # FIX SIM102: combined nested if into single if with `and`
        if self.cfg.enable_rigidity_detection and ev.p_rigidity is not None and ev.p_rigidity < self.cfg.min_rigidity_score:
            ev.flags.append(f"rejected_static_picture_rigidity_{ev.p_rigidity:.3f}")
            return NodeScore(tracklet=tl, score=0.0, weight=0.0, evidence=ev)

        w_age = self.cfg.w_age_default
        w_skel = self.cfg.w_skel_default
        w_rigidity = self.cfg.rigidity_weight

        weights_sum = 0.0
        score_sum = 0.0

        if ev.p_age is not None:
            score_sum += w_age * float(ev.p_age)
            weights_sum += w_age
        if ev.p_skeleton is not None:
            score_sum += w_skel * float(ev.p_skeleton)
            weights_sum += w_skel
        if ev.p_rigidity is not None:
            score_sum += w_rigidity * float(ev.p_rigidity)
            weights_sum += w_rigidity

        score = (score_sum / weights_sum) if weights_sum > 0 else 0.0
        weight = score * tl.duration_seconds()

        return NodeScore(tracklet=tl, score=score, weight=weight, evidence=ev)

    # -------- Edge building & scoring --------

    def _compute_age_inconsistency_penalty(
        self,
        src_node: NodeScore,
        dst_node: NodeScore
    ) -> float:
        """Penalize edges connecting nodes with inconsistent age evidence."""
        src_age = src_node.evidence.p_age
        dst_age = dst_node.evidence.p_age

        if src_age is None and dst_age is None:
            return 0.0

        if src_age is None:
            return abs(dst_age - 0.5) if dst_age is not None else 0.0
        if dst_age is None:
            return abs(src_age - 0.5)

        diff = abs(src_age - dst_age)
        if diff < self.cfg.age_inconsistency_threshold:
            return 0.0

        return min(1.0, diff / (1.0 - self.cfg.age_inconsistency_threshold))

    def _build_edges(self, nodes: list[NodeScore]) -> list[EdgeScore]:
        """Build candidate edges between non-overlapping, time-adjacent nodes."""
        edges: list[EdgeScore] = []
        if not nodes:
            return edges

        order = sorted(range(len(nodes)), key=lambda i: nodes[i].tracklet.start_frame)
        for i_idx in range(len(order)):
            i = order[i_idx]
            src = nodes[i].tracklet
            src_end = src.end_frame
            for j_idx in range(i_idx + 1, len(order)):
                j = order[j_idx]
                dst = nodes[j].tracklet
                if dst.start_frame <= src_end:
                    continue
                gap_frames = dst.start_frame - src_end
                gap_sec = float(gap_frames) / float(dst.fps or 1.0)
                if gap_sec > self.cfg.continuity_gap_seconds:
                    break

                temporal = max(0.0, 1.0 - (gap_sec / max(1e-6, self.cfg.continuity_gap_seconds)))

                bonus = 0.0
                reasons: dict[str, float] = {"temporal": temporal}
                if src.id == dst.id:
                    bonus = self.cfg.intra_id_gamma * math.exp(-gap_sec / max(1e-6, self.cfg.intra_id_tau))
                    reasons["same_id_bonus"] = bonus

                age_penalty = self._compute_age_inconsistency_penalty(nodes[i], nodes[j])
                weighted_penalty = age_penalty * self.cfg.age_inconsistency_penalty_weight
                if age_penalty > 0:
                    reasons["age_inconsistency_penalty"] = age_penalty

                score = temporal + bonus - weighted_penalty
                edges.append(EdgeScore(src_index=i, dst_index=j, score=score, reasons=reasons))

        return edges

    # -------- Path selection --------

    def _select_best_path(self, nodes: list[NodeScore], edges: list[EdgeScore]) -> list[int]:
        """Simple longest-path DP on a DAG defined by time ordering."""
        n = len(nodes)
        if n == 0:
            return []

        adj: dict[int, list[tuple[int, float]]] = {}
        for e in edges:
            adj.setdefault(e.src_index, []).append((e.dst_index, e.score))

        order = sorted(range(n), key=lambda i: nodes[i].tracklet.start_frame)

        best_score: dict[int, float] = {i: nodes[i].weight for i in range(n)}
        parent: dict[int, int | None] = {i: None for i in range(n)}

        for i in order:
            for (j, e_score) in adj.get(i, []):
                candidate = best_score[i] + nodes[j].weight + e_score
                if candidate > best_score.get(j, float("-inf")):
                    best_score[j] = candidate
                    parent[j] = i

        end_idx = max(best_score, key=lambda k: best_score[k])

        path: list[int] = []
        cur: int | None = end_idx
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    # -------- Confidence --------

    def _estimate_confidence(self, path_nodes: list[NodeScore]) -> float:
        if not path_nodes:
            return 0.0
        avg = sum(n.score for n in path_nodes) / float(len(path_nodes))
        return float(max(0.0, min(1.0, avg)))


# --------------------------- Public API ---------------------------


def identify_single_child(
    tracks: list[Track],
    annotations: AnnotationInfo,
    cfg: ChildIdentificationConfig | None = None,
) -> ChildResult:
    """Identify the single child across fragmented tracker IDs."""
    identifier = SingleChildIdentifier(cfg or ChildIdentificationConfig(), annotations)
    return identifier.identify_child(tracks)


# --------------------------- Age Utility ---------------------------

def _sigmoid(x: float, tau: float) -> float:
    """Simple logistic with temperature (p = 1 / (1 + exp(-x/tau)))."""
    t = max(1e-6, float(tau))
    return 1.0 / (1.0 + math.exp(-float(x) / t))


def deepface_predict_age(image: Any, face_conf_threshold: float) -> tuple[float | None, list[str]]:
    """Run DeepFace age on a single image and return (age, flags)."""
    if DeepFace is None:
        return None, ["deepface_not_available"]
    try:
        res = DeepFace.analyze(image, actions=["age"], enforce_detection=False)
        age_val: float | None = None
        if isinstance(res, list) and res:
            if len(res) > 1:
                face_attr = max(res, key=lambda r: r.get("face_confidence", 0.0))
            else:
                face_attr = res[0]
            if face_attr.get("face_confidence", 0.0) < face_conf_threshold:
                return None, ["deepface_low_face_confidence_{:.2f}".format(face_attr.get("face_confidence", 0.0))]
            age_val = face_attr.get("age")
        if age_val is None:
            return None, ["deepface_no_age"]
        return float(age_val), []
    except Exception:
        return None, ["deepface_error"]


def map_age_to_child_prob(age_years: float, child_years_threshold: float, tau: float) -> float:
    """Map age in years to child probability with a logistic mapping."""
    return _sigmoid((float(child_years_threshold) - float(age_years)), float(tau))


def siglip_predict_child_prob(image: Any, model: Any = None, processor: Any = None) -> tuple[float | None, list[str]]:
    """Run SigLIP age group classification and return (child_probability, flags)."""
    if not SIGLIP_AVAILABLE:
        return None, ["siglip_not_available"]

    try:
        if model is None or processor is None:
            return None, ["siglip_model_not_loaded"]

        if hasattr(image, 'shape'):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(image).convert("RGB")
            else:
                return None, ["siglip_invalid_image_format"]

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

        child_probability = float(probs[0])
        return child_probability, []
    except Exception as e:
        return None, [f"siglip_error_{type(e).__name__}"]


def _evenly_spaced_indices(n: int, k: int) -> list[int]:
    """Return k indices evenly covering [0, n-1]."""
    if k <= 1:
        return [max(0, (n - 1) // 2)] if n > 0 else []
    if n <= 0:
        return []
    step = (n - 1) / float(k - 1)
    return sorted({int(round(i * step)) for i in range(k)})


def _smart_frame_selection(
    keypoints_list: list[Any],
    k: int,
    min_pose_confidence: float = 0.7,
    body_keypoint_indices: list[int] | None = None
) -> list[int]:
    """Select k frames with highest pose confidence scores for specified body keypoints."""
    if not keypoints_list:
        return []

    if body_keypoint_indices is None:
        body_keypoint_indices = list(range(len(keypoints_list[0]) if keypoints_list and keypoints_list[0] else 17))

    frame_confidences = []
    for i, keypoints in enumerate(keypoints_list):
        if keypoints is None or len(keypoints) == 0:
            frame_confidences.append((i, 0.0))
            continue

        confidences = []
        for idx in body_keypoint_indices:
            # FIX SIM102: combined nested if
            if idx < len(keypoints) and len(keypoints[idx]) >= 3 and keypoints[idx][2] > 0.1:
                confidences.append(float(keypoints[idx][2]))

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        frame_confidences.append((i, avg_confidence))

    valid_frames = [
        (i, conf) for i, conf in frame_confidences
        if conf >= min_pose_confidence
    ]

    if len(valid_frames) < k:
        valid_frames = frame_confidences

    valid_frames.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [i for i, _ in valid_frames[:k]]

    return sorted(selected_indices)


def median_age_from_face_crops(
    face_crops: list[Any],
    face_conf_threshold: float = 0.9,
    sampling_percentage: float = 0.25,
    sampling_max: int = 40,
) -> tuple[float | None, list[str]]:
    """Sample face crops, run DeepFace age, and return median age."""
    n = len(face_crops or [])
    if n == 0:
        return None, ["no_face_crop"]
    k_by_perc = int(math.ceil(max(0.0, float(sampling_percentage)) * n))
    k_target = min(max(1, k_by_perc), max(1, int(sampling_max)), n)
    indices = _evenly_spaced_indices(n, k_target)
    ages: list[float] = []
    flags: list[str] = []
    for idx in indices:
        age, local_flags = deepface_predict_age(face_crops[idx], face_conf_threshold)
        if local_flags:
            flags.extend(local_flags)
        if age is not None:
            ages.append(float(age))
    if not ages:
        if not flags:
            flags.append("deepface_no_age_all")
        return None, flags
    ages.sort()
    m = ages[len(ages) // 2] if (len(ages) % 2 == 1) else (
        0.5 * (ages[len(ages)//2 - 1] + ages[len(ages)//2])
    )
    return float(m), flags


# --------------------------- Bbox-based Age Estimation ---------------------------


def load_video_frame(video_path: str, frame_number: int) -> Any | None:
    """Load a specific frame from video file."""
    if cv2 is None:
        return None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ret, frame = cap.read()
        cap.release()

        return frame if ret else None
    except Exception:
        return None


def crop_bbox_from_frame(frame: Any, bbox: tuple[float, float, float, float]) -> Any | None:
    """Crop a bounding box from a video frame."""
    if frame is None:
        return None
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(x1, min(x2, w))
    y2 = max(y1, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def is_body_in_bbox(
    person_roi: Any,
    keypoints: list[Any],
    min_visible_keypoints: int = 3,
    keypoint_conf_threshold: float = 0.3,
    enable_roi_size_filter: bool = False,
    min_roi_width: int = 30,
    min_roi_height: int = 50,
    body_keypoint_indices: list[int] | None = None
) -> dict[str, Any]:
    """Test if there's a valid body inside the given bounding box."""
    try:
        # FIX SIM102: combined nested if
        if enable_roi_size_filter and (person_roi.shape[0] < min_roi_height or person_roi.shape[1] < min_roi_width):
            return {
                'detected': False,
                'reason': 'Body region too small',
                'roi_size': person_roi.shape
            }

        if body_keypoint_indices is None:
            body_keypoint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        visible_keypoints = 0
        for idx in body_keypoint_indices:
            if (
                idx < len(keypoints)
                and len(keypoints[idx]) >= 3
                and keypoints[idx][2] > keypoint_conf_threshold
            ):
                visible_keypoints += 1

        if visible_keypoints < min_visible_keypoints:
            return {
                'detected': False,
                'reason': f'Too few visible keypoints: {visible_keypoints} < {min_visible_keypoints}',
                'visible_keypoints': visible_keypoints,
                'required_keypoints': min_visible_keypoints
            }

        total_visible = len([
            kp for kp in keypoints
            if len(kp) >= 3 and kp[2] > keypoint_conf_threshold
        ])

        return {
            'detected': True,
            'visible_keypoints': visible_keypoints,
            'total_keypoints': total_visible,
            'roi_size': person_roi.shape
        }

    except Exception as e:
        return {
            'detected': False,
            'reason': f'Error: {str(e)}'
        }


def median_age_from_bboxes(
    video_path: str,
    frame_numbers: list[int],
    bboxes: list[tuple[float, float, float, float]],
    keypoints: list[Any] | None = None,
    face_conf_threshold: float = 0.8,
    sampling_percentage: float = 0.25,
    sampling_max: int = 40,
    sampling_mode: str = "even",
    min_pose_confidence: float = 0.7,
    enable_body_visibility_filter: bool = False,
    min_visible_keypoints: int = 3,
    keypoint_conf_threshold: float = 0.3,
    enable_roi_size_filter: bool = False,
    min_roi_width: int = 30,
    min_roi_height: int = 50,
    body_keypoint_indices: list[int] | None = None,
) -> tuple[float | None, list[str]]:
    """Extract bbox crops from video frames and estimate median age using DeepFace."""
    flags: list[str] = []

    if cv2 is None:
        return None, ["cv2_not_available"]

    if not video_path or not frame_numbers or not bboxes:
        return None, ["missing_input_data"]

    n = len(frame_numbers)
    k_by_perc = int(math.ceil(max(0.0, float(sampling_percentage)) * n))
    k_target = min(max(1, k_by_perc), max(1, int(sampling_max)), n)

    if sampling_mode == "smart" and keypoints is not None:
        indices = _smart_frame_selection(keypoints, k_target, min_pose_confidence, body_keypoint_indices)
    else:
        indices = _evenly_spaced_indices(n, k_target)

    ages: list[float] = []

    for idx in indices:
        if idx >= len(frame_numbers) or idx >= len(bboxes):
            continue

        frame_num = frame_numbers[idx]
        bbox = bboxes[idx]

        frame = load_video_frame(video_path, frame_num)
        if frame is None:
            flags.append(f"frame_load_failed_{frame_num}")
            continue

        try:
            bbox_crop = crop_bbox_from_frame(frame, bbox)
        except Exception:
            flags.append(f"bbox_crop_error_{frame_num}")
            continue

        if enable_body_visibility_filter and keypoints is not None and idx < len(keypoints):
            frame_keypoints = keypoints[idx] if keypoints else []
            body_check = is_body_in_bbox(
                bbox_crop,
                frame_keypoints,
                min_visible_keypoints=min_visible_keypoints,
                keypoint_conf_threshold=keypoint_conf_threshold,
                enable_roi_size_filter=enable_roi_size_filter,
                min_roi_width=min_roi_width,
                min_roi_height=min_roi_height,
                body_keypoint_indices=body_keypoint_indices
            )

            if not body_check['detected']:
                flags.append(f"body_filter_failed_{frame_num}_{body_check['reason']}")
                continue

        age, local_flags = deepface_predict_age(bbox_crop, face_conf_threshold)
        if local_flags:
            flags.extend([f"{flag}_{frame_num}" for flag in local_flags])

        if age is not None:
            ages.append(float(age))

    if not ages:
        if not flags:
            flags.append("no_age_estimates")
        return None, flags

    ages.sort()
    if len(ages) % 2 == 1:
        median_age = ages[len(ages) // 2]
    else:
        median_age = 0.5 * (ages[len(ages)//2 - 1] + ages[len(ages)//2])

    return float(median_age), flags


def median_child_prob_from_bboxes_siglip(
    video_path: str,
    frame_numbers: list[int],
    bboxes: list[tuple[float, float, float, float]],
    keypoints: list[Any] | None = None,
    model: Any = None,
    processor: Any = None,
    sampling_percentage: float = 0.25,
    sampling_max: int = 40,
    sampling_mode: str = "even",
    min_pose_confidence: float = 0.7,
    enable_body_visibility_filter: bool = False,
    min_visible_keypoints: int = 3,
    keypoint_conf_threshold: float = 0.3,
    enable_roi_size_filter: bool = False,
    min_roi_width: int = 30,
    min_roi_height: int = 50,
    body_keypoint_indices: list[int] | None = None,
) -> tuple[float | None, list[str]]:
    """Extract bbox crops from video frames and estimate median child probability using SigLIP."""
    flags: list[str] = []

    if cv2 is None:
        return None, ["cv2_not_available"]

    if not SIGLIP_AVAILABLE:
        return None, ["siglip_not_available"]

    if not video_path or not frame_numbers or not bboxes:
        return None, ["missing_input_data"]

    n = len(frame_numbers)
    k_by_perc = int(math.ceil(max(0.0, float(sampling_percentage)) * n))
    k_target = min(max(1, k_by_perc), max(1, int(sampling_max)), n)

    if sampling_mode == "smart" and keypoints is not None:
        indices = _smart_frame_selection(keypoints, k_target, min_pose_confidence, body_keypoint_indices)
    else:
        indices = _evenly_spaced_indices(n, k_target)

    child_probs: list[float] = []

    for idx in indices:
        if idx >= len(frame_numbers) or idx >= len(bboxes):
            continue

        frame_num = frame_numbers[idx]
        bbox = bboxes[idx]

        frame = load_video_frame(video_path, frame_num)
        if frame is None:
            flags.append(f"frame_load_failed_{frame_num}")
            continue

        try:
            bbox_crop = crop_bbox_from_frame(frame, bbox)
        except Exception:
            flags.append(f"bbox_crop_error_{frame_num}")
            continue

        if enable_body_visibility_filter and keypoints is not None and idx < len(keypoints):
            frame_keypoints = keypoints[idx] if keypoints else []
            body_check = is_body_in_bbox(
                bbox_crop,
                frame_keypoints,
                min_visible_keypoints=min_visible_keypoints,
                keypoint_conf_threshold=keypoint_conf_threshold,
                enable_roi_size_filter=enable_roi_size_filter,
                min_roi_width=min_roi_width,
                min_roi_height=min_roi_height,
                body_keypoint_indices=body_keypoint_indices
            )

            if not body_check['detected']:
                flags.append(f"body_filter_failed_{frame_num}_{body_check['reason']}")
                continue

        child_prob, local_flags = siglip_predict_child_prob(bbox_crop, model, processor)
        if local_flags:
            flags.extend([f"{flag}_{frame_num}" for flag in local_flags])

        if child_prob is not None:
            child_probs.append(float(child_prob))

    if not child_probs:
        if not flags:
            flags.append("no_child_prob_estimates")
        return None, flags

    child_probs.sort()
    if len(child_probs) % 2 == 1:
        median_child_prob = child_probs[len(child_probs) // 2]
    else:
        median_child_prob = 0.5 * (child_probs[len(child_probs)//2 - 1] + child_probs[len(child_probs)//2])

    return float(median_child_prob), flags