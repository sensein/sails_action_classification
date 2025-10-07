"""
Single-child identification scaffold.

Goal: Trust per-ID temporal continuity but allow merging across IDs using
post-tracking evidence (age, skeleton ratios, quality) with a simple
graph-based selection. This file provides minimal structures and placeholders
so we can implement features incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
from math import inf

# Optional dependency for age estimation
try:  # DeepFace is optional; we guard all calls
    from deepface import DeepFace  # type: ignore
except Exception:  # pragma: no cover - environment may not have DeepFace
    DeepFace = None  # type: ignore

# Optional dependency for SigLIP age estimation
try:
    from transformers import AutoImageProcessor, SiglipForImageClassification
    from PIL import Image
    import torch
    SIGLIP_AVAILABLE = True
except Exception:  # pragma: no cover - environment may not have SigLIP
    SIGLIP_AVAILABLE = False

# Optional dependency for video frame loading
try:
    import cv2
except Exception:
    cv2 = None  # type: ignore

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore


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
    body_keypoint_indices: Optional[List[int]] = None  # Default: [0,1,2,3,4,5,6,7,8,9]
    enable_roi_size_filter: bool = False  # Enable filtering based on ROI size
    min_roi_width: int = 30
    min_roi_height: int = 50

    # Age mapping
    age_tau: float = 2.5
    age_child_years_threshold: float = 10.0

    # Weights
    w_age_default: float = 0.5
    w_skel_default: float = 0.5

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


@dataclass
class AnnotationInfo:
    """Minimal annotation info we use as priors."""

    age_in_months: Optional[float] = None
    quality_flags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Track:
    """A completed tracker ID timeline.

    Note: This is a minimal structure; downstream can add fields as needed.
    """

    id: int
    start_frame: int
    end_frame: int
    fps: float
    keypoints: Optional[List[Any]] = None  # per-frame keypoints (backend-specific)
    bboxes: Optional[List[Tuple[float, float, float, float]]] = None  # per-frame [x1,y1,x2,y2]
    face_crops: Optional[List[Any]] = None  # optional face crops or face ROIs
    video_path: Optional[str] = None  # path to source video for frame extraction
    frame_numbers: Optional[List[int]] = None  # corresponding frame numbers for keypoints/bboxes
    meta: Dict[str, Any] = field(default_factory=dict)

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
    keypoints: Optional[List[Any]] = None
    bboxes: Optional[List[Tuple[float, float, float, float]]] = None
    face_crops: Optional[List[Any]] = None
    video_path: Optional[str] = None
    frame_numbers: Optional[List[int]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> int:
        # For now, treat tracklet id as parent id; callers can supply unique ids if needed
        return self.parent_id

    def duration_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame + 1)

    def duration_seconds(self) -> float:
        return float(self.duration_frames()) / float(self.fps or 1.0)


# --------------------------- Evidence & Scores ---------------------------


@dataclass
class Evidence:
    p_age: Optional[float] = None
    p_skeleton: Optional[float] = None
    flags: List[str] = field(default_factory=list)


@dataclass
class NodeScore:
    tracklet: Tracklet
    score: float
    weight: float
    evidence: Evidence


@dataclass
class EdgeScore:
    src_index: int  # index into nodes list
    dst_index: int  # index into nodes list
    score: float
    reasons: Dict[str, float] = field(default_factory=dict)


@dataclass
class ChildResult:
    child_track_id_sequence: List[int]
    segments: List[Tracklet]
    confidence: float
    uncertainty: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# --------------------------- Skeleton Utilities ---------------------------


@dataclass
class SkeletonRatios:
    """Scale-invariant skeleton ratios for child detection."""
    head_shoulder: Optional[float] = None  # head_height / shoulder_width
    leg_torso: Optional[float] = None  # leg_length / torso_height
    shoulder_hip: Optional[float] = None  # shoulder_width / hip_width
    arm_torso: Optional[float] = None  # arm_length / torso_height


def compute_scale_invariant_ratios(
    keypoints: Any,
    min_confidence: float = 0.3
) -> SkeletonRatios:
    """Compute scale-invariant skeleton ratios from COCO pose keypoints.

    COCO pose keypoint indices:
    0: nose, 1-2: eyes, 3-4: ears
    5-6: shoulders (L, R), 7-8: elbows, 9-10: wrists
    11-12: hips (L, R), 13-14: knees, 15-16: ankles

    Args:
        keypoints: Array or list of keypoints, shape (17, 3) with [x, y, conf]
        min_confidence: Minimum confidence threshold for using a keypoint

    Returns:
        SkeletonRatios with computed ratios (None if keypoints insufficient)
    """
    if not NUMPY_AVAILABLE or keypoints is None:
        return SkeletonRatios()

    try:
        # Convert to numpy array if needed
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)

        if len(keypoints) < 17 or keypoints.shape[-1] < 3:
            return SkeletonRatios()

        # Extract keypoints with confidence check
        def get_kp(idx):
            if idx < len(keypoints) and keypoints[idx][2] > min_confidence:
                return keypoints[idx][:2]
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
        # Children have larger heads relative to shoulders
        if nose is not None and l_shoulder is not None and r_shoulder is not None:
            shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)
            shoulder_mid = (l_shoulder + r_shoulder) / 2
            head_height = np.linalg.norm(nose - shoulder_mid)
            if shoulder_width > 1e-6:
                ratios.head_shoulder = float(head_height / shoulder_width)

        # Ratio 2: leg_length / torso_height
        # Children have shorter legs relative to torso
        if l_shoulder is not None and r_shoulder is not None and l_hip is not None and r_hip is not None:
            shoulder_mid = (l_shoulder + r_shoulder) / 2
            hip_mid = (l_hip + r_hip) / 2
            torso_height = np.linalg.norm(shoulder_mid - hip_mid)

            # Use whichever leg is more visible (more keypoints)
            left_leg_visible = sum(x is not None for x in [l_hip, l_knee, l_ankle])
            right_leg_visible = sum(x is not None for x in [r_hip, r_knee, r_ankle])

            leg_length = None
            if left_leg_visible >= 2 and l_hip is not None:
                if l_knee is not None and l_ankle is not None:
                    leg_length = (np.linalg.norm(l_hip - l_knee) +
                                 np.linalg.norm(l_knee - l_ankle))
                elif l_knee is not None:
                    leg_length = np.linalg.norm(l_hip - l_knee) * 2  # Approximate full leg
            elif right_leg_visible >= 2 and r_hip is not None:
                if r_knee is not None and r_ankle is not None:
                    leg_length = (np.linalg.norm(r_hip - r_knee) +
                                 np.linalg.norm(r_knee - r_ankle))
                elif r_knee is not None:
                    leg_length = np.linalg.norm(r_hip - r_knee) * 2

            if leg_length is not None and torso_height > 1e-6:
                ratios.leg_torso = float(leg_length / torso_height)

        # Ratio 3: shoulder_width / hip_width
        # Children have more similar shoulder and hip widths
        if l_shoulder is not None and r_shoulder is not None and l_hip is not None and r_hip is not None:
            shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)
            hip_width = np.linalg.norm(r_hip - l_hip)
            if hip_width > 1e-6:
                ratios.shoulder_hip = float(shoulder_width / hip_width)

        # Ratio 4: arm_length / torso_height
        # Children have shorter arms relative to torso
        if l_shoulder is not None and r_shoulder is not None and l_hip is not None and r_hip is not None:
            shoulder_mid = (l_shoulder + r_shoulder) / 2
            hip_mid = (l_hip + r_hip) / 2
            torso_height = np.linalg.norm(shoulder_mid - hip_mid)

            # Use whichever arm is more visible
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


def ratio_to_child_score(ratios: SkeletonRatios) -> Optional[float]:
    """Map skeleton ratios to child-likelihood score.

    Based on anthropometric research:
    - Children have larger heads relative to body (head_shoulder: 0.8-1.2 vs 0.5-0.7 adults)
    - Children have shorter legs relative to torso (leg_torso: 1.2-1.8 vs 2.0-2.5 adults)
    - Children have more similar shoulder/hip widths (shoulder_hip: 0.9-1.1 vs 0.7-0.9 adults)
    - Children have shorter arms relative to torso (arm_torso: 1.0-1.5 vs 1.8-2.2 adults)

    Returns:
        Float in [0, 1] representing child likelihood, or None if insufficient data
    """
    scores = []

    # Head-shoulder ratio (higher = more child-like)
    if ratios.head_shoulder is not None:
        hs = ratios.head_shoulder
        if hs > 1.0:
            score = 0.9  # Very child-like
        elif hs > 0.8:
            score = 0.5 + (hs - 0.8) * 2.0  # Linear 0.8->0.5, 1.0->0.9
        elif hs > 0.6:
            score = 0.2 + (hs - 0.6) * 1.5  # Linear 0.6->0.2, 0.8->0.5
        else:
            score = 0.1  # Very adult-like
        scores.append(score)

    # Leg-torso ratio (lower = more child-like)
    if ratios.leg_torso is not None:
        lt = ratios.leg_torso
        if lt < 1.5:
            score = 0.9  # Very child-like
        elif lt < 2.0:
            score = 0.9 - (lt - 1.5) * 0.8  # Linear 1.5->0.9, 2.0->0.5
        elif lt < 2.5:
            score = 0.5 - (lt - 2.0) * 0.6  # Linear 2.0->0.5, 2.5->0.2
        else:
            score = 0.1  # Very adult-like
        scores.append(score)

    # Shoulder-hip ratio (closer to 1.0 = more child-like)
    if ratios.shoulder_hip is not None:
        sh = ratios.shoulder_hip
        deviation_from_1 = abs(sh - 1.0)
        if deviation_from_1 < 0.1:
            score = 0.9  # Very child-like
        elif deviation_from_1 < 0.2:
            score = 0.7  # Likely child
        elif deviation_from_1 < 0.3:
            score = 0.4  # Uncertain
        else:
            score = 0.2  # Likely adult
        scores.append(score)

    # Arm-torso ratio (lower = more child-like)
    if ratios.arm_torso is not None:
        at = ratios.arm_torso
        if at < 1.3:
            score = 0.9  # Very child-like
        elif at < 1.8:
            score = 0.9 - (at - 1.3) * 0.8  # Linear 1.3->0.9, 1.8->0.5
        elif at < 2.2:
            score = 0.5 - (at - 1.8) * 0.75  # Linear 1.8->0.5, 2.2->0.2
        else:
            score = 0.1  # Very adult-like
        scores.append(score)

    if not scores:
        return None

    # Return weighted average (could be refined with learned weights)
    return float(sum(scores) / len(scores))


def aggregate_skeleton_ratios_over_track(
    keypoints_list: List[Any],
    min_confidence: float = 0.3
) -> Optional[float]:
    """Aggregate skeleton ratios across all frames in a track.

    Computes scale-invariant ratios for each frame, then takes the median
    of each ratio type before computing the final child score.

    Args:
        keypoints_list: List of keypoint arrays for each frame
        min_confidence: Minimum confidence for keypoint visibility

    Returns:
        Aggregated child probability from skeleton ratios, or None if insufficient data
    """
    if not keypoints_list:
        return None

    # Collect ratios from all frames
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

    # Compute median ratios (robust to outliers/noise)
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

    # Map median ratios to child score
    return ratio_to_child_score(median_ratios)


# --------------------------- Main Identifier ---------------------------


class SingleChildIdentifier:
    """Identify and merge the single child across fragmented tracker IDs.

    This scaffold wires together:
    - tracklet splitting (currently: 1 tracklet per Track)
    - per-tracklet evidence (age & skeleton placeholders)
    - node scoring (childness × duration)
    - edge scoring (temporal adjacency; intra-ID bonus)
    - simple DAG DP to pick a best non-overlapping path
    """

    def __init__(self, cfg: ChildIdentificationConfig, annotations: AnnotationInfo):
        self.cfg = cfg
        self.ann = annotations
        self._siglip_model = None
        self._siglip_processor = None

        # Load SigLIP model if using SigLIP method
        if self.cfg.age_estimation_method == "siglip":
            self._load_siglip_model()

    # -------- Model Loading --------

    def _load_siglip_model(self):
        """Load SigLIP model and processor for age classification."""
        if not SIGLIP_AVAILABLE:
            return

        try:
            self._siglip_model = SiglipForImageClassification.from_pretrained(self.cfg.siglip_model_name)
            self._siglip_processor = AutoImageProcessor.from_pretrained(self.cfg.siglip_model_name)
        except Exception:
            # Model loading failed, will fall back to flags in prediction
            self._siglip_model = None
            self._siglip_processor = None

    # -------- Orchestration --------

    def identify_child(self, tracks: List[Track]) -> ChildResult:
        tracklets = self._split_into_tracklets(tracks)
        nodes = [self._score_node(tl) for tl in tracklets]
        edges = self._build_edges(nodes)
        path_indices = self._select_best_path(nodes, edges)
        path_segments = [nodes[i].tracklet for i in path_indices]
        confidence = self._estimate_confidence([nodes[i] for i in path_indices])
        result = ChildResult(
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
        return result

    # -------- Tracklet handling --------

    def _split_into_tracklets(self, tracks: List[Track]) -> List[Tracklet]:
        """For now, treat each Track as a single Tracklet.

        Later: split on gaps, low-quality spans, or co-presence events.
        """
        tracklets: List[Tracklet] = []
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

    def _compute_age_prob(self, tl: Tracklet, flags: List[str]) -> Optional[float]:
        """Compute age-based child probability for a tracklet."""
        if self.cfg.age_estimation_method == "siglip":
            return self._compute_age_prob_siglip(tl, flags)
        elif self.cfg.age_estimation_method == "deepface":
            return self._compute_age_prob_deepface(tl, flags)
        else:
            flags.append(f"unknown_age_method_{self.cfg.age_estimation_method}")
            return None

    def _compute_age_prob_siglip(self, tl: Tracklet, flags: List[str]) -> Optional[float]:
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
                # Body visibility filtering parameters
                enable_body_visibility_filter=self.cfg.enable_body_visibility_filter,
                min_visible_keypoints=self.cfg.min_visible_keypoints,
                keypoint_conf_threshold=self.cfg.keypoint_conf_threshold,
                enable_roi_size_filter=self.cfg.enable_roi_size_filter,
                min_roi_width=self.cfg.min_roi_width,
                min_roi_height=self.cfg.min_roi_height,
                body_keypoint_indices=self.cfg.body_keypoint_indices,
                # Smart sampling parameters
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

    def _compute_age_prob_deepface(self, tl: Tracklet, flags: List[str]) -> Optional[float]:
        """Compute age-based child probability using DeepFace (original implementation)."""
        # Try pre-extracted face crops first
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

        # Try extracting faces from bounding boxes
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
                # Body visibility filtering parameters
                enable_body_visibility_filter=self.cfg.enable_body_visibility_filter,
                min_visible_keypoints=self.cfg.min_visible_keypoints,
                keypoint_conf_threshold=self.cfg.keypoint_conf_threshold,
                enable_roi_size_filter=self.cfg.enable_roi_size_filter,
                min_roi_width=self.cfg.min_roi_width,
                min_roi_height=self.cfg.min_roi_height,
                body_keypoint_indices=self.cfg.body_keypoint_indices,
                # Smart sampling parameters
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

    def _compute_skeleton_prob(self, tl: Tracklet, flags: List[str]) -> Optional[float]:
        """Compute skeleton-based child probability for a tracklet.

        Uses scale-invariant ratios from keypoints:
        - head_height / shoulder_width
        - leg_length / torso_height
        - shoulder_width / hip_width
        - arm_length / torso_height

        Returns median-aggregated child probability across frames.
        """
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

    def _compute_evidence(self, tl: Tracklet) -> Evidence:
        """Compute evidence for child identification.

        Stage 0 (Age-only): DeepFace/SigLIP age estimation
        Stage 1 (Skeleton): Scale-invariant skeleton ratios

        - Evenly sample up to sampling_percentage frames (capped by
          sampling_max_frames_per_track) from face_crops.
        - Run age estimation and take the median.
        - Compute skeleton ratios if keypoints available.
        - Map to child probability via empirical mappings.

        Future: quality gating, calibration with priors, batching/caching.
        """
        flags: List[str] = []

        # Age evidence
        p_age: Optional[float] = None
        p_age = self._compute_age_prob(tl, flags)

        # Skeleton evidence (Stage 1)
        p_skel: Optional[float] = None
        if self.cfg.enable_skeleton_ratios:
            p_skel = self._compute_skeleton_prob(tl, flags)
        else:
            flags.append("skeleton_disabled")

        return Evidence(p_age=p_age, p_skeleton=p_skel, flags=flags)

    def _score_node(self, tl: Tracklet) -> NodeScore:
        ev = self._compute_evidence(tl)

        w_age = self.cfg.w_age_default
        w_skel = self.cfg.w_skel_default

        # If some signals missing, renormalize available weights
        weights_sum = 0.0
        score_sum = 0.0

        if ev.p_age is not None:
            score_sum += w_age * float(ev.p_age)
            weights_sum += w_age
        if ev.p_skeleton is not None:
            score_sum += w_skel * float(ev.p_skeleton)
            weights_sum += w_skel

        score = (score_sum / weights_sum) if weights_sum > 0 else 0.0
        weight = score * tl.duration_seconds()

        return NodeScore(tracklet=tl, score=score, weight=weight, evidence=ev)

    # -------- Edge building & scoring --------

    def _compute_age_inconsistency_penalty(
        self,
        src_node: NodeScore,
        dst_node: NodeScore
    ) -> float:
        """Penalize edges connecting nodes with inconsistent age evidence.

        Returns penalty in [0, 1] where:
        - 0 = no penalty (both have similar/missing evidence)
        - 1 = maximum penalty (high confidence in one, none/opposite in other)
        """
        src_age = src_node.evidence.p_age
        dst_age = dst_node.evidence.p_age

        # Case 1: Both missing → no penalty (can't judge)
        if src_age is None and dst_age is None:
            return 0.0

        # Case 2: One missing → penalize if the other is confident
        if src_age is None:
            return abs(dst_age - 0.5) if dst_age is not None else 0.0
        if dst_age is None:
            return abs(src_age - 0.5)

        # Case 3: Both present → penalize large differences
        diff = abs(src_age - dst_age)
        if diff < self.cfg.age_inconsistency_threshold:
            return 0.0  # Similar enough

        # Scale penalty: larger difference = higher penalty
        return min(1.0, diff / (1.0 - self.cfg.age_inconsistency_threshold))

    def _build_edges(self, nodes: List[NodeScore]) -> List[EdgeScore]:
        """Build candidate edges between non-overlapping, time-adjacent nodes.

        Minimal scaffold: edge exists if dst starts after src ends and within
        continuity_gap_seconds; score is temporal adjacency plus optional same-ID
        bonus; other terms (spatial, pose, age) are placeholders.
        """
        edges: List[EdgeScore] = []
        if not nodes:
            return edges

        # Sort by start time to build a DAG in-time
        order = sorted(range(len(nodes)), key=lambda i: nodes[i].tracklet.start_frame)
        for i_idx in range(len(order)):
            i = order[i_idx]
            src = nodes[i].tracklet
            src_end = src.end_frame
            for j_idx in range(i_idx + 1, len(order)):
                j = order[j_idx]
                dst = nodes[j].tracklet
                if dst.start_frame <= src_end:
                    # overlapping or touching; skip for now (resolve via node selection)
                    continue
                gap_frames = dst.start_frame - src_end
                gap_sec = float(gap_frames) / float(dst.fps or 1.0)
                if gap_sec > self.cfg.continuity_gap_seconds:
                    # too far; since list is sorted, further dst will also be too far
                    break

                # Temporal adjacency term (0..1)
                temporal = max(0.0, 1.0 - (gap_sec / max(1e-6, self.cfg.continuity_gap_seconds)))

                # Same-ID bonus with exponential decay by gap (Not relevant if IDs are distinct)
                bonus = 0.0
                reasons: Dict[str, float] = {"temporal": temporal}
                if src.id == dst.id:
                    bonus = self.cfg.intra_id_gamma * math.exp(-gap_sec / max(1e-6, self.cfg.intra_id_tau))
                    reasons["same_id_bonus"] = bonus

                # Age inconsistency penalty
                age_penalty = self._compute_age_inconsistency_penalty(nodes[i], nodes[j])
                weighted_penalty = age_penalty * self.cfg.age_inconsistency_penalty_weight
                if age_penalty > 0:
                    reasons["age_inconsistency_penalty"] = age_penalty

                score = temporal + bonus - weighted_penalty
                edges.append(EdgeScore(src_index=i, dst_index=j, score=score, reasons=reasons))

        return edges

    # -------- Path selection --------

    def _select_best_path(self, nodes: List[NodeScore], edges: List[EdgeScore]) -> List[int]:
        """Simple longest-path DP on a DAG defined by time ordering.

        This is a minimal scaffold: we use node.weight and edge.score. We do not
        yet enforce strict non-overlap beyond chronological ordering and gap
        checks performed during edge building.
        """
        n = len(nodes)
        if n == 0:
            return []

        # Build adjacency from edges
        adj: Dict[int, List[Tuple[int, float]]] = {}
        for e in edges:
            adj.setdefault(e.src_index, []).append((e.dst_index, e.score))

        # Topological order by start frame
        order = sorted(range(n), key=lambda i: nodes[i].tracklet.start_frame)

        # DP arrays
        best_score: Dict[int, float] = {i: nodes[i].weight for i in range(n)}
        parent: Dict[int, Optional[int]] = {i: None for i in range(n)}

        for i in order:
            for (j, e_score) in adj.get(i, []):
                candidate = best_score[i] + nodes[j].weight + e_score
                # Optional: small hysteresis/switch epsilon can be applied later
                if candidate > best_score.get(j, float("-inf")):
                    best_score[j] = candidate
                    parent[j] = i

        # Select best ending node
        end_idx = max(best_score, key=lambda k: best_score[k])

        # Reconstruct path
        path: List[int] = []
        cur: Optional[int] = end_idx
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    # -------- Confidence (placeholder) --------

    def _estimate_confidence(self, path_nodes: List[NodeScore]) -> float:
        if not path_nodes:
            return 0.0
        # Placeholder: average node score clipped to [0,1]
        avg = sum(n.score for n in path_nodes) / float(len(path_nodes))
        return float(max(0.0, min(1.0, avg)))


# --------------------------- Public API ---------------------------


def identify_single_child(
    tracks: List[Track],
    annotations: AnnotationInfo,
    cfg: Optional[ChildIdentificationConfig] = None,
) -> ChildResult:
    """Identify the single child across fragmented tracker IDs.

    This is a lightweight scaffold that wires together minimal pieces of the
    post-tracking identification pipeline and returns a merged timeline for the
    child in single-child videos. It trusts per-ID temporal continuity but does
    not assume each ID is a distinct person.

    Parameters
    - tracks: Completed per-ID timelines with frame indices, fps, and optional
      per-frame data (keypoints, bboxes, face crops). Very short tracks are
      currently ignored based on cfg.min_track_frames.
    - annotations: Minimal annotation priors (e.g., Age_in_months and optional
      quality flags) used to adapt future evidence weighting.
    - cfg: Optional configuration. If None, sensible defaults are used.

    Current behavior (Stage 0-1)
    - Splits: one Tracklet per Track (no intra-ID splitting yet).
    - Evidence:
      * Age via DeepFace/SigLIP on sampled frames
      * Skeleton via scale-invariant ratios (head/shoulder, leg/torso,
        shoulder/hip, arm/torso) with median aggregation
    - Node scoring: score = childness × duration (childness from available
      signals; missing signals are ignored via weight renormalization).
    - Edge scoring: temporal adjacency term plus a decaying same-ID bonus; no
      spatial/pose/penalty terms yet.
    - Selection: simple DAG dynamic program over node weights + edge scores to
      choose a non-overlapping path in time.

    Returns
    - ChildResult containing the ordered sequence of selected tracklet IDs,
      the corresponding segments, a coarse confidence estimate, and basic
      diagnostics (scored nodes/edges and selected path indices).

    Notes
    - Future steps will add age sampling/mapping, skeleton ratios, spatial/pose
      continuity, penalties/hysteresis, and tracklet splitting heuristics.
    """
    identifier = SingleChildIdentifier(cfg or ChildIdentificationConfig(), annotations)
    return identifier.identify_child(tracks)

# --------------------------- Age Utility ---------------------------

def _sigmoid(x: float, tau: float) -> float:
    """Simple logistic with temperature (p = 1 / (1 + exp(-x/tau)))."""
    t = max(1e-6, float(tau))
    return 1.0 / (1.0 + math.exp(-float(x) / t))

def deepface_predict_age(image: Any, face_conf_threshold: float) -> Tuple[Optional[float], List[str]]:
    """Run DeepFace age on a single image and return (age, flags)."""
    if DeepFace is None:
        return None, ["deepface_not_available"]
    try:
        res = DeepFace.analyze(image, actions=["age"], enforce_detection=False)
        age_val: Optional[float] = None
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

def siglip_predict_child_prob(image: Any, model=None, processor=None) -> Tuple[Optional[float], List[str]]:
    """Run SigLIP age group classification and return (child_probability, flags).

    Uses the direct method: returns the "Child 0-12" probability directly.

    Args:
        image: Input image (numpy array or PIL Image)
        model: Pre-loaded SigLIP model
        processor: Pre-loaded SigLIP processor

    Returns:
        (child_probability, flags) where child_probability is the "Child 0-12" probability
    """
    if not SIGLIP_AVAILABLE:
        return None, ["siglip_not_available"]

    try:
        if model is None or processor is None:
            return None, ["siglip_model_not_loaded"]

        # Convert to PIL Image if needed
        if hasattr(image, 'shape'):  # numpy array
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert RGB numpy array to PIL
                image = Image.fromarray(image).convert("RGB")
            else:
                return None, ["siglip_invalid_image_format"]

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

        # Return "Child 0-12" probability directly (index 0)
        child_probability = float(probs[0])

        return child_probability, []
    except Exception as e:
        return None, [f"siglip_error_{type(e).__name__}"]

def _evenly_spaced_indices(n: int, k: int) -> List[int]:
    """Return k indices evenly covering [0, n-1]."""
    if k <= 1:
        return [max(0, (n - 1) // 2)] if n > 0 else []
    if n <= 0:
        return []
    step = (n - 1) / float(k - 1)
    return sorted({int(round(i * step)) for i in range(k)})

def _smart_frame_selection(
    keypoints_list: List[Any],
    k: int,
    min_pose_confidence: float = 0.7,
    body_keypoint_indices: Optional[List[int]] = None
) -> List[int]:
    """Select k frames with highest pose confidence scores for specified body keypoints."""
    if not keypoints_list:
        return []

    # Default to upper body keypoints if not specified
    if body_keypoint_indices is None:
        body_keypoint_indices = list(range(len(keypoints_list[0]) if keypoints_list and keypoints_list[0] else 17))

    # Calculate pose confidence for each frame
    frame_confidences = []
    for i, keypoints in enumerate(keypoints_list):
        if keypoints is None or len(keypoints) == 0:
            frame_confidences.append((i, 0.0))
            continue

        # Calculate average confidence of specified body keypoints
        confidences = []
        for idx in body_keypoint_indices:
            if idx < len(keypoints) and len(keypoints[idx]) >= 3:
                if keypoints[idx][2] > 0.1:  # Basic visibility threshold
                    confidences.append(float(keypoints[idx][2]))

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        frame_confidences.append((i, avg_confidence))

    # Filter by minimum confidence and sort by confidence
    valid_frames = [
        (i, conf) for i, conf in frame_confidences
        if conf >= min_pose_confidence
    ]

    # If not enough high-confidence frames, fall back to all frames
    if len(valid_frames) < k:
        valid_frames = frame_confidences

    # Sort by confidence (descending) and take top k
    valid_frames.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [i for i, _ in valid_frames[:k]]

    return sorted(selected_indices)

def median_age_from_face_crops(
    face_crops: List[Any],
    face_conf_threshold: float = 0.9,
    sampling_percentage: float = 0.25,
    sampling_max: int = 40,
) -> Tuple[Optional[float], List[str]]:
    """Sample face crops, run DeepFace age, and return median age.

    Returns (median_age_years, flags). Flags indicate missing DeepFace, errors,
    or empty results.
    """
    n = len(face_crops or [])
    if n == 0:
        return None, ["no_face_crop"]
    k_by_perc = int(math.ceil(max(0.0, float(sampling_percentage)) * n))
    k_target = min(max(1, k_by_perc), max(1, int(sampling_max)), n)
    indices = _evenly_spaced_indices(n, k_target)
    ages: List[float] = []
    flags: List[str] = []
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
    # median to handle outliets/skew - can be improved with weighting by conf or more 
    m = ages[len(ages) // 2] if (len(ages) % 2 == 1) else (
        0.5 * (ages[len(ages)//2 - 1] + ages[len(ages)//2])
    )
    return float(m), flags


# --------------------------- Bbox-based Age Estimation ---------------------------


def load_video_frame(video_path: str, frame_number: int) -> Optional[Any]:
    """Load a specific frame from video file."""
    if cv2 is None:
        return None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)  # Convert 1-based to 0-based
        ret, frame = cap.read()
        cap.release()

        return frame if ret else None
    except Exception:
        return None

def crop_bbox_from_frame(frame: Any, bbox: Tuple[float, float, float, float]) -> Optional[Any]:
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
    keypoints: List[Any],
    min_visible_keypoints: int = 3,
    keypoint_conf_threshold: float = 0.3,
    enable_roi_size_filter: bool = False,
    min_roi_width: int = 30,
    min_roi_height: int = 50,
    body_keypoint_indices: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Test if there's a valid body inside the given bounding box.

    Args:
        person_roi: The cropped region of interest (ROI) representing the person.
        keypoints: List of keypoints where each keypoint is [x, y, confidence].
        min_visible_keypoints: Minimum number of visible keypoints required.
        keypoint_conf_threshold: Minimum confidence threshold for keypoints.
        min_roi_width: Minimum width for ROI to be considered valid.
        min_roi_height: Minimum height for ROI to be considered valid.
        body_keypoint_indices: List of keypoint indices to check (default: upper body).

    Returns:
        dict: Result with detection status, reason (if failed), and additional details if successful.
    """
    try:
        # Size validation (only if enabled)
        if enable_roi_size_filter:
            if person_roi.shape[0] < min_roi_height or person_roi.shape[1] < min_roi_width:
                return {
                    'detected': False,
                    'reason': 'Body region too small',
                    'roi_size': person_roi.shape
                }

        # Define indices for important upper body keypoints if not provided
        # (shoulders, hips, knees, ankles, nose, eyes, ears)
        if body_keypoint_indices is None:
            body_keypoint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Count visible keypoints with confidence > threshold
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

        # Count all visible keypoints (not just body_keypoints)
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
    frame_numbers: List[int],
    bboxes: List[Tuple[float, float, float, float]],
    keypoints: Optional[List[Any]] = None,
    face_conf_threshold: float = 0.8,
    sampling_percentage: float = 0.25,
    sampling_max: int = 40,
    # Smart sampling parameters
    sampling_mode: str = "even",
    min_pose_confidence: float = 0.7,
    # Body visibility filtering parameters
    enable_body_visibility_filter: bool = False,
    min_visible_keypoints: int = 3,
    keypoint_conf_threshold: float = 0.3,
    enable_roi_size_filter: bool = False,
    min_roi_width: int = 30,
    min_roi_height: int = 50,
    body_keypoint_indices: Optional[List[int]] = None,
) -> Tuple[Optional[float], List[str]]:
    """Extract bbox crops from video frames and estimate median age using DeepFace.

    Args:
        video_path: Path to source video
        frame_numbers: List of frame indices corresponding to bboxes
        bboxes: List of bounding boxes as (x1, y1, x2, y2)
        keypoints: Unused (kept for compatibility)
        face_conf_threshold: Minimum face confidence for DeepFace
        sampling_percentage: Percentage of frames to sample
        sampling_max: Maximum number of frames to sample

    Returns:
        (median_age_years, flags) tuple
    """
    flags: List[str] = []

    if cv2 is None:
        return None, ["cv2_not_available"]

    if not video_path or not frame_numbers or not bboxes:
        return None, ["missing_input_data"]

    # Sample frames for age estimation
    n = len(frame_numbers)
    k_by_perc = int(math.ceil(max(0.0, float(sampling_percentage)) * n))
    k_target = min(max(1, k_by_perc), max(1, int(sampling_max)), n)

    # Use smart sampling if enabled and keypoints are available
    if sampling_mode == "smart" and keypoints is not None:
        indices = _smart_frame_selection(keypoints, k_target, min_pose_confidence, body_keypoint_indices)
    else:
        indices = _evenly_spaced_indices(n, k_target)

    ages: List[float] = []

    for idx in indices:
        if idx >= len(frame_numbers) or idx >= len(bboxes):
            continue

        frame_num = frame_numbers[idx]
        bbox = bboxes[idx]

        # Load frame
        frame = load_video_frame(video_path, frame_num)
        if frame is None:
            flags.append(f"frame_load_failed_{frame_num}")
            continue

        try:
            bbox_crop = crop_bbox_from_frame(frame, bbox)
        except Exception:
            flags.append(f"bbox_crop_error_{frame_num}")
            continue

        # Check body visibility if enabled
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

        # Estimate age using DeepFace on bbox crop
        age, local_flags = deepface_predict_age(bbox_crop, face_conf_threshold)
        if local_flags:
            flags.extend([f"{flag}_{frame_num}" for flag in local_flags])

        if age is not None:
            ages.append(float(age))

    if not ages:
        if not flags:
            flags.append("no_age_estimates")
        return None, flags

    # Calculate median age
    ages.sort()
    if len(ages) % 2 == 1:
        median_age = ages[len(ages) // 2]
    else:
        median_age = 0.5 * (ages[len(ages)//2 - 1] + ages[len(ages)//2])

    return float(median_age), flags

def median_child_prob_from_bboxes_siglip(
    video_path: str,
    frame_numbers: List[int],
    bboxes: List[Tuple[float, float, float, float]],
    keypoints: Optional[List[Any]] = None,
    model=None,
    processor=None,
    sampling_percentage: float = 0.25,
    sampling_max: int = 40,
    # Smart sampling parameters
    sampling_mode: str = "even",
    min_pose_confidence: float = 0.7,
    # Body visibility filtering parameters
    enable_body_visibility_filter: bool = False,
    min_visible_keypoints: int = 3,
    keypoint_conf_threshold: float = 0.3,
    enable_roi_size_filter: bool = False,
    min_roi_width: int = 30,
    min_roi_height: int = 50,
    body_keypoint_indices: Optional[List[int]] = None,
) -> Tuple[Optional[float], List[str]]:
    """Extract bbox crops from video frames and estimate median child probability using SigLIP.

    Args:
        video_path: Path to source video
        frame_numbers: List of frame indices corresponding to bboxes
        bboxes: List of bounding boxes as (x1, y1, x2, y2)
        keypoints: Unused (kept for compatibility)
        model: Pre-loaded SigLIP model
        processor: Pre-loaded SigLIP processor
        sampling_percentage: Percentage of frames to sample
        sampling_max: Maximum number of frames to sample

    Returns:
        (median_child_probability, flags) tuple
    """
    flags: List[str] = []

    if cv2 is None:
        return None, ["cv2_not_available"]

    if not SIGLIP_AVAILABLE:
        return None, ["siglip_not_available"]

    if not video_path or not frame_numbers or not bboxes:
        return None, ["missing_input_data"]

    # Sample frames for child probability estimation
    n = len(frame_numbers)
    k_by_perc = int(math.ceil(max(0.0, float(sampling_percentage)) * n))
    k_target = min(max(1, k_by_perc), max(1, int(sampling_max)), n)

    # Use smart sampling if enabled and keypoints are available
    if sampling_mode == "smart" and keypoints is not None:
        indices = _smart_frame_selection(keypoints, k_target, min_pose_confidence, body_keypoint_indices)
    else:
        indices = _evenly_spaced_indices(n, k_target)

    child_probs: List[float] = []

    for idx in indices:
        if idx >= len(frame_numbers) or idx >= len(bboxes):
            continue

        frame_num = frame_numbers[idx]
        bbox = bboxes[idx]

        # Load frame
        frame = load_video_frame(video_path, frame_num)
        if frame is None:
            flags.append(f"frame_load_failed_{frame_num}")
            continue

        try:
            bbox_crop = crop_bbox_from_frame(frame, bbox)
        except Exception:
            flags.append(f"bbox_crop_error_{frame_num}")
            continue

        # Check body visibility if enabled
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

        # Estimate child probability using SigLIP on bbox crop
        child_prob, local_flags = siglip_predict_child_prob(bbox_crop, model, processor)
        if local_flags:
            flags.extend([f"{flag}_{frame_num}" for flag in local_flags])

        if child_prob is not None:
            child_probs.append(float(child_prob))

    if not child_probs:
        if not flags:
            flags.append("no_child_prob_estimates")
        return None, flags

    # Calculate median child probability
    child_probs.sort()
    if len(child_probs) % 2 == 1:
        median_child_prob = child_probs[len(child_probs) // 2]
    else:
        median_child_prob = 0.5 * (child_probs[len(child_probs)//2 - 1] + child_probs[len(child_probs)//2])

    return float(median_child_prob), flags
