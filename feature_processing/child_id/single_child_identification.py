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

    # Continuity / merging
    continuity_gap_seconds: float = 1.0
    intra_id_gamma: float = 0.2  # base same-ID bonus scale
    intra_id_tau: float = 0.75   # decay for same-ID bonus by gap (seconds)
    switch_epsilon: float = 0.05 # minimal gain to switch near boundaries


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

    def _compute_evidence(self, tl: Tracklet) -> Evidence:
        """Stage 0: age-only evidence via DeepFace on sampled frames.

        - Evenly sample up to sampling_percentage frames (capped by
          sampling_max_frames_per_track) from face_crops.
        - Run DeepFace age and take the median age.
        - Map to child probability via a simple logistic.

        Future: quality gating, calibration with priors, batching/caching.
        """
        flags: List[str] = []

        # Age via DeepFace (multi-sample median)
        p_age: Optional[float] = None
        p_age = self._compute_age_prob(tl, flags)

        # Skeleton evidence not implemented in Stage 0
        p_skel: Optional[float] = None
        flags.append("skeleton_unimplemented")

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

                score = temporal + bonus
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

    Current behavior (Stage 0)
    - Splits: one Tracklet per Track (no intra-ID splitting yet).
    - Evidence: age via DeepFace on a single middle face crop; skeleton
      unimplemented (flagged).
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
