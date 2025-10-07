# Child Identification Feature Design

## Overview
Post-tracking child identification for single-child videos that is robust to camera motion, uses age and scale-invariant skeleton evidence, and merges fragmented tracks via global time-interval optimization guided by annotations.

## Design Principles
1. Single Child: exactly one child per video; others are distractors.
2. Post-Processing: decide after tracking finishes to use full temporal context and allow retroactive merging.
3. Camera-Robust Evidence: avoid absolute scales; prefer age estimates and skeleton ratios invariant to zoom and camera motion.
4. Global Merging: select and merge track intervals by maximizing “childness” coverage over time (weighted interval scheduling), not by trusting ID continuity.
5. Annotation-Guided: use Age_in_months and quality flags to adapt weights, thresholds, and sampling.
6. Uncertainty Reporting: surface when evidence is insufficient or conflicting.

## Workflow
1. Complete tracking: obtain track fragments with time spans [start, end], keypoints per frame, optional face crops, and confidences.
2. Per-track evidence extraction:
   - Age signal: sample up to N high-quality frames per track (face/pose confidence); run age estimator; aggregate to a robust predicted age and confidence.
   - Skeleton signal: compute scale-invariant ratios per frame (only when keypoints confident), aggregate to a per-track child-likelihood.
3. Childness scoring per track: combine AgeChildProb and SkelChildProb with adaptive weights based on annotations (quality and visibility).
4. Interval selection: run weighted interval scheduling to pick a non-overlapping set of track intervals that maximizes total childness coverage; resolve overlaps in favor of higher marginal weight. (Current code still uses a gap-limited path search; see Implementation Status.)
5. Merge selected intervals: concatenate in time order into a single child timeline; assign a consistent child ID; produce per-frame mask indicating child vs not-child.
6. Fallbacks: handle sparse evidence (age-insufficient or skeleton-insufficient), relax thresholds, and provide uncertainty flags.

## Evidence Signals

### Age Signal
- Frame sampling: choose frames with high face visibility and keypoint confidence; cap per track (e.g., 20–40 frames) and per second to spread samples.
- Estimation: use an age model (e.g., DeepFace age or configured alternative) on cropped faces; record prediction and model confidence.
- Aggregation: confidence-weighted median or trimmed mean age for robustness to outliers.
- Probability mapping: convert to child-likelihood using known child age prior A (months):
  - define an adult threshold T (e.g., 10–12 years), or use |pred_age − A| closeness;
  - p_age = sigmoid((T − pred_age_years)/tau) or p_age = sigmoid((delta_max − |pred_age − A|)/tau).

### Skeleton Signal (Scale-Invariant)
- Use ratios computed only when keypoints exceed a confidence threshold; normalize by shoulder width to remove scale:
  - R1 = head_diameter / shoulder_width
  - R2 = shoulder_width / torso_length
  - R3 = torso_length / leg_length
  - R4 = upper_arm_length / forearm_length
  - R5 = thigh_length / shank_length
- Map each ratio to a child-likelihood via band-pass rules or simple logistics; average across ratios for a frame, then take the temporal median for the track.

## Annotation-Driven Adaptation
- Age_in_months: sets the age prior and tightens p_age when close to known age.
- Video_Quality_* and Body_Parts_Visible: adapt weights (w_age↓ when face poor; w_skel↑ when body visible; vice versa).
- Angle_of_Body: downweight skeleton when profile/oblique reduces shoulder/torso reliability.
- Child_of_interest_clear: increase switching penalty in overlap resolution.
- #_adults: weak prior to penalize short, low-score intervals when many adults present (reduces false positives).

## Scoring and Selection
- Per-track score: score_i = w_age * p_age_i + w_skel * p_skel_i, with w_age + w_skel = 1 and adapted by annotations.
- Interval weight: weight_i = score_i * duration_i (seconds or frames).
- Selection: run weighted interval scheduling to select non-overlapping intervals maximizing sum(weight_i). If overlaps remain (imperfect tracking timestamps), keep the interval with higher marginal weight in the overlap window; optionally add a small continuity bonus for adjacent intervals separated by short gaps and spatial proximity.

## Per-ID Continuity + Cross-ID Merging
Operate at tracklet granularity but prefer staying within the same tracker ID when reasonable.

- Tracklets: split IDs into contiguous segments at natural boundaries (gaps > Δt, sustained low quality, big motion/scale jumps, sustained co-presence with another ID). Minimum tracklet length (e.g., 30–50 frames).
- Node scoring: compute childness per tracklet as above (age + skeleton with annotation-adaptive weights). Node weight = score × duration.
- Candidate edges: only forward in time within a merge gap Δt_merge (e.g., 0.5–1.0 s). For each A→B, compute transition features (temporal adjacency, spatial continuity, pose similarity, age consistency), plus an intra-ID bonus if A.id == B.id.
- Path objective: maximize sum(node weights + edge scores) subject to non-overlap (longest path on DAG / Viterbi-style DP). Resolve brief overlaps by higher marginal weight; forbid merges across sustained co-presence.

### Intra-ID Continuity Bonus
Bias toward keeping continuity within the same tracker ID while respecting evidence.

- Edge score form: S_edge(A→B) = wT·Temporal + wS·Spatial + wP·Pose + wA·Age + Bonus_sameID − Penalties
- Same-ID bonus:
  - Bonus_sameID = γ · g(Δt) · r_conf · h_consistency
    - γ: base bonus scale (preference strength for same-ID continuation)
    - g(Δt): time-gap decay, e.g., exp(−Δt/τ), τ ≈ 0.5–1.0 s
    - r_conf: reliability near the boundary (mean keypoint confidence in tails, ∈[0,1])
    - h_consistency: downweight if age/skeleton boundary mismatch (∈[0,1])
- Apply only when: minimal overlap, no sustained co-presence, normalized center distance ≤ threshold, boundary pose similarity above threshold, age signals compatible.
- Guardrails: clamp ≥ 0 and cap at γ_max; zero out on hard contradictions (long co-presence, very large jump, child/adult disagreement).
- Hysteresis: add a small switch penalty for cross-ID edges near boundaries so switching requires a marginal gain (see Penalties).

Recommended defaults:
- γ ≈ 0.1–0.3 (about 10–30% of a typical 1 s node weight)
- τ ≈ 0.5–1.0 s; overlap_frames ≤ 5; normalized center distance ≤ 3–4 shoulder widths; pose cosine ≥ 0.6; |ageA−ageB| ≤ 3–4 years.

### Edge Penalties (Σ P_i)
Non-negative costs subtracted on an edge to discourage implausible merges when evidence contradicts same-person continuation.

- Overlap/co-presence penalty:
  - Hard forbid if overlap_sec > t_hard (e.g., 0.5 s);
  - else P_overlap = λ_overlap · min(1, overlap_sec / o_ref)
- Long gap penalty:
  - gap = start_B − end_A; P_gap = λ_gap · sigmoid((gap − g0)/τg) or λ_gap · min(1, gap / g_ref)
- Spatial jump penalty:
  - d_norm = center_distance / shoulder_width; P_jump = λ_jump · relu(d_norm − d0) / d_scale
- Pose discontinuity penalty:
  - cos = cosine(normalized_pose_A, normalized_pose_B);
  - P_pose = λ_pose · relu(cos0 − cos) / (1 − cos0)
- Age inconsistency penalty:
  - Δage = |ageA − ageB|; P_age = λ_age · relu(Δage − a0) / a_scale;
  - hard block if one adult-like (p_age < 0.3) and the other child-like (p_age > 0.7)
- Low boundary quality penalty:
  - r_conf ∈ [0,1]; P_quality = λ_q · (1 − r_conf)
- Very short fragment penalty:
  - P_short = λ_short · relu(L_min − duration_B) / L_min
- Motion/Kalman mismatch penalty (if available):
  - IoU_pred at boundary; P_motion = λ_motion · relu(IoU0 − IoU_pred) / IoU0
- Third-party conflict penalty:
  - Another high-childness ID overlaps both sides near the boundary; P_conflict = λ_conflict (fixed or proportional to overlap)

Suggested defaults (units ≈ “seconds-worth” of node weight):
- Hard forbid overlap > 0.5 s; else λ_overlap ≈ 1.0
- λ_gap ≈ 0.3 per second beyond g0 = 0.5 s, τg = 0.5
- λ_jump ≈ 0.2 per shoulder-width beyond d0 = 3
- λ_pose ≈ 0.3 when cosine < 0.5 (linear ramp from cos0 = 0.6)
- λ_age ≈ 0.2 per year beyond a0 = 4 years; hard block on child/adult disagreement
- λ_q ≈ 0.2 with r_conf from [0,1]
- λ_short ≈ 0.4 when duration < 30 frames (decays to 0 at 30)
- λ_motion ≈ 0.2 with IoU0 = 0.4
- Switch penalty (cross-ID near boundary): ε_switch ≈ 0.05 of a typical 1 s node weight
## Fallbacks and Uncertainty
- Age-insufficient: if face samples < M or low confidence, downweight p_age and flag.
- Skeleton-insufficient: if too few confident keypoint frames, downweight p_skel and flag.
- No clear winner: if top two solutions differ by < ε total weight, mark child-uncertain and emit both candidates with ranks and scores.
- Hard fallback: if both signals weak, choose the lowest predicted age track or the smallest skeleton height track as a last resort, but flag low confidence.

## Configuration
```python
@dataclass
class ChildIdentificationConfig:
    age_detection_model: str = "default"
    sampling_percentage: float = 0.25       # of frames per track, max cap also applied
    sampling_max_frames_per_track: int = 40
    min_track_frames: int = 30              # ignore very short tracks for robust stats
    keypoint_conf_threshold: float = 0.3
    age_tau: float = 2.5                    # sigmoid temperature for age mapping
    age_child_years_threshold: float = 10.0 # years considered child vs adult when no A
    w_age_default: float = 0.5
    w_skel_default: float = 0.5
    overlap_switch_epsilon: float = 0.05    # minimal gain to switch intervals in overlap
    continuity_gap_seconds: float = 1.0     # optional continuity bonus window
    prefer_high_quality_frames: bool = True
```

## Integration Points
```python
class ChildIdentificationModule:
    def __init__(self, cfg: ChildIdentificationConfig, annotation_row: dict):
        self.cfg = cfg
        self.ann = annotation_row  # contains Age_in_months, Video_Quality_*, etc.
        self.age_detector = load_age_detection_model(cfg.age_detection_model)

    def process_tracks(self, tracks: List[Track]) -> ChildResult:
        feats = [self._compute_evidence(t) for t in tracks]
        selected = weighted_interval_scheduling(feats, key=lambda f: f.weight)
        merged = merge_intervals(selected, overlap_eps=self.cfg.overlap_switch_epsilon)
        return ChildResult(merged_timeline=merged.timeline, child_track_id=merged.child_id,
                           confidence=merged.confidence, uncertainty=merged.uncertainty,
                           details=selected)
```

## Pseudocode
```python
tracks = get_completed_tracks(video_id)
for t in tracks:
    age_stats = sample_and_estimate_age(t, cfg, annotations)
    p_age    = child_prob_from_age(age_stats, annotations.Age_in_months, cfg)
    p_skel   = child_prob_from_skeleton(t.keypoints, cfg)
    w_age, w_skel = adapt_weights_from_annotations(annotations, cfg)
    score    = w_age * p_age + w_skel * p_skel
    weight   = score * duration(t)
    feats.append(Interval(start=t.start, end=t.end, score=score, weight=weight, track=t))

selected = weighted_interval_scheduling(feats)
child_timeline = resolve_overlaps_and_merge(selected, eps=cfg.overlap_switch_epsilon)
```

## Outputs
- Merged child timeline with consistent child ID and per-frame mask.
- Per-interval diagnostics: p_age, p_skel, weights, reasons for downweighting.
- Uncertainty flags and ranked alternatives when applicable.
- Export JSON/Parquet for downstream analysis and notebook joins.

## Evaluation and Metrics
- Track coverage: % of video time assigned to child.
- Consistency: frequency of switches and overlap conflicts.
- Robustness: performance under low face/body quality subsets.
- Sanity checks: predicted age vs annotation age distributions.

## Implementation Status

### ✅ Implemented Features

#### Evidence Signals
- **Age estimation** (Stage 0):
  - ✅ DeepFace age estimation with median aggregation
  - ✅ SigLIP age classification (direct child probability)
  - ✅ Configurable age estimation method (`age_estimation_method`)
  - ✅ Sigmoid mapping: age → child probability
  - ✅ Body visibility filtering before age estimation
  - ✅ Smart frame sampling (highest pose confidence frames)
  - ✅ Even frame sampling (evenly spaced)

- **Skeleton ratios** (Stage 1):
  - ✅ Scale-invariant ratios:
    - `head_shoulder = head_height / shoulder_width`
    - `leg_torso = leg_length / torso_height`
    - `shoulder_hip = shoulder_width / hip_width`
    - `arm_torso = arm_length / torso_height`
  - ✅ Median aggregation across frames (robust to outliers)
  - ✅ Configurable enable/disable (`enable_skeleton_ratios`)
  - ✅ Confidence thresholding for keypoint reliability

#### Scoring & Selection
- ✅ Node scoring with adaptive weight normalization
- ✅ Node weight = score × duration
- ✅ Basic edge scoring with temporal adjacency
- ✅ Same-ID bonus with exponential decay
- ✅ **Age inconsistency penalty** (partial edge penalties):
  - Penalizes connecting high-confidence child node to no-evidence node
  - Penalizes large age probability differences between connected nodes
  - Configurable penalty weight and threshold
- ✅ DAG longest-path dynamic programming for path selection
- ✅ Gap-limited path search (respects `continuity_gap_seconds`)

#### Quality & Filtering
- ✅ Body visibility filter: min visible keypoints before age estimation
- ✅ ROI size filter: min bbox dimensions (optional)
- ✅ Configurable body keypoint indices
- ✅ Min track frames filter (`min_track_frames`)
- ✅ Frame sampling limits (percentage + max cap)

#### Diagnostics & Logging
- ✅ Per-node diagnostics (score, weight, evidence flags)
- ✅ Per-edge diagnostics (score breakdown by reason)
- ✅ Evidence flags (body filter failures, missing data)
- ✅ Selected path indices and confidence estimation

### ❌ Not Yet Implemented

#### Tracklet Splitting
- ❌ Split tracks at natural boundaries (gaps, low quality, motion jumps)
- ❌ Co-presence detection and tracklet splitting
- ❌ Minimum tracklet length enforcement (currently uses full tracks)

#### Complete Edge Penalties
Partially implemented (age inconsistency only). Still needed:
- ❌ Overlap/co-presence penalty
- ❌ Long gap penalty (beyond hard cutoff)
- ❌ Spatial jump penalty (position discontinuity)
- ❌ Pose discontinuity penalty (pose vector cosine)
- ❌ Low boundary quality penalty
- ❌ Very short fragment penalty
- ❌ Motion/Kalman mismatch penalty
- ❌ Third-party conflict penalty
- ❌ Switch penalty (cross-ID hysteresis)

#### Advanced Selection
- ❌ Weighted interval scheduling (currently uses simpler DAG DP)
- ❌ Overlap resolution with marginal weight comparison
- ❌ Multi-solution ranking (uncertainty when close competitors)

#### Annotation-Driven Adaptation
- ❌ Adapt weights based on `Video_Quality_*` flags
- ❌ Adapt weights based on `Body_Parts_Visible`
- ❌ Downweight skeleton on `Angle_of_Body` (profile/oblique)
- ❌ Increase switch penalty when `Child_of_interest_clear`
- ❌ Use `#_adults` prior for false positive reduction
- ❌ Use `Age_in_months` to tighten age prior

#### Multi-Person Handling
- ❌ Keypoint-based person masking (remove interfering people)
- ❌ Segmentation-based masking
- ❌ Bbox purity check (detect multi-person bboxes)
- ❌ Temporal coherence check (detect parent-holding patterns)
- ❌ Multi-person scene detection
- ❌ Conservative crop strategy (torso-focused)
- ❌ Age estimate outlier rejection (IQR filtering)

#### Fallbacks & Uncertainty
- ❌ Age-insufficient detection and downweighting
- ❌ Skeleton-insufficient detection and downweighting
- ❌ Multi-solution emission when uncertain
- ❌ Hard fallback: choose lowest age or smallest skeleton

#### Appearance Cues (Stage 2)
- ❌ Upper-body color histogram for tie-breaks

## Roadmap

### Immediate Priorities (Address Known Bugs)
1. **Multi-person masking** - Handle parent-holding scenarios where bboxes overlap
2. **Age estimate outlier rejection** - IQR-based filtering of bimodal distributions
3. **Complete edge penalties** - Implement remaining penalty terms

### Stage 0 (Complete)
- ✅ Age-only evidence + basic path selection
- ✅ Export diagnostics and uncertainty flags

### Stage 1 (Nearly Complete)
- ✅ Skeleton ratios with median aggregation
- ✅ Basic edge penalties (age inconsistency)
- ⚠️ **Need:** Tracklet splitting, complete edge penalties

### Stage 2 (Future)
- Annotation-driven weight adaptation
- Weighted interval scheduling
- Appearance cues (color histogram for tie-breaks)
- Multi-solution ranking for uncertainty

### Stage 3 (Future)
- Advanced continuity modeling (spatial + pose + motion)
- Kalman filter integration for motion prediction
- Third-party conflict detection
