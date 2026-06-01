"""
Utilities for selecting the primary child track from per-track HDF5 outputs.

This module reuses the scoring infrastructure from `single_child_identification`
to rank tracks without merging them, so we can pick the best candidate track in
single-child videos. It exposes helpers to load tracks from the tracking export
and a lightweight selection routine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py

from .single_child_identification import (
    AnnotationInfo,
    ChildIdentificationConfig,
    NodeScore,
    SingleChildIdentifier,
    Track,
    Tracklet,
    Evidence,
    SigLipModel
)


@dataclass
class LoadedTrack:
    """Container holding a track and the source HDF5 path it came from."""

    track: Track
    h5_path: Path


@dataclass
class SingleTrackSelection:
    """Lightweight summary of the chosen track."""

    track: Track
    tracklet: Tracklet
    node: NodeScore


def _load_frame_group(frame_group: h5py.Group) -> dict:
    """Extract bbox/keypoint arrays from a single frame group."""
    frame_data = {}

    if "bbox" in frame_group:
        bbox_arr = frame_group["bbox"][:]
        frame_data["bbox"] = tuple(float(x) for x in bbox_arr.tolist())

    if "keypoints" in frame_group:
        kp_arr = frame_group["keypoints"][:]
        frame_data["keypoints"] = kp_arr.tolist()

    return frame_data


def load_track_from_h5(h5_path: Path, video_path: Optional[Path] = None) -> LoadedTrack:
    """Load a single tracking HDF5 file into a `Track` dataclass."""
    with h5py.File(str(h5_path), "r") as f:
        metadata = f["metadata"]
        start_frame = int(metadata.attrs.get("start_frame", 0))
        end_frame = int(metadata.attrs.get("end_frame", start_frame))
        fps = float(metadata.attrs.get("video_fps", 0.0) or 0.0)
        num_frames_attr = int(metadata.attrs.get("num_frames", 0))
        track_id_attr = metadata.attrs.get("track_id")

        if track_id_attr is None:
            # Fallback to file name (e.g., track_0007.h5)
            stem = h5_path.stem
            try:
                track_id_attr = int(stem.split("_")[-1])
            except Exception:
                track_id_attr = -1

        frames = f["frames"]
        frame_entries = sorted(frames.keys())

        frame_numbers: List[int] = []
        bboxes: List[Optional[tuple]] = []
        keypoints: List[Optional[list]] = []

        for frame_key in frame_entries:
            try:
                frame_number = int(frame_key.split("_")[-1])
            except Exception:
                frame_number = len(frame_numbers)

            frame_numbers.append(frame_number)

            frame = frames[frame_key]
            frame_data = _load_frame_group(frame)

            bboxes.append(frame_data.get("bbox"))
            keypoints.append(frame_data.get("keypoints"))

        track = Track(
            id=int(track_id_attr),
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            keypoints=keypoints,
            bboxes=bboxes,
            face_crops=None,
            video_path=str(video_path) if video_path else None,
            frame_numbers=frame_numbers,
            meta={
                "num_frames": num_frames_attr or len(frame_numbers),
                "video_width": metadata.attrs.get("video_width"),
                "video_height": metadata.attrs.get("video_height"),
                "source_h5": str(h5_path),
            },
        )

    return LoadedTrack(track=track, h5_path=h5_path)


def load_tracks_from_directory(
    tracking_dir: Path, video_path: Optional[Path] = None
) -> List[LoadedTrack]:
    """Load all `track_*.h5` files from a tracking directory."""
    loaded_tracks: List[LoadedTrack] = []

    for h5_path in sorted(tracking_dir.glob("track_*.h5")):
        try:
            loaded_tracks.append(load_track_from_h5(h5_path, video_path=video_path))
        except Exception:
            # Skip unreadable tracks; caller can inspect logs if needed.
            continue

    return loaded_tracks


def select_single_track(
    tracks: Sequence[Track],
    annotations: Optional[AnnotationInfo] = None,
    cfg: Optional[ChildIdentificationConfig] = None,
    siglip_model: Optional[SigLipModel] = None,
    include_diagnostics: bool = False,
) -> Union[
    Optional[SingleTrackSelection],
    Tuple[Optional[SingleTrackSelection], List[Dict[str, Any]]]
]:
    """
    Rank tracks using the single-child identification scoring and return the best one.

    The selector keeps the highest-weighted node (score × duration), which balances
    child-likeness and temporal coverage.
    """
    if not tracks:
        return None

    annotations = annotations or AnnotationInfo()
    cfg = cfg or ChildIdentificationConfig()
    identifier = SingleChildIdentifier(cfg, annotations, siglip_model=siglip_model)
    track_by_id = {track.id: track for track in tracks}

    tracklets = identifier._split_into_tracklets(list(tracks))
    if not tracklets:
        return None

    nodes: List[NodeScore] = [identifier._score_node(tl) for tl in tracklets]
    if not nodes:
        return (None, []) if include_diagnostics else None

    diagnostics: List[Dict[str, Any]] = []
    for tracklet, node in zip(tracklets, nodes):
        evidence = node.evidence or Evidence()
        parent_track = track_by_id.get(tracklet.parent_id)
        track_meta = parent_track.meta if parent_track and parent_track.meta else {}
        diagnostics.append(
            {
                "track_id": tracklet.parent_id,
                "tracklet_start_frame": tracklet.start_frame,
                "tracklet_end_frame": tracklet.end_frame,
                "duration_frames": tracklet.duration_frames(),
                "duration_seconds": tracklet.duration_seconds(),
                "score": node.score,
                "weight": node.weight,
                "evidence": {
                    "p_age": evidence.p_age,
                    "p_skeleton": evidence.p_skeleton,
                    "p_rigidity": evidence.p_rigidity,
                    "flags": list(evidence.flags) if evidence.flags else [],
                },
                "track_meta": track_meta,
            }
        )

    best_index, best_node = max(
        enumerate(nodes),
        key=lambda item: (item[1].weight, item[1].score),
    )
    best_tracklet = tracklets[best_index]

    # Find original track - same id as tracklet.parent_id
    best_track = next(
        (tr for tr in tracks if tr.id == best_tracklet.parent_id), None
    )
    if best_track is None:
        return (None, diagnostics) if include_diagnostics else None

    selection = SingleTrackSelection(
        track=best_track,
        tracklet=best_tracklet,
        node=best_node,
    )

    if include_diagnostics:
        return selection, diagnostics

    return selection


def select_from_directory(
    tracking_dir: Path,
    video_path: Optional[Path] = None,
    annotations: Optional[AnnotationInfo] = None,
    cfg: Optional[ChildIdentificationConfig] = None,
    siglip_model: Optional[SigLipModel] = None,
    include_diagnostics: bool = False,
) -> Optional[SingleTrackSelection]:
    """
    Convenience wrapper: load tracks from a directory and return the best candidate.
    """
    loaded_tracks = load_tracks_from_directory(tracking_dir, video_path=video_path)
    if not loaded_tracks:
        return None

    selection_result = select_single_track(
        [loaded.track for loaded in loaded_tracks],
        annotations=annotations,
        cfg=cfg,
        include_diagnostics=include_diagnostics,
        siglip_model=siglip_model,
    )

    diagnostics: Optional[List[Dict[str, Any]]] = None
    if include_diagnostics:
        if selection_result is None:
            return None, []
        selection, diagnostics = selection_result
    else:
        selection = selection_result

    if selection is None:
        return (None, diagnostics or []) if include_diagnostics else None

    # Replace track instance with the one carrying metadata/path information
    for loaded in loaded_tracks:
        if loaded.track.id == selection.track.id:
            final_selection = SingleTrackSelection(
                track=loaded.track,
                tracklet=selection.tracklet,
                node=selection.node,
            )
            if include_diagnostics:
                return final_selection, diagnostics or []
            return final_selection

    return (None, diagnostics or []) if include_diagnostics else None
