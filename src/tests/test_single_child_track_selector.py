"""
Tests for single_child_track_selector.

Run with:
    poetry run pytest src/tests/test_single_child_track_selector.py
"""

from __future__ import annotations

import numpy as np
import pytest
import h5py
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers to build lightweight stand-ins for the imported domain types.
# We mock at the module boundary so tests don't need the real
# single_child_identification package to be importable.
# ---------------------------------------------------------------------------

def _make_track(
    track_id: int = 0,
    start_frame: int = 0,
    end_frame: int = 10,
    fps: float = 30.0,
    bboxes: Optional[List] = None,
    keypoints: Optional[List] = None,
    frame_numbers: Optional[List[int]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> MagicMock:
    track = MagicMock()
    track.id = track_id
    track.start_frame = start_frame
    track.end_frame = end_frame
    track.fps = fps
    track.bboxes = bboxes
    track.keypoints = keypoints or []
    track.frame_numbers = frame_numbers or list(range(start_frame, end_frame + 1))
    track.meta = meta or {}
    track.face_crops = None
    track.video_path = None
    return track


def _make_tracklet(
    parent_id: int = 0,
    start_frame: int = 0,
    end_frame: int = 10,
    fps: float = 30.0,
) -> MagicMock:
    tl = MagicMock()
    tl.parent_id = parent_id
    tl.start_frame = start_frame
    tl.end_frame = end_frame
    tl.duration_frames.return_value = end_frame - start_frame
    tl.duration_seconds.return_value = (end_frame - start_frame) / fps
    return tl


def _make_node(score: float = 0.5, weight: float = 1.0, evidence: Any = None) -> MagicMock:
    node = MagicMock()
    node.score = score
    node.weight = weight
    node.evidence = evidence
    return node


def _make_evidence(
    p_age: float = 0.8,
    p_skeleton: float = 0.7,
    p_rigidity: float = 0.6,
    flags: Optional[List[str]] = None,
) -> MagicMock:
    ev = MagicMock()
    ev.p_age = p_age
    ev.p_skeleton = p_skeleton
    ev.p_rigidity = p_rigidity
    ev.flags = flags or []
    return ev


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_imports(monkeypatch):
    """
    Patch single_child_identification imports so the selector module can be
    imported without the real package being installed.
    """
    fake_module = MagicMock()
    fake_module.AnnotationInfo = MagicMock(return_value=MagicMock())
    fake_module.ChildIdentificationConfig = MagicMock(return_value=MagicMock())
    fake_module.Evidence = MagicMock(return_value=_make_evidence())
    fake_module.Track = MagicMock
    fake_module.Tracklet = MagicMock
    fake_module.NodeScore = MagicMock
    fake_module.SigLipModel = MagicMock
    fake_module.SingleChildIdentifier = MagicMock()

    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    import sys
    sys.modules.setdefault(
        "sailsprep.id_tracking_model.target_id.child_id.single_child_identification",
        fake_module,
    )
    return fake_module


@pytest.fixture()
def tmp_h5_track(tmp_path: Path) -> Path:
    """Write a minimal valid track_0001.h5 file and return its path."""
    h5_path = tmp_path / "track_0001.h5"
    with h5py.File(str(h5_path), "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["start_frame"] = 0
        meta.attrs["end_frame"] = 4
        meta.attrs["video_fps"] = 25.0
        meta.attrs["num_frames"] = 5
        meta.attrs["track_id"] = 1
        meta.attrs["video_width"] = 1920
        meta.attrs["video_height"] = 1080

        frames = f.create_group("frames")
        for i in range(5):
            fg = frames.create_group(f"frame_{i:06d}")
            fg.create_dataset("bbox", data=np.array([10.0, 20.0, 100.0, 200.0]))
            fg.create_dataset(
                "keypoints",
                data=np.zeros((17, 3), dtype=np.float32),
            )
    return h5_path


@pytest.fixture()
def tmp_h5_track_no_track_id(tmp_path: Path) -> Path:
    """HDF5 file whose metadata lacks a track_id — falls back to filename."""
    h5_path = tmp_path / "track_0007.h5"
    with h5py.File(str(h5_path), "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["start_frame"] = 0
        meta.attrs["end_frame"] = 2
        meta.attrs["video_fps"] = 30.0
        meta.attrs["num_frames"] = 3

        frames = f.create_group("frames")
        for i in range(3):
            fg = frames.create_group(f"frame_{i:06d}")
            fg.create_dataset("bbox", data=np.array([5.0, 5.0, 50.0, 50.0]))
    return h5_path


@pytest.fixture()
def tmp_h5_missing_bbox(tmp_path: Path) -> Path:
    """HDF5 track where one frame has no bbox → bboxes should be None."""
    h5_path = tmp_path / "track_0002.h5"
    with h5py.File(str(h5_path), "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["start_frame"] = 0
        meta.attrs["end_frame"] = 2
        meta.attrs["video_fps"] = 30.0
        meta.attrs["num_frames"] = 3
        meta.attrs["track_id"] = 2

        frames = f.create_group("frames")
        # frame 0 has bbox, frame 1 does not, frame 2 has bbox
        fg0 = frames.create_group("frame_000000")
        fg0.create_dataset("bbox", data=np.array([1.0, 2.0, 3.0, 4.0]))
        frames.create_group("frame_000001")          # no bbox
        fg2 = frames.create_group("frame_000002")
        fg2.create_dataset("bbox", data=np.array([5.0, 6.0, 7.0, 8.0]))
    return h5_path


# ---------------------------------------------------------------------------
# Import the module under test (after mock_imports patches sys.modules)
# We import lazily inside each test via a local import so the fixture runs
# first, OR we use a module-level import guarded by the fixture via
# importlib.  The cleanest pattern for this project is a helper:
# ---------------------------------------------------------------------------

def _selector(mock_imports):  # noqa: ANN001
    """Return the selector module, importing it fresh after mocks are in place."""
    import importlib
    import sys

    mod_name = (
        "sailsprep.id_tracking_model.target_id.child_id.single_child_track_selector"
    )
    # Remove cached version so the patched imports take effect
    sys.modules.pop(mod_name, None)

    # Provide a minimal package hierarchy so relative imports resolve
    pkg_parts = mod_name.split(".")
    for i in range(1, len(pkg_parts)):
        pkg = ".".join(pkg_parts[:i])
        if pkg not in sys.modules:
            sys.modules[pkg] = MagicMock()

    # Point the relative import target at our fake module
    sci_name = (
        "sailsprep.id_tracking_model.target_id.child_id.single_child_identification"
    )
    sys.modules[sci_name] = mock_imports

    # Now load the real source file directly
    import importlib.util

    src = Path(__file__).parent.parent / (
        "sailsprep/id_tracking_model/target_id/child_id/single_child_track_selector.py"
    )
    spec = importlib.util.spec_from_file_location(mod_name, src)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ===========================================================================
# _load_frame_group
# ===========================================================================

class TestLoadFrameGroup:
    def test_returns_bbox_and_keypoints(self, tmp_h5_track: Path) -> None:
        from sailsprep.id_tracking_model.target_id.child_id import (
            single_child_track_selector as sel,
        )
        with h5py.File(str(tmp_h5_track), "r") as f:
            frame = f["frames"]["frame_000000"]
            result = sel._load_frame_group(frame)

        assert "bbox" in result
        assert result["bbox"] == (10.0, 20.0, 100.0, 200.0)
        assert "keypoints" in result
        assert len(result["keypoints"]) == 17

    def test_missing_bbox_not_in_result(self, tmp_h5_missing_bbox: Path) -> None:
        from sailsprep.id_tracking_model.target_id.child_id import (
            single_child_track_selector as sel,
        )
        with h5py.File(str(tmp_h5_missing_bbox), "r") as f:
            frame = f["frames"]["frame_000001"]
            result = sel._load_frame_group(frame)

        assert "bbox" not in result
        assert "keypoints" not in result


# ===========================================================================
# load_track_from_h5
# ===========================================================================

class TestLoadTrackFromH5:
    def test_basic_fields(self, tmp_h5_track: Path) -> None:
        from sailsprep.id_tracking_model.target_id.child_id import (
            single_child_track_selector as sel,
        )
        loaded = sel.load_track_from_h5(tmp_h5_track)
        track = loaded.track

        assert loaded.h5_path == tmp_h5_track
        assert track.id == 1
        assert track.start_frame == 0
        assert track.end_frame == 4
        assert track.fps == 25.0
        assert len(track.frame_numbers) == 5

    def test_bboxes_all_present(self, tmp_h5_track: Path) -> None:
        from sailsprep.id_tracking_model.target_id.child_id import (
            single_child_track_selector as sel,
        )
        track = sel.load_track_from_h5(tmp_h5_track).track
        assert track.bboxes is not None
        assert len(track.bboxes) == 5
        assert track.bboxes[0] == (10.0, 20.0, 100.0, 200.0)

    def test_bboxes_none_when_any_missing(self, tmp_h5_missing_bbox: Path) -> None:
        from sailsprep.id_tracking_model.target_id.child_id import (
            single_child_track_selector as sel,
        )
        track = sel.load_track_from_h5(tmp_h5_missing_bbox).track
        assert track.bboxes is None

    def test_track_id_fallback_from_filename(
        self, tmp_h5_track_no_track_id: Path
    ) -> None:
        from sailsprep.id_tracking_model.target_id.child_id import (
            single_child_track_selector as sel,
        )
        track = sel.load_track_from_h5(tmp_h5_track_no_track_id).track
        assert track.id == 7

    def test_video_path_stored(self, tmp_h5_track: Path, tmp_path: Path) -> None:
        from sailsprep.id_tracking_model.target_id.child_id import (
            single_child_track_selector as sel,
        )
        video = tmp_path / "video.mp4"
        track = sel.load_track_from_h5(tmp_h5_track, video_path=video).track
        assert track.video_path == str(video)

    def test_meta_contains_source_h5(self, tmp_h5_track: Path) -> None:
        from sailsprep.id_tracking_model.target_id.child_id import (
            single_child_track_selector as sel,
        )
        track = sel.load_track_from_h5(tmp_h5_track).track
        assert track.meta["source_h5"] == str(tmp_h5_track)
        assert track.meta["video_width"] == 1920
        assert track.meta["video_height"] == 1080


# ===========================================================================
# load_tracks_from_directory
# ===========================================================================

class TestLoadTracksFromDirectory:
    def test_loads_all_matching_files(self, tmp_path: Path) -> None:
        # Write two valid track files and one non-matching file
        for name in ("track_0001.h5", "track_0002.h5"):
            p = tmp_path / name
            with h5py.File(str(p), "w") as f:
                m = f.create_group("metadata")
                m.attrs["start_frame"] = 0
                m.attrs["end_frame"] = 1
                m.attrs["video_fps"] = 30.0
                m.attrs["num_frames"] = 2
                fr = f.create_group("frames")
                for i in range(2):
                    fr.create_group(f"frame_{i:06d}")
        (tmp_path / "other_file.h5").touch()

        from sailsprep.id_tracking_model.target_id.child_id import (
            single_child_track_selector as sel,
        )
        loaded = sel.load_tracks_from_directory(tmp_path)
        assert len(loaded) == 2
        ids = {lt.track.id for lt in loaded}
        assert ids == {1, 2}

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        from sailsprep.id_tracking_model.target_id.child_id import (
            single_child_track_selector as sel,
        )
        assert sel.load_tracks_from_directory(tmp_path) == []

    def test_corrupt_file_is_skipped(self, tmp_path: Path) -> None:
        bad = tmp_path / "track_0001.h5"
        bad.write_bytes(b"not an hdf5 file")

        good = tmp_path / "track_0002.h5"
        with h5py.File(str(good), "w") as f:
            m = f.create_group("metadata")
            m.attrs["start_frame"] = 0
            m.attrs["end_frame"] = 0
            m.attrs["video_fps"] = 30.0
            m.attrs["num_frames"] = 1
            f.create_group("frames")

        from sailsprep.id_tracking_model.target_id.child_id import (
            single_child_track_selector as sel,
        )
        loaded = sel.load_tracks_from_directory(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].track.id == 2


# ===========================================================================
# select_single_track
# ===========================================================================

class TestSelectSingleTrack:
    def _patch_identifier(self, mock_imports, tracklets, nodes):
        """Make SingleChildIdentifier return the given tracklets/nodes."""
        identifier = MagicMock()
        identifier._split_into_tracklets.return_value = tracklets
        identifier._score_node.side_effect = lambda tl: nodes[tracklets.index(tl)]
        mock_imports.SingleChildIdentifier.return_value = identifier
        mock_imports.AnnotationInfo.return_value = MagicMock()
        mock_imports.ChildIdentificationConfig.return_value = MagicMock()
        mock_imports.Evidence.return_value = _make_evidence()

    def test_empty_tracks_returns_none(self, mock_imports) -> None:
        sel = _selector(mock_imports)
        assert sel.select_single_track([]) is None

    def test_no_tracklets_returns_none(self, mock_imports) -> None:
        self._patch_identifier(mock_imports, [], [])
        sel = _selector(mock_imports)
        track = _make_track(track_id=0)
        assert sel.select_single_track([track]) is None

    def test_single_track_selected(self, mock_imports) -> None:
        track = _make_track(track_id=42, start_frame=0, end_frame=10)
        tl = _make_tracklet(parent_id=42, start_frame=0, end_frame=10)
        node = _make_node(score=0.9, weight=5.0)
        self._patch_identifier(mock_imports, [tl], [node])

        sel = _selector(mock_imports)
        result = sel.select_single_track([track])

        assert result is not None
        assert result.track.id == 42
        assert result.tracklet is tl
        assert result.node is node

    def test_best_track_chosen_by_weight_then_score(self, mock_imports) -> None:
        track_a = _make_track(track_id=1)
        track_b = _make_track(track_id=2)
        tl_a = _make_tracklet(parent_id=1)
        tl_b = _make_tracklet(parent_id=2)
        node_a = _make_node(score=0.95, weight=3.0)   # lower weight
        node_b = _make_node(score=0.50, weight=10.0)  # higher weight → wins
        self._patch_identifier(mock_imports, [tl_a, tl_b], [node_a, node_b])

        sel = _selector(mock_imports)
        result = sel.select_single_track([track_a, track_b])

        assert result is not None
        assert result.track.id == 2

    def test_weight_tie_broken_by_score(self, mock_imports) -> None:
        track_a = _make_track(track_id=1)
        track_b = _make_track(track_id=2)
        tl_a = _make_tracklet(parent_id=1)
        tl_b = _make_tracklet(parent_id=2)
        node_a = _make_node(score=0.4, weight=5.0)
        node_b = _make_node(score=0.9, weight=5.0)  # same weight, higher score → wins
        self._patch_identifier(mock_imports, [tl_a, tl_b], [node_a, node_b])

        sel = _selector(mock_imports)
        result = sel.select_single_track([track_a, track_b])

        assert result is not None
        assert result.track.id == 2

    def test_include_diagnostics_returns_tuple(self, mock_imports) -> None:
        track = _make_track(track_id=1)
        tl = _make_tracklet(parent_id=1)
        node = _make_node(score=0.8, weight=4.0, evidence=_make_evidence())
        self._patch_identifier(mock_imports, [tl], [node])

        sel = _selector(mock_imports)
        result = sel.select_single_track([track], include_diagnostics=True)

        assert isinstance(result, tuple)
        selection, diagnostics = result
        assert selection is not None
        assert len(diagnostics) == 1
        diag = diagnostics[0]
        assert diag["track_id"] == 1
        assert diag["score"] == 0.8
        assert diag["weight"] == 4.0
        assert "evidence" in diag

    def test_diagnostics_evidence_fields(self, mock_imports) -> None:
        track = _make_track(track_id=5)
        tl = _make_tracklet(parent_id=5)
        ev = _make_evidence(p_age=0.9, p_skeleton=0.8, p_rigidity=0.7, flags=["small"])
        node = _make_node(score=0.7, weight=2.0, evidence=ev)
        self._patch_identifier(mock_imports, [tl], [node])

        sel = _selector(mock_imports)
        _, diagnostics = sel.select_single_track([track], include_diagnostics=True)

        ev_diag = diagnostics[0]["evidence"]
        assert ev_diag["p_age"] == 0.9
        assert ev_diag["p_skeleton"] == 0.8
        assert ev_diag["p_rigidity"] == 0.7
        assert ev_diag["flags"] == ["small"]

    def test_diagnostics_one_entry_per_tracklet(self, mock_imports) -> None:
        tracks = [_make_track(track_id=i) for i in range(3)]
        tracklets = [_make_tracklet(parent_id=i) for i in range(3)]
        nodes = [_make_node(score=0.1 * i, weight=float(i)) for i in range(3)]
        self._patch_identifier(mock_imports, tracklets, nodes)

        sel = _selector(mock_imports)
        _, diagnostics = sel.select_single_track(tracks, include_diagnostics=True)

        assert len(diagnostics) == 3

    def test_orphan_tracklet_parent_id_returns_none(self, mock_imports) -> None:
        """If best tracklet's parent_id doesn't match any track, return None."""
        track = _make_track(track_id=1)
        tl = _make_tracklet(parent_id=999)  # no track with id 999
        node = _make_node(score=0.9, weight=5.0)
        self._patch_identifier(mock_imports, [tl], [node])

        sel = _selector(mock_imports)
        result = sel.select_single_track([track])
        assert result is None

    def test_orphan_tracklet_with_diagnostics_returns_none_and_diag(
        self, mock_imports
    ) -> None:
        track = _make_track(track_id=1)
        tl = _make_tracklet(parent_id=999)
        node = _make_node(score=0.9, weight=5.0)
        self._patch_identifier(mock_imports, [tl], [node])

        sel = _selector(mock_imports)
        result = sel.select_single_track([track], include_diagnostics=True)
        assert isinstance(result, tuple)
        selection, diagnostics = result
        assert selection is None
        assert len(diagnostics) == 1


# ===========================================================================
# select_from_directory
# ===========================================================================

class TestSelectFromDirectory:
    def _write_minimal_h5(self, path: Path, track_id: int) -> None:
        with h5py.File(str(path), "w") as f:
            m = f.create_group("metadata")
            m.attrs["start_frame"] = 0
            m.attrs["end_frame"] = 4
            m.attrs["video_fps"] = 30.0
            m.attrs["num_frames"] = 5
            m.attrs["track_id"] = track_id
            fr = f.create_group("frames")
            for i in range(5):
                fg = fr.create_group(f"frame_{i:06d}")
                fg.create_dataset("bbox", data=np.array([1.0, 2.0, 3.0, 4.0]))

    def test_empty_directory_returns_none(
        self, tmp_path: Path, mock_imports
    ) -> None:
        sel = _selector(mock_imports)
        assert sel.select_from_directory(tmp_path) is None

    def test_empty_directory_with_diagnostics_returns_none_empty_list(
        self, tmp_path: Path, mock_imports
    ) -> None:
        sel = _selector(mock_imports)
        result = sel.select_from_directory(tmp_path, include_diagnostics=True)
        assert isinstance(result, tuple)
        selection, diagnostics = result
        assert selection is None
        assert diagnostics == []

    def test_returns_selection_with_loaded_track_metadata(
        self, tmp_path: Path, mock_imports
    ) -> None:
        self._write_minimal_h5(tmp_path / "track_0001.h5", track_id=1)

        tl = _make_tracklet(parent_id=1)
        node = _make_node(score=0.8, weight=3.0)
        identifier = MagicMock()
        identifier._split_into_tracklets.return_value = [tl]
        identifier._score_node.return_value = node
        mock_imports.SingleChildIdentifier.return_value = identifier
        mock_imports.AnnotationInfo.return_value = MagicMock()
        mock_imports.ChildIdentificationConfig.return_value = MagicMock()
        mock_imports.Evidence.return_value = _make_evidence()

        sel = _selector(mock_imports)
        result = sel.select_from_directory(tmp_path)

        assert result is not None
        assert result.track.id == 1
        # The track returned should carry the h5 source path in meta
        assert "source_h5" in result.track.meta

    def test_with_diagnostics_returns_tuple(
        self, tmp_path: Path, mock_imports
    ) -> None:
        self._write_minimal_h5(tmp_path / "track_0001.h5", track_id=1)

        tl = _make_tracklet(parent_id=1)
        node = _make_node(score=0.7, weight=2.0, evidence=_make_evidence())
        identifier = MagicMock()
        identifier._split_into_tracklets.return_value = [tl]
        identifier._score_node.return_value = node
        mock_imports.SingleChildIdentifier.return_value = identifier
        mock_imports.AnnotationInfo.return_value = MagicMock()
        mock_imports.ChildIdentificationConfig.return_value = MagicMock()
        mock_imports.Evidence.return_value = _make_evidence()

        sel = _selector(mock_imports)
        result = sel.select_from_directory(tmp_path, include_diagnostics=True)

        assert isinstance(result, tuple)
        selection, diagnostics = result
        assert selection is not None
        assert isinstance(diagnostics, list)
        assert len(diagnostics) == 1

    def test_best_track_replaced_with_loaded_instance(
        self, tmp_path: Path, mock_imports
    ) -> None:
        """
        select_from_directory replaces the Track instance returned by
        select_single_track with the LoadedTrack version that carries file metadata.
        """
        self._write_minimal_h5(tmp_path / "track_0001.h5", track_id=1)
        self._write_minimal_h5(tmp_path / "track_0002.h5", track_id=2)

        tl1 = _make_tracklet(parent_id=1)
        tl2 = _make_tracklet(parent_id=2)
        node1 = _make_node(score=0.3, weight=1.0)
        node2 = _make_node(score=0.9, weight=9.0)  # track 2 wins
        identifier = MagicMock()
        identifier._split_into_tracklets.return_value = [tl1, tl2]
        identifier._score_node.side_effect = lambda tl: (
            node1 if tl is tl1 else node2
        )
        mock_imports.SingleChildIdentifier.return_value = identifier
        mock_imports.AnnotationInfo.return_value = MagicMock()
        mock_imports.ChildIdentificationConfig.return_value = MagicMock()
        mock_imports.Evidence.return_value = _make_evidence()

        sel = _selector(mock_imports)
        result = sel.select_from_directory(tmp_path)

        assert result is not None
        assert result.track.id == 2
        # meta must come from the real LoadedTrack (file metadata present)
        assert result.track.meta.get("source_h5") is not None