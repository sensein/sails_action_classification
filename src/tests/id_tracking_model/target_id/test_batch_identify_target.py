"""
Tests for batch_identify_target.py

Run with:
    poetry run pytest src/tests/test_batch_identify_target.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


_mock_annotation_info = MagicMock(name="AnnotationInfo")
_mock_child_id_config = MagicMock(name="ChildIdentificationConfig")
_mock_child_id_config.return_value.age_estimation_method = "none"
_mock_child_id_config.return_value.enable_rigidity_detection = True
_mock_siglip = MagicMock(name="SigLipModel")
_mock_single_track_selection = MagicMock(name="SingleTrackSelection")
_mock_load_track = MagicMock(name="load_track_from_h5")
_mock_select_from_dir = MagicMock(name="select_from_directory")

_STUBBED_MODULE_NAMES = [
    "sailsprep",
    "sailsprep.id_tracking_model",
    "sailsprep.id_tracking_model.target_id",
    "sailsprep.id_tracking_model.target_id.child_id",
    "sailsprep.id_tracking_model.target_id.child_id.single_child_identification",
    "sailsprep.id_tracking_model.target_id.child_id.single_child_track_selector",
    "cv2",
    "h5py",
]

# Snapshot whatever was really in sys.modules (or None if nothing) before
# we overwrite it with stubs.
_real_modules: Dict[str, Any] = {
    name: sys.modules.get(name) for name in _STUBBED_MODULE_NAMES
}

for _name in [
    "sailsprep",
    "sailsprep.id_tracking_model",
    "sailsprep.id_tracking_model.target_id",
    "sailsprep.id_tracking_model.target_id.child_id",
]:
    sys.modules.setdefault(_name, MagicMock())

sys.modules["sailsprep.id_tracking_model.target_id.child_id.single_child_identification"] = MagicMock(
    AnnotationInfo=_mock_annotation_info,
    ChildIdentificationConfig=_mock_child_id_config,
    SigLipModel=_mock_siglip,
)
sys.modules["sailsprep.id_tracking_model.target_id.child_id.single_child_track_selector"] = MagicMock(
    SingleTrackSelection=_mock_single_track_selection,
    load_track_from_h5=_mock_load_track,
    select_from_directory=_mock_select_from_dir,
)
sys.modules.setdefault("cv2", MagicMock())
sys.modules.setdefault("h5py", MagicMock())

# ---------------------------------------------------------------------------
# Load the module under test directly from its file path so we bypass the
# sailsprep package hierarchy entirely (the intermediate nodes are MagicMocks
# and therefore not real packages — a regular import would fail with
# "not a package").
# ---------------------------------------------------------------------------
def _find_src_root(start: Path) -> Path:
    """Walk up from this file until we find the `src` directory."""
    for parent in start.parents:
        if parent.name == "src":
            return parent
    raise RuntimeError(f"Could not locate 'src' directory above {start}")


_SRC_ROOT = _find_src_root(Path(__file__))
_MODULE_PATH = (
    _SRC_ROOT
    / "sailsprep"
    / "id_tracking_model"
    / "target_id"
    / "batch_identify_target.py"
)
_spec = importlib.util.spec_from_file_location("batch_identify_target", str(_MODULE_PATH))
assert _spec is not None and _spec.loader is not None, (
    f"Could not build module spec from {_MODULE_PATH}"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["batch_identify_target"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

# Bind the public names we need in tests
EmbeddingProfile = _mod.EmbeddingProfile
TargetIdentifier = _mod.TargetIdentifier
TrackMatch = _mod.TrackMatch
_json_default = _mod._json_default
_sanitize_for_path = _mod._sanitize_for_path

# ---------------------------------------------------------------------------
# Restore the real modules now that batch_identify_target.py has finished
# importing, so our stubs don't leak into other test files collected later
# in the same pytest session.
# ---------------------------------------------------------------------------
for _name, _real in _real_modules.items():
    if _real is not None:
        sys.modules[_name] = _real
    else:
        sys.modules.pop(_name, None)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

MINIMAL_CSV_CONTENT = textwrap.dedent("""\
    ID,Coder,SourceFile,#_children,timepoint,Age
    CHILD01,CoderA,/videos/vid1.mp4,1,14_month,1.17
    CHILD01,CoderA,/videos/vid2.mp4,2,14_month,1.17
    CHILD02,CoderB,/videos/vid3.mp4,1,24_month,2.0
""")


@pytest.fixture()
def tmp_csv(tmp_path: Path) -> Path:
    p = tmp_path / "annotations.csv"
    p.write_text(MINIMAL_CSV_CONTENT)
    return p


@pytest.fixture()
def tmp_embeddings(tmp_path: Path) -> Path:
    d = tmp_path / "pipeline_outputs"
    d.mkdir()
    return d


@pytest.fixture()
def identifier(tmp_csv: Path, tmp_embeddings: Path, tmp_path: Path) -> TargetIdentifier:
    out = tmp_path / "output"
    return TargetIdentifier(
        csv_path=str(tmp_csv),
        embeddings_base_dir=str(tmp_embeddings),
        output_dir=str(out),
    )


def _make_profile(
    face: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    lower: Optional[np.ndarray] = None,
) -> EmbeddingProfile:
    return EmbeddingProfile(
        face_feature=face,
        upper_feature=upper,
        lower_feature=lower,
        num_observations=10,
        source_videos=["vid.mp4"],
    )


def _unit_vec(dim: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# _json_default
# ---------------------------------------------------------------------------

class TestJsonDefault:
    def test_numpy_scalar(self) -> None:
        assert _json_default(np.float32(3.14)) == pytest.approx(3.14, abs=1e-4)

    def test_numpy_int(self) -> None:
        assert _json_default(np.int64(7)) == 7

    def test_numpy_array(self) -> None:
        result = _json_default(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(TypeError):
            _json_default(object())


# ---------------------------------------------------------------------------
# _sanitize_for_path
# ---------------------------------------------------------------------------

class TestSanitizeForPath:
    def test_none_returns_unknown(self) -> None:
        assert _sanitize_for_path(None) == "unknown"

    def test_empty_string_returns_unknown(self) -> None:
        assert _sanitize_for_path("   ") == "unknown"

    def test_normal_string(self) -> None:
        assert _sanitize_for_path("14_month") == "14_month"

    def test_special_chars_replaced(self) -> None:
        result = _sanitize_for_path("hello world/foo")
        assert " " not in result
        assert "/" not in result

    def test_hyphens_preserved(self) -> None:
        result = _sanitize_for_path("child-001")
        assert "-" in result


# ---------------------------------------------------------------------------
# EmbeddingProfile
# ---------------------------------------------------------------------------

class TestEmbeddingProfile:
    def test_defaults(self) -> None:
        ep = EmbeddingProfile()
        assert ep.face_feature is None
        assert ep.upper_feature is None
        assert ep.lower_feature is None
        assert ep.num_observations == 0
        assert ep.source_videos == []

    def test_source_videos_not_shared(self) -> None:
        """Each instance must have its own list (field default_factory)."""
        a = EmbeddingProfile()
        b = EmbeddingProfile()
        a.source_videos.append("x")
        assert b.source_videos == []


# ---------------------------------------------------------------------------
# TrackMatch
# ---------------------------------------------------------------------------

class TestTrackMatch:
    def test_creation(self) -> None:
        tm = TrackMatch(
            video_id="v1",
            track_id=3,
            similarity_score=0.85,
            face_score=0.9,
            upper_score=0.8,
            lower_score=0.75,
            num_frames=120,
            start_frame=0,
            end_frame=119,
            confidence="high",
            timepoint="14_month",
            is_reference=False,
        )
        assert tm.confidence == "high"
        assert tm.is_reference is False

    def test_reference_flag(self) -> None:
        tm = TrackMatch("v", 1, 1.0, 1.0, 1.0, 1.0, 10, 0, 9, "high", is_reference=True)
        assert tm.is_reference is True


# ---------------------------------------------------------------------------
# TargetIdentifier.__init__
# ---------------------------------------------------------------------------

class TestTargetIdentifierInit:
    def test_loads_csv(self, identifier: TargetIdentifier) -> None:
        assert len(identifier.df) == 3

    def test_output_dir_created(self, identifier: TargetIdentifier) -> None:
        assert identifier.output_dir.exists()

    def test_filter_ids_stored(self, tmp_csv: Path, tmp_embeddings: Path, tmp_path: Path) -> None:
        ti = TargetIdentifier(
            str(tmp_csv),
            str(tmp_embeddings),
            str(tmp_path / "out"),
            filter_ids=["CHILD01"],
        )
        assert "CHILD01" in ti.filter_ids  # type: ignore[operator]

    def test_face_only_weights(self, tmp_csv: Path, tmp_embeddings: Path, tmp_path: Path) -> None:
        ti = TargetIdentifier(
            str(tmp_csv),
            str(tmp_embeddings),
            str(tmp_path / "out"),
            face_only=True,
        )
        assert ti.similarity_weights == {"face": 1.0, "upper": 0.0, "lower": 0.0}

    def test_default_weights(self, identifier: TargetIdentifier) -> None:
        assert identifier.similarity_weights["face"] == pytest.approx(0.5)
        assert identifier.similarity_weights["upper"] == pytest.approx(0.3)
        assert identifier.similarity_weights["lower"] == pytest.approx(0.2)

    def test_min_score_stored(self, tmp_csv: Path, tmp_embeddings: Path, tmp_path: Path) -> None:
        ti = TargetIdentifier(
            str(tmp_csv),
            str(tmp_embeddings),
            str(tmp_path / "out"),
            min_score=0.75,
        )
        assert ti.min_score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# _normalize_numeric_columns
# ---------------------------------------------------------------------------

class TestNormalizeNumericColumns:
    def test_children_column_becomes_integer(self, identifier: TargetIdentifier) -> None:
        assert pd.api.types.is_integer_dtype(identifier.df["#_children"])

    def test_age_column_numeric(self, identifier: TargetIdentifier) -> None:
        assert pd.api.types.is_numeric_dtype(identifier.df["Age"])


# ---------------------------------------------------------------------------
# _parse_timepoint_to_months
# ---------------------------------------------------------------------------

class TestParseTimepointToMonths:
    @pytest.mark.parametrize("tp,expected", [
        ("14_month", 14.0),
        ("36month", 36.0),
        ("2.5_months", 2.5),
        ("no_digits_here", None),
        ("", None),
    ])
    def test_parsing(self, tp: str, expected: Optional[float]) -> None:
        result = TargetIdentifier._parse_timepoint_to_months(tp)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _build_timepoint_index
# ---------------------------------------------------------------------------

class TestBuildTimepointIndex:
    def test_child_map_populated(self, identifier: TargetIdentifier) -> None:
        assert "CHILD01" in identifier.child_timepoint_months
        assert "CHILD02" in identifier.child_timepoint_months

    def test_global_map_populated(self, identifier: TargetIdentifier) -> None:
        assert len(identifier.global_timepoint_months) > 0

    def test_sorted_by_months(self, identifier: TargetIdentifier) -> None:
        months = [m for _, m in identifier.global_timepoint_months]
        assert months == sorted(months)

    def test_no_timepoint_column(self, tmp_embeddings: Path, tmp_path: Path) -> None:
        csv_path = tmp_path / "no_tp.csv"
        csv_path.write_text("ID,Coder,SourceFile,#_children\nC1,A,v.mp4,1\n")
        ti = TargetIdentifier(str(csv_path), str(tmp_embeddings), str(tmp_path / "out"))
        assert ti.child_timepoint_months == {}
        assert ti.global_timepoint_months == []


# ---------------------------------------------------------------------------
# _infer_timepoint_from_age
# ---------------------------------------------------------------------------

class TestInferTimepointFromAge:
    def test_snaps_to_nearest(self, identifier: TargetIdentifier) -> None:
        tp, months = identifier._infer_timepoint_from_age("CHILD01", 1.17)
        assert tp == "14_month"
        assert months == pytest.approx(14.04, abs=0.1)

    def test_invalid_age_returns_none(self, identifier: TargetIdentifier) -> None:
        tp, months = identifier._infer_timepoint_from_age("CHILD01", "not_a_number")
        assert tp is None
        assert months is None

    def test_none_age_returns_none(self, identifier: TargetIdentifier) -> None:
        tp, months = identifier._infer_timepoint_from_age("CHILD01", None)
        assert tp is None

    def test_nan_age_returns_none(self, identifier: TargetIdentifier) -> None:
        tp, months = identifier._infer_timepoint_from_age("CHILD01", float("nan"))
        assert tp is None

    def test_unknown_child_falls_back_to_global(self, identifier: TargetIdentifier) -> None:
        tp, _ = identifier._infer_timepoint_from_age("UNKNOWN_CHILD", 1.17)
        assert tp is not None


# ---------------------------------------------------------------------------
# _categorize_failure
# ---------------------------------------------------------------------------

class TestCategorizeFailure:
    @pytest.mark.parametrize("reason,expected", [
        (None, "unknown"),
        ("", "unknown"),
        ("no reference profile available", "reference_profile_missing"),
        ("timepoint missing in CSV", "timepoint_missing"),
        ("embeddings directory not found", "embeddings_missing"),
        ("no track files present", "no_track_files"),
        ("best score 0.3 below minimum threshold 0.5", "score_below_threshold"),
        ("could not select reference track in dir", "reference_selector_failed"),
        ("no solo videos found", "no_reference_videos"),
        ("some other weird reason", "other"),
    ])
    def test_categorization(
        self, identifier: TargetIdentifier, reason: Optional[str], expected: str
    ) -> None:
        assert identifier._categorize_failure(reason) == expected


# ---------------------------------------------------------------------------
# _record_match_outcome
# ---------------------------------------------------------------------------

class TestRecordMatchOutcome:
    def test_success_increments_correctly(self, identifier: TargetIdentifier) -> None:
        identifier._record_match_outcome("CHILD01", True, None)
        assert identifier.child_metrics["CHILD01"]["total_videos"] == 1
        assert identifier.child_metrics["CHILD01"]["successes"] == 1
        assert identifier.child_metrics["CHILD01"]["failures"] == 0
        assert identifier.global_metrics["successes"] == 1

    def test_failure_increments_correctly(self, identifier: TargetIdentifier) -> None:
        identifier._record_match_outcome("CHILD01", False, "embeddings directory not found")
        assert identifier.child_metrics["CHILD01"]["failures"] == 1
        assert identifier.child_metrics["CHILD01"]["failure_reasons"]["embeddings_missing"] == 1
        assert identifier.global_metrics["failures"] == 1

    def test_multiple_outcomes(self, identifier: TargetIdentifier) -> None:
        identifier._record_match_outcome("CHILD01", True, None)
        identifier._record_match_outcome("CHILD01", False, "timepoint missing in CSV")
        assert identifier.child_metrics["CHILD01"]["total_videos"] == 2
        assert identifier.global_metrics["total_videos"] == 2


# ---------------------------------------------------------------------------
# _format_failure_breakdown
# ---------------------------------------------------------------------------

class TestFormatFailureBreakdown:
    def test_empty_counter(self, identifier: TargetIdentifier) -> None:
        result = identifier._format_failure_breakdown(Counter(), 0)
        assert result == {}

    def test_correct_rate(self, identifier: TargetIdentifier) -> None:
        c: Counter[str] = Counter({"embeddings_missing": 2, "other": 1})
        result = identifier._format_failure_breakdown(c, 10)
        assert result["embeddings_missing"]["count"] == 2
        assert result["embeddings_missing"]["rate"] == pytest.approx(0.2)
        assert result["other"]["rate"] == pytest.approx(0.1)

    def test_zero_total_returns_empty(self, identifier: TargetIdentifier) -> None:
        c: Counter[str] = Counter({"x": 5})
        result = identifier._format_failure_breakdown(c, 0)
        assert result == {}


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self, identifier: TargetIdentifier) -> None:
        v = _unit_vec()
        assert identifier.cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self, identifier: TargetIdentifier) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert identifier.cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self, identifier: TargetIdentifier) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert identifier.cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-5)

    def test_none_returns_zero(self, identifier: TargetIdentifier) -> None:
        v = _unit_vec()
        assert identifier.cosine_similarity(None, v) == 0.0
        assert identifier.cosine_similarity(v, None) == 0.0
        assert identifier.cosine_similarity(None, None) == 0.0

    def test_zero_vector_returns_zero(self, identifier: TargetIdentifier) -> None:
        z = np.zeros(128)
        v = _unit_vec()
        assert identifier.cosine_similarity(z, v) == 0.0

    def test_returns_float(self, identifier: TargetIdentifier) -> None:
        v = _unit_vec()
        result = identifier.cosine_similarity(v, v)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_track_similarity
# ---------------------------------------------------------------------------

class TestComputeTrackSimilarity:
    def test_identical_profiles(self, identifier: TargetIdentifier) -> None:
        v = _unit_vec()
        prof = _make_profile(face=v, upper=v, lower=v)
        score, individual = identifier.compute_track_similarity(prof, prof)
        assert score == pytest.approx(1.0, abs=1e-5)
        assert individual["face"] == pytest.approx(1.0, abs=1e-5)

    def test_custom_weights_applied(self, identifier: TargetIdentifier) -> None:
        v = _unit_vec(seed=1)
        w = _unit_vec(seed=2)
        prof_a = _make_profile(face=v, upper=v, lower=v)
        prof_b = _make_profile(face=w, upper=w, lower=w)
        weights = {"face": 1.0, "upper": 0.0, "lower": 0.0}
        score, _ = identifier.compute_track_similarity(prof_a, prof_b, weights=weights)
        expected_face = identifier.cosine_similarity(v, w)
        assert score == pytest.approx(expected_face, abs=1e-5)

    def test_default_weights_sum_to_one(self, identifier: TargetIdentifier) -> None:
        w = identifier.similarity_weights
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-5)

    def test_missing_features_score_zero(self, identifier: TargetIdentifier) -> None:
        v = _unit_vec()
        prof_a = _make_profile(face=v)
        prof_b = _make_profile(face=v)
        score, individual = identifier.compute_track_similarity(prof_a, prof_b)
        assert individual["upper"] == 0.0
        assert individual["lower"] == 0.0

    def test_returns_tuple(self, identifier: TargetIdentifier) -> None:
        v = _unit_vec()
        prof = _make_profile(face=v)
        result = identifier.compute_track_similarity(prof, prof)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _video_basename
# ---------------------------------------------------------------------------

class TestVideoBasename:
    def test_format(self, identifier: TargetIdentifier) -> None:
        info: Dict[str, Any] = {
            "ID": "CHILD01",
            "Coder": "CoderA",
            "SourceFile": "/some/path/video_file.mp4",
        }
        assert identifier._video_basename(info) == "CHILD01_CoderA_video_file"

    def test_stem_only(self, identifier: TargetIdentifier) -> None:
        info: Dict[str, Any] = {
            "ID": "X",
            "Coder": "Y",
            "SourceFile": "plain.mp4",
        }
        assert identifier._video_basename(info) == "X_Y_plain"


# ---------------------------------------------------------------------------
# _get_embedding_path
# ---------------------------------------------------------------------------

class TestGetEmbeddingPath:
    def test_returns_last_candidate_when_none_exist(self, identifier: TargetIdentifier) -> None:
        info: Dict[str, Any] = {
            "ID": "CHILD01",
            "Coder": "CoderA",
            "SourceFile": "/videos/vid1.mp4",
        }
        path = identifier._get_embedding_path(info)
        assert "CHILD01_CoderA_vid1" in str(path)

    def test_returns_existing_candidate(self, identifier: TargetIdentifier) -> None:
        basename = "CHILD01_CoderA_vid1_tracking"
        emb_dir = identifier.embeddings_base_dir / "tracks_hdf5" / basename
        emb_dir.mkdir(parents=True)
        info: Dict[str, Any] = {
            "ID": "CHILD01",
            "Coder": "CoderA",
            "SourceFile": "/videos/vid1.mp4",
        }
        assert identifier._get_embedding_path(info) == emb_dir


# ---------------------------------------------------------------------------
# _resolve_track_path
# ---------------------------------------------------------------------------

class TestResolveTrackPath:
    def test_finds_zero_padded_file(self, identifier: TargetIdentifier, tmp_path: Path) -> None:
        tracking_dir = tmp_path / "tracking"
        tracking_dir.mkdir()
        (tracking_dir / "track_0003.h5").touch()
        result = identifier._resolve_track_path(tracking_dir, 3)
        assert result is not None
        assert result.name == "track_0003.h5"

    def test_returns_none_when_missing(self, identifier: TargetIdentifier, tmp_path: Path) -> None:
        tracking_dir = tmp_path / "empty_tracking"
        tracking_dir.mkdir()
        assert identifier._resolve_track_path(tracking_dir, 99) is None

    def test_finds_via_glob(self, identifier: TargetIdentifier, tmp_path: Path) -> None:
        tracking_dir = tmp_path / "tracking2"
        tracking_dir.mkdir()
        (tracking_dir / "track_0005_extra.h5").touch()
        result = identifier._resolve_track_path(tracking_dir, 5)
        assert result is not None


# ---------------------------------------------------------------------------
# _get_output_subdir
# ---------------------------------------------------------------------------

class TestGetOutputSubdir:
    def test_creates_dir(self, identifier: TargetIdentifier) -> None:
        subdir = identifier._get_output_subdir(identifier.output_dir, "CHILD01", "14_month")
        assert subdir.exists()

    def test_video_specific_subdir(self, identifier: TargetIdentifier) -> None:
        subdir = identifier._get_output_subdir(
            identifier.output_dir, "CHILD01", "14_month", "my_video"
        )
        assert subdir.exists()
        assert subdir.name == "my_video"


# ---------------------------------------------------------------------------
# identify_target_in_video
# ---------------------------------------------------------------------------

class TestIdentifyTargetInVideo:
    def _setup_profiles(self, identifier: TargetIdentifier) -> None:
        v = _unit_vec()
        identifier.reference_profiles["CHILD01"] = {
            "14_month": _make_profile(face=v, upper=v, lower=v)
        }

    def test_returns_none_when_no_reference_profile(self, identifier: TargetIdentifier) -> None:
        video_info: Dict[str, Any] = {
            "ID": "CHILD01", "Coder": "CoderA",
            "SourceFile": "/videos/vid2.mp4", "timepoint": "14_month", "Age": 1.17,
        }
        match, reason = identifier.identify_target_in_video("CHILD01", video_info)
        assert match is None
        assert reason is not None

    def test_returns_none_when_embeddings_dir_missing(self, identifier: TargetIdentifier) -> None:
        self._setup_profiles(identifier)
        video_info: Dict[str, Any] = {
            "ID": "CHILD01", "Coder": "CoderA",
            "SourceFile": "/videos/vid2.mp4", "timepoint": "14_month", "Age": 1.17,
        }
        match, reason = identifier.identify_target_in_video("CHILD01", video_info)
        assert match is None
        assert "embeddings" in (reason or "").lower()

    def test_returns_none_when_timepoint_and_age_missing(self, identifier: TargetIdentifier) -> None:
        self._setup_profiles(identifier)
        video_info: Dict[str, Any] = {
            "ID": "CHILD01", "Coder": "CoderA",
            "SourceFile": "/videos/vid2.mp4", "timepoint": None, "Age": None,
        }
        match, reason = identifier.identify_target_in_video("CHILD01", video_info)
        assert match is None
        assert reason is not None

    def test_uses_reference_track_when_available(self, identifier: TargetIdentifier) -> None:
        identifier.reference_videos["CHILD01"] = {"/videos/vid1.mp4": 3}
        video_info: Dict[str, Any] = {
            "ID": "CHILD01", "Coder": "CoderA",
            "SourceFile": "/videos/vid1.mp4", "timepoint": "14_month", "Age": 1.17,
        }
        mock_match = TrackMatch(
            "CHILD01_CoderA_vid1", 3, 1.0, 1.0, 1.0, 1.0, 10, 0, 9, "high",
            timepoint="14_month", is_reference=True,
        )
        with patch.object(identifier, "_create_match_from_track_id", return_value=mock_match):
            match, reason = identifier.identify_target_in_video("CHILD01", video_info)
        assert match is not None
        assert match.is_reference is True
        assert reason is None

    def test_score_below_threshold_returns_none(
        self, identifier: TargetIdentifier, tmp_path: Path
    ) -> None:
        self._setup_profiles(identifier)
        identifier.min_score = 0.99  # impossible to beat

        emb_dir = identifier.embeddings_base_dir / "tracks_hdf5" / "CHILD01_CoderA_vid2_tracking"
        emb_dir.mkdir(parents=True)
        (emb_dir / "track_0001.h5").touch()

        low_profile = _make_profile(face=_unit_vec(seed=99))

        with patch.object(identifier, "load_track_embeddings", return_value=low_profile):
            video_info: Dict[str, Any] = {
                "ID": "CHILD01", "Coder": "CoderA",
                "SourceFile": "/videos/vid2.mp4", "timepoint": "14_month", "Age": 1.17,
            }
            match, reason = identifier.identify_target_in_video("CHILD01", video_info)

        assert match is None
        assert "threshold" in (reason or "").lower()


# ---------------------------------------------------------------------------
# build_reference_profiles
# ---------------------------------------------------------------------------

class TestBuildReferenceProfiles:
    def test_no_embeddings_leaves_profile_empty(self, identifier: TargetIdentifier) -> None:
        with patch.object(identifier, "_build_profile_from_videos", return_value=None):
            identifier.build_reference_profiles("CHILD02")
        assert identifier.reference_profiles["CHILD02"] == {}

    def test_clears_previous_data(self, identifier: TargetIdentifier) -> None:
        identifier.reference_profiles["CHILD01"] = {"old_tp": _make_profile()}
        with patch.object(identifier, "_build_profile_from_videos", return_value=None):
            identifier.build_reference_profiles("CHILD01")
        assert "old_tp" not in identifier.reference_profiles["CHILD01"]

    def test_profile_stored_per_timepoint(self, identifier: TargetIdentifier) -> None:
        fake_profile = _make_profile(face=_unit_vec())
        with patch.object(identifier, "_build_profile_from_videos", return_value=fake_profile):
            identifier.build_reference_profiles("CHILD01")
        assert "14_month" in identifier.reference_profiles["CHILD01"]


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

class TestSaveResults:
    def test_json_file_created(self, identifier: TargetIdentifier) -> None:
        child_id = "CHILD01"
        identifier.match_results[child_id] = []
        identifier.child_metrics[child_id]["total_videos"] = 1
        identifier.child_metrics[child_id]["successes"] = 0
        identifier.child_metrics[child_id]["failures"] = 1
        identifier.save_results(child_id)
        summary_path = identifier.output_dir / "CHILD01" / "CHILD01_target_identification.json"
        assert summary_path.exists()

    def test_json_content_valid(self, identifier: TargetIdentifier) -> None:
        child_id = "CHILD01"
        tm = TrackMatch("v1", 1, 0.9, 0.9, 0.8, 0.7, 50, 0, 49, "high", "14_month")
        identifier.match_results[child_id] = [tm]
        identifier.child_metrics[child_id]["total_videos"] = 1
        identifier.child_metrics[child_id]["successes"] = 1
        identifier.child_metrics[child_id]["failures"] = 0
        identifier.save_results(child_id)
        summary_path = identifier.output_dir / "CHILD01" / "CHILD01_target_identification.json"
        data = json.loads(summary_path.read_text())
        assert data["child_id"] == "CHILD01"
        assert data["summary"]["matches_found"] == 1
        assert data["summary"]["success_rate"] == pytest.approx(1.0)

    def test_confidence_counts(self, identifier: TargetIdentifier) -> None:
        child_id = "CHILD01"
        matches = [
            TrackMatch("v1", 1, 0.9, 0.9, 0.8, 0.7, 50, 0, 49, "high", "14_month"),
            TrackMatch("v2", 2, 0.7, 0.7, 0.6, 0.5, 50, 0, 49, "medium", "14_month"),
            TrackMatch("v3", 3, 0.5, 0.5, 0.4, 0.3, 50, 0, 49, "low", "14_month"),
        ]
        identifier.match_results[child_id] = matches
        identifier.child_metrics[child_id]["total_videos"] = 3
        identifier.child_metrics[child_id]["successes"] = 3
        identifier.child_metrics[child_id]["failures"] = 0
        identifier.save_results(child_id)
        data = json.loads(
            (identifier.output_dir / "CHILD01" / "CHILD01_target_identification.json").read_text()
        )
        assert data["summary"]["high_confidence"] == 1
        assert data["summary"]["medium_confidence"] == 1
        assert data["summary"]["low_confidence"] == 1


# ---------------------------------------------------------------------------
# save_global_summary
# ---------------------------------------------------------------------------

class TestSaveGlobalSummary:
    def test_run_summary_created(self, identifier: TargetIdentifier) -> None:
        identifier.global_metrics["total_videos"] = 4
        identifier.global_metrics["successes"] = 3
        identifier.global_metrics["failures"] = 1
        identifier.global_metrics["children_processed"] = 1
        identifier.save_global_summary()
        assert (identifier.output_dir / "run_summary.json").exists()

    def test_run_summary_content(self, identifier: TargetIdentifier) -> None:
        identifier.global_metrics["total_videos"] = 4
        identifier.global_metrics["successes"] = 3
        identifier.global_metrics["failures"] = 1
        identifier.global_metrics["children_processed"] = 2
        identifier.child_metrics["CHILD01"]["total_videos"] = 2
        identifier.child_metrics["CHILD01"]["successes"] = 2
        identifier.child_metrics["CHILD01"]["failures"] = 0
        identifier.save_global_summary()
        data = json.loads((identifier.output_dir / "run_summary.json").read_text())
        assert data["success_rate"] == pytest.approx(0.75)
        assert data["children_processed"] == 2
        assert "CHILD01" in data["per_child"]


# ---------------------------------------------------------------------------
# process_all
# ---------------------------------------------------------------------------

class TestProcessAll:
    def test_filter_ids_restricts_processing(self, identifier: TargetIdentifier) -> None:
        identifier.filter_ids = {"CHILD02"}
        processed: List[str] = []

        def fake_process_child(child_id: str) -> None:
            processed.append(child_id)

        with patch.object(identifier, "process_child", side_effect=fake_process_child), \
            patch.object(identifier, "save_global_summary"):
            identifier.process_all()

        assert processed == ["CHILD02"]

    def test_all_children_processed_without_filter(self, identifier: TargetIdentifier) -> None:
        processed: List[str] = []

        def fake_process_child(child_id: str) -> None:
            processed.append(child_id)

        with patch.object(identifier, "process_child", side_effect=fake_process_child), \
            patch.object(identifier, "save_global_summary"):
            identifier.process_all()

        assert set(processed) == {"CHILD01", "CHILD02"}

    def test_exception_in_child_does_not_abort(self, identifier: TargetIdentifier) -> None:
        processed: List[str] = []

        def fake_process_child(child_id: str) -> None:
            if child_id == "CHILD01":
                raise RuntimeError("boom")
            processed.append(child_id)

        with patch.object(identifier, "process_child", side_effect=fake_process_child), \
            patch.object(identifier, "save_global_summary"):
            identifier.process_all()  # must not raise

        assert "CHILD02" in processed