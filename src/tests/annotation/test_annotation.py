"""Tests for the video annotation FastAPI app."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers – patch heavy app-level side-effects before importing the module
# ---------------------------------------------------------------------------

# Path constants used in patches throughout the tests
_CSV_FILE_PATH       = "sailsprep.annotation.annotation.CSV_FILE"
_OUTPUT_DIR_PATH     = "sailsprep.annotation.annotation.OUTPUT_DIR"
_ANNOTATION_DIR_PATH = "sailsprep.annotation.annotation.ANNOTATION_DIR"


# ---------------------------------------------------------------------------
# Import app AFTER patching static files
# ---------------------------------------------------------------------------
from sailsprep.annotation.annotation import (  # noqa: E402
    ALL_CATEGORIES,
    ACTION_CATEGORIES,
    CATEGORIES_WITH_REFERENCE,
    app,
    get_html_content,
    get_video_filename,
    is_empty_value,
    load_csv_data,
    update_status_in_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _make_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


SAMPLE_VIDEOS = [
    {"video": "sub-01/ses-01/video.mp4", "status": 0, "Locomotion": "Walking", "Repetitive_Motor_Movements": ""},
    {"video": "sub-02/ses-01/video.mp4", "status": 1, "Locomotion": "Running", "Repetitive_Motor_Movements": "Jumping"},
]

SAMPLE_CSV_BYTES = _make_csv_bytes(SAMPLE_VIDEOS)


def _mock_csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "video.csv"
    p.write_bytes(SAMPLE_CSV_BYTES)
    return p


# ---------------------------------------------------------------------------
# Unit tests – helper functions
# ---------------------------------------------------------------------------

class TestIsEmptyValue:
    def test_none(self) -> None:
        assert is_empty_value(None)

    def test_nan(self) -> None:
        assert is_empty_value(float("nan"))

    def test_empty_string(self) -> None:
        assert is_empty_value("")

    def test_whitespace(self) -> None:
        assert is_empty_value("   ")

    def test_nill(self) -> None:
        assert is_empty_value("nill")

    def test_na_string(self) -> None:
        assert is_empty_value("n/a")

    def test_none_string(self) -> None:
        assert is_empty_value("none")

    def test_valid_value(self) -> None:
        assert not is_empty_value("Walking")

    def test_zero(self) -> None:
        # 0 is not considered empty
        assert not is_empty_value(0)


class TestGetVideoFilename:
    def test_simple(self) -> None:
        assert get_video_filename("folder/file.mp4") == "file"

    def test_nested(self) -> None:
        assert get_video_filename("a/b/c/video.mkv") == "video"

    def test_no_extension(self) -> None:
        assert get_video_filename("video") == "video"


class TestGetHtmlContent:
    def test_returns_string(self) -> None:
        html = get_html_content()
        assert isinstance(html, str)
        assert "<html" in html
        assert "Video Annotation Tool" in html


# ---------------------------------------------------------------------------
# Unit tests – load_csv_data
# ---------------------------------------------------------------------------

class TestLoadCsvData:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        with patch(_CSV_FILE_PATH, tmp_path / "nonexistent.csv"):
            result = load_csv_data()
        assert result == ([], [], {})

    def test_loads_videos_and_status(self, tmp_path: Path) -> None:
        csv_path = _mock_csv_file(tmp_path)
        import sailsprep.annotation.annotation as mod
        mod._csv_cache = {"data": None, "mtime": 0}
        with patch(_CSV_FILE_PATH, csv_path):
            video_list, video_status, category_values = load_csv_data()
        assert len(video_list) == 2
        assert video_status == [0, 1]

    def test_category_values_parsed(self, tmp_path: Path) -> None:
        csv_path = _mock_csv_file(tmp_path)
        import sailsprep.annotation.annotation as mod
        mod._csv_cache = {"data": None, "mtime": 0}
        with patch(_CSV_FILE_PATH, csv_path):
            _, _, category_values = load_csv_data()
        key = "sub-01/ses-01/video.mp4"
        assert category_values[key]["Locomotion"] == "Walking"

    def test_cache_used_on_second_call(self, tmp_path: Path) -> None:
        csv_path = _mock_csv_file(tmp_path)
        import sailsprep.annotation.annotation as mod
        mod._csv_cache = {"data": None, "mtime": 0}
        with patch(_CSV_FILE_PATH, csv_path):
            first = load_csv_data()
            second = load_csv_data()
        assert first[0] == second[0]

    def test_no_status_column_defaults_to_zero(self, tmp_path: Path) -> None:
        rows = [{"video": "a.mp4"}, {"video": "b.mp4"}]
        csv_path = tmp_path / "video.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        import sailsprep.annotation.annotation as mod
        mod._csv_cache = {"data": None, "mtime": 0}
        with patch(_CSV_FILE_PATH, csv_path):
            _, video_status, _ = load_csv_data()
        assert video_status == [0, 0]


# ---------------------------------------------------------------------------
# Unit tests – update_status_in_csv
# ---------------------------------------------------------------------------

class TestUpdateStatusInCsv:
    def test_adds_status_column(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "video.csv"
        pd.DataFrame({"video": ["a.mp4", "b.mp4"]}).to_csv(csv_path, index=False)
        with patch(_CSV_FILE_PATH, csv_path):
            update_status_in_csv(["a.mp4", "b.mp4"], [0, 1])
        df = pd.read_csv(csv_path)
        assert "status" in df.columns
        assert df["status"].tolist() == [0, 1]

    def test_updates_existing_status_column(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "video.csv"
        pd.DataFrame({"video": ["a.mp4"], "status": [0]}).to_csv(csv_path, index=False)
        with patch(_CSV_FILE_PATH, csv_path):
            update_status_in_csv(["a.mp4"], [1])
        df = pd.read_csv(csv_path)
        assert df["status"].tolist() == [1]

    def test_bad_csv_does_not_raise(self, tmp_path: Path) -> None:
        # Points to a non-existent CSV; should swallow the exception silently.
        with patch(_CSV_FILE_PATH, tmp_path / "missing.csv"):
            update_status_in_csv(["a.mp4"], [0])  # must not raise


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

def _patch_load(video_list: list[str], video_status: list[int],
                category_values: dict[str, dict[str, str]]) -> Any:
    return patch(
        "sailsprep.annotation.annotation.load_csv_data",
        return_value=(video_list, video_status, category_values),
    )


class TestGetConfig:
    def test_returns_expected_keys(self, client: TestClient) -> None:
        r = client.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert "action_categories" in data
        assert "all_categories" in data
        assert "categories_with_reference" in data
        assert data["video_fps"] == 15

    def test_action_categories_match(self, client: TestClient) -> None:
        r = client.get("/api/config")
        assert r.json()["action_categories"] == ACTION_CATEGORIES


class TestGetVideos:
    def test_empty_csv(self, client: TestClient) -> None:
        with _patch_load([], [], {}):
            r = client.get("/api/videos")
        assert r.status_code == 200
        assert r.json() == {"videos": [], "total": 0}

    def test_returns_videos(self, client: TestClient) -> None:
        vl = ["a.mp4", "b.mp4"]
        vs = [0, 1]
        cv: dict[str, dict[str, str]] = {}
        with _patch_load(vl, vs, cv):
            r = client.get("/api/videos")
        data = r.json()
        assert data["total"] == 2
        assert data["videos"][0]["path"] == "a.mp4"
        assert data["videos"][1]["status"] == 1


class TestGetVideoInfo:
    def test_valid_index(self, client: TestClient) -> None:
        with _patch_load(["a.mp4"], [0], {}):
            r = client.get("/api/video/0")
        assert r.status_code == 200
        assert r.json()["path"] == "a.mp4"

    def test_invalid_index_404(self, client: TestClient) -> None:
        with _patch_load(["a.mp4"], [0], {}):
            r = client.get("/api/video/99")
        assert r.status_code == 404

    def test_negative_index_404(self, client: TestClient) -> None:
        with _patch_load(["a.mp4"], [0], {}):
            r = client.get("/api/video/-1")
        assert r.status_code == 404


class TestGetAnnotations:
    def test_invalid_index_404(self, client: TestClient) -> None:
        with _patch_load([], [], {}):
            r = client.get("/api/annotations/0")
        assert r.status_code == 404

    def test_no_annotation_file_returns_not_found(self, client: TestClient, tmp_path: Path) -> None:
        with _patch_load(["a.mp4"], [0], {}), \
             patch(_ANNOTATION_DIR_PATH, tmp_path), \
             patch(_OUTPUT_DIR_PATH, tmp_path):
            r = client.get("/api/annotations/0")
        assert r.status_code == 200
        assert r.json()["found"] is False

    def test_annotation_file_loaded(self, client: TestClient, tmp_path: Path) -> None:
        # Create a minimal annotation CSV
        frames = 10
        ann_data = {"Frame": list(range(frames))}
        for cat in ALL_CATEGORIES:
            ann_data[cat] = ["Walking"] * frames
        ann_path = tmp_path / "a_actions_corrected.csv"
        pd.DataFrame(ann_data).to_csv(ann_path, index=False)

        # Provide non-empty reference values so N/A logic is not triggered
        cv = {"a.mp4": {c: "Walking" for c in CATEGORIES_WITH_REFERENCE}}
        with _patch_load(["a.mp4"], [0], cv), \
             patch(_ANNOTATION_DIR_PATH, tmp_path), \
             patch(_OUTPUT_DIR_PATH, tmp_path):
            r = client.get("/api/annotations/0")
        assert r.status_code == 200
        body = r.json()
        assert body["found"] is True
        assert "Locomotion" in body["categories"]

    def test_empty_ref_value_forces_na(self, client: TestClient, tmp_path: Path) -> None:
        frames = 5
        ann_data = {"Frame": list(range(frames)), "Locomotion": ["Walking"] * frames,
                    "Repetitive_Motor_Movements": ["Jumping"] * frames}
        ann_path = tmp_path / "a_actions_corrected.csv"
        pd.DataFrame(ann_data).to_csv(ann_path, index=False)

        # Empty reference value → should be set to N/A
        cv = {"a.mp4": {"Locomotion": "", "Repetitive_Motor_Movements": ""}}
        with _patch_load(["a.mp4"], [0], cv), \
             patch(_ANNOTATION_DIR_PATH, tmp_path), \
             patch(_OUTPUT_DIR_PATH, tmp_path):
            r = client.get("/api/annotations/0")
        body = r.json()
        assert body["found"] is True
        segs = body["categories"]["Locomotion"]["segments"]
        assert segs[0]["label"] == "N/A"


class TestSaveAnnotations:
    def _payload(self, video_index: int = 0, total_frames: int = 5) -> dict[str, Any]:
        return {
            "video_index": video_index,
            "total_frames": total_frames,
            "categories": {
                "Locomotion": {
                    "breakpoints": [],
                    "segments": [{"start_frame": 0, "end_frame": 5, "label": "Walking"}],
                },
                "Repetitive_Motor_Movements": {
                    "breakpoints": [],
                    "segments": [{"start_frame": 0, "end_frame": 5, "label": "N/A"}],
                },
            },
        }

    def test_invalid_index_404(self, client: TestClient) -> None:
        with _patch_load([], [], {}):
            r = client.post("/api/save", json=self._payload())
        assert r.status_code == 404

    def test_save_creates_csv(self, client: TestClient, tmp_path: Path) -> None:
        with _patch_load(["a.mp4"], [0], {}), \
             patch(_OUTPUT_DIR_PATH, tmp_path), \
             patch("sailsprep.annotation.annotation.update_status_in_csv"):
            r = client.post("/api/save", json=self._payload())
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert (tmp_path / "a_actions_corrected.csv").exists()

    def test_saved_csv_has_correct_frames(self, client: TestClient, tmp_path: Path) -> None:
        with _patch_load(["a.mp4"], [0], {}), \
             patch(_OUTPUT_DIR_PATH, tmp_path), \
             patch("sailsprep.annotation.annotation.update_status_in_csv"):
            client.post("/api/save", json=self._payload(total_frames=5))
        df = pd.read_csv(tmp_path / "a_actions_corrected.csv")
        assert len(df) == 5
        assert df["Locomotion"].tolist() == ["Walking"] * 5


class TestUpdateStatus:
    def test_invalid_index_404(self, client: TestClient) -> None:
        with _patch_load([], [], {}):
            r = client.post("/api/update-status", json={"video_index": 0, "status": 1})
        assert r.status_code == 404

    def test_updates_status(self, client: TestClient, tmp_path: Path) -> None:
        csv_path = tmp_path / "video.csv"
        pd.DataFrame({"video": ["a.mp4"], "status": [0]}).to_csv(csv_path, index=False)
        with _patch_load(["a.mp4"], [0], {}), patch(_CSV_FILE_PATH, csv_path):
            r = client.post("/api/update-status", json={"video_index": 0, "status": 1})
        assert r.status_code == 200
        assert r.json()["status"] == 1


class TestFirstUnprocessed:
    def test_all_processed(self, client: TestClient) -> None:
        with _patch_load(["a.mp4", "b.mp4"], [1, 1], {}):
            r = client.get("/api/first-unprocessed")
        assert r.json() == {"index": -1, "found": False}

    def test_finds_first_unprocessed(self, client: TestClient) -> None:
        with _patch_load(["a.mp4", "b.mp4"], [1, 0], {}):
            r = client.get("/api/first-unprocessed")
        assert r.json() == {"index": 1, "found": True}

    def test_empty_list(self, client: TestClient) -> None:
        with _patch_load([], [], {}):
            r = client.get("/api/first-unprocessed")
        assert r.json() == {"index": -1, "found": False}


class TestRootEndpoint:
    def test_returns_html(self, client: TestClient) -> None:
        # index.html won't exist in test env → falls back to get_html_content()
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]