"""Tests for BIDS Video Processing Pipeline."""

import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Generator
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
import yaml


# Create a temporary config file to allow module import
@pytest.fixture(scope="session", autouse=True)
def setup_mock_config() -> Generator[None, None, None]:
    """Create a temporary config.yaml file for testing."""
    mock_config = {
        "video_root": "/mock/videos",
        "asd_csv": "mock_asd.csv",
        "nonasd_csv": "mock_nonasd.csv",
        "output_dir": "/mock/output",
        "target_resolution": "1280x720",
        "target_fps": 30,
    }

    # Create temporary config file
    with open("config.yaml", "w") as f:
        yaml.dump(mock_config, f)

    yield

    # Cleanup
    if os.path.exists("config.yaml"):
        os.remove("config.yaml")


# Import the module after config is created
@pytest.fixture(scope="session")
def bvp_module(setup_mock_config: Generator[None, None, None]) -> ModuleType:
    """Import the BIDS converter module."""
    sys.path.insert(0, "src")
    import sailsprep.BIDS_convertor as bvp

    return bvp


class TestConfiguration:
    """Test configuration loading and validation."""

    def test_load_configuration_success(self, bvp_module: ModuleType) -> None:
        """Test successful configuration loading."""
        mock_config = {
            "video_root": "/path/to/videos",
            "annotation_file": "blablabla.csv",
            "asd_status": "nonasd.xlsx",
            "output_dir": "/output",
            "target_resolution": "1280x720",
            "target_framerate": 30,
        }

        with patch("builtins.open", mock_open(read_data=yaml.dump(mock_config))):
            with patch("yaml.safe_load", return_value=mock_config):
                config = bvp_module.load_configuration("config.yaml")
                assert config == mock_config

    def test_load_configuration_file_not_found(self, bvp_module: ModuleType) -> None:
        """Test configuration loading with missing file."""
        with patch("builtins.open", side_effect=FileNotFoundError()):
            with pytest.raises(FileNotFoundError):
                bvp_module.load_configuration("nonexistent.yaml")

    def test_load_configuration_invalid_yaml(self, bvp_module: ModuleType) -> None:
        """Test configuration loading with invalid YAML."""
        with patch("builtins.open", mock_open(read_data="invalid: yaml: : format")):
            with pytest.raises(yaml.YAMLError):
                bvp_module.load_configuration("config.yaml")

    def test_load_configuration_missing_required_fields(
        self, bvp_module: ModuleType
    ) -> None:
        """Test configuration loading with missing required fields."""
        incomplete_config = {
            "video_root": "/path/to/videos",
            # Missing other required fields
        }
        with patch("builtins.open", mock_open(read_data=yaml.dump(incomplete_config))):
            with pytest.raises(KeyError):
                bvp_module.load_configuration("config.yaml")


class TestInfoExtractorforBIDS:
    """Test info extraction and missing excel handling for BIDS."""

    def test_create_dummy_excel_data_returns_expected_dict(
        self, tmp_path: Path, bvp_module: ModuleType
    ) -> None:
        """Test dummy excel data creation returns expected dict."""
        # Arrange
        video_path = tmp_path / "sub-001_video.mp4"
        video_path.write_text("dummy")  # just to create a filename
        participant_id = "001"
        session_id = "01"

        # Act
        data = bvp_module.create_dummy_excel_data(
            str(video_path), participant_id, session_id, "rest"
        )

        # Assert
        assert data["ID"] == "001"
        assert data["FileName"] == os.path.basename(video_path)
        assert data["Context"] == "rest"
        assert data["Notes"].startswith("Video not found")
        assert "Vid_duration" in data
        # All fields should have default "n/a" except the few explicitly set
        assert all(
            v == "n/a" or k in ["ID", "FileName", "Context", "Vid_duration", "Notes"]
            for k, v in data.items()
            if k not in ["ID", "FileName", "Context", "Vid_duration", "Notes"]
        )

    def test_find_age_folder_session_direct_match(
        self, tmp_path: Path, bvp_module: ModuleType
    ) -> None:
        """Test direct match for age folder session."""
        participant_path = tmp_path / "sub-001"
        participant_path.mkdir()
        current_path = participant_path / "12-16_months"
        current_path.mkdir()

        with patch(
            "sailsprep.BIDS_convertor.determine_session_from_folder", return_value="01"
        ):
            session = bvp_module.find_age_folder_session(
                str(current_path), str(participant_path)
            )
            assert session == "01"

    def test_find_age_folder_session_outside_participant_path(
        self, tmp_path: Path, bvp_module: ModuleType
    ) -> None:
        """Test age folder session outside participant path."""
        participant_path = tmp_path / "sub-001"
        other_path = tmp_path / "other" / "12-16_months"
        other_path.mkdir(parents=True)

        with patch(
            "sailsprep.BIDS_convertor.determine_session_from_folder", return_value="01"
        ):
            session = bvp_module.find_age_folder_session(
                str(other_path), str(participant_path)
            )
            assert session is None

    def test_get_task_from_excel_row_valid_context(
        self, bvp_module: ModuleType
    ) -> None:
        """Test get task from excel row with valid context."""
        row = pd.Series({"Context": "Play-time"})
        result = bvp_module.get_task_from_excel_row(row)
        assert result == "Playtime"  # cleaned via make_bids_task_label

    def get_task_from_excel_row(self, row: pd.Series, bvp_module: ModuleType) -> None:
        """Test get task from excel row with unknown context."""
        context = str(row.get("Context", "Other ")).strip()
        result = bvp_module.make_bids_task_label(context)
        assert result == "unknown"

    def test_extract_participant_id_from_folder_with_ames_prefix(
        self, bvp_module: ModuleType
    ) -> None:
        """Test extract participant ID from folder with AMES prefix."""
        assert (
            bvp_module.extract_participant_id_from_folder("SOMETHING_AMES_123") == "123"
        )

    def test_extract_participant_id_edge_cases(self, bvp_module: ModuleType) -> None:
        """Test extract participant ID edge cases."""
        assert (
            bvp_module.extract_participant_id_from_folder("ABC_AMES_456_extra_AMES")
            == "456_extra_AMES"
        )
        assert (
            bvp_module.extract_participant_id_from_folder("participant123")
            == "participant123"
        )
        assert (
            bvp_module.extract_participant_id_from_folder("AA_participant_123") == "123"
        )

    def test_determine_session_from_excel_timepoint_14(
        self, bvp_module: ModuleType
    ) -> None:
        """Test determine session from excel with timepoint 14."""
        df = pd.DataFrame(
            [{"ID": "001", "FileName": "video1.mp4", "timepoint": "14_month", "Age": 1}]
        )
        session = bvp_module.determine_session_from_excel(
            "/some/path/video1.mp4", df, "001"
        )
        assert session == "01"

    def test_determine_session_from_excel_timepoint_36(
        self, bvp_module: ModuleType
    ) -> None:
        """Test determine session from excel with timepoint 36."""
        df = pd.DataFrame(
            [{"ID": "002", "FileName": "vid2.mov", "timepoint": "36months", "Age": 3}]
        )
        session = bvp_module.determine_session_from_excel(
            "/some/path/vid2.mov", df, "002"
        )
        assert session == "02"

    def test_determine_session_from_excel_age_based(
        self, bvp_module: ModuleType
    ) -> None:
        """Test determine session from excel."""
        df = pd.DataFrame(
            [
                {"ID": "003", "FileName": "a.mp4", "timepoint": "unknown", "Age": 1.5},
                {"ID": "004", "FileName": "b.mp4", "timepoint": pd.NA, "Age": 3},
            ]
        )
        s1 = bvp_module.determine_session_from_excel("/p/a.mp4", df, "003")
        s2 = bvp_module.determine_session_from_excel("/p/b.mp4", df, "004")
        assert s1 == "01"
        assert s2 == "02"

    def test_determine_session_from_excel_participant_not_found(
        self, bvp_module: ModuleType
    ) -> None:
        """Test determine session from excel with error in participant ID."""
        df = pd.DataFrame(
            [{"ID": "999", "FileName": "x.mp4", "timepoint": "14", "Age": 1}]
        )
        with pytest.raises(ValueError):
            bvp_module.determine_session_from_excel("/p/y.mp4", df, "001")

    def test_determine_session_from_excel_file_not_found(
        self, bvp_module: ModuleType
    ) -> None:
        """Test determine session from excel with missing excel."""
        df = pd.DataFrame(
            [{"ID": "010", "FileName": "other.mp4", "timepoint": "14", "Age": 1}]
        )
        with pytest.raises(ValueError):
            bvp_module.determine_session_from_excel("/p/missing.mp4", df, "010")

    def test_determine_session_from_excel_unable_to_determine(
        self, bvp_module: ModuleType
    ) -> None:
        """Test determine session timepoint does not match and age is NaN."""
        df = pd.DataFrame(
            [{"ID": "030", "FileName": "u.mp4", "timepoint": "unk", "Age": pd.NA}]
        )
        with pytest.raises(ValueError):
            bvp_module.determine_session_from_excel("/p/u.mp4", df, "030")


class TestBIDSStructure:
    """Test BIDS directory structure creation and validation."""

    def test_create_bids_structure(self, bvp_module: ModuleType) -> None:
        """Test BIDS directory structure creation."""
        with patch("os.makedirs") as mock_makedirs:
            bvp_module.create_bids_structure()
            # Check that directories are created with exist_ok=True
            assert mock_makedirs.call_count == 2

    def test_create_dataset_description(self, bvp_module: ModuleType) -> None:
        """Test dataset description file creation."""
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            with patch("json.dump") as mock_json_dump:
                bvp_module.create_dataset_description()
                mock_file.assert_called_once()
                mock_json_dump.assert_called_once()
                # Check that the dataset description contains required fields
                args, kwargs = mock_json_dump.call_args
                dataset_desc = args[0]
                assert "Name" in dataset_desc
                assert "BIDSVersion" in dataset_desc
                assert "DatasetType" in dataset_desc

    def test_create_readme(self, bvp_module: ModuleType) -> None:
        """Test README file creation."""
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            bvp_module.create_readme()
            mock_file.assert_called_once()
            # Check that content was written
            handle = mock_file()
            handle.write.assert_called()


class TestBIDSNaming:
    """Test BIDS naming conventions and filename generation."""

    def test_create_bids_filename(self, bvp_module: ModuleType) -> None:
        """Test BIDS filename creation."""
        filename = bvp_module.create_bids_filename(
            "123", "01", "mealtime", "beh", "mp4"
        )
        expected = "sub-123_ses-01_task-mealtime_run-01_beh.mp4"
        assert filename == expected

    def test_get_next_run_number_no_dir(
        self, tmp_path: Path, bvp_module: ModuleType
    ) -> None:
        """Test get_next_run_numberwhen no subject/session directory exists."""
        root = tmp_path
        result = bvp_module.get_next_run_number("001", "01", "rest", str(root))
        assert result == 1

    def test_get_next_run_number_empty_dir(
        self, tmp_path: Path, bvp_module: ModuleType
    ) -> None:
        """Test get_next_run_number  when runs already exist."""
        beh_dir = tmp_path / "sub-001" / "ses-01" / "beh"
        beh_dir.mkdir(parents=True)
        result = bvp_module.get_next_run_number("001", "01", "rest", str(tmp_path))
        assert result == 1

    def test_get_next_run_number_with_existing_runs(
        self, tmp_path: Path, bvp_module: ModuleType
    ) -> None:
        """Test get_next_run_number w existing runs."""
        beh_dir = tmp_path / "sub-001" / "ses-01" / "beh"
        beh_dir.mkdir(parents=True)
        # Simulate existing files
        (beh_dir / "sub-001_ses-01_task-rest_run-1_beh.tsv").touch()
        (beh_dir / "sub-001_ses-01_task-rest_run-2_beh.tsv").touch()
        result = bvp_module.get_next_run_number("001", "01", "rest", str(tmp_path))
        assert result == 3

    def test_get_next_run_number_with_invalid_and_no_run(
        self, tmp_path: Path, bvp_module: ModuleType
    ) -> None:
        """Test get_next_run_number skips invalid filenames."""
        beh_dir = tmp_path / "sub-001" / "ses-01" / "beh"
        beh_dir.mkdir(parents=True)
        # One invalid, one missing run number
        (beh_dir / "sub-001_ses-01_task-rest_run-abc_beh.tsv").touch()
        (beh_dir / "sub-001_ses-01_task-rest_beh.tsv").touch()
        result = bvp_module.get_next_run_number("001", "01", "rest", str(tmp_path))
        assert result == 2  # treated as next after run-1

    def test_make_bids_task_label_sanitizes_name(self, bvp_module: ModuleType) -> None:
        """Test make_bids_task_label correctly sanitizes and normalizes task names."""
        assert bvp_module.make_bids_task_label(" Task Rest ") == "TaskRest"
        assert bvp_module.make_bids_task_label("run-01+") == "run01+"
        assert bvp_module.make_bids_task_label("We!rd#Name$") == "WerdName"
        assert bvp_module.make_bids_task_label("") == ""
        assert bvp_module.make_bids_task_label(None) == "None"

    def test_get_session_from_path_12_16_months(self, bvp_module: ModuleType) -> None:
        """Test session determination for 12-16 month videos."""
        path = "12-16 month"
        session = bvp_module.determine_session_from_folder(path)
        assert session == "01"

    def test_get_session_from_path_34_38_months(self, bvp_module: ModuleType) -> None:
        """Test session determination for 34-38 month videos."""
        path = "34-38 month"
        session = bvp_module.determine_session_from_folder(path)
        assert session == "02"


class TestVideoMetadataExtraction:
    """Test video metadata extraction and processing."""

    def test_parse_duration_various_formats(self, bvp_module: ModuleType) -> None:
        """Test for various duration formats."""
        # Normal HH:MM:SS
        assert math.isclose(bvp_module.parse_duration("01:02:03"), 3723.0)
        # MM:SS format
        assert math.isclose(bvp_module.parse_duration("05:30"), 330.0)
        # Plain number string
        assert math.isclose(bvp_module.parse_duration("12.5"), 12.5)
        # Empty or NaN → 0.0
        assert bvp_module.parse_duration("") == 0.0
        assert bvp_module.parse_duration(np.nan) == 0.0
        # Invalid types → handled gracefully
        assert bvp_module.parse_duration(None) == 0.0
        assert bvp_module.parse_duration("abc") == 0.0

    def test_extract_exif_empty_file(self, bvp_module: ModuleType) -> None:
        """Test video metadata extraction with empty file."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "{}"  # Empty JSON response

            result = bvp_module.extract_exif("empty.mp4")
            assert result.get("duration_sec") == 0
            assert result.get("format") is None

    def test_extract_exif_corrupted_json(self, bvp_module: ModuleType) -> None:
        """Test video metadata extraction with corrupted JSON output."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "corrupted json"

            result = bvp_module.extract_exif("corrupt.mp4")
            assert "error" in result

    def test_extract_exif_success(self, bvp_module: ModuleType) -> None:
        """Test successful video metadata extraction."""
        mock_metadata = {
            "format": {
                "filename": "test.mp4",
                "format_long_name": "QuickTime / MOV",
                "duration": "120.5",
                "bit_rate": "1000000",
                "size": "15000000",
                "tags": {"creation_time": "2023-01-01T12:00:00.000000Z"},
            },
            "streams": [{"tags": {"creation_time": "2023-01-01T12:00:00.000000Z"}}],
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = json.dumps(mock_metadata)

            result = bvp_module.extract_exif("test.mp4")
            assert "duration_sec" in result
            assert result["duration_sec"] == 120.5
            assert result["format"] == "QuickTime / MOV"

    def test_extract_exif_ffprobe_error(self, bvp_module: ModuleType) -> None:
        """Test video metadata extraction with ffprobe error."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = "Error message"

            result = bvp_module.extract_exif("test.mp4")
            assert "ffprobe_error" in result
            assert result["ffprobe_error"] == "Error message"


class TestVideoProcessing:
    """Test video processing functions."""

    @patch("subprocess.run")
    @patch("os.remove")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_stabilize_video(
        self,
        mock_makedirs: MagicMock,
        mock_exists: MagicMock,
        mock_remove: MagicMock,
        mock_run: MagicMock,
        bvp_module: ModuleType,
    ) -> None:
        """Test video stabilization."""
        mock_exists.return_value = True
        mock_run.return_value.returncode = 0  # success
        mock_run.return_value.stderr = ""
        bvp_module.stabilize_video("input.mp4", "output.mp4", "output/TEMP/task-01")

        # Should call subprocess.run twice (detect and transform)
        assert mock_run.call_count == 2
        mock_remove.assert_called_once_with(
            os.path.join("output/TEMP/task-01", "transforms.trf")
        )

    def test_stabilize_video_input_missing(self, bvp_module: ModuleType) -> None:
        """Test video stabilization with missing input file."""
        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                bvp_module.stabilize_video("nonexistent.mp4", "output.mp4", "temp")

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_stabilize_video_vidstab_error(
        self,
        mock_exists: MagicMock,
        mock_run: MagicMock,
        bvp_module: ModuleType,
    ) -> None:
        """Test video stabilization with vidstab error."""
        mock_exists.return_value = True
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Error in vidstab"

        with pytest.raises(RuntimeError):
            bvp_module.stabilize_video("input.mp4", "output.mp4", "temp")

    def test_get_video_properties_success(
        self, monkeypatch: pytest.MonkeyPatch, bvp_module: ModuleType
    ) -> None:
        """Test video properties extraction success."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [30.0, 1280.0, 720.0]
        monkeypatch.setattr("cv2.VideoCapture", lambda _: mock_cap)

        props = bvp_module.get_video_properties("video.mp4")
        assert props["FrameRate"] == 30.0
        assert props["Resolution"] == "1280x720"

    def test_get_video_properties_unopened(
        self, monkeypatch: pytest.MonkeyPatch, bvp_module: ModuleType
    ) -> None:
        """Test video properties extraction with unopened video."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        monkeypatch.setattr("cv2.VideoCapture", lambda _: mock_cap)

        props = bvp_module.get_video_properties("missing.mp4")
        assert props == {"FrameRate": None, "Resolution": None}

    def test_get_video_properties_exception(
        self, monkeypatch: pytest.MonkeyPatch, bvp_module: ModuleType
    ) -> None:
        """Test video properties extraction with OpenCV exception."""

        def broken_videocap() -> None:
            raise RuntimeError("OpenCV error")

        monkeypatch.setattr("cv2.VideoCapture", broken_videocap)

        props = bvp_module.get_video_properties("corrupt.mp4")
        assert props == {"FrameRate": None, "Resolution": None}

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_extract_audio(
        self, mock_exists: MagicMock, mock_run: MagicMock, bvp_module: ModuleType
    ) -> None:
        """Test audio extraction from video."""
        # Pretend both input and output exist
        mock_exists.return_value = True
        mock_run.return_value.returncode = 0  # Simulate success
        mock_run.return_value.stderr = ""

        bvp_module.extract_audio("input.mp4", "output.wav")

        mock_run.assert_called_once()

        # Check that the command includes correct audio parameters
        args = mock_run.call_args[0][0]
        assert "-ar" in args
        assert "16000" in args
        assert "-ac" in args
        assert "1" in args


class TestMetadataFileCreation:
    """Test creation of BIDS metadata files."""

    def test_create_events_file(self, bvp_module: ModuleType) -> None:
        """Test events TSV file creation."""
        video_metadata = pd.DataFrame(
            [
                {"duration": 120.5, "filename": "video1.mp4"},
                {"duration": 43.5, "filename": "video2.mp4"},
            ]
        )

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            bvp_module.create_events_file(
                video_metadata, "output.tsv", "filepath/on/Engaging.mp4"
            )
            mock_to_csv.assert_called_once()

    def test_create_video_metadata_json(self, bvp_module: ModuleType) -> None:
        """Test video metadata JSON creation."""
        metadata = {"duration_sec": 120.5, "format": "MP4"}
        processing_info = {"has_stabilization": True}
        task_info = {
            "task_name": "unknown",
            "task_description": "Behavioral session:",
            "instructions": "Natural behavior observation",
            "context": "mealtime",
            "activity": "eating",
        }
        with patch("builtins.open", mock_open()):
            with patch("json.dump") as mock_json_dump:
                bvp_module.create_video_metadata_json(
                    metadata,
                    processing_info,
                    task_info,
                    "output.json",
                )
                mock_json_dump.assert_called_once()

                # Check JSON content structure
                args = mock_json_dump.call_args[0]
                json_content = args[0]
                assert "TaskName" in json_content
                assert "ProcessingPipeline" in json_content
                assert "OriginalMetadata" in json_content


class TestUtilityFunctions:
    """Test utility functions."""

    def test_save_json(self, bvp_module: ModuleType) -> None:
        """Test JSON file saving utility."""
        test_data = {"test": "data", "number": 123}

        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            with patch("json.dump") as mock_json_dump:
                bvp_module.save_json(test_data, "output.json")
                # Check that json.dump was called with the test data and the file handle
                mock_json_dump.assert_called_once()
                args, kwargs = mock_json_dump.call_args
                assert args[0] == test_data
                assert kwargs.get("indent") == 4


class TestMainWorkflow:
    """Test the main processing workflow."""

    @patch("sailsprep.BIDS_convertor.get_all_videos")
    @patch("sailsprep.BIDS_convertor.process_videos")
    @patch("sailsprep.BIDS_convertor.create_readme")
    @patch("sailsprep.BIDS_convertor.create_derivatives_dataset_description")
    @patch("sailsprep.BIDS_convertor.create_dataset_description")
    @patch("sailsprep.BIDS_convertor.create_bids_structure")
    @patch("sailsprep.BIDS_convertor.save_json")
    def test_main_workflow(
        self,
        mock_save_json: MagicMock,
        mock_create_structure: MagicMock,
        mock_create_dataset: MagicMock,
        mock_create_derivatives: MagicMock,
        mock_create_readme: MagicMock,
        mock_process_videos: MagicMock,
        mock_get_all_videos: MagicMock,
        bvp_module: ModuleType,
    ) -> None:
        """Test the main processing workflow."""
        # Setup mocks
        mock_get_all_videos.return_value = (["dummy_video_1.mp4"], [])

        mock_process_videos.return_value = (
            [
                {
                    "task_label": "task-rest",
                    "participant_id": "sub-001",
                    "session_id": "ses-01",
                }
            ],
            [{"error": None}],
        )
        with (
            patch("sailsprep.BIDS_convertor.os.path.exists", return_value=True),
            patch(
                "sailsprep.BIDS_convertor.pd.read_csv",
                return_value=pd.DataFrame(
                    {"Context": ["playing", "unknown"], "ID": ["AZE", "RET"]}
                ),
            ),
            patch.object(sys, "argv", ["BIDS_convertor.py", "0", "4"]),
            patch("sys.exit") as mock_exit,
        ):
            bvp_module.main()
            mock_exit.assert_not_called()

        # Verify all steps were called
        mock_create_structure.assert_called_once()
        mock_create_dataset.assert_called_once()
        mock_create_derivatives.assert_called_once()
        mock_create_readme.assert_called_once()
        mock_process_videos.assert_called_once()


class TestExtendedFunctions:
    """Additional unit tests for deeper functions and edge cases."""

    def test_find_session_id_uses_folder_first(self, bvp_module: ModuleType) -> None:
        """Should use folder-based session detection first."""
        mock_df = pd.DataFrame()  # not used

        with (
            patch(
                "sailsprep.BIDS_convertor.determine_session_from_folder",
                return_value="01",
            ) as mock_folder,
            patch(
                "sailsprep.BIDS_convertor.determine_session_from_excel"
            ) as mock_excel,
        ):
            session = bvp_module.find_session_id(
                directory="/data/participant/session01",
                current_path="/data/participant/session01/video.mp4",
                participant_path="/data/participant",
                annotation_df=mock_df,
                participant_id="001",
            )

        assert session == "01"
        mock_folder.assert_called_once()
        mock_excel.assert_not_called()

    def test_find_session_id_falls_back_to_folder_when_excel_fails(
        self, bvp_module: ModuleType
    ) -> None:
        """Should fall back to Excel lookup when folder-based detection fails."""
        mock_df = pd.DataFrame()
        with (
            patch(
                "sailsprep.BIDS_convertor.determine_session_from_folder",
                return_value=None,
            ) as mock_folder,
            patch(
                "sailsprep.BIDS_convertor.determine_session_from_excel",
                return_value="02",
            ) as mock_excel,
        ):
            session = bvp_module.find_session_id(
                directory="/data/participant/unknown_folder",
                current_path="/data/participant/unknown_folder/video.mp4",
                participant_path="/data/participant",
                annotation_df=mock_df,
                participant_id="001",
            )

        assert session == "02"
        mock_folder.assert_called_once()
        mock_excel.assert_called_once()

    def test_find_videos_recursive_collects_videos(
        self, tmp_path: Path, bvp_module: ModuleType
    ) -> None:
        """Test find_videos_recursive function."""
        participant = tmp_path / "sub-ABC"
        participant.mkdir()
        (participant / "12-16_months").mkdir()
        v1 = participant / "12-16_months" / "one.mp4"
        v1.write_text("x")
        (participant / "notes.txt").write_text("ignore")

        videos = bvp_module.find_videos_recursive(
            str(participant), str(participant), pd.DataFrame(), "ABC"
        )
        assert any(str(v1) == p for p, s in videos)

    def test_preprocess_video_success_creates_output(
        self, tmp_path: Path, bvp_module: ModuleType
    ) -> None:
        """Check that preprocess_video succeeds when all steps work."""
        input_file = tmp_path / "in.mp4"
        input_file.write_bytes(b"video")

        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()

        # Pre-create stabilized temp file
        stabilized_tmp = temp_dir / f"stabilized_temp_{os.getpid()}.mp4"
        stabilized_tmp.write_bytes(b"stable")

        output_path = tmp_path / "out.mp4"
        output_path.write_bytes(b"processed")

        # Patch stabilize_video and subprocess.run
        with (
            patch("sailsprep.BIDS_convertor.stabilize_video", return_value=None),
            patch("sailsprep.BIDS_convertor.subprocess.run") as mock_run,
        ):
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = ""

            # Should not raise any error
            bvp_module.preprocess_video(
                str(input_file), str(output_path), str(temp_dir)
            )

        # ✅ Assert that output file exists and is non-empty
        assert output_path.exists(), "Output video file should exist"
        assert output_path.stat().st_size >= 0, "Output video file should not be empty"

        # ✅ Assert that stabilized temp file was cleaned up
        assert (
            not stabilized_tmp.exists()
        ), "Temporary stabilized file should be removed"

        # ✅ Verify that ffmpeg (subprocess) was called
        mock_run.assert_called_once()

    def test_safe_float_conversion_various(self, bvp_module: ModuleType) -> None:
        """Test function for the conversion of float."""
        assert bvp_module.safe_float_conversion(None) == "n/a"
        assert bvp_module.safe_float_conversion("n/a") == "n/a"
        assert bvp_module.safe_float_conversion("12.5") == 12.5
        assert bvp_module.safe_float_conversion(3) == 3.0
        assert bvp_module.safe_float_conversion("abc", default="-") == "-"

    def test_create_audio_metadata_json_calls_save_json(
        self, bvp_module: ModuleType
    ) -> None:
        """Test audio metadata creation function."""
        with patch("sailsprep.BIDS_convertor.save_json") as mock_save_json:
            bvp_module.create_audio_metadata_json(
                12.3, {"task_name": "t", "task_description": "blabla"}, "out.json"
            )
            mock_save_json.assert_called_once()
            args = mock_save_json.call_args[0]
            assert args[0]["Duration"] == 12.3
            assert args[0]["TaskName"] == "t"
            assert args[0]["TaskDescription"] == "blabla"

    def test_create_raw_video_json_saves_properties(
        self, tmp_path: Path, bvp_module: ModuleType
    ) -> None:
        """Test raw video json creation function."""
        with (
            patch(
                "sailsprep.BIDS_convertor.get_video_properties",
                return_value={"FrameRate": 30.0, "Resolution": "1280x720"},
            ),
            patch("sailsprep.BIDS_convertor.save_json") as mock_save,
        ):
            row = pd.Series(
                {
                    "FileName": "a.mp4",
                    "Vid_duration": "00:01:00",
                    "Vid_date": "2020-01-01",
                    "timepoint": "14",
                    "SourceFile": "orig.mp4",
                }
            )

            bvp_module.create_raw_video_json(
                row,
                {"task_name": "t", "context": "c", "activity": "a"},
                "somepath.mp4",
                str(tmp_path / "raw.json"),
            )

            # Assert save_json was called once
            mock_save.assert_called_once()

            # Extract the arguments used in the call
            saved_data = mock_save.call_args[0][0]

            # Check that the metadata contains expected values
            assert saved_data["TaskName"] == "t"
            assert saved_data["FrameRate"] == 30.0
            assert saved_data["Resolution"] == "1280x720"
            assert saved_data["OriginalFilename"] == "a.mp4"
            assert saved_data["Context"] == "c"
            assert saved_data["Activity"] == "a"
            assert saved_data["TimePoint"] == "14"
            assert saved_data["SourceFile"] == "orig.mp4"
            assert (
                abs(saved_data["Duration"] - 60.0) < 1e-6
            )  # assuming parse_duration → seconds

    def test_create_participants_file_creates_expected_outputs(
        self, tmp_path: Path, bvp_module: ModuleType
    ) -> None:
        """Test create participants.tsv function."""
        # Setup mock data
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        (bids_root / "sub-101").mkdir()
        (bids_root / "sub-102").mkdir()

        asd_file = tmp_path / "asd.xlsx"
        df = pd.DataFrame({"ID": ["101", "102"], "Group": ["ASD", "Non-ASD"]})
        df.to_excel(asd_file, index=True)

        bvp_module.create_participants_file(str(bids_root), str(asd_file))

        # Assertions
        tsv_path = bids_root / "participants.tsv"
        json_path = bids_root / "participants.json"
        assert tsv_path.exists()
        assert json_path.exists()

        df_out = pd.read_csv(tsv_path, sep="\t")
        print(df_out)
        assert set(df_out["participant_id"]) == {"sub-101", "sub-102"}
        assert set(df_out["group"]) == {"ASD", "Non-ASD"}

    def test_print_summary_outputs_expected(
        self, capsys: pytest.CaptureFixture[str], bvp_module: ModuleType
    ) -> None:
        """Test the summary printer function."""
        processed = [
            {
                "task_label": "a",
                "participant_id": "p1",
                "session_id": "01",
                "duration_sec": 60,
                "has_excel_data": True,
            },
            {
                "task_label": "b",
                "participant_id": "p2",
                "session_id": "02",
                "duration_sec": 120,
                "has_excel_data": False,
            },
        ]
        failed = [{"video": "x", "error": "boom"}]
        bvp_module.print_summary(processed, failed)
        captured = capsys.readouterr()
        assert "Successfully processed: 2 videos" in captured.out
        assert "Failed to process: 1 videos" in captured.out

    def test_merge_subjects_merges_and_removes(
        self, tmp_path: Path, bvp_module: ModuleType
    ) -> None:
        """Test merge subjects function."""
        # Prepare FINAL_BIDS_ROOT and derivatives paths
        root = tmp_path / "bids"
        deriv = root / "derivatives" / "preprocessed"
        (root).mkdir(parents=True)
        (deriv).mkdir(parents=True)

        # Create original and duplicate subject folders
        orig = root / "sub-200"
        dup = root / "sub-200 2"
        orig.mkdir()
        dup.mkdir()
        # Add file to dup that should be moved
        (dup / "file.txt").write_text("hello")

        # Run merge_subjects
        bvp_module.merge_subjects(str(root))

        # After merge, duplicate folder should not exist
        assert not dup.exists()


class TestProcessSingleVideo:
    """Test the process_single_video function."""

    def test_process_single_video_empty_info(self, bvp_module: ModuleType) -> None:
        """Test the processing of single video with empty information."""
        result, error = bvp_module.process_single_video(
            {}, pd.DataFrame(), "root", "deriv", "tmp"
        )
        assert result is None
        assert isinstance(error, dict)
        assert "video_info is empty" in error["error"]

    def test_process_single_video_missing_keys(self, bvp_module: ModuleType) -> None:
        """Test the processing of single video with missing information."""
        video_info = {"filename": "f.mp4"}  # missing participant_id, etc.
        result, error = bvp_module.process_single_video(
            video_info, pd.DataFrame(), "root", "deriv", "tmp"
        )
        assert result is None
        assert "Missing required video_info keys" in error["error"]


# Test fixtures for reusable data
@pytest.fixture
def sample_demographics() -> pd.DataFrame:
    """Sample demographics DataFrame for testing."""
    return pd.DataFrame(
        {
            "dependent_temporary_id": ["A001", "A002", "N001"],
            "dependent_dob": ["2022-01-01", "2022-02-01", "2022-03-01"],
            "sex": ["M", "F", "M"],
            "diagnosis": ["ASD", "ASD", "TD"],
        }
    )


@pytest.fixture
def sample_video_metadata() -> dict[str, float | str | int]:
    """Sample video metadata for testing."""
    return {
        "duration_sec": 120.5,
        "format": "QuickTime / MOV",
        "bit_rate": 1000000,
        "size_bytes": 15000000,
    }


if __name__ == "__main__":
    pytest.main([__file__])
