"""BIDS Video Processing Pipeline.

This module processes home videos from ASD screening studies and organizes them
according to the Brain Imaging Data Structure (BIDS) specification version 1.9.0.

The pipeline includes video stabilization, denoising, standardization, and audio
extraction for behavioral analysis research.

Example:
    Basic usage:
        $ python bids_video_processor.py

Todo:
    * check with actual data
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import pandas as pd
import yaml


def load_configuration(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load and validate configuration from YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Configuration dictionary containing video processing parameters.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If the YAML file is malformed.
        KeyError: If required keys are missing in the configuration.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    required_keys = [
        "annotation_file",
        "video_root",
        "output_dir",
        "target_resolution",
        "target_framerate",
        "asd_status",
    ]

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing configuration keys: {', '.join(missing_keys)}")
    return config


# Load configuration
config_path = (
    Path(__file__).resolve().parents[2] / "configs" / "config_bids_convertor.yaml"
)
config = load_configuration(str(config_path))
# Unpack configuration
ANNOTATION_FILE = config["annotation_file"]
VIDEO_ROOT = config["video_root"]
OUTPUT_DIR = config["output_dir"]
TARGET_RESOLUTION = config["target_resolution"]
TARGET_FRAMERATE = config["target_framerate"]
ASD_STATUS_FILE = config["asd_status"]

# BIDS directory structure
FINAL_BIDS_ROOT = os.path.join(
    OUTPUT_DIR, config.get("final_bids_root", "final_bids-dataset")
)
FINAL_DERIVATIVES_DIR = os.path.join(
    FINAL_BIDS_ROOT, config.get("derivatives_subdir", "derivatives/preprocessed")
)


def create_bids_structure() -> None:
    """Create the BIDS directory structure.

    Creates the main BIDS dataset directory and derivatives subdirectory
    following BIDS specification requirements.

    Note:
        This function creates directories with exist_ok=True to prevent
        errors if directories already exist.
    """
    os.makedirs(FINAL_BIDS_ROOT, exist_ok=True)
    os.makedirs(FINAL_DERIVATIVES_DIR, exist_ok=True)


def save_json(data: Union[List[Any], Dict[str, Any]], path: str) -> None:
    """Save data to JSON file.

    Utility function to save Python data structures to JSON files with
    proper formatting and error handling.

    Args:
        data (list or dict): Data structure to save as JSON.
        path (str): Output file path for JSON file.

    Raises:
        IOError: If unable to write to the specified path.
        TypeError: If data contains non-serializable objects.

    Note:
        Uses 4-space indentation for readable JSON output.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def safe_print(message: str) -> None:
    """Print with timestamps."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{timestamp} [MAIN] {message}")


# Helper functions
def parse_duration(duration_str: str) -> float:
    """Parse duration string to seconds."""
    try:
        if pd.isna(duration_str) or duration_str == "":
            return 0.0
        duration_str = str(duration_str)
        if ":" in duration_str:
            parts = duration_str.split(":")
            if len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
        return float(duration_str)
    except (ValueError, TypeError):
        return 0.0


def make_bids_task_label(task_name: str) -> str:
    """Convert TaskName to BIDS-compatible task label for filenames."""
    s = str(task_name).strip()
    s = re.sub(r"[^0-9a-zA-Z+]", "", s)  # Keep only alphanumeric and +
    return s


def get_video_properties(video_path: str) -> dict:
    """Extract video properties using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"FrameRate": None, "Resolution": None}

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return {
            "FrameRate": fps,
            "Resolution": f"{width}x{height}",
        }

    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return {"FrameRate": None, "Resolution": None}


def determine_session_from_folder(folder_name: str) -> Optional[str]:
    """Determine the session ID from a folder name based on known age-related patterns.

    Args:
        folder_name (str): The name of the folder to check.

    Returns:
        Optional[str]: "01" for 12–16 month sessions, "02" for 34–38 month sessions,
        or None if no match.
    """
    folder_lower = folder_name.lower()

    # Session 01 patterns
    if any(
        pattern in folder_lower
        for pattern in [
            "12-16 month",
            "12-14 month",
            "12_16",
            "12_14",
            "12-16month",
            "12-14month",
            "12-16_month_videos",
        ]
    ):
        return "01"

    # Session 02 patterns (typos and variants included)
    if any(
        pattern in folder_lower
        for pattern in [
            "34-38 month",
            "34-28 month",
            "34-48 month",
            "34_38",
            "34_28",
            "34_48",
            "34-38month",
            "34-28month",
            "34-48month",
            "34-38_month_videos",
        ]
    ):
        return "02"

    return None


def find_age_folder_session(current_path: str, participant_path: str) -> Optional[str]:
    """Recursively seek the timepoint folder.

    Args:
        current_path (str): Current directory path to inspect.
        participant_path (str): Root path of the participant.

    Returns:
        Optional[str]: Session ID ("01" or "02") if detected, else None.
    """
    if (
        not current_path.startswith(participant_path)
        or current_path == participant_path
    ):
        return None

    current_folder = os.path.basename(current_path)
    session_id = determine_session_from_folder(current_folder)
    if session_id:
        return session_id

    parent_path = os.path.dirname(current_path)
    return find_age_folder_session(parent_path, participant_path)


def extract_participant_id_from_folder(folder_name: str) -> str:
    """Extract the participant ID from folder names.

    Args:
        folder_name (str): Folder name containing participant info.

    Returns:
        str: Extracted participant ID.
    """
    if "AMES_" in folder_name:
        parts = folder_name.split("AMES_")
        if len(parts) > 1:
            return parts[1].strip()

    if "_" in folder_name:
        return folder_name.split("_")[-1]

    return folder_name


def determine_session_from_excel(
    current_path: str, annotation_df: pd.DataFrame, participant_id: str
) -> Optional[str]:
    """Determine the session ID for a video based on the annotation file.

    Args:
        current_path (str): Path to the video file.
        annotation_df (pd.DataFrame): Excel data containing 'ID',
        'FileName', 'timepoint', and 'Age' columns.
        participant_id (str): Participant identifier.

    Returns:
        Optional[str]: Session ID ("01" or "02"), or None if not found.
    """
    filename = os.path.splitext(os.path.basename(current_path))[0]
    if participant_id.endswith(" 2"):
        participant_id = participant_id[:-2].strip()
    # Filter for the participant
    participant_excel = annotation_df[
        annotation_df["ID"].astype(str) == str(participant_id)
    ]
    if participant_excel.empty:
        raise ValueError(
            f"Participant ID '{participant_id}' not found in Excel metadata"
            f" for file '{filename}'."
        )

    # Match the video filename (without extension)
    mask = participant_excel["FileName"].str.split(".").str[0] == filename
    video_entry = participant_excel[mask]

    if video_entry.empty:
        raise ValueError(
            f"No matching Excel entry found for video '{filename}'"
            f"(participant {participant_id})."
        )

    timepoint = video_entry["timepoint"].iloc[0]
    age = video_entry["Age"].iloc[0]

    # Normalize timepoint to string for pattern matching
    timepoint_str = str(timepoint)

    if "14" in timepoint_str:
        return "01"
    elif "36" in timepoint_str:
        return "02"
    elif pd.notna(age):
        return "01" if age < 2 else "02"
    else:
        raise ValueError(
            f"Unable to determine session ID: timepoint={timepoint}, age={age}"
        )


def find_session_id(
    directory: str,
    current_path: str,
    participant_path: str,
    annotation_df: pd.DataFrame,
    participant_id: str,
    excel: bool = True,
) -> Optional[str]:
    """Determine session ID by checking folder names first, then Excel data if needed.

    Args:
        directory (str): Current directory being scanned.
        current_path (str): Full path to the file.
        participant_path (str): Root participant directory.
        annotation_df (pd.DataFrame): Excel metadata.
        participant_id (str): Participant identifier.
        excel (bool) : Whether to use Excel data for session determination.

    Returns:
        Optional[str]: Session ID ("01" or "02"), or None.
    """
    if (
        not current_path.startswith(participant_path)
        or current_path == participant_path
    ):
        return None

    try:
        folder_name = os.path.basename(directory)
        session_id = determine_session_from_folder(folder_name)

        if not session_id and excel:
            try:
                session_id = determine_session_from_excel(
                    current_path, annotation_df, participant_id
                )
            except ValueError as e:
                print(f"Excel lookup failed for {participant_id}: {e}")

        if session_id:
            return session_id

        # Recurse upward if not found
        parent_path = os.path.dirname(directory)
        if parent_path != directory:
            return find_session_id(
                parent_path,
                current_path,
                participant_path,
                annotation_df,
                participant_id,
                False,
            )

    except PermissionError:
        print(f"Permission denied: {current_path}")
    except Exception as e:
        print(f"Error accessing {current_path}: {e}")

    return None


def find_videos_recursive(
    directory: str,
    participant_path: str,
    annotation_df: pd.DataFrame,
    participant_id: str,
) -> List[Tuple[str, Optional[str]]]:
    """Recursively find video files and determine their session IDs.

    Args:
        directory (str): Directory to search in.
        participant_path (str): Root path of the participant.
        annotation_df (pd.DataFrame): Excel data for metadata lookup.
        participant_id (str): Participant identifier.

    Returns:
        List[Tuple[str, Optional[str]]]: List of (video_path, session_id) pairs.
    """
    videos = []
    try:
        for item in os.listdir(directory):
            if item.startswith("."):
                continue  # Skip hidden files

            item_path = os.path.join(directory, item)

            if os.path.isfile(item_path) and item.lower().endswith(
                (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp", ".mts")
            ):
                session_id = find_session_id(
                    directory,
                    item_path,
                    participant_path,
                    annotation_df,
                    participant_id,
                )
                videos.append((item_path, session_id))

            elif os.path.isdir(item_path):
                videos.extend(
                    find_videos_recursive(
                        item_path, participant_path, annotation_df, participant_id
                    )
                )

    except PermissionError:
        print(f"Permission denied: {directory}")
    except Exception as e:
        print(f"Error accessing {directory}: {e}")

    return videos


def get_all_videos(video_root: str, annotation_df: pd.DataFrame) -> List[dict]:
    """Find and label all participant videos with their corresponding session IDs.

    Args:
        video_root (str): Root directory containing all participant folders.
        annotation_df (pd.DataFrame): Excel data with metadata.

    Returns:
        List[dict]: List of video metadata dictionaries.
    """
    all_videos = []

    try:
        for participant_folder in os.listdir(video_root):
            participant_path = os.path.join(video_root, participant_folder)
            if not os.path.isdir(participant_path):
                continue

            participant_id = extract_participant_id_from_folder(participant_folder)
            if not participant_id:
                continue

            videos = find_videos_recursive(
                participant_path, participant_path, annotation_df, participant_id
            )

            for video_path, session_id in videos:
                if session_id in {"01", "02"}:
                    all_videos.append(
                        {
                            "participant_id": participant_id,
                            "filename": os.path.basename(video_path),
                            "full_path": video_path,
                            "session_id": session_id,
                            "age_folder": os.path.basename(os.path.dirname(video_path)),
                        }
                    )

    except Exception as e:
        print(f"Error scanning video folders: {e}")

    return all_videos


def create_dummy_excel_data(
    video_path: str, participant_id: str, session_id: str, task_label: str = "unknown"
) -> dict[str, str]:
    """Create dummy behavioral data for videos not in Excel file."""
    video_filename = os.path.basename(video_path)

    dummy_row_data = {
        "ID": participant_id,
        "FileName": video_filename,
        "Context": task_label,
        "Location": "n/a",
        "Activity": "n/a",
        "Child_of_interest_clear": "n/a",
        "#_adults": "n/a",
        "#_children": "n/a",
        "#_people_background": "n/a",
        "Interaction_with_child": "n/a",
        "#_people_interacting": "n/a",
        "Child_constrained": "n/a",
        "Constraint_type": "n/a",
        "Supports": "n/a",
        "Support_type": "n/a",
        "Example_support_type": "n/a",
        "Gestures": "n/a",
        "Gesture_type": "n/a",
        "Vocalizations": "n/a",
        "RMM": "n/a",
        "RMM_type": "n/a",
        "Response_to_name": "n/a",
        "Locomotion": "n/a",
        "Locomotion_type": "n/a",
        "Grasping": "n/a",
        "Grasp_type": "n/a",
        "Body_Parts_Visible": "n/a",
        "Angle_of_Body": "n/a",
        "time_point": "n/a",
        "DOB": "n/a",
        "Vid_date": "n/a",
        "Video_Quality_Child_Face_Visibility": "n/a",
        "Video_Quality_Child_Body_Visibility": "n/a",
        "Video_Quality_Child_Hand_Visibility": "n/a",
        "Video_Quality_Lighting": "n/a",
        "Video_Quality_Resolution": "n/a",
        "Video_Quality_Motion": "n/a",
        "Coder": "n/a",
        "SourceFile": "n/a",
        "Vid_duration": "00:00:00",
        "Notes": "Video not found in Excel file - behavioral data unavailable",
    }

    return dummy_row_data


def get_task_from_excel_row(row: pd.Series) -> str:
    """Extract and create task label from Excel row data."""
    context = str(row.get("Context", "")).strip()

    if context and context.lower() not in ["nan", "n/a", ""]:
        return make_bids_task_label(context)
    else:
        return "unknown"


def get_next_run_number(
    participant_id: str, session_id: str, task_label: str, final_bids_root: str
) -> int:
    """Find the next available run number for this participant/session/task."""
    beh_dir = os.path.join(
        final_bids_root, f"sub-{participant_id}", f"ses-{session_id}", "beh"
    )

    if not os.path.exists(beh_dir):
        return 1

    # Look for existing files with this task
    pattern = f"sub-{participant_id}_ses-{session_id}_task-{task_label}_"
    existing_files = [f for f in os.listdir(beh_dir) if f.startswith(pattern)]

    if not existing_files:
        return 1

    # Extract run numbers from existing files
    run_numbers = []
    for filename in existing_files:
        if "_run-" in filename:
            run_part = filename.split("_run-")[1].split("_")[0]
            try:
                run_numbers.append(int(run_part))
            except ValueError:
                continue
        else:
            run_numbers.append(1)  # Files without run numbers are considered run-1

    return max(run_numbers) + 1 if run_numbers else 1


def create_bids_filename(
    participant_id: str,
    session_id: str,
    task_label: str,
    suffix: str,
    extension: str,
    run_id: int = 1,
) -> str:
    """Create BIDS-compliant filename w run identifier for multiple videos per task."""
    return (
        f"sub-{participant_id}_"
        f"ses-{session_id}_"
        f"task-{task_label}_"
        f"run-{run_id:02d}_"
        f"{suffix}.{extension}"
    )


# Video processing functions
def extract_exif(video_path: str) -> Dict[str, Any]:
    """Extract video metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {"ffprobe_error": result.stderr.strip()}

        metadata = json.loads(result.stdout)
        extracted = {}

        format_info = metadata.get("format", {})
        extracted["filename"] = format_info.get("filename")
        extracted["format"] = format_info.get("format_long_name")
        extracted["duration_sec"] = float(format_info.get("duration", 0))
        extracted["bit_rate"] = int(format_info.get("bit_rate", 0))
        extracted["size_bytes"] = int(format_info.get("size", 0))

        return extracted
    except Exception as e:
        return {"error": str(e)}


def stabilize_video(input_path: str, stabilized_path: str, temp_dir: str) -> None:
    """Stabilize video using FFmpeg vidstab filters, with error checks."""
    os.makedirs(temp_dir, exist_ok=True)
    transforms_file = os.path.join(temp_dir, "transforms.trf")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Video to stabilize not found: {input_path}")

    # Step 1: Detect transforms
    detect_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        f"vidstabdetect=shakiness=5:accuracy=15:result={transforms_file}",
        "-f",
        "null",
        "-",
    ]
    detect_proc = subprocess.run(detect_cmd, capture_output=True, text=True)

    if detect_proc.returncode != 0:
        print(f"[ERROR] vidstabdetect failed for {input_path}:\n{detect_proc.stderr}")
        raise RuntimeError(f"FFmpeg vidstabdetect failed for {input_path}")

    if not os.path.exists(transforms_file):
        raise FileNotFoundError(f"Transform file not created: {transforms_file}")

    # Step 2: Apply transforms
    transform_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        f"vidstabtransform=smoothing=30:input={transforms_file}",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "23",
        "-c:a",
        "copy",
        stabilized_path,
    ]
    print(f"[DEBUG] Running: {' '.join(transform_cmd)}")
    transform_proc = subprocess.run(transform_cmd, capture_output=True, text=True)

    if transform_proc.returncode != 0:
        print(
            f"[ERROR] vidstabtransform failed for {input_path}:"
            f"\n{transform_proc.stderr}"
        )
        raise RuntimeError(f"FFmpeg vidstabtransform failed for {input_path}")

    if not os.path.exists(stabilized_path):
        raise FileNotFoundError(f"Stabilized video not created: {stabilized_path}")

    # Cleanup
    os.remove(transforms_file)


def preprocess_video(input_path: str, output_path: str, temp_dir: str) -> None:
    """Preprocess video with stabilization, denoising, and standardization."""
    if not os.path.exists(input_path):
        raise ValueError(f"Input video not found: {input_path}")

    stabilized_tmp = os.path.join(temp_dir, f"stabilized_temp_{os.getpid()}.mp4")

    try:
        stabilize_video(input_path, stabilized_tmp, temp_dir)

        # Verify stabilization succeeded
        if not os.path.exists(stabilized_tmp):
            raise ValueError(
                "Video stabilization failed - no intermediate file created"
            )

        width, height = TARGET_RESOLUTION.split("x")
        vf_filters = (
            "yadif,"
            "hqdn3d,"
            "eq=contrast=1.0:brightness=0.0:saturation=1.0,"
            f"scale=-2:{height},"
            "pad=ceil(iw/2)*2:ceil(ih/2)*2,"
            f"fps={TARGET_FRAMERATE}"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            stabilized_tmp,
            "-vf",
            vf_filters,
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-preset",
            "fast",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            output_path,
        ]

        # Capture and check stderr
        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            raise ValueError(f"Video processing failed: {result.stderr}")

        # Verify output file was created and has content
        if not os.path.exists(output_path):
            raise ValueError(f"Video processing failed - no output file: {output_path}")
        if os.path.getsize(output_path) == 0:
            raise ValueError(
                f"Video processing failed - empty output file: {output_path}"
            )

    finally:
        # Clean up temp file
        if os.path.exists(stabilized_tmp):
            os.remove(stabilized_tmp)


def extract_audio(input_path: str, output_audio_path: str) -> None:
    """Extract audio from video file."""
    if not os.path.exists(input_path):
        raise ValueError(f"Input video not found: {input_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_audio_path,
    ]

    # Check return code and stderr
    result = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        raise ValueError(f"Audio extraction failed: {result.stderr}")

    # Verify output file was created
    if not os.path.exists(output_audio_path):
        raise ValueError(
            f"Audio extraction failed - no output file: {output_audio_path}"
        )


def safe_float_conversion(
    value: float | int | str | None, default: str = "n/a"
) -> float | str:
    """Convert value to float, return default if conversion fails."""
    if value is None or pd.isna(value):
        return default

    # Convert to string and check for common non-numeric indicators
    str_val = str(value).strip().lower()
    if str_val in ["", "n/a", "na", "nan", "none", "null"]:
        return default

    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# BIDS file creation functions
def create_events_file(
    group_df: pd.DataFrame, output_path: str, full_filepath: str
) -> None:
    """Create events.tsv file from Excel data with all columns."""
    events_data = []

    for idx, row in group_df.iterrows():
        event = {
            "onset": 0.0,
            "duration": parse_duration(row.get("Vid_duration", "00:00:00")),
            "coder": str(row.get("Coder", "n/a")),
            "filepath_engaging": str(full_filepath),
            "source_file": str(row.get("SourceFile", "n/a")),
            "context": str(row.get("Context", "n/a")),
            "location": str(row.get("Location", "n/a")),
            "activity": str(row.get("Activity", "n/a")),
            "child_clear": str(row.get("Child_of_interest_clear", "n/a")),
            "num_adults": str(row.get("#_adults", "n/a")),
            "num_children": str(row.get("#_children", "n/a")),
            "num_people_background": str(row.get("#_people_background", "n/a")),
            "interaction_with_child": str(row.get("Interaction_with_child", "n/a")),
            "num_people_interacting": str(row.get("#_people_interacting", "n/a")),
            "child_constrained": str(row.get("Child_constrained", "n/a")),
            "constraint_type": str(row.get("Constraint_type", "n/a")),
            "supports": str(row.get("Supports", "n/a")),
            "support_type": str(row.get("Support_type", "n/a")),
            "example_support_type": str(row.get("Example_support_type", "n/a")),
            "gestures": str(row.get("Gestures", "n/a")),
            "gesture_type": str(row.get("Gesture_type", "n/a")),
            "vocalizations": str(row.get("Vocalizations", "n/a")),
            "rmm": str(row.get("RMM", "n/a")),
            "rmm_type": str(row.get("RMM_type", "n/a")),
            "response_to_name": str(row.get("Response_to_name", "n/a")),
            "locomotion": str(row.get("Locomotion", "n/a")),
            "locomotion_type": str(row.get("Locomotion_type", "n/a")),
            "grasping": str(row.get("Grasping", "n/a")),
            "grasp_type": str(row.get("Grasp_type", "n/a")),
            "body_parts_visible": str(row.get("Body_Parts_Visible", "n/a")),
            "angle_of_body": str(row.get("Angle_of_Body", "n/a")),
            "timepoint": str(row.get("time_point", "n/a")),
            "dob": str(row.get("DOB", "n/a")),
            "vid_date": str(row.get("Vid_date", "n/a")),
            "video_quality_face": safe_float_conversion(
                row.get("Video_Quality_Child_Face_Visibility")
            ),
            "video_quality_body": safe_float_conversion(
                row.get("Video_Quality_Child_Body_Visibility")
            ),
            "video_quality_hand": safe_float_conversion(
                row.get("Video_Quality_Child_Hand_Visibility")
            ),
            "video_quality_lighting": safe_float_conversion(
                row.get("Video_Quality_Lighting")
            ),
            "video_quality_resolution": safe_float_conversion(
                row.get("Video_Quality_Resolution")
            ),
            "video_quality_motion": safe_float_conversion(
                row.get("Video_Quality_Motion")
            ),
            "notes": str(row.get("Notes", "n/a")),
        }
        events_data.append(event)

    events_df = pd.DataFrame(events_data)
    events_df.to_csv(output_path, sep="\t", index=False, na_rep="n/a")


def create_video_metadata_json(
    metadata: Dict[str, Any],
    processing_info: Dict[str, Any],
    task_info: Dict[str, Any],
    output_path: str,
) -> None:
    """Create JSON metadata file for processed video with dynamic task info."""
    video_json = {
        "TaskName": task_info.get("task_name", "unknown"),
        "TaskDescription": task_info.get(
            "task_description", "Video recorded during behavioral session"
        ),
        "Instructions": task_info.get(
            "instructions", "Natural behavior in home environment"
        ),
        "Context": task_info.get("context", "n/a"),
        "Activity": task_info.get("activity", "n/a"),
        "FrameRate": TARGET_FRAMERATE,
        "Resolution": TARGET_RESOLUTION,
        "ProcessingPipeline": {
            "Stabilization": processing_info.get("has_stabilization", False),
            "Denoising": processing_info.get("has_denoising", False),
            "Equalization": processing_info.get("has_equalization", False),
            "StandardizedFPS": TARGET_FRAMERATE,
            "StandardizedResolution": TARGET_RESOLUTION,
        },
        "OriginalMetadata": metadata,
    }
    save_json(video_json, output_path)


def create_audio_metadata_json(
    duration_sec: float, task_info: Dict[str, Any], output_path: str
) -> None:
    """Create JSON metadata file for extracted audio with dynamic task info."""
    audio_json = {
        "SamplingFrequency": 16000,
        "Channels": 1,
        "SampleEncoding": "16bit",
        "Duration": duration_sec,
        "TaskName": task_info.get("task_name", "unknown"),
        "TaskDescription": task_info.get(
            "task_description", "Audio extracted from behavioral session"
        ),
        "Context": task_info.get("context", "n/a"),
        "Activity": task_info.get("activity", "n/a"),
    }
    save_json(audio_json, output_path)


def create_raw_video_json(
    row: pd.Series, task_info: Dict[str, Any], video_path: str, output_path: str
) -> None:
    """Create JSON metadata for raw video."""
    video_props = get_video_properties(video_path)

    video_json = {
        "TaskName": task_info.get("task_name", "unknown"),
        "TaskDescription": task_info.get(
            "task_description", "Raw video from behavioral session"
        ),
        "FrameRate": video_props.get("FrameRate", "n/a"),
        "Resolution": video_props.get("Resolution", "n/a"),
        "OriginalFilename": str(row.get("FileName", "")),
        "Duration": parse_duration(row.get("Vid_duration", "00:00:00")),
        "RecordingDate": str(row.get("Vid_date", "n/a")),
        "Context": task_info.get("context", "n/a"),
        "Activity": task_info.get("activity", "n/a"),
        "TimePoint": str(row.get("timepoint", "n/a")),
        "SourceFile": str(row.get("SourceFile", "n/a")),
    }
    save_json(video_json, output_path)


def process_single_video(
    video_info: Dict,
    annotation_df: pd.DataFrame,
    final_bids_root: str,
    final_derivatives_dir: str,
    temp_dir: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Process a single video with all BIDS structures robustly."""
    try:
        # --- Validate input --------------------------------------------------
        if not video_info or not isinstance(video_info, dict):
            raise ValueError("video_info is empty or invalid")

        required_keys = ["participant_id", "filename", "session_id", "full_path"]
        missing = [k for k in required_keys if k not in video_info]
        if missing:
            raise ValueError(f"Missing required video_info keys: {missing}")

        participant_id = video_info["participant_id"]
        filename = video_info["filename"]
        session_id = video_info["session_id"]
        input_video_path = video_info["full_path"]

        safe_print(f"Processing: {participant_id}/{filename}")
        filename_without_extension = os.path.splitext(filename)[0]

        # --- Handle empty or invalid annotation_df ---------------------------
        if annotation_df is None or annotation_df.empty:
            safe_print("Annotation DataFrame is empty - using dummy data")
            video_excel = pd.DataFrame(
                [create_dummy_excel_data(input_video_path, participant_id, session_id)]
            )
            has_excel_data = False
        else:
            # Ensure expected columns exist
            expected_cols = {"ID", "FileName"}
            if not expected_cols.issubset(annotation_df.columns):
                safe_print(
                    "Annotation DataFrame missing required columns - using dummy data"
                )
                video_excel = pd.DataFrame(
                    [
                        create_dummy_excel_data(
                            input_video_path, participant_id, session_id
                        )
                    ]
                )
                has_excel_data = False
            else:
                # Normal Excel lookup
                participant_excel = annotation_df[
                    annotation_df["ID"].astype(str) == str(participant_id)
                ]
                mask = (
                    participant_excel["FileName"].str.split(".").str[0]
                    == filename_without_extension
                )
                video_excel = participant_excel[mask]
                if video_excel.empty:
                    safe_print("No Excel data found - using dummy data")
                    video_excel = pd.DataFrame(
                        [
                            create_dummy_excel_data(
                                input_video_path, participant_id, session_id
                            )
                        ]
                    )
                    has_excel_data = False
                else:
                    has_excel_data = True

        excel_row = video_excel.iloc[0]
        task_label = get_task_from_excel_row(excel_row)
        activity = excel_row.get("Activity", "unknown activity")

        # --- Build task info -------------------------------------------------
        task_info = {
            "task_name": task_label,
            "task_description": f"Behavioral session: {activity}",
            "instructions": "Natural behavior observation",
            "context": str(excel_row.get("Context", "n/a")),
            "activity": str(excel_row.get("Activity", "n/a")),
        }

        # --- Directory setup -------------------------------------------------
        raw_subj_dir = os.path.join(
            final_bids_root, f"sub-{participant_id}", f"ses-{session_id}", "beh"
        )
        deriv_subj_dir = os.path.join(
            final_derivatives_dir, f"sub-{participant_id}", f"ses-{session_id}", "beh"
        )
        os.makedirs(raw_subj_dir, exist_ok=True)
        os.makedirs(deriv_subj_dir, exist_ok=True)

        # --- File naming -----------------------------------------------------
        ext = os.path.splitext(filename)[1]
        run_number = get_next_run_number(
            participant_id, session_id, task_label, final_bids_root
        )

        raw_video_name = create_bids_filename(
            participant_id, session_id, task_label, "beh", "mp4", run_number
        )
        processed_video_name = create_bids_filename(
            participant_id,
            session_id,
            task_label,
            "desc-processed_beh",
            "mp4",
            run_number,
        )
        audio_name = create_bids_filename(
            participant_id, session_id, task_label, "audio", "wav", run_number
        )
        events_name = create_bids_filename(
            participant_id, session_id, task_label, "events", "tsv", run_number
        )

        # --- Paths -----------------------------------------------------------
        raw_video_path = os.path.join(raw_subj_dir, raw_video_name)
        processed_video_path = os.path.join(deriv_subj_dir, processed_video_name)
        audio_path = os.path.join(deriv_subj_dir, audio_name)
        events_path = os.path.join(raw_subj_dir, events_name)

        # --- Raw video preparation ------------------------------------------
        if not os.path.exists(raw_video_path):
            if ext.lower() != ".mp4":
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_video_path,
                    "-c",
                    "copy",
                    raw_video_path,
                ]
                result = subprocess.run(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
                )
                if result.returncode != 0 or not os.path.exists(raw_video_path):
                    raise ValueError(f"FFmpeg conversion failed: {result.stderr}")
                safe_print("  Converted to raw BIDS format")
            else:
                shutil.copy2(input_video_path, raw_video_path)
                if not os.path.exists(raw_video_path):
                    raise ValueError(f"Failed to copy to raw BIDS: {raw_video_path}")
                safe_print("  Copied to raw BIDS")

        # --- Metadata extraction --------------------------------------------
        exif_data = extract_exif(raw_video_path)
        if (
            not isinstance(exif_data, dict)
            or "error" in exif_data
            or "ffprobe_error" in exif_data
        ):
            raise ValueError("Unreadable or unsupported video format")

        # --- Video processing -----------------------------------------------
        if not os.path.exists(processed_video_path):
            safe_print("  Starting video processing...")
            preprocess_video(raw_video_path, processed_video_path, temp_dir)
            if (
                not os.path.exists(processed_video_path)
                or os.path.getsize(processed_video_path) == 0
            ):
                raise ValueError("Video processing failed - no valid output")
            safe_print("  Video processing complete")

        # --- Audio extraction -----------------------------------------------
        if not os.path.exists(audio_path):
            safe_print("  Extracting audio...")
            extract_audio(processed_video_path, audio_path)
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                raise ValueError("Audio extraction failed - no valid output")
            safe_print("  Audio extraction complete")

        # --- Events file ----------------------------------------------------
        create_events_file(video_excel, events_path, input_video_path)
        if not os.path.exists(events_path):
            raise ValueError(f"Failed to create events file: {events_path}")

        # --- Metadata JSONs -------------------------------------------------
        processing_info = {
            "has_stabilization": True,
            "has_denoising": True,
            "has_equalization": True,
        }

        raw_video_json_path = raw_video_path.replace(".mp4", ".json")
        create_raw_video_json(excel_row, task_info, raw_video_path, raw_video_json_path)
        if not os.path.exists(raw_video_json_path):
            raise ValueError(f"Failed to create raw video JSON: {raw_video_json_path}")

        processed_video_json_path = processed_video_path.replace(".mp4", ".json")
        create_video_metadata_json(
            exif_data, processing_info, task_info, processed_video_json_path
        )
        if not os.path.exists(processed_video_json_path):
            raise ValueError(
                f"Failed to create processed video JSON: {processed_video_json_path}"
            )

        audio_json_path = audio_path.replace(".wav", ".json")
        create_audio_metadata_json(
            exif_data.get("duration_sec", 0), task_info, audio_json_path
        )
        if not os.path.exists(audio_json_path):
            raise ValueError(f"Failed to create audio JSON: {audio_json_path}")

        # --- Success return -------------------------------------------------
        entry = {
            "participant_id": participant_id,
            "session_id": session_id,
            "task_label": task_label,
            "original_video": input_video_path,
            "raw_video_bids": raw_video_path,
            "processed_video_bids": processed_video_path,
            "audio_file_bids": audio_path,
            "events_file_bids": events_path,
            "filename": filename,
            "age_folder": video_info.get("age_folder", "n/a"),
            "duration_sec": exif_data.get("duration_sec", 0),
            "has_excel_data": has_excel_data,
            "excel_metadata": excel_row.to_dict(),
            "task_info": task_info,
            "processing_info": processing_info,
        }

        safe_print(f"  Successfully processed: {participant_id}/{filename}")
        return entry, None

    except Exception as e:
        safe_print(
            f"  ERROR processing {video_info.get('full_path', 'unknown file')}:"
            f" {str(e)}"
        )
        return None, {"video": video_info.get("full_path", "unknown"), "error": str(e)}


def create_dataset_description() -> None:
    """Create dataset_description.json for main BIDS dataset."""
    dataset_desc = {
        "Name": "SAILS Phase III Home Videos",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
    }
    try:
        filepath = os.path.join(FINAL_BIDS_ROOT, "dataset_description.json")
        save_json(dataset_desc, filepath)

    except Exception as e:
        raise ValueError(
            f"Failed to create dataset_description.json at {filepath}: {e}"
        )


def create_derivatives_dataset_description() -> None:
    """Create dataset_description.json for derivatives."""
    os.makedirs(FINAL_DERIVATIVES_DIR, exist_ok=True)

    derivatives_desc = {
        "Name": "SAILS Phase III Home Videos - Preprocessed",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "Video Preprocessing Pipeline",
                "Version": "1.0.0",
                "Description": (
                    "FFmpeg-based video stabilization, denoising, "
                    "and standardization pipeline with audio extraction"
                ),
                "CodeURL": "local",
            }
        ],
        "SourceDatasets": [{"URL": "", "Version": "1.0.0"}],
        "HowToAcknowledge": "Please cite the original study",
    }

    filepath = os.path.join(FINAL_DERIVATIVES_DIR, "dataset_description.json")
    save_json(derivatives_desc, filepath)
    if not os.path.exists(filepath):
        raise ValueError(
            f"Failed to create derivatives dataset_description.json at {filepath}"
        )


def create_readme() -> None:
    """Create README file for the BIDS dataset."""
    readme_content = """# SAILS Phase III Home Videos BIDS Dataset

## Overview
This dataset contains home videos from the SAILS Phase III study,
organized according to the Brain Imaging Data Structure (BIDS) specification.

## Data Collection
Videos were collected from home environments during various activities.
Two main age groups were included:
- Session 01: 12-16 month old children
- Session 02: 34-38 month old children

## Dataset Structure
### Raw Data
- sub-*/ses-*/beh/: Raw behavioral videos (converted to mp4) and event
annotations (contains also the original filepath of the video processed)

### Derivatives
- derivatives/preprocessed/sub-*/ses-*/beh/: Processed videos and extracted audio
  - Videos: Stabilized, denoised, standardized to 720p/30fps
  - Audio: Extracted to 16kHz mono WAV format

## Data Processing
All videos underwent standardized preprocessing including:
- Video stabilization using vidstab
- Denoising and quality enhancement
- Standardization to 720p resolution and 30fps
- Audio extraction for speech analysis
- Filename modication according to subject ID and task label
- Extraction of ASD status for every subject stored in the participants.tsv file.

## Behavioral Coding
Events files include manual annotations from csv file and Engaging
location of the raw video.

## Task Labels
Task labels are derived from the Context column in the csv.
It allows to capture what kind of interaction was happening in the video.
Videos without behavioral coding data use "unknown" task label.
"""

    filepath = os.path.join(FINAL_BIDS_ROOT, "README")
    try:
        with open(filepath, "w") as f:
            f.write(readme_content)
    except Exception as e:
        raise ValueError(f"Failed to create README at {filepath}: {e}")


def create_participants_file(
    final_bids_root: str = FINAL_BIDS_ROOT, asd_status_file: str = ASD_STATUS_FILE
) -> None:
    """Create participants.tsv and participants.json files."""
    if not os.path.exists(asd_status_file):
        raise FileNotFoundError(f"ASD status file not found: {asd_status_file}")

    asd_status = pd.read_excel(asd_status_file)
    ids_processed_participants = []
    for name in os.listdir(final_bids_root):
        full_path = os.path.join(final_bids_root, name)
        if os.path.isdir(full_path) and name.startswith("sub-"):
            ids_processed_participants.append(name.split("sub-")[1])
    participants_data = []
    for participant_id in sorted(ids_processed_participants):
        asd_info = asd_status[asd_status["ID"].astype(str) == str(participant_id)]
        participants_data.append(
            {
                "participant_id": f"sub-{participant_id}",
                "group": asd_info["Group"].values[0] if not asd_info.empty else "n/a",
            }
        )

    participants_df = pd.DataFrame(participants_data)
    participants_df.to_csv(
        os.path.join(final_bids_root, "participants.tsv"),
        sep="\t",
        index=False,
        na_rep="n/a",
    )

    participants_json = {
        "participant_id": {"Description": "Unique BIDS participant identifier"},
        "Group": {"Description": "ASD status"},
    }

    save_json(participants_json, os.path.join(final_bids_root, "participants.json"))


def print_summary(all_processed: List[Dict], all_failed: List[Dict]) -> None:
    """Print processing summary statistics."""
    print("PROCESSING SUMMARY")

    print(f"Successfully processed: {len(all_processed)} videos")
    print(f"Failed to process: {len(all_failed)} videos")
    print(f"Total videos attempted: {len(all_processed) + len(all_failed)}")

    if all_processed:
        # Excel data availability
        with_excel = sum(
            1 for entry in all_processed if entry.get("has_excel_data", False)
        )
        without_excel = len(all_processed) - with_excel
        print("\nData sources:")
        print(f"  With Excel behavioral data: {with_excel} videos")
        print(f"  With dummy behavioral data: {without_excel} videos")

        # Task distribution
        task_counts: dict[str, int] = {}
        participant_counts: dict[str, int] = {}
        session_counts: dict[str, int] = {}

        for entry in all_processed:
            task = entry["task_label"]
            participant = entry["participant_id"]
            session = entry["session_id"]
            task_counts[task] = task_counts.get(task, 0) + 1
            participant_counts[participant] = participant_counts.get(participant, 0) + 1
            session_counts[session] = session_counts.get(session, 0) + 1

        print("\nTask distribution:")
        for task, count in sorted(task_counts.items()):
            print(f"  {task}: {count} videos")

        print("\nSession distribution:")
        for session, count in sorted(session_counts.items()):
            print(f"  Session {session}: {count} videos")

        print(f"\nUnique participants processed: {len(participant_counts)}")

        # Duration statistics
        durations = [entry.get("duration_sec", 0) for entry in all_processed]
        total_duration = sum(durations)
        avg_duration = total_duration / len(durations) if durations else 0

        print("\nDuration statistics:")
        print(f"  Total video duration: {total_duration/3600:.1f} hours")
        print(f"  Average video duration: {avg_duration/60:.1f} minutes")

    if all_failed:
        print("\nFailed videos breakdown:")
        error_types: dict[str, int] = {}
        for entry in all_failed:
            error = entry.get("error", "Unknown error")
            error_types[error] = error_types.get(error, 0) + 1

        for error, count in sorted(error_types.items()):
            print(f"  {error}: {count} videos")


def merge_subjects(final_bids_root: str = FINAL_BIDS_ROOT) -> None:
    """Merge duplicated subject folders safely."""
    paths_to_check = [
        Path(final_bids_root),
        Path(final_bids_root) / "derivatives" / "preprocessed",
    ]

    for folder in paths_to_check:
        if not folder.exists():
            continue

        subs = [d for d in folder.iterdir() if d.is_dir() and d.name.startswith("sub-")]
        sub_names = {d.name for d in subs}

        for sub in subs:
            if sub.name.endswith(" 2"):
                original_name = sub.name[:-2]
                original_path = folder / original_name
                if original_name in sub_names and original_path.exists():
                    print(f"Merging {sub} → {original_path}")

                    for item in sub.iterdir():
                        dest = original_path / item.name
                        if item.is_dir():
                            if dest.exists():
                                if dest.is_file():
                                    print(
                                        f"Conflict: {dest} is a file, "
                                        "expected a folder. Skipping."
                                    )
                                    continue
                                # merge recursively if same session already exists
                                for subitem in item.iterdir():
                                    dest_sub = dest / subitem.name
                                    if dest_sub.exists():
                                        # type conflict handling
                                        if dest_sub.is_file() != subitem.is_file():
                                            print(
                                                f"Type conflict for {dest_sub}, "
                                                "skipping."
                                            )
                                            continue
                                    if subitem.is_dir():
                                        shutil.copytree(
                                            subitem, dest_sub, dirs_exist_ok=True
                                        )
                                    else:
                                        shutil.copy2(subitem, dest_sub)
                            else:
                                shutil.copytree(item, dest)
                        else:
                            if dest.exists():
                                if dest.is_dir():
                                    print(
                                        f"Conflict: {dest} is a directory,"
                                        " expected a file. Skipping."
                                    )
                                    continue
                            shutil.copy2(item, dest)

                    shutil.rmtree(sub)
                else:
                    print(f"No base subject found for {sub}, skipping.")


def process_videos(
    task_id: int,
    num_tasks: int,
    annotation_df: pd.DataFrame,
    all_videos: list,
    final_bids_root: str,
    final_derivatives_dir: str,
    output_dir: str,
) -> tuple[list, list]:
    """Process the subset of videos assigned to this task.

    Returns:
        (all_processed, all_failed)
    """
    safe_print(f"Task {task_id}: Processing videos...")
    video_chunks = all_videos[task_id::num_tasks]

    if not video_chunks:
        safe_print(f"No videos assigned to task {task_id}")
        return [], []

    temp_dir = os.path.join(output_dir, str(task_id), "temp")
    os.makedirs(temp_dir, exist_ok=True)

    all_processed, all_failed = [], []

    for i, video_info in enumerate(video_chunks, 1):
        safe_print(f"[Task {task_id}] Video {i}/{len(video_chunks)}")
        processed_entry, failed_entry = process_single_video(
            video_info,
            annotation_df,
            final_bids_root,
            final_derivatives_dir,
            temp_dir,
        )
        if processed_entry:
            all_processed.append(processed_entry)
        if failed_entry:
            all_failed.append(failed_entry)

    # Save per-task logs
    task_dir = os.path.join(output_dir, str(task_id))
    os.makedirs(task_dir, exist_ok=True)
    save_json(all_processed, os.path.join(task_dir, "processing_log.json"))
    save_json(all_failed, os.path.join(task_dir, "not_processed.json"))

    # Cleanup temp dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    return all_processed, all_failed


def main() -> None:
    """Main entry point for multi-task BIDS video processing."""
    parser = argparse.ArgumentParser(
        description="Run updated_bids with task and total number of tasks."
    )
    parser.add_argument("task_id", type=int, help="ID of the current task")
    parser.add_argument("num_tasks", type=int, help="Total number of tasks")

    args = parser.parse_args()
    my_task_id = args.task_id
    num_tasks = args.num_tasks

    print(f"Running task {my_task_id}/{num_tasks}")

    start_time = time.time()

    # --- Validate paths ---
    for path, label in [(VIDEO_ROOT, "Video root"), (ANNOTATION_FILE, "Excel file")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found at {path}")
            sys.exit(1)
            return

    # --- Load metadata ---
    try:
        annotation_df = pd.read_csv(ANNOTATION_FILE)
        annotation_df.columns = annotation_df.columns.str.strip()
        safe_print(f"Loaded {len(annotation_df)} rows from Excel file")
    except Exception as e:
        safe_print(f"ERROR: Failed to load Excel file: {e}")
        sys.exit(1)
        return

    # --- Discover videos ---
    safe_print("Discovering videos...")
    all_videos = get_all_videos(VIDEO_ROOT, annotation_df)
    if not all_videos:
        safe_print("ERROR: No videos found.")
        sys.exit(1)
    safe_print(f"Found {len(all_videos)} video files.")

    # --- Create BIDS structure (only once) ---
    if my_task_id == 0:
        try:
            safe_print("Creating BIDS structure files...")
            create_bids_structure()
            create_dataset_description()
            create_derivatives_dataset_description()
            create_readme()
        except Exception as e:
            safe_print(f"CRITICAL ERROR: Failed to create BIDS structure files: {e}")
            sys.exit(1)

    # --- Process this task’s subset ---
    all_processed, all_failed = process_videos(
        my_task_id,
        num_tasks,
        annotation_df,
        all_videos,
        FINAL_BIDS_ROOT,
        FINAL_DERIVATIVES_DIR,
        OUTPUT_DIR,
    )

    # --- Final summary ---
    total_time = time.time() - start_time
    print_summary(all_processed, all_failed)
    safe_print(
        f"Total processing time: {total_time / 3600:.1f}"
        f" hours ({total_time / 60:.1f} minutes)"
    )

    if all_processed:
        avg_time = total_time / len(all_processed)
        safe_print(f"Average time per video: {avg_time:.1f} seconds")

    safe_print("Processing complete ✅")


if __name__ == "__main__":
    main()
