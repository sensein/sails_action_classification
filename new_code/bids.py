# Standard library imports
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

# Third-party imports
import pandas as pd
import numpy as np
import cv2

def safe_print(message: str):
    """Print with timestamps."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{timestamp} [MAIN] {message}")

# Helper functions
def parse_duration(duration_str) -> float:
    """Parse duration string to seconds"""
    try:
        if pd.isna(duration_str) or duration_str == '':
            return 0.0
        duration_str = str(duration_str)
        if ':' in duration_str:
            parts = duration_str.split(':')
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
    except:
        return 0.0

def make_bids_task_label(task_name):
    """Convert TaskName to BIDS-compatible task label for filenames."""
    s = str(task_name).strip()
    s = re.sub(r'[^0-9a-zA-Z+]', '', s)  # Keep only alphanumeric and +
    return s

def get_video_properties(video_path):
    """Extract video properties using OpenCV"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"SamplingFrequency": None, "Resolution": None}

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return {
            "SamplingFrequency": fps,
            "Resolution": f"{width}x{height}",
        }
    except:
        return {"SamplingFrequency": None, "Resolution": None}

def determine_session_from_folder(folder_name: str) -> Optional[str]:
    """Determine session ID from folder names with spaces."""
    folder_lower = folder_name.lower()

    # Check for 12-16 month patterns (including spaces and variations)
    if any(pattern in folder_lower for pattern in [
        '12-16 month', '12-14 month', '12_16', '12_14', '12-16month', '12-14month', '12-16_month_videos'
    ]):
        return "01"

    # Check for 34-38 month patterns (including spaces, typos, and variations)
    elif any(pattern in folder_lower for pattern in [
        '34-38 month', '34-28 month', '34-48 month', '34_38', '34_28', '34_48',
        '34-38month', '34-28month', '34-48month','34-38_month_videos'
    ]):
        return "02"

    return None

def find_age_folder_session(current_path: str, participant_path: str) -> Optional[str]:
    """Recursively check if current path or any parent path contains age-related folder pattern."""
    if not current_path.startswith(participant_path) or current_path == participant_path:
        return None

    current_folder = os.path.basename(current_path)
    session_id = determine_session_from_folder(current_folder)
    if session_id:
        return session_id

    parent_path = os.path.dirname(current_path)
    return find_age_folder_session(parent_path, participant_path)

def find_all_videos_recursive(directory: str, participant_path: str) -> List[Tuple[str, Optional[str]]]:
    """Recursively find all video files in a directory and determine their session."""
    videos = []

    try:
        for item in os.listdir(directory):
            if item.startswith('.'):  # Skip hidden files
                continue

            item_path = os.path.join(directory, item)

            if os.path.isfile(item_path):
                if item.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.m4v', '.3gp', '.mts')):
                    session_id = find_age_folder_session(directory, participant_path)
                    videos.append((item_path, session_id))

            elif os.path.isdir(item_path):
                videos.extend(find_all_videos_recursive(item_path, participant_path))

    except PermissionError:
        print(f"Permission denied: {directory}")
    except Exception as e:
        print(f"Error accessing {directory}: {e}")

    return videos

def extract_participant_id_from_folder(folder_name: str) -> str:
    """Extract participant ID from folder names like 'A.A._Home_Videos_AMES_A2P7X9N8L7'."""
    if 'AMES_' in folder_name:
        parts = folder_name.split('AMES_')
        if len(parts) > 1:
            return parts[1].strip()

    if '_' in folder_name:
        return folder_name.split('_')[-1]

    return folder_name

def get_all_videos_from_age_folders(video_root):
    """Find ALL videos in age folders regardless of Excel file."""
    all_videos = []

    try:
        for participant_folder in os.listdir(video_root):
            participant_path = os.path.join(video_root, participant_folder)
            if not os.path.isdir(participant_path):
                continue

            participant_id = extract_participant_id_from_folder(participant_folder)
            if not participant_id:
                continue

            participant_videos = find_all_videos_recursive(participant_path, participant_path)

            for video_path, session_id in participant_videos:
                if session_id in ['01', '02']:
                    all_videos.append({
                        'participant_id': participant_id,
                        'filename': os.path.basename(video_path),
                        'full_path': video_path,
                        'session_id': session_id,
                        'age_folder': os.path.basename(os.path.dirname(video_path))
                    })

    except Exception as e:
        print(f"Error scanning video folders: {e}")

    return all_videos

def create_dummy_excel_data(video_path, participant_id, session_id, task_label="unknown"):
    """Create dummy behavioral data for videos not in Excel file."""
    video_filename = os.path.basename(video_path)

    dummy_row_data = {
        'ID': participant_id,
        'FileName': video_filename,
        'Context': task_label,
        'Location': 'n/a',
        'Activity': 'n/a',
        'Child_of_interest_clear': 'n/a',
        '#_adults': 'n/a',
        '#_children': 'n/a',
        '#_people_background': 'n/a',
        'Interaction_with_child': 'n/a',
        '#_people_interacting': 'n/a',
        'Child_constrained': 'n/a',
        'Constraint_type': 'n/a',
        'Supports': 'n/a',
        'Support_type': 'n/a',
        'Example_support_type': 'n/a',
        'Gestures': 'n/a',
        'Gesture_type': 'n/a',
        'Vocalizations': 'n/a',
        'RMM': 'n/a',
        'RMM_type': 'n/a',
        'Response_to_name': 'n/a',
        'Locomotion': 'n/a',
        'Locomotion_type': 'n/a',
        'Grasping': 'n/a',
        'Grasp_type': 'n/a',
        'Body_Parts_Visible': 'n/a',
        'Angle_of_Body': 'n/a',
        'time_point': 'n/a',
        'DOB': 'n/a',
        'Vid_date': 'n/a',
        'Video_Quality_Child_Face_Visibility': 'n/a',
        'Video_Quality_Child_Body_Visibility': 'n/a',
        'Video_Quality_Child_Hand_Visibility': 'n/a',
        'Video_Quality_Lighting': 'n/a',
        'Video_Quality_Resolution': 'n/a',
        'Video_Quality_Motion': 'n/a',
        'Coder': 'n/a',
        'SourceFile': 'n/a',
        'Vid_duration': '00:00:00',
        'Notes': 'Video not found in Excel file - behavioral data unavailable'
    }

    return dummy_row_data

def get_task_from_excel_row(row: pd.Series) -> str:
    """Extract and create task label from Excel row data."""
    context = str(row.get('Context', '')).strip()

    if context and context.lower() not in ['nan', 'n/a', '']:
        return make_bids_task_label(context)
    else:
        return "unknown"
        
def get_next_run_number(participant_id: str, session_id: str, task_label: str, 
                       final_bids_root: str) -> int:
    """Find the next available run number for this participant/session/task."""
    beh_dir = os.path.join(final_bids_root, f"sub-{participant_id}", f"ses-{session_id}", "beh")
    
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
    
def create_bids_filename(participant_id: str, session_id: str, task_label: str, 
                        suffix: str, extension: str, run_id: int = 1) -> str:
    """Create BIDS-compliant filename with run identifier for multiple videos per task."""
    return f"sub-{participant_id}_ses-{session_id}_task-{task_label}_run-{run_id:02d}_{suffix}.{extension}"

# Video processing functions
def extract_exif(video_path: str) -> Dict[str, Any]:
    """Extract video metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
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
    """Stabilize video using ffmpeg vidstab."""
    transforms_file = os.path.join(temp_dir, "transforms.trf")

    detect_cmd = [
        "ffmpeg", "-i", input_path,
        "-vf", f"vidstabdetect=shakiness=5:accuracy=15:result={transforms_file}",
        "-f", "null", "-"
    ]
    subprocess.run(detect_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    transform_cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"vidstabtransform=smoothing=30:input={transforms_file}",
        "-c:v", "libx264", "-preset", "slow", "-crf", "23",
        "-c:a", "copy", stabilized_path
    ]
    subprocess.run(transform_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(transforms_file):
        os.remove(transforms_file)

def preprocess_video(input_path: str, output_path: str, temp_dir: str, target_framerate: int) -> None:
    """Preprocess video with stabilization, denoising, and standardization."""
    if not os.path.exists(input_path):
        raise ValueError(f"Input video not found: {input_path}")
        
    stabilized_tmp = os.path.join(temp_dir, f"stabilized_temp_{os.getpid()}.mp4")

    try:
        stabilize_video(input_path, stabilized_tmp, temp_dir)
        
        # Verify stabilization succeeded
        if not os.path.exists(stabilized_tmp):
            raise ValueError("Video stabilization failed - no intermediate file created")

        vf_filters = (
            "yadif,"
            "hqdn3d,"
            "eq=contrast=1.0:brightness=0.0:saturation=1.0,"
            "scale=-2:720,"
            "pad=ceil(iw/2)*2:ceil(ih/2)*2,"
            f"fps={target_framerate}"
        )

        cmd = [
            "ffmpeg", "-y", "-i", stabilized_tmp,
            "-vf", vf_filters,
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]
        
        # Capture and check stderr
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise ValueError(f"Video processing failed: {result.stderr}")
            
        # Verify output file was created and has content
        if not os.path.exists(output_path):
            raise ValueError(f"Video processing failed - no output file: {output_path}")
        if os.path.getsize(output_path) == 0:
            raise ValueError(f"Video processing failed - empty output file: {output_path}")

    finally:
        # Clean up temp file
        if os.path.exists(stabilized_tmp):
            os.remove(stabilized_tmp)

def extract_audio(input_path: str, output_audio_path: str) -> None:
    """Extract audio from video file."""
    if not os.path.exists(input_path):
        raise ValueError(f"Input video not found: {input_path}")
        
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output_audio_path,
    ]
    
    # Check return code and stderr
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise ValueError(f"Audio extraction failed: {result.stderr}")
    
    # Verify output file was created
    if not os.path.exists(output_audio_path):
        raise ValueError(f"Audio extraction failed - no output file: {output_audio_path}")


def safe_float_conversion(value, default='n/a'):
    """Safely convert value to float, return default if conversion fails."""
    if pd.isna(value):
        return default

    # Convert to string and check for common non-numeric indicators
    str_val = str(value).strip().lower()
    if str_val in ['', 'n/a', 'na', 'nan', 'none', 'null']:
        return default

    try:
        return float(value)
    except (ValueError, TypeError):
        return default
        
# BIDS file creation functions
def create_events_file(group_df: pd.DataFrame, output_path: str) -> None:
    """Create events.tsv file from Excel data with all columns."""
    events_data = []

    for idx, row in group_df.iterrows():
        event = {
            'onset': 0.0,
            'duration': parse_duration(row.get('Vid_duration', '00:00:00')),
            'coder': str(row.get('Coder', 'n/a')),
            'source_file': str(row.get('SourceFile', 'n/a')),
            'context': str(row.get('Context', 'n/a')),
            'location': str(row.get('Location', 'n/a')),
            'activity': str(row.get('Activity', 'n/a')),
            'child_clear': str(row.get('Child_of_interest_clear', 'n/a')),
            'num_adults': str(row.get('#_adults', 'n/a')),
            'num_children': str(row.get('#_children', 'n/a')),
            'num_people_background': str(row.get('#_people_background', 'n/a')),
            'interaction_with_child': str(row.get('Interaction_with_child', 'n/a')),
            'num_people_interacting': str(row.get('#_people_interacting', 'n/a')),
            'child_constrained': str(row.get('Child_constrained', 'n/a')),
            'constraint_type': str(row.get('Constraint_type', 'n/a')),
            'supports': str(row.get('Supports', 'n/a')),
            'support_type': str(row.get('Support_type', 'n/a')),
            'example_support_type': str(row.get('Example_support_type', 'n/a')),
            'gestures': str(row.get('Gestures', 'n/a')),
            'gesture_type': str(row.get('Gesture_type', 'n/a')),
            'vocalizations': str(row.get('Vocalizations', 'n/a')),
            'rmm': str(row.get('RMM', 'n/a')),
            'rmm_type': str(row.get('RMM_type', 'n/a')),
            'response_to_name': str(row.get('Response_to_name', 'n/a')),
            'locomotion': str(row.get('Locomotion', 'n/a')),
            'locomotion_type': str(row.get('Locomotion_type', 'n/a')),
            'grasping': str(row.get('Grasping', 'n/a')),
            'grasp_type': str(row.get('Grasp_type', 'n/a')),
            'body_parts_visible': str(row.get('Body_Parts_Visible', 'n/a')),
            'angle_of_body': str(row.get('Angle_of_Body', 'n/a')),
            'timepoint': str(row.get('time_point', 'n/a')),
            'dob': str(row.get('DOB', 'n/a')),
            'vid_date': str(row.get('Vid_date', 'n/a')),
            'video_quality_face': safe_float_conversion(row.get('Video_Quality_Child_Face_Visibility')),
            'video_quality_body': safe_float_conversion(row.get('Video_Quality_Child_Body_Visibility')),
            'video_quality_hand': safe_float_conversion(row.get('Video_Quality_Child_Hand_Visibility')),
            'video_quality_lighting': safe_float_conversion(row.get('Video_Quality_Lighting')),
            'video_quality_resolution': safe_float_conversion(row.get('Video_Quality_Resolution')),
            'video_quality_motion': safe_float_conversion(row.get('Video_Quality_Motion')),
            'notes': str(row.get('Notes', 'n/a'))
        }
        events_data.append(event)

    events_df = pd.DataFrame(events_data)
    events_df.to_csv(output_path, sep='\t', index=False, na_rep='n/a')

def create_video_metadata_json(metadata: Dict[str, Any], processing_info: Dict[str, Any], task_info: Dict[str, Any], output_path: str, target_framerate: int, target_resolution: str) -> None:
    """Create JSON metadata file for processed video with dynamic task info."""
    video_json = {
        "TaskName": task_info.get("task_name", "unknown"),
        "TaskDescription": task_info.get("task_description", "Video recorded during behavioral session"),
        "Instructions": task_info.get("instructions", "Natural behavior in home environment"),
        "Context": task_info.get("context", "n/a"),
        "Activity": task_info.get("activity", "n/a"),
        "SamplingFrequency": target_framerate,
        "Resolution": target_resolution,
        "ProcessingPipeline": {
            "Stabilization": processing_info.get("has_stabilization", False),
            "Denoising": processing_info.get("has_denoising", False),
            "Equalization": processing_info.get("has_equalization", False),
            "StandardizedFPS": target_framerate,
            "StandardizedResolution": target_resolution,
        },
        "OriginalMetadata": metadata,
    }

    with open(output_path, "w") as f:
        json.dump(video_json, f, indent=4)

def create_audio_metadata_json(duration_sec: float, task_info: Dict[str, Any], output_path: str) -> None:
    """Create JSON metadata file for extracted audio with dynamic task info."""
    audio_json = {
        "SamplingFrequency": 16000,
        "Channels": 1,
        "SampleEncoding": "16bit",
        "Duration": duration_sec,
        "TaskName": task_info.get("task_name", "unknown"),
        "TaskDescription": task_info.get("task_description", "Audio extracted from behavioral session"),
        "Context": task_info.get("context", "n/a"),
        "Activity": task_info.get("activity", "n/a"),
    }

    with open(output_path, "w") as f:
        json.dump(audio_json, f, indent=4)

def create_raw_video_json(row, task_info: Dict[str, Any], video_path: str, output_path: str) -> None:
    """Create JSON metadata for raw video."""
    video_props = get_video_properties(video_path)

    video_json = {
        "TaskName": task_info.get("task_name", "unknown"),
        "TaskDescription": task_info.get("task_description", "Raw video from behavioral session"),
        "SamplingFrequency": video_props.get("SamplingFrequency", "n/a"),
        "Resolution": video_props.get("Resolution", "n/a"),
        "OriginalFilename": str(row.get('FileName', '')),
        "Duration": parse_duration(row.get('Vid_duration', '00:00:00')),
        "RecordingDate": str(row.get('Vid_date', 'n/a')),
        "Context": task_info.get("context", "n/a"),
        "Activity": task_info.get("activity", "n/a"),
        "TimePoint": str(row.get('time_point', 'n/a')),
        "SourceFile": str(row.get('SourceFile', 'n/a'))
    }

    with open(output_path, 'w') as f:
        json.dump(video_json, f, indent=4)

def process_single_video(video_info: Dict, excel_df: pd.DataFrame,
                        final_bids_root: str, final_derivatives_dir: str,
                        final_sourcedata_dir: str, temp_dir: str,
                        target_framerate: int, target_resolution: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Process a single video with all BIDS structures."""

    participant_id = video_info['participant_id']
    filename = video_info['filename']
    session_id = video_info['session_id']
    input_video_path = video_info['full_path']

    safe_print(f"Processing: {participant_id}/{filename}")

    try:
        # Check if video exists in Excel or create dummy data
        participant_excel = excel_df[excel_df['ID'].astype(str) == str(participant_id)]
        video_excel = participant_excel[participant_excel['FileName'].astype(str) == filename]

        if video_excel.empty:
            # Create dummy data for missing Excel entries
            dummy_data = create_dummy_excel_data(input_video_path, participant_id, session_id)
            video_excel = pd.DataFrame([dummy_data])
            has_excel_data = False
            safe_print(f"  No Excel data found - using dummy data")
        else:
            has_excel_data = True

        excel_row = video_excel.iloc[0]
        task_label = get_task_from_excel_row(excel_row)

        # Create task information
        task_info = {
            "task_name": task_label,
            "task_description": f"Behavioral session: {excel_row.get('Activity', 'unknown activity')}",
            "instructions": "Natural behavior observation",
            "context": str(excel_row.get('Context', 'n/a')),
            "activity": str(excel_row.get('Activity', 'n/a'))
        }

        # Create BIDS directory structure
        raw_subj_dir = os.path.join(final_bids_root, f"sub-{participant_id}", f"ses-{session_id}", "beh")
        deriv_subj_dir = os.path.join(final_derivatives_dir, f"sub-{participant_id}", f"ses-{session_id}", "beh")
        source_subj_dir = os.path.join(final_sourcedata_dir, f"sub-{participant_id}", f"ses-{session_id}", "video")

        os.makedirs(raw_subj_dir, exist_ok=True)
        os.makedirs(deriv_subj_dir, exist_ok=True)
        os.makedirs(source_subj_dir, exist_ok=True)

        # Create BIDS filenames with run number
        ext = os.path.splitext(filename)[1][1:] 
        run_number = get_next_run_number(participant_id, session_id, task_label, final_bids_root)
        
        raw_video_name = create_bids_filename(participant_id, session_id, task_label, "beh", "mp4", run_number)
        processed_video_name = create_bids_filename(participant_id, session_id, task_label, "desc-processed_beh", "mp4", run_number)
        audio_name = create_bids_filename(participant_id, session_id, task_label, "audio", "wav", run_number)
        events_name = create_bids_filename(participant_id, session_id, task_label, "events", "tsv", run_number)
        source_video_name = create_bids_filename(participant_id, session_id, task_label, "video", ext, run_number)

        # File paths
        raw_video_path = os.path.join(raw_subj_dir, raw_video_name)
        processed_video_path = os.path.join(deriv_subj_dir, processed_video_name)
        audio_path = os.path.join(deriv_subj_dir, audio_name)
        events_path = os.path.join(raw_subj_dir, events_name)
        source_video_path = os.path.join(source_subj_dir, source_video_name)

        # Copy to sourcedata (original, unmodified)
        if not os.path.exists(source_video_path):
            shutil.copy2(input_video_path, source_video_path)
            if not os.path.exists(source_video_path):
                raise ValueError(f"Failed to copy to sourcedata: {source_video_path}")
            safe_print(f"  Copied to sourcedata")

        if not os.path.exists(raw_video_path):
            if ext.lower() != '.mp4':
                # Convert to mp4 without processing
                cmd = ["ffmpeg", "-y", "-i", source_video_path, "-c", "copy", raw_video_path]
                result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                # Check return code and verify output file
                if result.returncode != 0:
                    raise ValueError(f"FFmpeg conversion failed: {result.stderr}")
                if not os.path.exists(raw_video_path):
                    raise ValueError(f"FFmpeg did not create output file: {raw_video_path}")
                safe_print(f"  Converted to raw BIDS format")
            else:
                shutil.copy2(source_video_path, raw_video_path)
                # FIX: Verify copy succeeded
                if not os.path.exists(raw_video_path):
                    raise ValueError(f"Failed to copy to raw BIDS: {raw_video_path}")
                safe_print(f"  Copied to raw BIDS")       

        # Extract metadata from raw video
        exif_data = extract_exif(raw_video_path)
        if "error" in exif_data or "ffprobe_error" in exif_data:
            raise ValueError("Unreadable or unsupported video format")


        # Process video for derivatives
        if not os.path.exists(processed_video_path):
            safe_print(f"  Starting video processing...")
            preprocess_video(raw_video_path, processed_video_path, temp_dir, target_framerate)
            # Verify processing succeeded
            if not os.path.exists(processed_video_path):
                raise ValueError(f"Video processing failed - no output file: {processed_video_path}")
            if os.path.getsize(processed_video_path) == 0:
                raise ValueError(f"Video processing failed - empty output file: {processed_video_path}")
            safe_print(f"  Video processing complete")


        if not os.path.exists(audio_path):
            safe_print(f"  Extracting audio...")
            extract_audio(processed_video_path, audio_path)
            # Verify audio extraction succeeded
            if not os.path.exists(audio_path):
                raise ValueError(f"Audio extraction failed - no output file: {audio_path}")
            if os.path.getsize(audio_path) == 0:
                raise ValueError(f"Audio extraction failed - empty output file: {audio_path}")
            safe_print(f"  Audio extraction complete")

        # Create events files
        create_events_file(video_excel, events_path)
        if not os.path.exists(events_path):
            raise ValueError(f"Failed to create events file: {events_path}")
            
        # Create metadata JSON files
        processing_info = {
            "has_stabilization": True,
            "has_denoising": True,
            "has_equalization": True,
        }

        # Raw video JSON
        raw_video_json_path = raw_video_path.replace(".mp4", ".json")
        create_raw_video_json(excel_row, task_info, raw_video_path, raw_video_json_path)
        if not os.path.exists(raw_video_json_path):
            raise ValueError(f"Failed to create raw video JSON: {raw_video_json_path}")
            
        # Processed video JSON
        processed_video_json_path = processed_video_path.replace(".mp4", ".json")
        create_video_metadata_json(exif_data, processing_info, task_info, processed_video_json_path, target_framerate, target_resolution)
        if not os.path.exists(processed_video_json_path):
            raise ValueError(f"Failed to create processed video JSON: {processed_video_json_path}")
            
        # Audio JSON
        audio_json_path = audio_path.replace(".wav", ".json")
        create_audio_metadata_json(exif_data.get("duration_sec", 0), task_info, audio_json_path)
        if not os.path.exists(audio_json_path):
            raise ValueError(f"Failed to create audio JSON: {audio_json_path}")
            
        # Store processing information
        entry = {
            "participant_id": participant_id,
            "session_id": session_id,
            "task_label": task_label,
            "original_video": input_video_path,
            "source_video_bids": source_video_path,
            "raw_video_bids": raw_video_path,
            "processed_video_bids": processed_video_path,
            "audio_file_bids": audio_path,
            "events_file_bids": events_path,
            "filename": filename,
            "age_folder": video_info['age_folder'],
            "duration_sec": exif_data.get("duration_sec", 0),
            "has_excel_data": has_excel_data,
            "excel_metadata": excel_row.to_dict(),
            "task_info": task_info,
            "processing_info": processing_info,
        }

        safe_print(f"  Successfully processed: {participant_id}/{filename}")
        return entry, None

    except Exception as e:
        safe_print(f"  ERROR processing {input_video_path}: {str(e)}")
        return None, {"video": input_video_path, "error": str(e)}

def create_dataset_description(final_bids_root: str) -> None:
    """Create dataset_description.json for main BIDS dataset."""
    os.makedirs(final_bids_root, exist_ok=True)
    
    dataset_desc = {
        "Name": "SAILS Phase III Home Videos",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "License": "na",
        "Authors": ["Research Team"],
        "Acknowledgements": "participants and families",
        "HowToAcknowledge": "na",
        "Funding": ["na"],
        "EthicsApprovals": ["na"],
        "ReferencesAndLinks": ["na"],
        "DatasetDOI": "doi:",
    }

    filepath = os.path.join(final_bids_root, "dataset_description.json")
    with open(filepath, "w") as f:
        json.dump(dataset_desc, f, indent=4)
    
    if not os.path.exists(filepath):
        raise ValueError(f"Failed to create dataset_description.json at {filepath}")



def create_derivatives_dataset_description(final_derivatives_dir: str) -> None:
    """Create dataset_description.json for derivatives."""
    os.makedirs(final_derivatives_dir, exist_ok=True)
    
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

    filepath = os.path.join(final_derivatives_dir, "dataset_description.json")
    with open(filepath, "w") as f:
        json.dump(derivatives_desc, f, indent=4)
    
    if not os.path.exists(filepath):
        raise ValueError(f"Failed to create derivatives dataset_description.json at {filepath}")


def create_readme(final_bids_root: str) -> None:
    """Create README file for the BIDS dataset."""

    os.makedirs(final_bids_root, exist_ok=True)
    
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
- sub-*/ses-*/beh/: Raw behavioral videos (converted to mp4) and event annotations
- sourcedata/: Original unmodified video files in their native formats

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

## Behavioral Coding
Events files include annotations from csv file.

## Task Labels
Task labels are derived from the Context column in the csv.
Videos without behavioral coding data use "unknown" task label.
"""

    filepath = os.path.join(final_bids_root, "README")
    with open(filepath, "w") as f:
        f.write(readme_content)
    
    # FIX: Verify file was created
    if not os.path.exists(filepath):
        raise ValueError(f"Failed to create README at {filepath}")

def create_participants_files(processed_data: List[Dict[str, Any]], final_bids_root: str) -> None:
    """Create participants.tsv and participants.json files."""
    processed_participants = set(entry["participant_id"] for entry in processed_data)

    participants_data = []
    for participant_id in sorted(processed_participants):
        participants_data.append({
            'participant_id': f'sub-{participant_id}',
            'age': 'n/a',
            'validity': 'n/a'
        })

    participants_df = pd.DataFrame(participants_data)
    participants_df.to_csv(os.path.join(final_bids_root, "participants.tsv"), sep='\t', index=False, na_rep='n/a')

    participants_json = {
        "participant_id": {"Description": "Unique participant identifier"},
        "age": {"Description": "Age information", "Units": "months"},
        "validity": {"Description": "data validity information"},
    }

    with open(os.path.join(final_bids_root, "participants.json"), "w") as f:
        json.dump(participants_json, f, indent=4)

def print_summary(all_processed: List[Dict], all_failed: List[Dict]) -> None:
    """Print processing summary statistics."""

    print("PROCESSING SUMMARY")


    print(f"Successfully processed: {len(all_processed)} videos")
    print(f"Failed to process: {len(all_failed)} videos")
    print(f"Total videos attempted: {len(all_processed) + len(all_failed)}")

    if all_processed:
        # Excel data availability
        with_excel = sum(1 for entry in all_processed if entry.get('has_excel_data', False))
        without_excel = len(all_processed) - with_excel
        print(f"\nData sources:")
        print(f"  With Excel behavioral data: {with_excel} videos")
        print(f"  With dummy behavioral data: {without_excel} videos")

        # Task distribution
        task_counts = {}
        participant_counts = {}
        session_counts = {}

        for entry in all_processed:
            task = entry['task_label']
            participant = entry['participant_id']
            session = entry['session_id']

            task_counts[task] = task_counts.get(task, 0) + 1
            participant_counts[participant] = participant_counts.get(participant, 0) + 1
            session_counts[session] = session_counts.get(session, 0) + 1

        print(f"\nTask distribution:")
        for task, count in sorted(task_counts.items()):
            print(f"  {task}: {count} videos")

        print(f"\nSession distribution:")
        for session, count in sorted(session_counts.items()):
            print(f"  Session {session}: {count} videos")

        print(f"\nUnique participants processed: {len(participant_counts)}")

        # Duration statistics
        durations = [entry.get('duration_sec', 0) for entry in all_processed]
        total_duration = sum(durations)
        avg_duration = total_duration / len(durations) if durations else 0

        print(f"\nDuration statistics:")
        print(f"  Total video duration: {total_duration/3600:.1f} hours")
        print(f"  Average video duration: {avg_duration/60:.1f} minutes")

    if all_failed:
        print(f"\nFailed videos breakdown:")
        error_types = {}
        for entry in all_failed:
            error = entry.get('error', 'Unknown error')
            error_types[error] = error_types.get(error, 0) + 1

        for error, count in sorted(error_types.items()):
            print(f"  {error}: {count} videos")

def main():
    """Main function."""

    if len(sys.argv) != 3:
        print("Usage: python bids.py <task_id> <num_tasks>")
        sys.exit(1)

    # Configuration
    EXCEL_FILE = "/orcd/data/satra/002/datasets/SAILS/data4analysis/Video Rating Data/SAILS_RATINGS_ALL_8.8.25.xlsx"
    VIDEO_ROOT = "/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external/"
    OUTPUT_DIR = "/home/aparnabg/orcd/scratch/BIDS"
    TARGET_RESOLUTION = "1280x720"
    TARGET_FRAMERATE = 30

    FINAL_BIDS_ROOT = os.path.join(OUTPUT_DIR, "final_bids-dataset")
    FINAL_DERIVATIVES_DIR = os.path.join(FINAL_BIDS_ROOT, "derivatives", "preprocessed")
    FINAL_SOURCEDATA_DIR = os.path.join(FINAL_BIDS_ROOT, "sourcedata")

    # Parse command line arguments
    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])

    # Create task-specific temp directory
    TEMP_DIR = os.path.join(OUTPUT_DIR, str(my_task_id), "temp")
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Start timing
    start_time = time.time()

    # Check if paths exist
    if not os.path.exists(VIDEO_ROOT):
        print(f"ERROR: Video root directory not found: {VIDEO_ROOT}")
        sys.exit(1)

    if not os.path.exists(EXCEL_FILE):
        print(f"ERROR: Excel file not found: {EXCEL_FILE}")
        sys.exit(1)

    # Load Excel file
    try:
        excel_df = pd.read_excel(EXCEL_FILE)
        excel_df.columns = excel_df.columns.str.strip()
        safe_print(f"Loaded {len(excel_df)} rows from Excel file")
    except Exception as e:
        safe_print(f"ERROR: Failed to load Excel file: {e}")
        sys.exit(1)

    # Discover videos
    print("Discovering all video files from age folders")
    all_videos = get_all_videos_from_age_folders(VIDEO_ROOT)
    print(f"Found {len(all_videos)} video files in age-specific folders")

    if not all_videos:
        print("ERROR: No video files found")
        sys.exit(1)

    # Create BIDS structure files (only for task 0 to avoid conflicts)
    if my_task_id == 0:
        try:
            safe_print("Creating BIDS structure files...")
            create_dataset_description(FINAL_BIDS_ROOT)
            create_derivatives_dataset_description(FINAL_DERIVATIVES_DIR)
            create_readme(FINAL_BIDS_ROOT)
            safe_print("Successfully created BIDS structure files")
        except Exception as e:
            safe_print(f"CRITICAL ERROR: Failed to create BIDS structure files: {e}")
            sys.exit(1)

    # Divide videos among tasks
    video_chunks = all_videos[my_task_id::num_tasks]
    safe_print(f"Task {my_task_id}: Processing {len(video_chunks)} videos")

    # Process videos
    all_processed = []
    all_failed = []

    for i, video_info in enumerate(video_chunks, 1):
        safe_print(f"Video {i}/{len(video_chunks)}")

        processed_entry, failed_entry = process_single_video(
            video_info, excel_df, FINAL_BIDS_ROOT, FINAL_DERIVATIVES_DIR,
            FINAL_SOURCEDATA_DIR, TEMP_DIR, TARGET_FRAMERATE, TARGET_RESOLUTION
        )

        if processed_entry:
            all_processed.append(processed_entry)
        if failed_entry:
            all_failed.append(failed_entry)

    # Save processing logs
    task_output_dir = os.path.join(OUTPUT_DIR, str(my_task_id))
    os.makedirs(task_output_dir, exist_ok=True)

    log_path = os.path.join(task_output_dir, "processing_log.json")
    failed_path = os.path.join(task_output_dir, "not_processed.json")

    try:
        with open(log_path, "w") as f:
            json.dump(all_processed, f, indent=4, default=str)

        with open(failed_path, "w") as f:
            json.dump(all_failed, f, indent=4, default=str)
    except Exception as e:
        safe_print(f"ERROR: Failed to save processing logs: {e}")

    # Clean up temp directory
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    # Print summary
    end_time = time.time()
    total_time = end_time - start_time
    print_summary(all_processed, all_failed)
    safe_print(f"Total processing time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")

    if all_processed:
        avg_time_per_video = total_time / len(all_processed)
        safe_print(f"Average time per video: {avg_time_per_video:.1f} seconds")

    safe_print("Processing complete")

if __name__ == "__main__":
    main()