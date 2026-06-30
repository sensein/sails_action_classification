import cv2
import numpy as np
import pytest
from pathlib import Path

from sailsprep.tracking_pose_model_testing.entitysam import (
    get_cpu_memory_info,
    extract_frames_from_video,
)


def make_dummy_video(tmp_path: Path) -> Path:
    video_path = tmp_path / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 64))
    for _ in range(10):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


@pytest.mark.unit
def test_get_cpu_memory_info_keys() -> None:
    info = get_cpu_memory_info()
    assert set(info) == {
        "process_rss_gb",
        "process_vms_gb",
        "system_available_gb",
        "system_used_percent",
    }


@pytest.mark.unit
def test_extract_frames_from_video(tmp_path: Path) -> None:
    video_path = make_dummy_video(tmp_path)
    frame_names, fps = extract_frames_from_video(str(video_path), str(tmp_path))
    assert len(frame_names) > 0
    assert fps > 0