"""
Tests for src/sailsprep/action_model_testing/pyskl/scripts/pyskl_dataset.py

 src/tests/action_model_testing/pyskl/scripts/test_pyskl_dataset.py

"""
import json
import pickle
import subprocess

import numpy as np
import pandas as pd
import pytest

from sailsprep.action_model_testing.pyskl.scripts import pyskl_dataset as pds


# ---------------------------------------------------------------------------
# ffprobe_dims
# ---------------------------------------------------------------------------

def test_ffprobe_dims_returns_none_if_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.mp4"
    assert pds.ffprobe_dims(str(missing)) is None


def test_ffprobe_dims_parses_dimensions(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")

    monkeypatch.setattr(
        subprocess, "check_output",
        lambda *a, **k: b"1920x1080\n",
    )

    assert pds.ffprobe_dims(str(video)) == (1080, 1920)


def test_ffprobe_dims_handles_subprocess_failure(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")

    def _raise(*a, **k):
        raise subprocess.CalledProcessError(1, "ffprobe")

    monkeypatch.setattr(subprocess, "check_output", _raise)

    assert pds.ffprobe_dims(str(video)) is None


def test_ffprobe_dims_handles_malformed_output(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")

    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: b"garbage\n")

    assert pds.ffprobe_dims(str(video)) is None


# ---------------------------------------------------------------------------
# build_video_to_split
# ---------------------------------------------------------------------------

def test_build_video_to_split_success(tmp_path):
    csv_path = tmp_path / "split.csv"
    df = pd.DataFrame({
        "video_path": ["/data/videos/vid_001.mp4", "/data/videos/vid_002.mp4"],
        "split": ["Train", "VAL"],
    })
    df.to_csv(csv_path, index=False)

    mapping = pds.build_video_to_split(str(csv_path))

    assert mapping["vid_001"] == ("train", "/data/videos/vid_001.mp4")
    assert mapping["vid_002"] == ("val", "/data/videos/vid_002.mp4")


def test_build_video_to_split_missing_columns_raises(tmp_path):
    csv_path = tmp_path / "bad_split.csv"
    pd.DataFrame({"path": ["a.mp4"], "set": ["train"]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        pds.build_video_to_split(str(csv_path))


# ---------------------------------------------------------------------------
# parse_video_key_from_filename
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fname,expected", [
    ("subject01_session_1_2_3_4_clip5.json", "subject01_session"),
    ("subject01_session_clip5.json", "subject01_session"),
    ("no_numeric_suffix_clip1.json", "no_numeric_suffix"),
    ("only_two_trailing_9_8_clip3.json", "only_two_trailing"),
    ("plainname.json", "plainname"),
])
def test_parse_video_key_from_filename(fname, expected):
    assert pds.parse_video_key_from_filename(fname) == expected


# ---------------------------------------------------------------------------
# resolve_split_for_clip
# ---------------------------------------------------------------------------

@pytest.fixture
def video_split_map():
    return {
        "vid_alpha": ("train", "/data/vid_alpha.mp4"),
        "vid_beta": ("val", "/data/vid_beta.mp4"),
    }


def test_resolve_split_for_clip_matches_via_video_field(video_split_map):
    data = {"video": "vid_alpha"}
    sp, vp, key = pds.resolve_split_for_clip(data, "irrelevant_clip1.json", video_split_map)
    assert (sp, vp, key) == ("train", "/data/vid_alpha.mp4", "vid_alpha")


def test_resolve_split_for_clip_matches_via_filename(video_split_map):
    data = {}  # no "video" key present
    sp, vp, key = pds.resolve_split_for_clip(data, "vid_beta_1_2_3_4_clip2.json", video_split_map)
    assert (sp, vp, key) == ("val", "/data/vid_beta.mp4", "vid_beta")


def test_resolve_split_for_clip_startswith_fallback(video_split_map):
    data = {"video": "vid_alpha_extra_suffix"}
    sp, vp, key = pds.resolve_split_for_clip(data, "unrelated_clip1.json", video_split_map)
    assert sp == "train"
    assert vp == "/data/vid_alpha.mp4"


def test_resolve_split_for_clip_no_match_returns_none(video_split_map):
    data = {"video": "totally_unknown_video"}
    sp, vp, key = pds.resolve_split_for_clip(data, "totally_unknown_video_clip1.json", video_split_map)
    assert sp is None
    assert vp is None
    assert key == "totally_unknown_video"


# ---------------------------------------------------------------------------
# json_to_arrays
# ---------------------------------------------------------------------------

def _make_frame(x=1.0, y=2.0, confidence=0.9):
    return {name: {"x": x, "y": y, "confidence": confidence} for name in pds.COCO_17_NAMES}


def test_json_to_arrays_empty_frames_returns_none():
    kp, score, total = pds.json_to_arrays({"frames": {}})
    assert kp is None
    assert score is None
    assert total == 0


def test_json_to_arrays_missing_frames_key_returns_none():
    kp, score, total = pds.json_to_arrays({})
    assert kp is None
    assert score is None
    assert total == 0


def test_json_to_arrays_basic_shape_and_values():
    frames = {
        "10": _make_frame(x=1.0, y=2.0, confidence=0.9),
        "11": _make_frame(x=3.0, y=4.0, confidence=0.8),
        "12": _make_frame(x=5.0, y=6.0, confidence=0.7),
    }
    kp, score, total = pds.json_to_arrays({"frames": frames}, clip_t=pds.CLIP_T)

    assert kp.shape == (1, pds.CLIP_T, pds.NUM_KP, 2)
    assert score.shape == (1, pds.CLIP_T, pds.NUM_KP)
    assert total == 3

    # clip_start is min ann-frame index (10), so frame "10" lands at t_idx=0
    assert np.allclose(kp[0, 0, :, 0], 1.0)
    assert np.allclose(kp[0, 0, :, 1], 2.0)
    assert np.allclose(score[0, 0, :], 0.9)

    assert np.allclose(kp[0, 1, :, 0], 3.0)
    assert np.allclose(score[0, 2, :], 0.7)

    # frames beyond the populated range should remain all-zero
    assert np.allclose(score[0, 3:, :], 0.0)


def test_json_to_arrays_drops_frames_outside_clip_t():
    frames = {
        "0": _make_frame(),
        str(pds.CLIP_T + 5): _make_frame(),  # far outside clip_t window -> dropped
    }
    kp, score, total = pds.json_to_arrays({"frames": frames}, clip_t=pds.CLIP_T)
    # only the first frame contributes a non-zero score
    assert total == 1


def test_json_to_arrays_handles_missing_keypoint_entries():
    frame = _make_frame()
    del frame["Nose"]  # simulate a keypoint that wasn't detected in this frame
    kp, score, total = pds.json_to_arrays({"frames": {"0": frame}}, clip_t=pds.CLIP_T)
    nose_idx = pds.COCO_17_NAMES.index("Nose")
    assert score[0, 0, nose_idx] == 0.0
    assert total == 1


# ---------------------------------------------------------------------------
# convert_task (integration)
# ---------------------------------------------------------------------------

def _write_json_clip(class_dir, fname, video_key, n_frames=6):
    frames = {str(i): _make_frame(x=float(i), y=float(i) + 1, confidence=0.9)
              for i in range(n_frames)}
    payload = {"video": video_key, "frames": frames}
    with open(class_dir / fname, "w") as f:
        json.dump(payload, f)


@pytest.fixture
def rmm_fixture(tmp_path, monkeypatch):
    """Builds a minimal pose_root/rmm/<class>/*.json tree plus a split CSV
    covering all four rmm classes, and stubs out ffprobe_dims."""
    pose_root = tmp_path / "pose_root"
    task = "rmm"
    classes = pds.TASK_CLASSES[task]

    video_rows = []
    for i, class_name in enumerate(classes):
        class_dir = pose_root / task / class_name
        class_dir.mkdir(parents=True)
        video_key = f"vid_{class_name.lower()}"
        _write_json_clip(class_dir, f"{video_key}_1_2_3_4_clip0.json", video_key)
        split_name = "train" if i % 2 == 0 else "val"
        video_rows.append({"video_path": f"/data/{video_key}.mp4", "split": split_name})

    split_csv = tmp_path / "split.csv"
    pd.DataFrame(video_rows).to_csv(split_csv, index=False)

    monkeypatch.setattr(pds, "ffprobe_dims", lambda video_path, timeout=10: (480, 640))

    return pose_root, split_csv, classes


def test_convert_task_writes_expected_pickle(tmp_path, rmm_fixture):
    pose_root, split_csv, classes = rmm_fixture
    out_path = tmp_path / "out" / "rmm_pyskl.pkl"

    pds.convert_task("rmm", str(out_path), str(pose_root), str(split_csv))

    assert out_path.exists()
    with open(out_path, "rb") as f:
        result = pickle.load(f)

    assert set(result.keys()) == {"split", "annotations"}
    assert len(result["annotations"]) == len(classes)

    total_in_splits = sum(len(v) for v in result["split"].values())
    assert total_in_splits == len(classes)

    for ann in result["annotations"]:
        assert ann["label"] in range(len(classes))
        assert ann["img_shape"] == (480, 640)
        assert ann["keypoint"].shape == (1, pds.CLIP_T, pds.NUM_KP, 2)
        assert ann["keypoint_score"].shape == (1, pds.CLIP_T, pds.NUM_KP)
        assert ann["total_frames"] >= 1


def test_convert_task_respects_min_frames_filter(tmp_path, rmm_fixture):
    pose_root, split_csv, classes = rmm_fixture

    # Overwrite one class's clip with too few populated frames to pass the filter.
    class_dir = pose_root / "rmm" / classes[0]
    fname = next(f for f in class_dir.iterdir() if f.suffix == ".json")
    with open(fname) as f:
        payload = json.load(f)
    payload["frames"] = {"0": _make_frame()}  # only 1 frame
    with open(fname, "w") as f:
        json.dump(payload, f)

    out_path = tmp_path / "out" / "rmm_pyskl.pkl"
    pds.convert_task("rmm", str(out_path), str(pose_root), str(split_csv), min_frames=5)

    with open(out_path, "rb") as f:
        result = pickle.load(f)

    # The under-filled clip should have been dropped, leaving one fewer annotation.
    assert len(result["annotations"]) == len(classes) - 1


def test_convert_task_exits_when_no_annotations_survive(tmp_path):
    pose_root = tmp_path / "pose_root"
    task = "rmm"
    classes = pds.TASK_CLASSES[task]

    class_dir = pose_root / task / classes[0]
    class_dir.mkdir(parents=True)
    # video key that will never match anything in the split CSV
    _write_json_clip(class_dir, "unmatched_clip0.json", "unmatched_video")

    split_csv = tmp_path / "split.csv"
    pd.DataFrame({
        "video_path": ["/data/some_other_video.mp4"],
        "split": ["train"],
    }).to_csv(split_csv, index=False)

    out_path = tmp_path / "out" / "rmm_pyskl.pkl"

    with pytest.raises(SystemExit) as exc_info:
        pds.convert_task("rmm", str(out_path), str(pose_root), str(split_csv))

    assert exc_info.value.code == 1
    assert not out_path.exists()


def test_convert_task_warns_on_missing_class_dir(tmp_path, capsys, monkeypatch):
    """If one class's directory doesn't exist on disk, convert_task should
    warn and continue rather than crashing, still processing the other
    classes that are present."""
    pose_root = tmp_path / "pose_root"
    task = "rmm"
    classes = pds.TASK_CLASSES[task]

    # Only create the directory for the first class; leave the rest missing.
    present_class = classes[0]
    class_dir = pose_root / task / present_class
    class_dir.mkdir(parents=True)
    video_key = f"vid_{present_class.lower()}"
    _write_json_clip(class_dir, f"{video_key}_1_2_3_4_clip0.json", video_key)

    split_csv = tmp_path / "split.csv"
    pd.DataFrame({
        "video_path": [f"/data/{video_key}.mp4"],
        "split": ["train"],
    }).to_csv(split_csv, index=False)

    monkeypatch.setattr(pds, "ffprobe_dims", lambda video_path, timeout=10: (480, 640))

    out_path = tmp_path / "out" / "rmm_pyskl.pkl"
    pds.convert_task("rmm", str(out_path), str(pose_root), str(split_csv))

    captured = capsys.readouterr()
    assert "missing class dir" in captured.out

    with open(out_path, "rb") as f:
        result = pickle.load(f)
    assert len(result["annotations"]) == 1
    assert result["annotations"][0]["label"] == classes.index(present_class)