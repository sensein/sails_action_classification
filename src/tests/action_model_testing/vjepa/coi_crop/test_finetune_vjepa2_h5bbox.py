"""
Tests for sailsprep/action_model_testing/vjepa/coi_crop/finetune_vjepa2_h5bbox.py
No disk I/O beyond tmp_path, no GPU, no HF downloads (AutoModel.from_pretrained mocked).

Run: poetry run pytest
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# The module calls os.makedirs(OUTPUT_DIR, ...) at import time against a
# hardcoded cluster path (/orcd/...) that doesn't exist/isn't writable in CI.
# Swallow that failure for the duration of the import only.
_real_makedirs = os.makedirs
os.makedirs = lambda *a, **k: None
try:
    from sailsprep.action_model_testing.vjepa.coi_crop.finetune_vjepa2_h5bbox import (
        CROP_SIZE,
        EMBED_DIM,
        FRAMES_PER_CLIP,
        NUM_CLASSES,
        AttentivePoolHead,
        VJEPAFineTune,
        VJEPASegmentDataset,
        build_samples,
        find_action_runs,
        load_bbox_map,
        make_collate,
    )
finally:
    os.makedirs = _real_makedirs

DEVICE = torch.device("cpu")


# ============================================================
# Helpers
# ============================================================

def _make_h5_bbox(path: str, frames: range) -> None:
    dtype = np.dtype([
        ("index", "i8"),
        ("values_block_1", "i8", (6,)),
    ])
    table = np.zeros(len(frames), dtype=dtype)
    for i, f in enumerate(frames):
        table[i] = (f, (f, 0, 5, 5, 50, 50))
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("bboxes/table", data=table)


class _FakeEncoderOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_tokens=5):
        super().__init__()
        self.dummy = nn.Linear(1, 1)
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

    def forward(self, **kwargs):
        pixel_values = kwargs.get("pixel_values_videos")
        b = pixel_values.shape[0] if pixel_values is not None else 1
        return _FakeEncoderOutput(torch.randn(b, self.num_tokens, self.embed_dim))


# ============================================================
# 1. load_bbox_map
# ============================================================

class TestLoadBboxMap:
    def test_maps_frame_to_bbox(self, tmp_path):
        h5_path = str(tmp_path / "bbox.h5")
        _make_h5_bbox(h5_path, range(3))
        bbox_map = load_bbox_map(h5_path)
        assert set(bbox_map.keys()) == {0, 1, 2}
        assert bbox_map[0] == (5, 5, 50, 50)


# ============================================================
# 2. find_action_runs
# ============================================================

class TestFindActionRuns:
    def _df(self, frames, labels, col="Repetitive_Motor_Movements"):
        return pd.DataFrame({"Frame": frames, col: labels})

    def test_run_meeting_min_frames_kept(self):
        df = self._df(list(range(20)), ["spin"] * 20)
        runs = find_action_runs(df, "Repetitive_Motor_Movements", min_frames=15)
        assert runs == [(0, 19, "spin")]

    def test_run_below_min_frames_dropped(self):
        df = self._df(list(range(5)), ["spin"] * 5)
        runs = find_action_runs(df, "Repetitive_Motor_Movements", min_frames=15)
        assert runs == []

    def test_na_breaks_run(self):
        df = self._df(list(range(10)), ["spin"] * 4 + ["N/A"] + ["spin"] * 5)
        runs = find_action_runs(df, "Repetitive_Motor_Movements", min_frames=4)
        assert runs == [(0, 3, "spin"), (5, 9, "spin")]


# ============================================================
# 3. build_samples
# ============================================================

class TestBuildSamples:
    def test_missing_column_raises(self, tmp_path):
        csv_path = tmp_path / "split.csv"
        pd.DataFrame({"video_path": ["a.mp4"]}).to_csv(csv_path, index=False)
        try:
            build_samples(str(csv_path))
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_skips_rows_with_missing_files(self, tmp_path):
        csv_path = tmp_path / "split.csv"
        pd.DataFrame({
            "video_path": ["/nonexistent/video.mp4"],
            "label_path": ["/nonexistent/labels.csv"],
            "interpolated_anno_h5": ["/nonexistent/anno.h5"],
        }).to_csv(csv_path, index=False)
        samples = build_samples(str(csv_path))
        assert samples == []

    def test_builds_segments_for_valid_row(self, tmp_path):
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake")
        h5_path = tmp_path / "anno.h5"
        _make_h5_bbox(str(h5_path), range(20))
        label_path = tmp_path / "labels.csv"
        pd.DataFrame({
            "Frame": list(range(20)),
            "Repetitive_Motor_Movements": ["spin"] * 20,
        }).to_csv(label_path, index=False)

        csv_path = tmp_path / "split.csv"
        pd.DataFrame({
            "video_path": [str(video_path)],
            "label_path": [str(label_path)],
            "interpolated_anno_h5": [str(h5_path)],
        }).to_csv(csv_path, index=False)

        samples = build_samples(str(csv_path), "Repetitive_Motor_Movements")
        assert len(samples) == 1
        assert samples[0]["label_str"] == "spin"
        assert samples[0]["start_frame"] == 0
        assert samples[0]["end_frame"] == 19


# ============================================================
# 4. VJEPASegmentDataset
# ============================================================

class TestVJEPASegmentDataset:
    def test_getitem_falls_back_to_zeros_on_load_error(self):
        # NOTE: the source's except-branch fallback uses the module-level
        # FRAMES_PER_CLIP/CROP_SIZE constants rather than self.num_frames /
        # self.crop_size, so the fallback shape is fixed regardless of the
        # dataset's constructor args.
        samples = [{
            "video_path": "/nonexistent/video.mp4",
            "h5_path": "/nonexistent/anno.h5",
            "start_frame": 0,
            "end_frame": 10,
            "label_str": "spin",
        }]
        ds = VJEPASegmentDataset(samples, {"spin": 0}, num_frames=4, crop_size=8)
        frames, label = ds[0]
        assert frames.shape == (FRAMES_PER_CLIP, CROP_SIZE, CROP_SIZE, 3)
        assert label == 0

    def test_len(self):
        samples = [{"video_path": "a", "h5_path": "b", "start_frame": 0,
                    "end_frame": 1, "label_str": "spin"}] * 3
        ds = VJEPASegmentDataset(samples, {"spin": 0})
        assert len(ds) == 3


# ============================================================
# 5. make_collate
# ============================================================

class TestMakeCollate:
    def test_collate_calls_processor_and_stacks_labels(self):
        def fake_processor(clip_list, return_tensors="pt"):
            return {"pixel_values_videos": torch.randn(len(clip_list), 3, 2, 2, 2)}

        collate = make_collate(fake_processor)
        batch = [
            (np.zeros((4, 8, 8, 3), dtype=np.uint8), 0),
            (np.zeros((4, 8, 8, 3), dtype=np.uint8), 1),
        ]
        inputs, labels = collate(batch)
        assert inputs["pixel_values_videos"].shape[0] == 2
        assert labels.tolist() == [0, 1]


# ============================================================
# 6. AttentivePoolHead
# ============================================================

class TestAttentivePoolHead:
    def test_forward_shape(self):
        head = AttentivePoolHead(dim=32, num_heads=4, num_classes=4)
        tokens = torch.randn(3, 6, 32)
        assert head(tokens).shape == (3, 4)


# ============================================================
# 7. VJEPAFineTune (encoder mocked, no HF download)
# ============================================================

class TestVJEPAFineTune:
    def test_forward_shape_frozen_encoder(self, mocker):
        mocker.patch(
            "sailsprep.action_model_testing.vjepa.coi_crop.finetune_vjepa2_h5bbox."
            "AutoModel.from_pretrained",
            return_value=_FakeEncoder(),
        )
        model = VJEPAFineTune(num_classes=NUM_CLASSES, full_finetune=False)
        inputs = {"pixel_values_videos": torch.randn(2, 3, 4, 4, 4)}
        logits = model(inputs)
        assert logits.shape == (2, NUM_CLASSES)

    def test_frozen_encoder_params_not_trainable(self, mocker):
        mocker.patch(
            "sailsprep.action_model_testing.vjepa.coi_crop.finetune_vjepa2_h5bbox."
            "AutoModel.from_pretrained",
            return_value=_FakeEncoder(),
        )
        model = VJEPAFineTune(num_classes=NUM_CLASSES, full_finetune=False)
        assert all(not p.requires_grad for p in model.encoder.parameters())
        assert all(p.requires_grad for p in model.head.parameters())

    def test_class_weights_registered_as_buffer(self, mocker):
        mocker.patch(
            "sailsprep.action_model_testing.vjepa.coi_crop.finetune_vjepa2_h5bbox."
            "AutoModel.from_pretrained",
            return_value=_FakeEncoder(),
        )
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        model = VJEPAFineTune(num_classes=NUM_CLASSES, full_finetune=False,
                              class_weights=weights)
        assert torch.allclose(model.class_weights, weights)
