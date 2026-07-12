# src/tests/test_videomae2.py
"""
Unit tests for VideoMAE V2 fine-tuning pipeline.

Tests run without GPU, without the VideoMAEv2 checkpoint,
and without real video data — passes under `poetry run pytest`.
"""

import pathlib
import sys
from collections import Counter
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

# ============================================================
# Path resolution
# ============================================================
# __file__ = <repo>/src/tests/test_videomae2.py
# .parents[0] = src/tests
# .parents[1] = src
# .parents[2] = <repo root>
_SRC       = pathlib.Path(__file__).parents[3]          # .../src
_VMAE2_DIR = _SRC / "sailsprep" / "action_model_testing" / "Videomaev2"

_HEAVY_MOCKS = {
    "pytorch_lightning":          MagicMock(),
    "pytorch_lightning.callbacks": MagicMock(),
    "sklearn":                    MagicMock(),
    "sklearn.metrics":            MagicMock(),
    "sklearn.preprocessing":      MagicMock(),
    "cv2":                        MagicMock(),
    "h5py":                       MagicMock(),
}


def _load(script_name: str, module_alias: str):
    """Import one of the VideoMAE V2 scripts, mocking heavy runtime deps."""
    src = _VMAE2_DIR / script_name
    if not src.exists():
        pytest.skip(f"{script_name} not found at {src}")
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_alias, src)
    mod  = importlib.util.module_from_spec(spec)
    # videomae2_*.py scripts do `from utils.bbox import load_bbox_map`,
    # relying on Videomaev2/ being on sys.path so `utils` resolves.
    sys.path.insert(0, str(_VMAE2_DIR))
    try:
        with patch.dict(sys.modules, _HEAVY_MOCKS):
            spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(_VMAE2_DIR))
    return mod


# ============================================================
# Stub for modeling_finetune (avoids checkpoint download)
# ============================================================

def _inject_stub_modeling_finetune(monkeypatch):
    """
    Replace modeling_finetune with a lightweight stub whose
    vit_base_patch16_224 returns a 768-d TinyViT.
    """
    import types

    class _TinyViT(nn.Module):
        def __init__(self, num_classes=710, **kw):
            super().__init__()
            self.fc_norm      = nn.LayerNorm(768)
            self.head         = nn.Linear(768, num_classes)
            self.head_dropout = nn.Dropout(0.0)

        def forward(self, x):
            B    = x.size(0)
            feat = self.fc_norm(torch.zeros(B, 768))
            return self.head(self.head_dropout(feat))

    fake = types.ModuleType("modeling_finetune")
    fake.vit_base_patch16_224 = lambda num_classes=710, **kw: _TinyViT(num_classes)
    monkeypatch.setitem(sys.modules, "modeling_finetune", fake)
    return fake


def _dummy_ckpt(tmp_path: pathlib.Path) -> str:
    """Write a minimal state-dict .pth the loader will accept."""
    class _TinyViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_norm      = nn.LayerNorm(768)
            self.head         = nn.Linear(768, 710)
            self.head_dropout = nn.Dropout(0.0)

    p = tmp_path / "vit_b_k710_dl_from_giant.pth"
    torch.save(_TinyViT().state_dict(), str(p))
    return str(p)


# ============================================================
# 1.  modeling_finetune.py — ViT architecture
# ============================================================

class TestModelingFinetune:

    @pytest.fixture(autouse=True)
    def _mod(self):
        src = _VMAE2_DIR / "modeling_finetune.py"
        if not src.exists():
            pytest.skip(f"modeling_finetune.py not found at {src}")
        import importlib.util
        spec = importlib.util.spec_from_file_location("modeling_finetune", src)
        mod  = importlib.util.module_from_spec(spec)
        # timm's @register_model does sys.modules[fn.__module__]
        # so the module MUST be in sys.modules before exec
        sys.modules["modeling_finetune"] = mod
        spec.loader.exec_module(mod)
        self.mod = mod
        yield
        # Clean up so other tests get a fresh import if needed
        sys.modules.pop("modeling_finetune", None)

    def test_patch_embed_output_shape(self):
        pe  = self.mod.PatchEmbed(img_size=224, patch_size=16, in_chans=3,
                                  embed_dim=64, num_frames=16, tubelet_size=2)
        out = pe(torch.zeros(2, 3, 16, 224, 224))
        # (16//2) * (224//16)**2 = 8 * 196 = 1568 patches
        assert out.shape == (2, 1568, 64)

    def test_sinusoid_table_shape(self):
        tbl = self.mod.get_sinusoid_encoding_table(100, 64)
        assert tbl.shape == (1, 100, 64)
        assert not tbl.requires_grad

    def test_attention_forward(self):
        attn = self.mod.Attention(dim=64, num_heads=4, qkv_bias=True)
        out  = attn(torch.randn(2, 10, 64))
        assert out.shape == (2, 10, 64)

    def test_cos_attention_forward(self):
        attn = self.mod.CosAttention(dim=64, num_heads=4, qkv_bias=True)
        out  = attn(torch.randn(2, 10, 64))
        assert out.shape == (2, 10, 64)

    def test_block_no_gamma(self):
        block = self.mod.Block(dim=64, num_heads=4, init_values=0.)
        out   = block(torch.randn(2, 10, 64))
        assert out.shape == (2, 10, 64)

    def test_block_with_gamma(self):
        block = self.mod.Block(dim=64, num_heads=4, init_values=1e-4)
        out   = block(torch.randn(2, 10, 64))
        assert out.shape == (2, 10, 64)

    def test_vit_small_forward(self):
        model = self.mod.vit_small_patch16_224(num_classes=5, all_frames=16)
        model.eval()
        out = model(torch.zeros(1, 3, 16, 224, 224))
        assert out.shape == (1, 5)

    def test_vit_base_num_patches(self):
        model = self.mod.vit_base_patch16_224(num_classes=10, all_frames=16)
        assert model.patch_embed.num_patches == 1568

    def test_vit_base_reset_classifier(self):
        model = self.mod.vit_base_patch16_224(num_classes=10, all_frames=16)
        model.reset_classifier(3)
        assert model.head.out_features == 3

    def test_cos_attn_scale_clamped(self):
        attn = self.mod.CosAttention(dim=32, num_heads=2)
        with torch.no_grad():
            attn.scale.fill_(100.0)
        out = attn(torch.randn(1, 5, 32))
        assert torch.isfinite(out).all()

    def test_mlp_forward(self):
        mlp = self.mod.Mlp(in_features=32, hidden_features=64)
        out = mlp(torch.randn(4, 10, 32))
        assert out.shape == (4, 10, 32)

    def test_drop_path_eval_identity(self):
        dp = self.mod.DropPath(drop_prob=0.5)
        x  = torch.ones(8, 4)
        dp.eval()
        assert torch.allclose(dp(x), x)


# ============================================================
# 2.  chunk_run / find_action_runs  (videomae2_finetune.py)
# ============================================================

class TestChunkRun:

    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _load("videomae2_finetune.py", "vmae2_ft_cr")

    def test_too_short_returns_empty(self):
        assert self.mod.chunk_run(0, 13) == []

    def test_exactly_min_frames(self):
        clips = self.mod.chunk_run(0, 14)   # 15 frames == MIN_FRAMES
        assert len(clips) == 1

    def test_single_clip(self):
        clips = self.mod.chunk_run(0, 28)
        assert clips == [(0, 28)]

    def test_two_clips_at_boundary(self):
        clips = self.mod.chunk_run(0, 44)   # 45 frames
        assert len(clips) == 2

    def test_multi_chunk(self):
        clips = self.mod.chunk_run(0, 89)   # 90 frames → 3 × 30
        assert len(clips) == 3
        for s, e in clips:
            assert e - s + 1 >= self.mod.MIN_FRAMES


class TestFindActionRuns:

    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _load("videomae2_finetune.py", "vmae2_ft_far")

    def _ann(self, frames, labels, col="Locomotion"):
        return pd.DataFrame({"Frame": frames, col: labels})

    def test_single_run(self):
        runs = self.mod.find_action_runs(
            self._ann([0,1,2,3], ["walk"]*4), "Locomotion")
        assert runs == [(0, 3, "walk")]

    def test_na_skipped(self):
        runs = self.mod.find_action_runs(
            self._ann([0,1,2,3], ["N/A","walk","walk","N/A"]), "Locomotion")
        assert runs == [(1, 2, "walk")]

    def test_two_separate_runs(self):
        runs = self.mod.find_action_runs(
            self._ann([0,1,2,3,4,5],
                      ["walk","walk","walk","run","run","run"]),
            "Locomotion")
        assert len(runs) == 2
        assert runs[0][2] == "walk"
        assert runs[1][2] == "run"

    def test_empty_annotation(self):
        assert self.mod.find_action_runs(self._ann([], []), "Locomotion") == []

    def test_all_na(self):
        assert self.mod.find_action_runs(
            self._ann([0,1,2], ["N/A"]*3), "Locomotion") == []


# ============================================================
# 3.  get_window_label — fullvideo sliding script
# ============================================================

class TestGetWindowLabelFullVideo:

    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _load("videomae2_fullvideo_sliding.py", "vmae2_fw_wl")

    def test_majority_wins(self):
        f2l = {0: "walk", 1: "walk", 2: "run"}
        assert self.mod.get_window_label(f2l, 0, 3) == "walk"

    def test_empty_map_returns_na(self):
        assert self.mod.get_window_label({}, 0, 5) == "N/A"

    def test_empty_range_returns_na(self):
        assert self.mod.get_window_label({}, 5, 5) == "N/A"

    def test_missing_frames_counted_as_na(self):
        # 1 walk vs 4 implicit N/A
        assert self.mod.get_window_label({0: "walk"}, 0, 5) == "N/A"

    def test_all_active_label(self):
        f2l = {i: "run" for i in range(10)}
        assert self.mod.get_window_label(f2l, 0, 10) == "run"


# ============================================================
# 4.  get_window_label — two-stage script
# ============================================================

class TestGetWindowLabelTwoStage:

    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _load("videomae2_twostage_sliding.py", "vmae2_ts_wl")

    def test_majority_active(self):
        f2l = {i: "walk" for i in range(20)}
        f2l.update({20: "N/A", 21: "N/A"})
        assert self.mod.get_window_label(f2l, 0, 22) == "walk"

    def test_empty_returns_na(self):
        assert self.mod.get_window_label({}, 0, 10) == "N/A"

    def test_binary_constants(self):
        assert self.mod.BIN_NA     == 0
        assert self.mod.BIN_ACTIVE == 1
        assert self.mod.NA_LABEL   == "N/A"


# ============================================================
# 5.  Dataset error handling — zero tensor on bad video
# ============================================================

class TestDatasetErrorHandling:

    _BAD = {
        "video_path":  "/nonexistent/video.mp4",
        "h5_path":     "/nonexistent/bbox.h5",
        "start_frame": 0,
        "end_frame":   15,
        "label_str":   "walk",
        "ann_fps":     15.0,
    }

    def test_finetune_bad_video_returns_zeros(self):
        mod = _load("videomae2_finetune.py", "vmae2_ft_ds")
        ds  = mod.BBoxCropVideoDataset([self._BAD], {"walk": 0})
        frames, label = ds[0]
        assert label == 0
        assert frames.shape == (3, mod.NUM_FRAMES, mod.CROP_SIZE, mod.CROP_SIZE)
        assert frames.sum().item() == pytest.approx(0.0)

    def test_fullvideo_bad_video_returns_zeros(self):
        mod = _load("videomae2_fullvideo_sliding.py", "vmae2_fw_ds")
        ds  = mod.BBoxCropVideoDataset([self._BAD], {"walk": 0})
        frames, label = ds[0]
        assert frames.shape == (3, mod.NUM_FRAMES, mod.CROP_SIZE, mod.CROP_SIZE)
        assert frames.sum().item() == pytest.approx(0.0)

    def test_twostage_bad_video_returns_zeros(self):
        mod    = _load("videomae2_twostage_sliding.py", "vmae2_ts_ds")
        sample = {**self._BAD, "bin_label": 0, "fg_label": -1}
        ds     = mod.TwoStageVideoDataset([sample])
        frames, bin_lbl, fg_lbl = ds[0]
        assert frames.shape    == (3, mod.NUM_FRAMES, mod.CROP_SIZE, mod.CROP_SIZE)
        assert bin_lbl.item()  == 0
        assert fg_lbl.item()   == -1


# ============================================================
# 6.  Collate functions
# ============================================================

class TestCollate:

    def test_finetune_collate_shapes(self):
        mod   = _load("videomae2_finetune.py", "vmae2_ft_col")
        batch = [(torch.zeros(3,16,224,224), 0),
                 (torch.ones(3,16,224,224),  1)]
        videos, labels = mod.collate(batch)
        assert videos.shape    == (2, 3, 16, 224, 224)
        assert labels.tolist() == [0, 1]

    def test_fullvideo_collate_shapes(self):
        mod   = _load("videomae2_fullvideo_sliding.py", "vmae2_fw_col")
        batch = [(torch.zeros(3,16,224,224), 0),
                 (torch.ones(3,16,224,224),  2)]
        videos, labels = mod.collate(batch)
        assert videos.shape == (2, 3, 16, 224, 224)

    def test_twostage_collate_shapes(self):
        mod   = _load("videomae2_twostage_sliding.py", "vmae2_ts_col")
        batch = [
            (torch.zeros(3,16,224,224), torch.tensor(0), torch.tensor(-1)),
            (torch.ones(3,16,224,224),  torch.tensor(1), torch.tensor(2)),
        ]
        videos, bin_labels, fg_labels = mod.collate_fn(batch)
        assert videos.shape        == (2, 3, 16, 224, 224)
        assert bin_labels.tolist() == [0, 1]
        assert fg_labels.tolist()  == [-1, 2]


# ============================================================
# 7.  Two-stage model architecture
# ============================================================

class TestTwoStageModel:

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, tmp_path):
        _inject_stub_modeling_finetune(monkeypatch)
        ckpt = _dummy_ckpt(tmp_path)

        mod = _load("videomae2_twostage_sliding.py", "vmae2_ts_arch")
        # Point the module at our dummy checkpoint
        mod.VMAE2_CKPT = ckpt
        self.mod = mod

    def test_two_heads_exist(self):
        m = self.mod.TwoStageVideoMAE2(num_fg_classes=4)
        assert m.binary_head.out_features == 2
        assert m.fg_head.out_features     == 4

    def test_forward_output_shapes(self):
        m = self.mod.TwoStageVideoMAE2(num_fg_classes=5)
        m.eval()
        bin_l, fg_l = m(torch.zeros(2, 3, 16, 224, 224))
        assert bin_l.shape == (2, 2)
        assert fg_l.shape  == (2, 5)

    def test_feat_dim_constant(self):
        assert self.mod.TwoStageVideoMAE2.FEAT_DIM == 768

    def test_heads_trainable_after_freeze(self):
        m = self.mod.TwoStageVideoMAE2(num_fg_classes=3)
        m.freeze_except_last_block()
        for name, p in m.named_parameters():
            if name.startswith("binary_head.") or name.startswith("fg_head."):
                assert p.requires_grad, f"{name} should be trainable"


# ============================================================
# 8.  Class-weight formula
# ============================================================

class TestClassWeights:

    @staticmethod
    def _w(counts):
        c = np.maximum(np.array(counts, dtype=np.float64), 1.0)
        return c.sum() / (len(c) * c)

    def test_balanced_gives_equal_weights(self):
        assert np.allclose(self._w([100, 100, 100]), [1.0, 1.0, 1.0])

    def test_rare_class_gets_higher_weight(self):
        w = self._w([10, 100])
        assert w[0] > w[1]

    def test_zero_count_clamped(self):
        assert np.isfinite(self._w([0, 50])).all()

    def test_balanced_weights_sum_to_n(self):
        assert np.isclose(self._w([50, 50, 50]).sum(), 3.0)


# ============================================================
# 9.  Task-config constants
# ============================================================

@pytest.mark.parametrize("script,task,key,expected", [
    ("videomae2_finetune.py",          "loco", "num_classes",    5),
    ("videomae2_finetune.py",          "rmm",  "num_classes",    4),
    ("videomae2_fullvideo_sliding.py", "loco", "num_classes",    6),
    ("videomae2_fullvideo_sliding.py", "rmm",  "num_classes",    5),
    ("videomae2_twostage_sliding.py",  "loco", "num_fg_classes", 5),
    ("videomae2_twostage_sliding.py",  "rmm",  "num_fg_classes", 4),
])
def test_task_config_num_classes(script, task, key, expected):
    mod = _load(script, f"_cfg_{script}_{task}")
    assert mod.TASK_CONFIG[task][key] == expected


@pytest.mark.parametrize("script", [
    "videomae2_finetune.py",
    "videomae2_fullvideo_sliding.py",
    "videomae2_twostage_sliding.py",
])
def test_label_cols_defined(script):
    mod = _load(script, f"_lbl_{script}")
    for task in ("loco", "rmm"):
        assert "label_col" in mod.TASK_CONFIG[task]


# ============================================================
# 10. Global hyper-parameter sanity
# ============================================================

@pytest.mark.parametrize("script", [
    "videomae2_finetune.py",
    "videomae2_fullvideo_sliding.py",
    "videomae2_twostage_sliding.py",
])
def test_num_frames_is_16(script):
    assert _load(script, f"_nf_{script}").NUM_FRAMES == 16


@pytest.mark.parametrize("script", [
    "videomae2_finetune.py",
    "videomae2_fullvideo_sliding.py",
    "videomae2_twostage_sliding.py",
])
def test_crop_size_is_224(script):
    assert _load(script, f"_cs_{script}").CROP_SIZE == 224


@pytest.mark.parametrize("script", [
    "videomae2_finetune.py",
    "videomae2_fullvideo_sliding.py",
    "videomae2_twostage_sliding.py",
])
def test_learning_rate_positive(script):
    assert _load(script, f"_lr_{script}").LEARNING_RATE > 0


@pytest.mark.parametrize("script", [
    "videomae2_fullvideo_sliding.py",
    "videomae2_twostage_sliding.py",
])
def test_sliding_window_params(script):
    mod = _load(script, f"_sw_{script}")
    assert mod.WINDOW_SEC    == 2.0
    assert mod.WINDOW_STRIDE == 1.0
    assert mod.ANN_FPS       == 15.0