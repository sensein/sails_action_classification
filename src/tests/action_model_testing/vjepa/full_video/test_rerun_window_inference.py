"""
Tests for sailsprep/action_model_testing/vjepa/full_video/rerun_window_inference.py
No disk I/O beyond tmp_path, no GPU required.

Run: poetry run pytest
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

from sailsprep.action_model_testing.vjepa.full_video.rerun_window_inference import (
    EMBED_DIM,
    WINDOW_FRAMES,
    AttentiveProbe,
    HierarchicalProbe,
    infer_one_video_flat,
    infer_one_video_hierarchical,
    load_video_data,
    run_model_task_seed,
)

DEVICE = torch.device("cpu")


def _make_feat_npy(tmp_dir: str, T: int = 90, embed: int = EMBED_DIM) -> str:
    path = os.path.join(tmp_dir, "feat.npy")
    np.save(path, np.random.randn(embed, T).astype(np.float32))
    return path


def _make_label_csv(tmp_dir: str, T: int = 90, column: str = "Locomotion") -> str:
    path = os.path.join(tmp_dir, "labels.csv")
    labels = (["walk"] * (T // 2)) + (["run"] * (T - T // 2))
    pd.DataFrame({column: labels}).to_csv(path, index=False)
    return path


class TestLoadVideoData:
    def test_bad_feat_returns_none(self, tmp_path):
        label_path = _make_label_csv(str(tmp_path))
        feat, labels = load_video_data("/bad/path.npy", label_path, "Locomotion")
        assert feat is None and labels is None

    def test_missing_column_returns_none(self, tmp_path):
        feat_path = _make_feat_npy(str(tmp_path))
        label_path = _make_label_csv(str(tmp_path), column="Other")
        feat, labels = load_video_data(feat_path, label_path, "Locomotion")
        assert feat is None and labels is None

    def test_correct_shapes(self, tmp_path):
        T = 40
        feat_path = _make_feat_npy(str(tmp_path), T=T)
        label_path = _make_label_csv(str(tmp_path), T=T)
        feat, labels = load_video_data(feat_path, label_path, "Locomotion")
        assert feat.shape == (T, EMBED_DIM)
        assert len(labels) == T


class TestInferOneVideoFlat:
    def test_output_lengths_match_video(self):
        T = 90
        probe = AttentiveProbe(embed_dim=EMBED_DIM, num_classes=2, num_heads=8)
        feat = np.random.randn(T, EMBED_DIM).astype(np.float32)
        label_map = {"walk": 0, "run": 1}
        predictions, confidences = infer_one_video_flat(probe, feat, label_map, DEVICE)
        assert len(predictions) == T
        assert len(confidences) == T

    def test_too_short_video_returns_none(self):
        T = WINDOW_FRAMES - 1
        probe = AttentiveProbe(embed_dim=EMBED_DIM, num_classes=2, num_heads=8)
        feat = np.random.randn(T, EMBED_DIM).astype(np.float32)
        label_map = {"walk": 0, "run": 1}
        result = infer_one_video_flat(probe, feat, label_map, DEVICE)
        assert result is None


class TestInferOneVideoHierarchical:
    def test_output_lengths_match_video(self):
        T = 90
        probe = HierarchicalProbe(embed_dim=EMBED_DIM, num_stage2_classes=2, num_heads=8)
        feat = np.random.randn(T, EMBED_DIM).astype(np.float32)
        stage2_map = {"walk": 0, "run": 1}
        predictions, confidences = infer_one_video_hierarchical(probe, feat, stage2_map, DEVICE)
        assert len(predictions) == T
        assert len(confidences) == T
        assert all(p in {"None", "walk", "run"} for p in predictions)


class TestRunModelTaskSeed:
    def test_skips_missing_seed_dir(self, tmp_path, capsys):
        run_model_task_seed("window", "locomotion", 42, DEVICE)
        # No exception; nothing to assert on filesystem since dir is absent.

    def test_flat_model_writes_per_video_predictions(self, tmp_path, monkeypatch):
        import sailsprep.action_model_testing.vjepa.full_video.rerun_window_inference as mod

        base_dir = tmp_path / "base"
        seed_dir = base_dir / "window" / "locomotion" / "seed_42"
        seed_dir.mkdir(parents=True)

        probe = AttentiveProbe(embed_dim=EMBED_DIM, num_classes=2, num_heads=8)
        torch.save(probe.state_dict(), seed_dir / "best_probe.pt")
        label_map = {"walk": 0, "run": 1}
        (seed_dir / "label_mapping.json").write_text(json.dumps(label_map))

        T = 60
        feat_path = _make_feat_npy(str(tmp_path), T=T)
        label_path = _make_label_csv(str(tmp_path), T=T)
        splits_csv = tmp_path / "splits.csv"
        pd.DataFrame({
            "split": ["test"],
            "vjpe_features_full_video_vit_h_features": [feat_path],
            "label_path": [label_path],
        }).to_csv(splits_csv, index=False)

        monkeypatch.setattr(mod, "BASE_DIR", str(base_dir))
        monkeypatch.setattr(mod, "SPLITS_CSV", str(splits_csv))

        run_model_task_seed("window", "locomotion", 42, DEVICE)

        pred_dir = seed_dir / "per_video_predictions"
        assert pred_dir.exists()
        assert len(list(pred_dir.glob("*_predictions.csv"))) == 1
