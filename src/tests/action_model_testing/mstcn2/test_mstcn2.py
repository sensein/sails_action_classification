"""
Tests for mstcn2.py
Run with:
    poetry run pytest test_mstcn2.py -v
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

# ---------------------------------------------------------------------------
# Import the module under test.
# Adjust the path if your project layout differs.
# ---------------------------------------------------------------------------

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "sailsprep", "action_model_testing", "mstcn2"),
)
import  mstcn2 as M  # noqa: E402  (after sys.path tweak)


# ============================================================
# Shared fixtures
# ============================================================

NUM_CLASSES = 3          # background + 2 action classes
FEATURE_DIM = 16         # tiny dim for fast tests
T = 40                   # number of frames
BATCH = 2

LABEL_MAP = {"background": 0, "walk": 1, "run": 2}


@pytest.fixture()
def device():
    return torch.device("cpu")


@pytest.fixture()
def label_map():
    return dict(LABEL_MAP)


@pytest.fixture()
def tiny_model():
    """MS-TCN++ with deliberately small hyper-params so tests stay fast."""
    return M.MultiStageModel(
        num_layers_pg=3,
        num_layers_r=2,
        num_r=2,
        num_f_maps=8,
        dim=FEATURE_DIM,
        num_classes=NUM_CLASSES,
    )


@pytest.fixture()
def dummy_batch():
    """(feats, labels, masks) tensors — single video."""
    feats  = torch.randn(1, FEATURE_DIM, T)
    labels = torch.randint(0, NUM_CLASSES, (1, T))
    masks  = torch.ones(1, 1, T)
    return feats, labels, masks


# ============================================================
# 1. MODEL ARCHITECTURE — forward pass & output shapes
# ============================================================

class TestModelArchitecture:

    def test_output_shape(self, tiny_model, dummy_batch):
        feats, _, masks = dummy_batch
        outputs = tiny_model(feats, masks)
        # outputs: [num_stages, B, C, T]
        num_stages = 1 + 2   # PG + 2 refinement stages
        assert outputs.shape == (num_stages, 1, NUM_CLASSES, T), (
            f"Expected ({num_stages}, 1, {NUM_CLASSES}, {T}), got {tuple(outputs.shape)}"
        )

    def test_output_is_finite(self, tiny_model, dummy_batch):
        feats, _, masks = dummy_batch
        outputs = tiny_model(feats, masks)
        assert torch.isfinite(outputs).all(), "Model output contains NaN or Inf"

    def test_mask_zeros_respected(self, tiny_model):
        """Padded positions (mask=0) should produce zero logits."""
        feats = torch.randn(1, FEATURE_DIM, T)
        masks = torch.zeros(1, 1, T)
        outputs = tiny_model(feats, masks)
        assert outputs.abs().max().item() < 1e-6, (
            "Expected all-zero output when mask is all zeros"
        )

    def test_batch_size_two(self, tiny_model):
        feats  = torch.randn(BATCH, FEATURE_DIM, T)
        masks  = torch.ones(BATCH, 1, T)
        outputs = tiny_model(feats, masks)
        assert outputs.shape[1] == BATCH

    def test_different_sequence_lengths(self, tiny_model):
        for t in [10, 50, 128]:
            feats  = torch.randn(1, FEATURE_DIM, t)
            masks  = torch.ones(1, 1, t)
            outputs = tiny_model(feats, masks)
            assert outputs.shape[-1] == t, f"Output T mismatch for input T={t}"

    def test_prediction_generation_module(self):
        pg = M.PredictionGeneration(num_layers=3, num_f_maps=8,
                                     dim=FEATURE_DIM, num_classes=NUM_CLASSES)
        x    = torch.randn(1, FEATURE_DIM, T)
        mask = torch.ones(1, 1, T)
        out  = pg(x, mask)
        assert out.shape == (1, NUM_CLASSES, T)

    def test_refinement_module(self):
        ref = M.Refinement(num_layers=2, num_f_maps=8,
                           dim=NUM_CLASSES, num_classes=NUM_CLASSES)
        x    = torch.randn(1, NUM_CLASSES, T)
        mask = torch.ones(1, 1, T)
        out  = ref(x, mask)
        assert out.shape == (1, NUM_CLASSES, T)

    def test_dilated_residual_layer(self):
        layer = M.DilatedResidualLayer(dilation=2, in_channels=8, out_channels=8)
        x    = torch.randn(1, 8, T)
        mask = torch.ones(1, 1, T)
        out  = layer(x, mask)
        assert out.shape == (1, 8, T)


# ============================================================
# 2. LOSS
# ============================================================

class TestMSTCNLoss:

    def test_loss_positive(self, tiny_model, dummy_batch):
        feats, labels, masks = dummy_batch
        outputs  = tiny_model(feats, masks)
        criterion = M.MSTCNLoss(NUM_CLASSES)
        loss = criterion(outputs, labels, masks)
        assert loss.item() > 0, "Loss should be positive"

    def test_loss_is_finite(self, tiny_model, dummy_batch):
        feats, labels, masks = dummy_batch
        outputs  = tiny_model(feats, masks)
        criterion = M.MSTCNLoss(NUM_CLASSES)
        loss = criterion(outputs, labels, masks)
        assert torch.isfinite(loss), "Loss is NaN or Inf"

    def test_ignore_index_minus100(self, tiny_model):
        """Frames labelled -100 should be ignored (no crash, finite loss)."""
        feats  = torch.randn(1, FEATURE_DIM, T)
        masks  = torch.ones(1, 1, T)
        labels = torch.full((1, T), -100, dtype=torch.long)
        outputs   = tiny_model(feats, masks)
        criterion = M.MSTCNLoss(NUM_CLASSES)
        loss = criterion(outputs, labels, masks)
        assert torch.isfinite(loss)

    def test_loss_decreases_with_training(self, tiny_model, dummy_batch, device):
        """Loss should drop after a handful of gradient steps."""
        tiny_model.to(device)
        feats, labels, masks = [t.to(device) for t in dummy_batch]
        criterion = M.MSTCNLoss(NUM_CLASSES)
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-2)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            out  = tiny_model(feats, masks)
            loss = criterion(out, labels, masks)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


# ============================================================
# 3. DATA LOADING & PRE-PROCESSING
# ============================================================

class TestDataLoading:

    # ---- load_feature_file ----

    def test_load_npy_correct_shape(self, tmp_path):
        arr = np.random.randn(FEATURE_DIM, T).astype(np.float32)
        p   = tmp_path / "feat.npy"
        np.save(p, arr)
        loaded = M.load_feature_file(str(p), FEATURE_DIM)
        assert loaded.shape == (FEATURE_DIM, T)

    def test_load_npy_auto_transpose(self, tmp_path):
        """(T, D) arrays should be auto-transposed to (D, T)."""
        arr = np.random.randn(T, FEATURE_DIM).astype(np.float32)
        p   = tmp_path / "feat_td.npy"
        np.save(p, arr)
        loaded = M.load_feature_file(str(p), FEATURE_DIM)
        assert loaded.shape == (FEATURE_DIM, T)

    def test_load_npy_wrong_dim_raises(self, tmp_path):
        arr = np.random.randn(99, 77).astype(np.float32)
        p   = tmp_path / "bad.npy"
        np.save(p, arr)
        with pytest.raises(ValueError, match="Feature dim mismatch"):
            M.load_feature_file(str(p), FEATURE_DIM)

    def test_load_npy_1d_raises(self, tmp_path):
        arr = np.random.randn(FEATURE_DIM).astype(np.float32)
        p   = tmp_path / "onedim.npy"
        np.save(p, arr)
        with pytest.raises(ValueError, match="Expected 2D array"):
            M.load_feature_file(str(p), FEATURE_DIM)

    def test_load_pt_file(self, tmp_path):
        arr = torch.randn(FEATURE_DIM, T)
        p   = tmp_path / "feat.pt"
        torch.save(arr, p)
        loaded = M.load_feature_file(str(p), FEATURE_DIM)
        assert loaded.shape == (FEATURE_DIM, T)

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "feat.txt"
        p.write_text("nope")
        with pytest.raises(ValueError, match="Unsupported feature file"):
            M.load_feature_file(str(p), FEATURE_DIM)

    # ---- load_label_sequence ----

    def test_load_label_sequence_basic(self, tmp_path, label_map):
        df = pd.DataFrame({"Frame": [0, 1, 2, 3], "Locomotion": ["walk", "walk", "run", "background"]})
        p  = tmp_path / "labels.csv"
        df.to_csv(p, index=False)
        seq = M.load_label_sequence(str(p), "Locomotion", label_map, num_frames=5)
        assert seq[0] == label_map["walk"]
        assert seq[2] == label_map["run"]
        assert seq[4] == label_map["background"]   # unfilled frame defaults to background

    def test_load_label_sequence_missing_file(self, label_map):
        seq = M.load_label_sequence("/nonexistent/path.csv", "Locomotion", label_map, num_frames=10)
        assert all(v == label_map[M.BACKGROUND_LABEL] for v in seq)

    def test_load_label_sequence_missing_column(self, tmp_path, label_map):
        df = pd.DataFrame({"Frame": [0, 1], "OtherCol": ["x", "y"]})
        p  = tmp_path / "labels.csv"
        df.to_csv(p, index=False)
        seq = M.load_label_sequence(str(p), "Locomotion", label_map, num_frames=3)
        assert all(v == label_map[M.BACKGROUND_LABEL] for v in seq)

    def test_load_label_sequence_out_of_range_frames(self, tmp_path, label_map):
        df = pd.DataFrame({"Frame": [100, 200], "Locomotion": ["walk", "run"]})
        p  = tmp_path / "labels.csv"
        df.to_csv(p, index=False)
        seq = M.load_label_sequence(str(p), "Locomotion", label_map, num_frames=5)
        assert all(v == label_map[M.BACKGROUND_LABEL] for v in seq)

    def test_load_label_sequence_nan_becomes_background(self, tmp_path, label_map):
        df = pd.DataFrame({"Frame": [0], "Locomotion": ["nan"]})
        p  = tmp_path / "labels.csv"
        df.to_csv(p, index=False)
        seq = M.load_label_sequence(str(p), "Locomotion", label_map, num_frames=2)
        assert seq[0] == label_map[M.BACKGROUND_LABEL]

    # ---- Dataset & collate ----

    def _make_sample(self, tmp_path, feat_dim=FEATURE_DIM, num_frames=T,
                     label_col="Locomotion", vid_name="vid_001.mp4"):
        feat = np.random.randn(feat_dim, num_frames).astype(np.float32)
        feat_path = tmp_path / f"{vid_name}.npy"
        np.save(feat_path, feat)

        df = pd.DataFrame({
            "Frame":     list(range(num_frames)),
            label_col:   ["walk"] * (num_frames // 2) + ["run"] * (num_frames - num_frames // 2),
        })
        label_path = tmp_path / f"{vid_name}_labels.csv"
        df.to_csv(label_path, index=False)

        vid_path = tmp_path / vid_name
        vid_path.write_bytes(b"")   # empty placeholder

        return {
            "video_path":   str(vid_path),
            "label_path":   str(label_path),
            "feat_col":     str(feat_path),
        }

    def test_full_video_dataset_length(self, tmp_path, label_map):
        samples = [self._make_sample(tmp_path, vid_name=f"v{i}.mp4") for i in range(3)]
        ds = M.FullVideoDataset(samples, "Locomotion", label_map,
                                feature_col="feat_col", feature_dim=FEATURE_DIM)
        assert len(ds) == 3

    def test_full_video_dataset_item_shapes(self, tmp_path, label_map):
        sample = self._make_sample(tmp_path)
        ds     = M.FullVideoDataset([sample], "Locomotion", label_map,
                                     feature_col="feat_col", feature_dim=FEATURE_DIM)
        feat, labels, mask, vid_id = ds[0]
        assert feat.shape   == (FEATURE_DIM, T)
        assert labels.shape == (T,)
        assert mask.shape   == (1, T)
        assert isinstance(vid_id, str)

    def test_collate_fn_pads_correctly(self, tmp_path, label_map):
        s1 = self._make_sample(tmp_path, num_frames=20, vid_name="v1.mp4")
        s2 = self._make_sample(tmp_path, num_frames=35, vid_name="v2.mp4")
        ds = M.FullVideoDataset([s1, s2], "Locomotion", label_map,
                                 feature_col="feat_col", feature_dim=FEATURE_DIM)
        batch = M.collate_fn([ds[0], ds[1]])
        feats, labels, masks, vids = batch
        assert feats.shape  == (2, FEATURE_DIM, 35)
        assert labels.shape == (2, 35)
        assert masks.shape  == (2, 1, 35)
        # Padded positions should be -100 in labels and 0 in masks
        assert labels[0, 20:].unique().tolist() == [-100]
        assert masks[0, :, 20:].sum().item() == 0.0


# ============================================================
# 4. TRAINER — predict & evaluate (with mocks)
# ============================================================

class TestTrainerPredictEvaluate:

    def _make_trainer(self, tmp_dir):
        return M.MSTCNTrainer(
            num_classes=NUM_CLASSES,
            feature_dim=FEATURE_DIM,
            output_dir=tmp_dir,
            label_map=LABEL_MAP,
            seed=0,
        )

    def _make_checkpoint(self, tmp_dir, trainer):
        """Save a freshly initialised model as best_model.pt."""
        seed_dir = trainer._seed_dir()
        model    = M.MultiStageModel(
            num_layers_pg=3, num_layers_r=2, num_r=2,
            num_f_maps=8, dim=FEATURE_DIM, num_classes=NUM_CLASSES,
        )
        ckpt = os.path.join(seed_dir, "best_model.pt")
        torch.save(model.state_dict(), ckpt)
        return ckpt

    def _make_sample(self, tmp_path, vid_name="v0.mp4", num_frames=T):
        feat = np.random.randn(FEATURE_DIM, num_frames).astype(np.float32)
        feat_path  = os.path.join(tmp_path, f"{vid_name}.npy")
        np.save(feat_path, feat)

        df = pd.DataFrame({
            "Frame":      list(range(num_frames)),
            "Locomotion": ["walk"] * num_frames,
        })
        label_path = os.path.join(tmp_path, f"{vid_name}_labels.csv")
        df.to_csv(label_path, index=False)

        vid_path = os.path.join(tmp_path, vid_name)
        open(vid_path, "w").close()

        return {"video_path": vid_path, "label_path": label_path, "feat_col": feat_path}

    # --- seed_dir helper ---

    def test_seed_dir_created(self, tmp_path):
        trainer  = self._make_trainer(str(tmp_path))
        seed_dir = trainer._seed_dir()
        assert os.path.isdir(seed_dir)
        assert seed_dir.endswith("seed_0")

    # --- predict ---

    def test_predict_returns_lists(self, tmp_path):
        trainer = self._make_trainer(str(tmp_path))

        # Monkey-patch _build_model to use tiny arch
        def _tiny_model():
            return M.MultiStageModel(3, 2, 2, 8, FEATURE_DIM, NUM_CLASSES)
        trainer._build_model = _tiny_model

        ckpt    = self._make_checkpoint(str(tmp_path), trainer)
        sample  = self._make_sample(str(tmp_path))

        all_true, all_pred = trainer.predict(
            [sample], "Locomotion", "feat_col", ckpt, torch.device("cpu")
        )
        assert isinstance(all_true, list)
        assert isinstance(all_pred, list)
        assert len(all_true) == T
        assert len(all_pred) == T

    def test_predict_saves_csv(self, tmp_path):
        trainer = self._make_trainer(str(tmp_path))
        trainer._build_model = lambda: M.MultiStageModel(3, 2, 2, 8, FEATURE_DIM, NUM_CLASSES)
        ckpt   = self._make_checkpoint(str(tmp_path), trainer)
        sample = self._make_sample(str(tmp_path))

        trainer.predict([sample], "Locomotion", "feat_col", ckpt, torch.device("cpu"))

        pred_csv = os.path.join(trainer._seed_dir(), "test_frame_predictions.csv")
        assert os.path.exists(pred_csv)
        df = pd.read_csv(pred_csv)
        assert set(["video_id", "frame", "true_label", "pred_label", "correct"]).issubset(df.columns)

    def test_predict_saves_segment_summary(self, tmp_path):
        trainer = self._make_trainer(str(tmp_path))
        trainer._build_model = lambda: M.MultiStageModel(3, 2, 2, 8, FEATURE_DIM, NUM_CLASSES)
        ckpt   = self._make_checkpoint(str(tmp_path), trainer)
        sample = self._make_sample(str(tmp_path))

        trainer.predict([sample], "Locomotion", "feat_col", ckpt, torch.device("cpu"))

        seg_csv = os.path.join(trainer._seed_dir(), "test_segment_summary.csv")
        assert os.path.exists(seg_csv)

    # --- evaluate ---

    def test_evaluate_returns_dict(self, tmp_path):
        trainer = self._make_trainer(str(tmp_path))
        all_true = ["walk", "walk", "run",  "background"] * 10
        all_pred = ["walk", "run",  "run",  "background"] * 10
        metrics  = trainer.evaluate(all_true, all_pred, tag="test")
        assert isinstance(metrics, dict)
        for key in ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"):
            assert key in metrics, f"Missing key: {key}"

    def test_evaluate_saves_files(self, tmp_path):
        trainer = self._make_trainer(str(tmp_path))
        all_true = ["walk"] * 20 + ["run"] * 20
        all_pred = ["walk"] * 15 + ["run"] * 25
        trainer.evaluate(all_true, all_pred, tag="test")

        seed_dir = trainer._seed_dir()
        assert os.path.exists(os.path.join(seed_dir, "test_metrics.json"))
        assert os.path.exists(os.path.join(seed_dir, "test_report.txt"))
        assert os.path.exists(os.path.join(seed_dir, "test_confusion_matrix.csv"))

    def test_evaluate_empty_returns_empty_dict(self, tmp_path):
        trainer = self._make_trainer(str(tmp_path))
        result  = trainer.evaluate([], [], tag="test")
        assert result == {}

    def test_evaluate_perfect_predictions(self, tmp_path):
        trainer  = self._make_trainer(str(tmp_path))
        labels   = ["walk", "run", "background"] * 10
        metrics  = trainer.evaluate(labels, labels, tag="test")
        assert abs(metrics["accuracy"] - 1.0) < 1e-6
        assert abs(metrics["macro_f1"] - 1.0) < 1e-6

    def test_evaluate_metrics_json_valid(self, tmp_path):
        trainer  = self._make_trainer(str(tmp_path))
        all_true = ["walk", "run"] * 10
        all_pred = ["run",  "walk"] * 10
        trainer.evaluate(all_true, all_pred, tag="test")

        json_path = os.path.join(trainer._seed_dir(), "test_metrics.json")
        with open(json_path) as f:
            data = json.load(f)
        assert data["seed"] == 0
        assert 0.0 <= data["accuracy"] <= 1.0

    # --- aggregate_seeds ---

    def test_aggregate_seeds(self, tmp_path):
        seeds = [0, 1, 2]
        for seed in seeds:
            seed_dir = os.path.join(str(tmp_path), f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            metrics = {
                "seed": seed,
                "accuracy": 0.8 + seed * 0.01,
                "macro_f1": 0.75,
                "weighted_f1": 0.77,
                "macro_precision": 0.76,
                "macro_recall": 0.74,
                "per_class_f1": {"walk": 0.8, "run": 0.7, "background": 0.9},
                "num_frames": 100,
            }
            with open(os.path.join(seed_dir, "test_metrics.json"), "w") as f:
                json.dump(metrics, f)

        M.aggregate_seeds(str(tmp_path), seeds, tag="test")

        agg_path = os.path.join(str(tmp_path), "test_aggregate_seeds.csv")
        assert os.path.exists(agg_path)
        df = pd.read_csv(agg_path)
        assert "mean" in df.columns and "std" in df.columns
        assert "accuracy" in df["metric"].values