"""
MS-TCN2 Frame-Level Action Segmentation using pre-extracted features (I3D / VJEPA / R2+1D)

Usage:
    python mstcn2.py --feature_type i3d  --label loco  --action train  --seed 42
    python mstcn2.py --feature_type vjepa --label rmm  --action train  --seed 123
    python mstcn2.py --feature_type i3d  --label loco  --action predict --seed 42
    python mstcn2.py --feature_type i3d  --label loco  --action evaluate --seed 42

"""

import argparse
import json
import os
import random
from collections import defaultdict
from itertools import groupby
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset


# ============================================================
# ARGUMENT PARSING
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--label",        choices=["loco", "rmm"], required=True)
    p.add_argument("--feature_type", choices=["i3d", "vjepa", "r2plus1d"], default="i3d")
    p.add_argument("--action",       choices=["train", "predict", "evaluate"], default="train")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--split",        default="1")
    return p.parse_args()


# ============================================================
# STATIC CONFIG
# ============================================================
SPLIT_CSV = "/orcd/data/satra/002/projects/SAILS/action_outputs_features/labels_and_clips/latest_split_csv.csv"

LABEL_CONFIGS: dict[str, dict[str, str]] = {
    "loco": {
        "label_col":  "Locomotion",
        "output_dir": "/orcd/data/satra/002/projects/SAILS/action_outputs_features/models_output_seeds/mstcn2/loco/{feature_type}/",
    },
    "rmm": {
        "label_col":  "Repetitive_Motor_Movements",
        "output_dir": "/orcd/data/satra/002/projects/SAILS/action_outputs_features/models_output_seeds/mstcn2/rmm/{feature_type}/",
    },
}

FEATURE_DIM: dict[str, int] = {
    "i3d":      512,
    "vjepa":    1408,
    "r2plus1d": 512,
}

FEATURE_COL: dict[str, str] = {
    "i3d":      "i3d_full_path",
    "vjepa":    "vjepa_full_path",
    "r2plus1d": "r2plus1d_full_path",
}

# MS-TCN++ arch
NUM_LAYERS_PG = 11   # prediction generation stage layers
NUM_LAYERS_R  = 10   # refinement stage layers
NUM_R         = 3    # number of refinement stages
NUM_F_MAPS    = 64   # feature maps

# Training
NUM_EPOCHS    = 50
BATCH_SIZE    = 1
LEARNING_RATE = 5e-4

BACKGROUND_LABEL = "background"


# ============================================================
# MS-TCN++ MODEL
# ============================================================
class MultiStageModel(nn.Module):
    """Full MS-TCN++ with PG module + R refinement modules."""

    def __init__(
        self,
        num_layers_pg: int,
        num_layers_r: int,
        num_r: int,
        num_f_maps: int,
        dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.PG = PredictionGeneration(num_layers_pg, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList([
            Refinement(num_layers_r, num_f_maps, num_classes, num_classes)
            for _ in range(num_r)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.PG(x, mask)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        result: torch.Tensor = outputs
        return result   # [num_stages, B, C, T]


class PredictionGeneration(nn.Module):
    """
    MS-TCN++ PG module with DUAL dilated pathways:
      - conv_dilated_1: decreasing dilation (2^(L-1-i))
      - conv_dilated_2: increasing dilation (2^i)
    Fused via 1x1 conv at each layer.
    """

    def __init__(self, num_layers: int, num_f_maps: int, dim: int, num_classes: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1: nn.ModuleList = nn.ModuleList()
        self.conv_dilated_2: nn.ModuleList = nn.ModuleList()
        self.conv_fusion: nn.ModuleList    = nn.ModuleList()
        self.dropout: nn.ModuleList        = nn.ModuleList()

        for i in range(num_layers):
            dil_1 = 2 ** (num_layers - 1 - i)
            dil_2 = 2 ** i
            self.conv_dilated_1.append(
                nn.Conv1d(num_f_maps, num_f_maps, 3, padding=dil_1, dilation=dil_1)
            )
            self.conv_dilated_2.append(
                nn.Conv1d(num_f_maps, num_f_maps, 3, padding=dil_2, dilation=dil_2)
            )
            self.conv_fusion.append(nn.Conv1d(2 * num_f_maps, num_f_maps, 1))
            self.dropout.append(nn.Dropout())

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.conv_1x1_in(x)
        for i in range(self.num_layers):
            d1 = self.conv_dilated_1[i](out)
            d2 = self.conv_dilated_2[i](out)
            fused = self.conv_fusion[i](torch.cat([d1, d2], dim=1))
            fused = F.relu(fused)
            fused = self.dropout[i](fused)
            out = (out + fused) * mask[:, 0:1, :]
        result: torch.Tensor = self.conv_out(out) * mask[:, 0:1, :]
        return result


class Refinement(nn.Module):
    """Refinement module: single dilated pathway (increasing dilation)."""

    def __init__(self, num_layers: int, num_f_maps: int, dim: int, num_classes: int) -> None:
        super().__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        result: torch.Tensor = self.conv_out(out) * mask[:, 0:1, :]
        return result


class DilatedResidualLayer(nn.Module):

    def __init__(self, dilation: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3,
                                      padding=dilation, dilation=dilation)
        self.conv_1x1     = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout      = nn.Dropout()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        result: torch.Tensor = (x + out) * mask[:, 0:1, :]
        return result


# ============================================================
# MS-TCN2 LOSS (CE + smoothing)
# ============================================================
class MSTCNLoss(nn.Module):

    def __init__(self, num_classes: int, smoothing_weight: float = 0.15, clamp: float = 16.0) -> None:
        super().__init__()
        self.ce      = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        self.sm_w    = smoothing_weight
        self.clamp   = clamp
        self.num_classes = num_classes

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        loss: torch.Tensor = torch.tensor(0.0, device=outputs.device)
        for out in outputs:
            ce_loss = self.ce(out.transpose(1, 2).contiguous().view(-1, self.num_classes),
                              targets.view(-1))
            ce_loss = ce_loss.view(targets.shape) * mask[:, 0, :]
            loss   = loss + ce_loss.mean()

            sm_loss = torch.clamp(
                (F.log_softmax(out[:, :, 1:], dim=1) -
                 F.log_softmax(out[:, :, :-1], dim=1)) ** 2,
                min=0, max=self.clamp
            )
            sm_loss = self.sm_w * sm_loss * mask[:, :, 1:]
            loss    = loss + sm_loss.mean()
        result: torch.Tensor = loss
        return result


# ============================================================
# DATA LOADING
# ============================================================
def load_feature_file(path: str, expected_dim: int) -> npt.NDArray[np.float32]:
    """Load feature file and validate shape is (D, T)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr: npt.NDArray[np.float32] = np.load(path)
    elif ext in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            key = list(f.keys())[0]
            arr = f[key][()]
    elif ext == ".pt":
        arr = torch.load(path, map_location="cpu").numpy()
    else:
        raise ValueError(f"Unsupported feature file: {path}")

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape} in {path}")

    if arr.shape[0] != expected_dim and arr.shape[1] == expected_dim:
        print(f"  [shape fix] Transposing {arr.shape} -> ({arr.shape[1]}, {arr.shape[0]}) for {os.path.basename(path)}")
        arr = arr.T
    elif arr.shape[0] != expected_dim and arr.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dim mismatch in {path}: shape={arr.shape}, expected_dim={expected_dim}"
        )

    return arr.astype(np.float32)   # (D, T)


def load_label_sequence(
    label_path: str,
    label_col: str,
    label_map: dict[str, int],
    num_frames: int,
) -> npt.NDArray[np.int64]:
    try:
        df = pd.read_csv(label_path, encoding="utf-8-sig", keep_default_na=False)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"  Warning: cannot read {label_path}: {e}")
        return np.full(num_frames, label_map[BACKGROUND_LABEL], dtype=np.int64)

    labels: npt.NDArray[np.int64] = np.full(num_frames, label_map[BACKGROUND_LABEL], dtype=np.int64)

    if label_col not in df.columns or "Frame" not in df.columns:
        return labels

    for _, row in df.iterrows():
        frame_idx = int(row["Frame"])
        if frame_idx < 0 or frame_idx >= num_frames:
            continue
        raw = str(row[label_col]).strip()
        if raw in ("", "N/A", "nan", "NaN"):
            raw = BACKGROUND_LABEL
        if raw in label_map:
            labels[frame_idx] = label_map[raw]

    return labels


def gather_all_labels(df_csv: pd.DataFrame, label_col: str) -> list[str]:
    all_labels: set[str] = {BACKGROUND_LABEL}
    for _, row in df_csv.iterrows():
        lp = str(row.get("label_path", "")).strip()
        if not os.path.exists(lp):
            continue
        try:
            df = pd.read_csv(lp, encoding="utf-8-sig", keep_default_na=False)
            df.columns = df.columns.str.strip()
        except Exception:
            continue
        if label_col not in df.columns:
            continue
        for v in df[label_col].astype(str).unique():
            v = v.strip()
            if v not in ("", "N/A", "nan", "NaN"):
                all_labels.add(v)
    return sorted(all_labels)


# ============================================================
# DATASET
# ============================================================
# Each item: (feat_tensor, labels_tensor, mask_tensor, vid_id_str)
_DatasetItem = tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]


class FullVideoDataset(Dataset[_DatasetItem]):

    def __init__(
        self,
        samples: list[dict[str, Any]],
        label_col: str,
        label_map: dict[str, int],
        feature_col: str,
        feature_dim: int,
    ) -> None:
        self.samples     = samples
        self.label_col   = label_col
        self.label_map   = label_map
        self.feature_col = feature_col
        self.feature_dim = feature_dim

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> _DatasetItem:
        row = self.samples[idx]
        feat_path  = str(row[self.feature_col]).strip()
        label_path = str(row["label_path"]).strip()
        vid_id     = os.path.basename(str(row["video_path"]))

        feat = load_feature_file(feat_path, self.feature_dim)  # (D, T)
        T    = feat.shape[1]
        labels = load_label_sequence(label_path, self.label_col, self.label_map, T)
        mask = np.ones((1, T), dtype=np.float32)

        return (
            torch.from_numpy(feat),
            torch.from_numpy(labels),
            torch.from_numpy(mask),
            vid_id,
        )


def collate_fn(
    batch: list[_DatasetItem],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    feats_list, labels_list, masks_list, vids = zip(*batch, strict=False)
    T_max = max(f.shape[1] for f in feats_list)
    D     = feats_list[0].shape[0]

    feats_pad  = torch.zeros(len(feats_list), D,     T_max)
    labels_pad = torch.full( (len(feats_list), T_max), -100, dtype=torch.long)
    masks_pad  = torch.zeros(len(feats_list), 1,     T_max)

    for i, (f, lbl, m) in enumerate(zip(feats_list, labels_list, masks_list, strict=False)):
        T = f.shape[1]
        feats_pad[i,  :, :T] = f
        labels_pad[i,    :T] = lbl
        masks_pad[i,  :, :T] = m

    return feats_pad, labels_pad, masks_pad, list(vids)


# ============================================================
# TRAINER
# ============================================================
class MSTCNTrainer:

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        output_dir: str,
        label_map: dict[str, int],
        seed: int,
    ) -> None:
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.output_dir  = output_dir
        self.label_map   = label_map
        self.id_to_label: dict[int, str] = {v: k for k, v in label_map.items()}
        self.seed        = seed

    def _build_model(self) -> MultiStageModel:
        return MultiStageModel(
            num_layers_pg = NUM_LAYERS_PG,
            num_layers_r  = NUM_LAYERS_R,
            num_r         = NUM_R,
            num_f_maps    = NUM_F_MAPS,
            dim           = self.feature_dim,
            num_classes   = self.num_classes,
        )

    def _seed_dir(self) -> str:
        """Per-seed output subdirectory."""
        d = os.path.join(self.output_dir, f"seed_{self.seed}")
        os.makedirs(d, exist_ok=True)
        return d

    def train(
        self,
        train_samples: list[dict[str, Any]],
        val_samples: list[dict[str, Any]],
        label_col: str,
        feature_col: str,
        device: torch.device,
    ) -> str:
        seed_dir = self._seed_dir()

        train_ds = FullVideoDataset(train_samples, label_col, self.label_map,
                                    feature_col, self.feature_dim)
        val_ds   = FullVideoDataset(val_samples,   label_col, self.label_map,
                                    feature_col, self.feature_dim)

        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_fn, num_workers=0)
        val_dl   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)

        model     = self._build_model().to(device)
        criterion = MSTCNLoss(self.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        best_val_loss: float = float("inf")
        start_epoch    = 1
        history: list[dict[str, Any]] = []

        best_ckpt    = os.path.join(seed_dir, "best_model.pt")
        resume_ckpt  = os.path.join(seed_dir, "resume_checkpoint.pt")
        hist_path    = os.path.join(seed_dir, "training_history.csv")

        # ---- RESUME if checkpoint exists ----
        if os.path.exists(resume_ckpt):
            print(f"\n*** Resuming from checkpoint: {resume_ckpt}")
            ckpt = torch.load(resume_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            best_val_loss = ckpt["best_val_loss"]
            start_epoch   = ckpt["epoch"] + 1
            print(f"    Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

            if os.path.exists(hist_path):
                history = pd.read_csv(hist_path).to_dict("records")
        else:
            print(f"\n*** No resume checkpoint found — training from scratch (seed={self.seed})")

        if start_epoch > NUM_EPOCHS:
            print("Training already completed for this seed. Skipping.")
            return best_ckpt

        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            # ---- TRAIN ----
            model.train()
            train_loss, n_correct, n_total = 0.0, 0, 0
            for feats, labels, masks, _ in train_dl:
                feats  = feats.to(device)
                labels = labels.to(device)
                masks  = masks.to(device)

                optimizer.zero_grad()
                outputs = model(feats, masks)
                loss    = criterion(outputs, labels, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds       = outputs[-1].argmax(dim=1)
                valid       = (labels != -100)
                n_correct  += int(((preds == labels) & valid).sum().item())
                n_total    += int(valid.sum().item())

            train_acc  = n_correct / max(n_total, 1)
            train_loss /= len(train_dl)

            # ---- VAL ----
            model.eval()
            val_loss, nv_correct, nv_total = 0.0, 0, 0
            with torch.no_grad():
                for feats, labels, masks, _ in val_dl:
                    feats  = feats.to(device)
                    labels = labels.to(device)
                    masks  = masks.to(device)
                    outputs = model(feats, masks)
                    loss    = criterion(outputs, labels, masks)
                    val_loss += loss.item()
                    preds     = outputs[-1].argmax(dim=1)
                    valid     = (labels != -100)
                    nv_correct += int(((preds == labels) & valid).sum().item())
                    nv_total   += int(valid.sum().item())

            val_acc  = nv_correct / max(nv_total, 1)
            val_loss /= max(len(val_dl), 1)
            scheduler.step(val_loss)

            history.append({
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
            })

            print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_ckpt)
                print(f"  -> Saved best model (val_loss={val_loss:.4f})")

            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss":   best_val_loss,
            }, resume_ckpt)

            pd.DataFrame(history).to_csv(hist_path, index=False)

        if os.path.exists(resume_ckpt):
            os.remove(resume_ckpt)
            print("Removed resume checkpoint (training complete).")

        print(f"Training history -> {hist_path}")
        print(f"Training complete. Best checkpoint: {best_ckpt}")
        return best_ckpt

    def predict(
        self,
        test_samples: list[dict[str, Any]],
        label_col: str,
        feature_col: str,
        ckpt_path: str,
        device: torch.device,
    ) -> tuple[list[str], list[str]]:
        """Run inference, save frame-level predictions + segment summary."""
        seed_dir = self._seed_dir()
        model = self._build_model().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        test_ds = FullVideoDataset(test_samples, label_col, self.label_map,
                                   feature_col, self.feature_dim)

        rows: list[dict[str, Any]] = []
        all_true: list[str] = []
        all_pred: list[str] = []

        print(f"\nRunning inference on {len(test_ds)} videos (seed={self.seed})...")

        with torch.no_grad():
            for i in range(len(test_ds)):
                feat, label_seq, mask, vid_id = test_ds[i]
                feat      = feat.unsqueeze(0).to(device)
                mask      = mask.unsqueeze(0).to(device)
                label_seq = label_seq.to(device)

                outputs = model(feat, mask)
                preds   = outputs[-1].squeeze(0).argmax(dim=0)

                T = feat.shape[2]
                for t in range(T):
                    gt   = int(label_seq[t].item())
                    pred = int(preds[t].item())
                    rows.append({
                        "video_id":   vid_id,
                        "frame":      t,
                        "true_label": self.id_to_label.get(gt,   str(gt)),
                        "pred_label": self.id_to_label.get(pred,  str(pred)),
                        "correct":    int(gt == pred) if gt != -100 else -1,
                    })
                    if gt != -100:
                        all_true.append(self.id_to_label.get(gt,   str(gt)))
                        all_pred.append(self.id_to_label.get(pred,  str(pred)))

                valid_rows = [r for r in rows if r["video_id"] == vid_id and r["correct"] != -1]
                if valid_rows:
                    video_acc = sum(r["correct"] for r in valid_rows) / len(valid_rows)
                    print(f"  [{i+1}/{len(test_ds)}] {vid_id}  acc={video_acc:.3f}")

        pred_csv = os.path.join(seed_dir, "test_frame_predictions.csv")
        pd.DataFrame(rows).to_csv(pred_csv, index=False)
        print(f"Frame-level predictions saved -> {pred_csv}")

        self._save_segment_summary(rows, seed_dir)

        return all_true, all_pred

    def evaluate(
        self,
        all_true: list[str],
        all_pred: list[str],
        tag: str = "test",
    ) -> dict[str, Any]:
        """Compute and save detailed metrics."""
        seed_dir = self._seed_dir()

        if not all_true:
            print("No valid predictions to evaluate.")
            return {}

        acc         = accuracy_score(all_true, all_pred)
        macro_f1    = f1_score(all_true, all_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(all_true, all_pred, average="weighted", zero_division=0)
        macro_prec  = precision_score(all_true, all_pred, average="macro", zero_division=0)
        macro_rec   = recall_score(all_true, all_pred, average="macro", zero_division=0)

        labels_sorted = sorted(set(all_true) | set(all_pred))

        per_class_f1 = f1_score(all_true, all_pred, average=None,
                                labels=labels_sorted, zero_division=0)
        per_class_dict = {lab: float(f) for lab, f in zip(labels_sorted, per_class_f1, strict=True)}

        report_str = classification_report(all_true, all_pred,
                                           labels=labels_sorted, zero_division=0)
        cm    = confusion_matrix(all_true, all_pred, labels=labels_sorted)
        cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)

        metrics: dict[str, Any] = {
            "seed":            self.seed,
            "accuracy":        float(acc),
            "macro_f1":        float(macro_f1),
            "weighted_f1":     float(weighted_f1),
            "macro_precision": float(macro_prec),
            "macro_recall":    float(macro_rec),
            "per_class_f1":    per_class_dict,
            "num_frames":      len(all_true),
        }

        print(f"\n{'='*60}")
        print(f"  EVALUATION RESULTS  (seed={self.seed}, tag={tag})")
        print(f"{'='*60}")
        print(f"  Accuracy       : {acc:.4f}")
        print(f"  Macro F1       : {macro_f1:.4f}")
        print(f"  Weighted F1    : {weighted_f1:.4f}")
        print(f"  Macro Precision: {macro_prec:.4f}")
        print(f"  Macro Recall   : {macro_rec:.4f}")
        print(f"  Num frames     : {len(all_true)}")
        print(f"\nClassification Report:\n{report_str}")
        print(f"Confusion Matrix:\n{cm_df}\n")

        metrics_path = os.path.join(seed_dir, f"{tag}_metrics.json")
        with open(metrics_path, "w") as fp:
            json.dump(metrics, fp, indent=2)

        report_path = os.path.join(seed_dir, f"{tag}_report.txt")
        with open(report_path, "w") as fp:
            fp.write(f"Seed: {self.seed}\n")
            fp.write(f"Accuracy:        {acc:.4f}\n")
            fp.write(f"Macro F1:        {macro_f1:.4f}\n")
            fp.write(f"Weighted F1:     {weighted_f1:.4f}\n")
            fp.write(f"Macro Precision: {macro_prec:.4f}\n")
            fp.write(f"Macro Recall:    {macro_rec:.4f}\n\n")
            fp.write(report_str)
            fp.write(f"\n\nConfusion Matrix:\n{cm_df.to_string()}\n")

        cm_csv_path = os.path.join(seed_dir, f"{tag}_confusion_matrix.csv")
        cm_df.to_csv(cm_csv_path)

        print(f"Metrics JSON  -> {metrics_path}")
        print(f"Report        -> {report_path}")
        print(f"Confusion mat -> {cm_csv_path}")

        return metrics

    def _save_segment_summary(self, rows: list[dict[str, Any]], seed_dir: str) -> None:
        seg_rows: list[dict[str, Any]] = []
        vid_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            vid_groups[r["video_id"]].append(r)

        for vid_id, frames in vid_groups.items():
            frames = sorted(frames, key=lambda x: x["frame"])
            for label, group in groupby(frames, key=lambda x: x["pred_label"]):
                grp   = list(group)
                start = grp[0]["frame"]
                end   = grp[-1]["frame"]
                seg_rows.append({
                    "video_id":        vid_id,
                    "start_frame":     start,
                    "end_frame":       end,
                    "duration_frames": end - start + 1,
                    "pred_label":      label,
                })

        seg_csv = os.path.join(seed_dir, "test_segment_summary.csv")
        pd.DataFrame(seg_rows).to_csv(seg_csv, index=False)
        print(f"Segment summary -> {seg_csv}")


# ============================================================
# AGGREGATE MULTI-SEED RESULTS
# ============================================================
def aggregate_seeds(output_dir: str, seeds: list[int], tag: str = "test") -> None:
    """Load per-seed metrics JSONs and compute mean ± std."""
    all_metrics: list[dict[str, Any]] = []
    for seed in seeds:
        path = os.path.join(output_dir, f"seed_{seed}", f"{tag}_metrics.json")
        if not os.path.exists(path):
            print(f"  Warning: missing {path}, skipping seed {seed}")
            continue
        with open(path) as fp:
            all_metrics.append(json.load(fp))

    if len(all_metrics) < 2:
        print("Not enough seed results to aggregate.")
        return

    scalar_keys = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    rows: list[dict[str, Any]] = []
    for key in scalar_keys:
        vals = [m[key] for m in all_metrics]
        rows.append({
            "metric": key,
            "mean":   np.mean(vals),
            "std":    np.std(vals),
            "min":    np.min(vals),
            "max":    np.max(vals),
            "values": vals,
        })

    all_classes: set[str] = set()
    for m in all_metrics:
        all_classes.update(m.get("per_class_f1", {}).keys())

    for cls in sorted(all_classes):
        vals = [m.get("per_class_f1", {}).get(cls, 0.0) for m in all_metrics]
        rows.append({
            "metric": f"f1_{cls}",
            "mean":   np.mean(vals),
            "std":    np.std(vals),
            "min":    np.min(vals),
            "max":    np.max(vals),
            "values": vals,
        })

    df = pd.DataFrame(rows)

    agg_path = os.path.join(output_dir, f"{tag}_aggregate_seeds.csv")
    df.to_csv(agg_path, index=False)

    print(f"\n{'='*60}")
    print(f"  AGGREGATE RESULTS across seeds {seeds}")
    print(f"{'='*60}")
    for _, row in df.iterrows():
        print(f"  {row['metric']:25s} : {row['mean']:.4f} ± {row['std']:.4f}  "
              f"(min={row['min']:.4f}, max={row['max']:.4f})")
    print(f"\nSaved -> {agg_path}")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    args = parse_args()
    cfg  = LABEL_CONFIGS[args.label]

    label_col   = cfg["label_col"]
    output_dir  = cfg["output_dir"].format(feature_type=args.feature_type)
    feature_col = FEATURE_COL[args.feature_type]
    feature_dim = FEATURE_DIM[args.feature_type]
    seed        = args.seed

    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nMode        : {args.label.upper()}")
    print(f"Feature     : {args.feature_type}  (col={feature_col}, dim={feature_dim})")
    print(f"Label col   : {label_col}")
    print(f"Seed        : {seed}")
    print(f"Output dir  : {output_dir}")

    df_csv = pd.read_csv(SPLIT_CSV)
    df_csv.columns = df_csv.columns.str.strip()

    required = ["video_path", "label_path", feature_col, "split"]
    missing  = [c for c in required if c not in df_csv.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    def row_ok(row: pd.Series) -> bool:
        return (os.path.exists(str(row["video_path"])) and
                os.path.exists(str(row["label_path"])) and
                os.path.exists(str(row[feature_col])))

    df_csv = df_csv[df_csv.apply(row_ok, axis=1)].reset_index(drop=True)
    print(f"Valid rows (all files exist): {len(df_csv)}")

    print("Scanning labels across all annotation files...")
    all_label_strs = gather_all_labels(df_csv, label_col)
    label_map = {lab: i for i, lab in enumerate(all_label_strs)}
    num_classes = len(label_map)
    print(f"Labels ({num_classes}): {label_map}")

    label_map_path = os.path.join(output_dir, "label_mapping.json")
    with open(label_map_path, "w") as fp:
        json.dump(label_map, fp, indent=2)

    splits = df_csv["split"].str.strip().str.lower()
    train_samples = df_csv[splits == "train"].to_dict("records")
    val_samples   = df_csv[splits == "val"].to_dict("records")
    test_samples  = df_csv[splits == "test"].to_dict("records")
    print(f"Split -> train: {len(train_samples)} | val: {len(val_samples)} | test: {len(test_samples)}")

    trainer = MSTCNTrainer(num_classes, feature_dim, output_dir, label_map, seed)
    ckpt_path = os.path.join(trainer._seed_dir(), "best_model.pt")

    if args.action == "train":
        if not train_samples:
            raise RuntimeError("No training samples found.")
        ckpt_path = trainer.train(train_samples, val_samples, label_col, feature_col, device)
        if test_samples:
            all_true, all_pred = trainer.predict(test_samples, label_col, feature_col,
                                                 ckpt_path, device)
            trainer.evaluate(all_true, all_pred, tag="test")
        if val_samples:
            all_true_v, all_pred_v = trainer.predict(val_samples, label_col, feature_col,
                                                     ckpt_path, device)
            trainer.evaluate(all_true_v, all_pred_v, tag="val")

    elif args.action == "predict":
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train first.")
        all_true, all_pred = trainer.predict(test_samples, label_col, feature_col,
                                             ckpt_path, device)
        trainer.evaluate(all_true, all_pred, tag="test")

    elif args.action == "evaluate":
        seeds = [42, 123, 456]
        aggregate_seeds(output_dir, seeds, tag="test")


if __name__ == "__main__":
    main()