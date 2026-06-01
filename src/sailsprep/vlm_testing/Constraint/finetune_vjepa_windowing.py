"""
Fine-tune V-JEPA2 for Constraint Prediction with Temporal Windowing
Strategy:
  1. Divide video into N windows
  2. Sample frames from each window
  3. Get prediction for each window
  4. Majority vote across windows for final prediction
"""

import argparse
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoVideoProcessor
from pathlib import Path
import av
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
from collections import Counter


class VideoDataset(Dataset):
    """Dataset for video constraint prediction with temporal windowing"""
    
    def __init__(self, data, processor, num_frames=64, num_windows=3, use_windowing=False):
        self.data = data
        self.processor = processor
        self.num_frames = num_frames
        self.num_windows = num_windows
        self.use_windowing = use_windowing
    
    def __len__(self):
        return len(self.data)
    
    def extract_frames(self, video_path):
        """Extract evenly spaced frames from entire video"""
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            
            total_frames = stream.frames
            if total_frames == 0:
                total_frames = int(stream.duration * stream.time_base * stream.average_rate)
            
            if total_frames <= self.num_frames:
                indices = list(range(total_frames))
            else:
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            frames = []
            container.seek(0)
            frame_count = 0
            
            for frame in container.decode(video=0):
                if frame_count in indices:
                    frames.append(frame.to_ndarray(format='rgb24'))
                frame_count += 1
                if len(frames) == self.num_frames:
                    break
            
            container.close()
            
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
            
            return frames[:self.num_frames]
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames
    
    def extract_windowed_frames(self, video_path):
        """
        Extract frames from multiple temporal windows
        
        Example with num_windows=3, num_frames=64:
            Window 1: frames 0-20    (sample 21-22 frames)
            Window 2: frames 21-41   (sample 21-22 frames)  
            Window 3: frames 42-63   (sample 21-22 frames)
            Total: ~64 frames
        
        Returns:
            List of frame lists, one per window
        """
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            
            total_frames = stream.frames
            if total_frames == 0:
                total_frames = int(stream.duration * stream.time_base * stream.average_rate)
            
            # Calculate frames per window
            frames_per_window = self.num_frames // self.num_windows
            
            all_windows = []
            
            for window_idx in range(self.num_windows):
                # Define window boundaries
                window_start = (total_frames // self.num_windows) * window_idx
                window_end = (total_frames // self.num_windows) * (window_idx + 1)
                
                if window_idx == self.num_windows - 1:
                    window_end = total_frames  # Last window goes to end
                
                # Sample frames within this window
                if window_end - window_start <= frames_per_window:
                    indices = list(range(window_start, window_end))
                else:
                    indices = np.linspace(window_start, window_end - 1, frames_per_window, dtype=int)
                
                # Extract frames for this window
                window_frames = []
                container.seek(0)
                frame_count = 0
                
                for frame in container.decode(video=0):
                    if frame_count in indices:
                        window_frames.append(frame.to_ndarray(format='rgb24'))
                    frame_count += 1
                    if len(window_frames) == frames_per_window:
                        break
                
                # Pad if needed
                while len(window_frames) < frames_per_window:
                    window_frames.append(window_frames[-1] if window_frames else np.zeros((224, 224, 3), dtype=np.uint8))
                
                all_windows.append(window_frames[:frames_per_window])
            
            container.close()
            
            return all_windows
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            dummy_frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * (self.num_frames // self.num_windows)
            return [dummy_frames for _ in range(self.num_windows)]
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.use_windowing:
            # Extract frames from multiple windows
            windows = self.extract_windowed_frames(item['video_path'])
            
            # Process each window separately
            pixel_values_list = []
            for window_frames in windows:
                video = np.stack(window_frames)
                video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)
                inputs = self.processor(video_tensor, return_tensors='pt')
                pixel_values_list.append(inputs['pixel_values_videos'].squeeze(0))
            
            # Stack all windows: [num_windows, num_frames, C, H, W]
            pixel_values = torch.stack(pixel_values_list)
            
        else:
            # Original approach: single pass through video
            frames = self.extract_frames(item['video_path'])
            video = np.stack(frames)
            video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)
            inputs = self.processor(video_tensor, return_tensors='pt')
            pixel_values = inputs['pixel_values_videos'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'constrained': item['constrained'],
            'constraint_type': item['constraint_type'],
            'video_path': item['video_path'],
            'use_windowing': self.use_windowing
        }


class VJEPAConstraintModel(nn.Module):
    """V-JEPA2 with constraint prediction heads"""
    
    def __init__(self, vjepa_model, num_status_classes=3, num_type_classes=10):
        super().__init__()
        self.vjepa = vjepa_model
        self.embed_dim = vjepa_model.config.hidden_size
        
        for param in self.vjepa.parameters():
            param.requires_grad = False
        
        self.status_head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_status_classes)
        )
        
        self.type_head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_type_classes)
        )
    
    def forward(self, pixel_values, use_windowing=False):
        """
        Args:
            pixel_values: 
                If use_windowing=False: [B, T, C, H, W]
                If use_windowing=True: [B, num_windows, T, C, H, W]
            use_windowing: Whether to process multiple windows
        
        Returns:
            status_logits: [B, num_status_classes] or [B, num_windows, num_status_classes]
            type_logits: [B, num_type_classes] or [B, num_windows, num_type_classes]
        """
        
        if use_windowing:
            # Process each window separately
            batch_size, num_windows = pixel_values.shape[:2]
            
            # Reshape: [B*num_windows, T, C, H, W]
            pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
            
            # Get features for all windows
            outputs = self.vjepa.get_vision_features(pixel_values)
            features = outputs.mean(dim=1)  # [B*num_windows, D]
            
            # Get predictions for each window
            status_logits = self.status_head(features)  # [B*num_windows, num_status]
            type_logits = self.type_head(features)      # [B*num_windows, num_types]
            
            # Reshape back: [B, num_windows, num_classes]
            status_logits = status_logits.view(batch_size, num_windows, -1)
            type_logits = type_logits.view(batch_size, num_windows, -1)
            
            return status_logits, type_logits
        
        else:
            # Original: process entire video at once
            outputs = self.vjepa.get_vision_features(pixel_values)
            features = outputs.mean(dim=1)
            
            status_logits = self.status_head(features)
            type_logits = self.type_head(features)
            
            return status_logits, type_logits


def majority_vote(predictions):
    """
    Perform majority voting across windows
    
    Args:
        predictions: List of predictions from each window
    
    Returns:
        Most common prediction
    """
    counter = Counter(predictions)
    return counter.most_common(1)[0][0]


def train_epoch(model, dataloader, optimizer, criterion, device, status_to_idx, type_to_idx, use_windowing=False):
    """Train for one epoch with optional windowing"""
    model.train()
    total_loss = 0
    status_preds, status_labels = [], []
    
    for batch in tqdm(dataloader, desc="Training"):
        pixel_values = batch['pixel_values'].to(device)
        use_batch_windowing = batch['use_windowing'][0].item() if use_windowing else False
        
        status_batch = [status_to_idx.get(s, 0) for s in batch['constrained']]
        type_batch = [type_to_idx.get(t, 0) for t in batch['constraint_type']]
        
        status_batch = torch.tensor(status_batch, device=device)
        type_batch = torch.tensor(type_batch, device=device)
        
        optimizer.zero_grad()
        status_logits, type_logits = model(pixel_values, use_windowing=use_batch_windowing)
        
        if use_batch_windowing:
            # Average predictions across windows for training
            status_logits = status_logits.mean(dim=1)
            type_logits = type_logits.mean(dim=1)
        
        loss_status = criterion(status_logits, status_batch)
        
        constrained_mask = (status_batch != status_to_idx['n'])
        if constrained_mask.sum() > 0:
            loss_type = criterion(type_logits[constrained_mask], type_batch[constrained_mask])
            loss = loss_status + 0.5 * loss_type
        else:
            loss = loss_status
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        status_preds.extend(status_logits.argmax(dim=1).cpu().tolist())
        status_labels.extend(status_batch.cpu().tolist())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(status_labels, status_preds)
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, status_to_idx, type_to_idx, idx_to_status, idx_to_type, use_windowing=False):
    """Validate model with optional majority voting"""
    model.eval()
    total_loss = 0
    status_preds, status_labels = [], []
    type_preds, type_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            pixel_values = batch['pixel_values'].to(device)
            use_batch_windowing = batch['use_windowing'][0].item() if use_windowing else False
            
            status_batch = [status_to_idx.get(s, 0) for s in batch['constrained']]
            type_batch = [type_to_idx.get(t, 0) for t in batch['constraint_type']]
            
            status_batch = torch.tensor(status_batch, device=device)
            type_batch = torch.tensor(type_batch, device=device)
            
            status_logits, type_logits = model(pixel_values, use_windowing=use_batch_windowing)
            
            if use_batch_windowing:
                # Majority voting across windows
                for b in range(status_logits.shape[0]):
                    # Get prediction for each window
                    window_status_preds = status_logits[b].argmax(dim=1).cpu().tolist()
                    window_type_preds = type_logits[b].argmax(dim=1).cpu().tolist()
                    
                    # Majority vote
                    final_status_pred = majority_vote([idx_to_status[p] for p in window_status_preds])
                    final_type_pred = majority_vote([idx_to_type[p] for p in window_type_preds])
                    
                    status_preds.append(final_status_pred)
                    status_labels.append(idx_to_status[status_batch[b].item()])
                    
                    if status_batch[b].item() != status_to_idx['n']:
                        type_preds.append(final_type_pred)
                        type_labels.append(idx_to_type[type_batch[b].item()])
                
                # Loss: average across windows
                status_logits_mean = status_logits.mean(dim=1)
                type_logits_mean = type_logits.mean(dim=1)
                loss_status = criterion(status_logits_mean, status_batch)
                constrained_mask = (status_batch != status_to_idx['n'])
                if constrained_mask.sum() > 0:
                    loss_type = criterion(type_logits_mean[constrained_mask], type_batch[constrained_mask])
                    loss = loss_status + 0.5 * loss_type
                else:
                    loss = loss_status
            else:
                # Original: no windowing
                loss_status = criterion(status_logits, status_batch)
                constrained_mask = (status_batch != status_to_idx['n'])
                if constrained_mask.sum() > 0:
                    loss_type = criterion(type_logits[constrained_mask], type_batch[constrained_mask])
                    loss = loss_status + 0.5 * loss_type
                else:
                    loss = loss_status
                
                status_pred_idx = status_logits.argmax(dim=1)
                type_pred_idx = type_logits.argmax(dim=1)
                
                for i in range(len(status_batch)):
                    status_preds.append(idx_to_status[status_pred_idx[i].item()])
                    status_labels.append(idx_to_status[status_batch[i].item()])
                    
                    if status_batch[i].item() != status_to_idx['n']:
                        type_preds.append(idx_to_type[type_pred_idx[i].item()])
                        type_labels.append(idx_to_type[type_batch[i].item()])
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    status_acc = accuracy_score(status_labels, status_preds)
    status_f1 = f1_score(status_labels, status_preds, average='weighted', zero_division=0)
    type_acc = accuracy_score(type_labels, type_preds) if type_labels else 0.0
    
    return {
        'loss': avg_loss,
        'status_accuracy': status_acc,
        'status_f1': status_f1,
        'type_accuracy': type_acc,
        'status_preds': status_preds,
        'status_labels': status_labels
    }


def main():
    parser = argparse.ArgumentParser(description='Fine-tune V-JEPA2 with temporal windowing')
    parser.add_argument('--data', type=str, required=True, help='Prepared data pickle file')
    parser.add_argument('--model_id', type=str, default='facebook/vjepa2-vitl-fpc64-256')
    parser.add_argument('--model_cache', type=str, default='./models')
    parser.add_argument('--output', type=str, default='./finetuned_vjepa')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_frames', type=int, default=64)
    parser.add_argument('--num_windows', type=int, default=3, help='Number of temporal windows')
    parser.add_argument('--use_windowing', action='store_true', help='Enable temporal windowing with majority voting')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=8)
    
    args = parser.parse_args()

    print(f"\nModel: {args.model_id}")
    print(f"Frames per video: {args.num_frames}")
    print(f"Temporal windowing: {'ENABLED' if args.use_windowing else 'DISABLED'}")
    if args.use_windowing:
        print(f"Number of windows: {args.num_windows}")
        print(f"Frames per window: {args.num_frames // args.num_windows}")
        print(f"Prediction: Majority vote across windows")
    
    # Load prepared data
    print(f"\nLoading data from {args.data}")
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Train: {len(data['train'])} videos")
    print(f"Val:   {len(data['val'])} videos")
    if 'test' in data:
        print(f"Test:  {len(data['test'])} videos (held out)")
    
    # Create label mappings
    status_labels = sorted(set(d['constrained'] for d in data['train']))
    type_labels = sorted(set(d['constraint_type'] for d in data['train'] if d['constrained'] in ['y', 'partial']))
    
    status_to_idx = {label: idx for idx, label in enumerate(status_labels)}
    idx_to_status = {idx: label for label, idx in status_to_idx.items()}
    
    type_to_idx = {label: idx for idx, label in enumerate(type_labels)}
    idx_to_type = {idx: label for label, idx in type_to_idx.items()}
    
    print(f"\nStatus classes: {status_labels}")
    print(f"Type classes: {len(type_labels)} unique types")

    print(f"\nLoading V-JEPA2 from {args.model_cache}")
    try:
        processor = AutoVideoProcessor.from_pretrained(args.model_id, cache_dir=args.model_cache)
        vjepa = AutoModel.from_pretrained(args.model_id, cache_dir=args.model_cache)
    except Exception as e:
        print(f"\n ERROR: {e}")
        return 1
    
    # Create model
    model = VJEPAConstraintModel(vjepa, len(status_labels), len(type_labels))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model on: {device}")
    
    # Create datasets
    print("\nCreating datasets")
    train_dataset = VideoDataset(data['train'], processor, args.num_frames, args.num_windows, args.use_windowing)
    val_dataset = VideoDataset(data['val'], processor, args.num_frames, args.num_windows, args.use_windowing)
    batch_size = args.batch_size if not args.use_windowing else max(1, args.batch_size // args.num_windows)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Effective batch size: {batch_size}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    
    best_f1 = 0.0
    history = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            status_to_idx, type_to_idx, args.use_windowing
        )
        
        val_metrics = validate(
            model, val_loader, criterion, device, 
            status_to_idx, type_to_idx, idx_to_status, idx_to_type, args.use_windowing
        )
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['status_accuracy']:.4f} | Val F1: {val_metrics['status_f1']:.4f}")
        print(f"Type Acc: {val_metrics['type_accuracy']:.4f}")
        
        if val_metrics['status_f1'] > best_f1:
            best_f1 = val_metrics['status_f1']
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'status_to_idx': status_to_idx,
                'type_to_idx': type_to_idx,
                'idx_to_status': idx_to_status,
                'idx_to_type': idx_to_type,
                'vjepa_model_id': args.model_id,
                'num_windows': args.num_windows,
                'use_windowing': args.use_windowing,
                'config': vars(args)
            }, output_dir / 'best_model.pt')
            
            print(f"✓ Saved best model (F1: {best_f1:.4f})")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['status_accuracy'],
            'val_f1': val_metrics['status_f1'],
            'type_acc': val_metrics['type_accuracy']
        })
    
    output_dir = Path(args.output)
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nBest Val F1: {best_f1:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    return 0


if __name__ == "__main__":
    exit(main())
