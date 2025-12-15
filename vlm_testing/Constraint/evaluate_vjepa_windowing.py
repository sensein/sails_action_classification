"""
Evaluation script.
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoModel, AutoVideoProcessor
import pandas as pd
from pathlib import Path
import av
import numpy as np
from tqdm import tqdm
import json
import pickle
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import re


def normalize_type(type_str):
    """
    Normalize type string for fair comparison
    Removes: spaces, hyphens, underscores, extra punctuation
    Converts to lowercase
    
    Examples:
        "high chair" → "highchair"
        "car-seat" → "carseat"
        "Baby Carrier  " → "babycarrier"
    """
    if pd.isna(type_str) or type_str == '':
        return 'none'
    
    s = str(type_str).lower().strip()
    
    # Remove common separators
    s = s.replace(' ', '').replace('-', '').replace('_', '')
    
    # Remove extra punctuation
    s = re.sub(r'[^\w]', '', s)
    
    return s


class VJEPAConstraintModel(nn.Module):
    """V-JEPA with constraint prediction heads"""
    
    def __init__(self, vjepa_model, num_status_classes=3, num_type_classes=10):
        super().__init__()
        self.vjepa = vjepa_model
        self.embed_dim = vjepa_model.config.hidden_size
        
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
    
    def forward(self, pixel_values):
        outputs = self.vjepa.get_vision_features(pixel_values)
        
        # Pool features (mean pooling over patches)
        features = outputs.mean(dim=1)  # B x D
        
        status_logits = self.status_head(features)
        type_logits = self.type_head(features)
        
        return status_logits, type_logits


def extract_frames(video_path, num_frames=64):
    """Extract evenly spaced frames from video"""
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        
        total_frames = stream.frames
        if total_frames == 0:
            total_frames = int(stream.duration * stream.time_base * stream.average_rate)
        
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        container.seek(0)
        frame_count = 0
        
        for frame in container.decode(video=0):
            if frame_count in indices:
                frames.append(frame.to_ndarray(format='rgb24'))
            frame_count += 1
            if len(frames) == num_frames:
                break
        
        container.close()
        
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        return frames[:num_frames]
        
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        return [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames


def predict_video(video_path, model, processor, device, idx_to_status, idx_to_type, num_frames=64):
    """Predict constraint for a single video"""
    
    frames = extract_frames(video_path, num_frames)
    
    # V-JEPA2 expects video tensor: T x C x H x W
    video = np.stack(frames)  # T x H x W x C
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W
    
    inputs = processor(video_tensor, return_tensors='pt')
    pixel_values = inputs['pixel_values_videos'].to(device)
    
    with torch.no_grad():
        status_logits, type_logits = model(pixel_values)
    
    status_idx = status_logits.argmax(dim=1).item()
    pred_status = idx_to_status[status_idx]
    
    if pred_status in ['y', 'partial']:
        type_idx = type_logits.argmax(dim=1).item()
        pred_type = idx_to_type[type_idx]
    else:
        pred_type = "none"
    
    return pred_status, pred_type


def calculate_normalized_accuracy(y_true, y_pred):
    """
    Calculate accuracy with normalized string comparison
    
    Returns:
        dict with exact and normalized accuracy
    """
    exact_matches = 0
    normalized_matches = 0
    total = len(y_true)
    
    normalization_examples = []
    
    for gt, pred in zip(y_true, y_pred):
        # Exact comparison
        if gt == pred:
            exact_matches += 1
            normalized_matches += 1
        # Normalized comparison
        elif normalize_type(gt) == normalize_type(pred):
            normalized_matches += 1
            normalization_examples.append({
                'ground_truth': gt,
                'predicted': pred,
                'normalized': normalize_type(gt)
            })
    
    return {
        'exact_accuracy': exact_matches / total if total > 0 else 0,
        'normalized_accuracy': normalized_matches / total if total > 0 else 0,
        'exact_matches': exact_matches,
        'normalized_matches': normalized_matches,
        'total': total,
        'normalization_examples': normalization_examples
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate V-JEPA with normalized type comparison'
    )
    parser.add_argument('--data', type=str, required=True, help='Prepared data pickle file')
    parser.add_argument('--model', type=str, required=True, help='Fine-tuned model directory')
    parser.add_argument('--model_cache', type=str, default='./models', help='V-JEPA cache')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], 
                        help='Which split to evaluate')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--num_frames', type=int, default=64, help='Frames per video')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    
    args = parser.parse_args()
    
    # Setup output
    if args.output is None:
        args.output = f"evaluation_{args.split}"
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nLoading data from {args.data}")
    with open(args.data, 'rb') as f:
        data_dict = pickle.load(f)
    
    test_data = data_dict[args.split]
    print(f"{args.split.capitalize()} set: {len(test_data)} videos")
    
    print(f"\nLoading fine-tuned model from {args.model}")
    checkpoint = torch.load(Path(args.model) / 'best_model.pt', map_location='cpu')
    
    status_to_idx = checkpoint['status_to_idx']
    type_to_idx = checkpoint['type_to_idx']
    idx_to_status = checkpoint['idx_to_status']
    idx_to_type = checkpoint['idx_to_type']
    
    print(f"Status classes: {list(idx_to_status.values())}")
    print(f"Type classes: {len(idx_to_type)} unique types")
    
    vjepa_model_id = checkpoint.get('vjepa_model_id', 'facebook/vjepa2-vitl-fpc64-256')
    print(f"\nLoading V-JEPA2")
    print(f"Model ID: {vjepa_model_id}")
    
    processor = AutoVideoProcessor.from_pretrained(vjepa_model_id, cache_dir=args.model_cache)
    vjepa = AutoModel.from_pretrained(vjepa_model_id, cache_dir=args.model_cache)
    
    # Create model
    model = VJEPAConstraintModel(vjepa, len(status_to_idx), len(type_to_idx))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on: {device}")
    
    # Predict on test set
    
    results = []
    start_time = datetime.now()
    
    for item in tqdm(test_data, desc=f"Processing {args.split} set"):
        video_path = item['video_path']
        gt_constrained = item['constrained']
        gt_type = item['constraint_type']
        
        pred_constrained, pred_type = predict_video(
            video_path,
            model,
            processor,
            device,
            idx_to_status,
            idx_to_type,
            args.num_frames
        )
        
        results.append({
            'video_path': video_path,
            'gt_constrained': gt_constrained,
            'pred_constrained': pred_constrained,
            'gt_constraint_type': gt_type,
            'pred_constraint_type': pred_type
        })
    
    
    results_df = pd.DataFrame(results)
    results_path = output_dir / f'{args.split}_predictions.csv'
    results_df.to_csv(results_path, index=False, quoting=1)
    
    print(f"\nPredictions saved to: {results_path}")
    
    y_true_status = results_df['gt_constrained'].tolist()
    y_pred_status = results_df['pred_constrained'].tolist()
    
    status_acc = accuracy_score(y_true_status, y_pred_status)
    status_f1 = f1_score(y_true_status, y_pred_status, average='weighted', zero_division=0)
    
    print(f"\nConstraint Status:")
    print(f"Accuracy:    {status_acc:.4f}")
    print(f"Weighted F1: {status_f1:.4f}")
    
    print("\nClassification Report (Status):")
    print(classification_report(y_true_status, y_pred_status, zero_division=0))
    
    print("\nConfusion Matrix (Status):")
    cm = confusion_matrix(y_true_status, y_pred_status)
    print(cm)

    
    constrained_df = results_df[results_df['gt_constrained'].isin(['y', 'partial'])].copy()
    
    if len(constrained_df) > 0:
        y_true_type = constrained_df['gt_constraint_type'].tolist()
        y_pred_type = constrained_df['pred_constraint_type'].tolist()
        
        # Calculate both exact and normalized accuracy
        exact_acc = accuracy_score(y_true_type, y_pred_type)
        norm_results = calculate_normalized_accuracy(y_true_type, y_pred_type)
        
        print(f"\nConstraint Type ({len(constrained_df)} constrained videos):")
        print(f"Exact Accuracy:       {exact_acc:.4f} ({norm_results['exact_matches']}/{norm_results['total']})")
        print(f"Normalized Accuracy:  {norm_results['normalized_accuracy']:.4f} ({norm_results['normalized_matches']}/{norm_results['total']})")
        print(f"Improvement:          +{(norm_results['normalized_accuracy'] - exact_acc):.4f}")
        print(f"Formatting differences caught: {norm_results['normalized_matches'] - norm_results['exact_matches']}")
        
        # Show normalization examples
        if norm_results['normalization_examples']:
            print(f"\nExamples of formatting differences handled:")
            for i, ex in enumerate(norm_results['normalization_examples'][:10], 1):
                print(f"{i}. GT: '{ex['ground_truth']}' | Pred: '{ex['predicted']}' → Normalized: '{ex['normalized']}'")
        
        print("\n\nTop 10 Predicted Types:")
        type_counts = constrained_df['pred_constraint_type'].value_counts().head(10)
        for ctype, count in type_counts.items():
            pct = count / len(constrained_df) * 100
            print(f"{str(ctype)[:40]:42s}: {count:5d} ({pct:5.1f}%)")
    else:
        exact_acc = 0
        norm_results = {'normalized_accuracy': 0, 'exact_matches': 0, 'normalized_matches': 0, 'total': 0}
    
    # Save metrics
    metrics = {
        'split': args.split,
        'model': args.model,
        'total_videos': len(results_df),
        'status': {
            'accuracy': status_acc,
            'weighted_f1': status_f1,
            'confusion_matrix': cm.tolist()
        },
        'type': {
            'constrained_videos': len(constrained_df) if len(constrained_df) > 0 else 0,
            'exact_accuracy': exact_acc,
            'normalized_accuracy': norm_results['normalized_accuracy'],
            'exact_matches': norm_results['exact_matches'],
            'normalized_matches': norm_results['normalized_matches'],
            'improvement': norm_results['normalized_accuracy'] - exact_acc
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / f'{args.split}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {output_dir / f'{args.split}_metrics.json'}")

    status_counts = results_df['pred_constrained'].value_counts()
    print(f"\nPredicted Status Distribution:")
    for status, count in status_counts.items():
        pct = count / len(results_df) * 100
        print(f"{status:10s}: {count:5d} ({pct:5.1f}%)")
    
    elapsed = datetime.now() - start_time
    print(f"\nTotal time: {elapsed.total_seconds():.1f}s")
    print(f"Average: {len(results_df) / elapsed.total_seconds():.2f} videos/sec")

    print(f"\nStatus Accuracy: {status_acc:.2%}")
    if len(constrained_df) > 0:
        print(f"Type Exact Accuracy: {exact_acc:.2%}")
        print(f"Type Normalized Accuracy: {norm_results['normalized_accuracy']:.2%}")
    

if __name__ == "__main__":
    main()
