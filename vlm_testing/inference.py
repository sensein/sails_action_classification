"""Main inference script with train/test split support"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

from config import list_models
from models import create_vlm
from prompts import get_prompts
from evaluator import VideoActivityEvaluator


def load_and_normalize_data(csv_path):
    """Load CSV"""
    df = pd.read_csv(csv_path)
    
    context_mapping = {
        'special occasion': 1.0,
        'general social communication interaction': 2.0,
        'general social interaction': 2.0,
        'motor play': 3.0,
        'daily routine': 4.0,
        'toy play': 5.0,
        'social routine': 6.0,
        'other': 7.0,
        'book share': 8.0
    }
    
    df['Context'] = df['Context'].map(context_mapping)
    df = df.dropna(subset=['Context'])
    
    return df


def main():
    parser = argparse.ArgumentParser(description='VLM inference with train/test split')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--test_csv', type=str, required=True, help='Test CSV path')
    parser.add_argument('--train_csv', type=str, default=None, help='Train CSV for prompts (optional)')
    parser.add_argument('--start_row', type=int, default=0)
    parser.add_argument('--end_row', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--list_models', action='store_true', help='List available models')
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for model in list_models():
            print(f"  {model}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading test data from: {args.test_csv}")
    test_df = load_and_normalize_data(args.test_csv)
    
    if args.train_csv:
        print(f"Loading train data from: {args.train_csv}")
        train_df = load_and_normalize_data(args.train_csv)
        print(f"Using train data for prompts, test data for evaluation")
    else:
        train_df = test_df
        print(f"Warning: Using test data for prompts (potential data leak)")
    
    end_row = args.end_row if args.end_row else len(test_df)
    test_subset = test_df.iloc[args.start_row:end_row].reset_index(drop=True)
    
    print(f"Test subset: {len(test_subset)} samples")
    print(f"Context distribution:\n{test_subset['Context'].value_counts().sort_index()}")
    
    print(f"\nInitializing model: {args.model}")
    vlm = create_vlm(args.model, device=args.device)
    
    activity_prompt, context_prompt = get_prompts(version="v1")
    
    results = []
    
    print(f"\nStarting inference")
    for idx, row in tqdm(test_subset.iterrows(), total=len(test_subset)):
        video_path = row['BidsProcessed']
        ground_truth_activity = row.get('Activity', '')
        ground_truth_context = row['Context']
        
        predicted_activity = vlm.predict_activity(video_path, activity_prompt)
        predicted_context = vlm.map_activity_to_context(
            video_path,
            predicted_activity if predicted_activity else '',
            context_prompt
        )
        
        results.append({
            'video_path': video_path,
            'ground_truth_activity': ground_truth_activity,
            'predicted_activity': predicted_activity,
            'ground_truth_context': int(ground_truth_context),
            'predicted_context': predicted_context
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    print("\nEvaluating predictions")
    evaluator = VideoActivityEvaluator()
    
    predicted_activities = results_df['predicted_activity'].tolist()
    ground_truth_activities = results_df['ground_truth_activity'].tolist()
    predicted_contexts = results_df['predicted_context'].tolist()
    ground_truth_contexts = results_df['ground_truth_context'].tolist()
    
    evaluation = evaluator.evaluate_all(
        predicted_activities,
        ground_truth_activities,
        predicted_contexts,
        ground_truth_contexts
    )
    
    evaluation['metadata'] = {
        'model': args.model,
        'test_csv': args.test_csv,
        'train_csv': args.train_csv,
        'test_samples': len(test_subset),
        'timestamp': datetime.now().isoformat()
    }
    
    # Convert numpy types to Python types for JSON
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    evaluation = convert_to_serializable(evaluation)
    
    with open(output_dir / 'evaluation.json', 'w') as f:
        json.dump(evaluation, f, indent=2)
    
    print("\nResults")
    print(f"Accuracy: {evaluation['context_metrics']['accuracy']:.4f}")
    print(f"Weighted F1: {evaluation['context_metrics']['f1_scores']['weighted_f1']:.4f}")
    print(f"Macro F1: {evaluation['context_metrics']['f1_scores']['macro_f1']:.4f}")
    print(f"BLEU: {evaluation['activity_metrics']['bleu']['mean_bleu']:.4f}")
    
    print(f"\nSaved to: {output_dir}")
    
    vlm.cleanup()


if __name__ == '__main__':
    main()
