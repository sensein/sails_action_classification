#!/usr/bin/env python3
"""
Unified VLM Context Prediction

Usage:
    python context.py --model llava-next-qwen --csv_path data.csv
    python context.py --model videollama2-7b --csv_path data.csv --start_row 0 --end_row 100
    python context.py --list-models  # Show available models
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys

from config import list_available_models, print_model_info, get_model_config
from models import create_vlm
from prompts import get_prompts, list_prompt_versions
from evaluator import VideoActivityEvaluator


def normalize_ground_truth_context(ctx):
    """
    Convert text context labels to numeric codes (8-category system)
    """
    if pd.isna(ctx):
        return None
    
    # Mapping for 8-category system
    text_to_numeric = {
        'special occasion': 1,
        'general social communication interaction': 2,
        'general social interaction': 2, 
        'motor play': 3,
        'daily routine': 4,
        'toy play': 5,
        'social routine': 6,
        'other': 7,
        'book share': 8,
    }
    
    # Try numeric first (if already numeric in CSV)
    try:
        num = int(ctx)
        # Validate it's in range 1-8
        if 1 <= num <= 8:
            return num
        else:
            print(f"Warning: Invalid numeric context {num}, setting to None")
            return None
    except (ValueError, TypeError):
        # Text label - convert to numeric
        ctx_lower = str(ctx).lower().strip()
        result = text_to_numeric.get(ctx_lower, None)
        if result is None:
            print(f"Warning: Unknown context label '{ctx}', setting to None")
        return result


def main():
    parser = argparse.ArgumentParser(
        description='Unified VLM Context Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run LLaVA-Next-Qwen on full dataset
  python context.py --model llava-next-qwen --csv_path data.csv
  
  # Run VideoLLaMA2 on subset (rows 0-100)
  python context.py --model videollama2-7b --csv_path data.csv --start_row 0 --end_row 100
  
  # List available models
  python context.py --list-models
  
  # Show model details
  python context.py --show-models
        """
    )
    
    # Model selection
    parser.add_argument('--model', type=str, help='Model name (see --list-models)')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    parser.add_argument('--show-models', action='store_true', help='Show detailed model info')
    
    # Data arguments
    parser.add_argument('--csv_path', type=str, help='Path to CSV file')
    parser.add_argument('--video_col', type=str, default='BidsProcessed', help='Column name for video paths')
    parser.add_argument('--activity_col', type=str, default='Activity', help='Column name for activities')
    parser.add_argument('--context_col', type=str, default='Context', help='Column name for contexts')
    
    # Processing arguments
    parser.add_argument('--start_row', type=int, default=0, help='Start row index (inclusive)')
    parser.add_argument('--end_row', type=int, default=None, help='End row index (exclusive, None = end of file)')
    parser.add_argument('--resume_from', type=int, default=0, help='Resume from index (within sliced data)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (auto-generated if not specified)')
    parser.add_argument('--checkpoint_every', type=int, default=100, help='Save checkpoint every N videos')
    
    # Prompt arguments
    parser.add_argument('--prompt_version', type=str, default='v1', help='Prompt version to use')
    parser.add_argument('--list-prompts', action='store_true', help='List available prompt versions')
    
    # Model-specific arguments
    parser.add_argument('--videollama_path', type=str, default='/home/aparnabg/orcd/scratch/VideoLLaMA2',
                       help='Path to VideoLLaMA2 installation (for VideoLLaMA models)')
    
    args = parser.parse_args()
    
    # Handle info commands
    if args.list_models:
        print("\nAvailable Models:")
        print("="*80)
        for model in list_available_models():
            config = get_model_config(model)
            print(f"  • {model:<25} ({config.model_family}, {config.max_frames} frames)")
        print("\nUse --show-models for detailed information")
        return
    
    if args.show_models:
        print_model_info()
        return
    
    if args.list_prompts:
        list_prompt_versions()
        return
    
    # Validate required arguments
    if not args.model:
        parser.error("--model is required (use --list-models to see options)")
    
    if not args.csv_path:
        parser.error("--csv_path is required")
    
    # Auto-generate output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./results_{args.model}_{timestamp}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    # Load data
    df = pd.read_csv(args.csv_path)
    print(f"Dataframe shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Store original labels
    df['ground_truth_context_original'] = df[args.context_col].copy()
    
    # Normalize ground truth contexts
    print("\nNormalizing ground truth contexts (8 categories)")
    print("Mapping 'general social interaction' typo to 'general social communication interaction'")
    df[args.context_col] = df[args.context_col].apply(normalize_ground_truth_context)
    print(f"Normalized context values: {sorted(df[args.context_col].dropna().unique())}")
    print(f"Context value counts:\n{df[args.context_col].value_counts().sort_index()}")
    
    # Slice dataframe
    end_idx = args.end_row if args.end_row is not None else len(df)
    df = df.iloc[args.start_row:end_idx]
    print(f"\nProcessing rows {args.start_row} to {end_idx} ({len(df)} total)")
    
    # Load prompts
    print(f"\nLoading prompts (version: {args.prompt_version})")
    activity_prompt, context_prompt = get_prompts(args.prompt_version)
    
    # Initialize model
    print(f"\nInitializing model: {args.model}")
    vlm = create_vlm(args.model, device='cuda:0', videollama_path=args.videollama_path)
    
    # Print model info
    model_info = vlm.get_model_info()
    print(f"\nModel Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Initialize evaluator
    evaluator = VideoActivityEvaluator()
    
    # Storage for predictions
    predicted_activities = []
    predicted_contexts = []
    predicted_context_labels = []
    ground_truth_activities = []
    ground_truth_contexts = []
    ground_truth_context_originals = []
    
    
    start_time = datetime.now()
    
    for idx in tqdm(range(args.resume_from, len(df)), desc="Processing videos"):
        row = df.iloc[idx]
        video_path = row[args.video_col]
        gt_activity = row[args.activity_col]
        gt_context = row[args.context_col]
        gt_context_original = row['ground_truth_context_original']
        
        # Predict
        pred_activity, pred_context = vlm.process_video(video_path, activity_prompt, context_prompt)
        
        # Get text label
        pred_context_label = vlm.numeric_to_context.get(pred_context, 'unknown') if pred_context else None
        
        # Store results
        predicted_activities.append(pred_activity)
        predicted_contexts.append(pred_context)
        predicted_context_labels.append(pred_context_label)
        ground_truth_activities.append(gt_activity)
        ground_truth_contexts.append(gt_context)
        ground_truth_context_originals.append(gt_context_original)
        
        # Print progress every 10 videos
        if (idx + 1) % 10 == 0:
            elapsed = datetime.now() - start_time
            videos_per_sec = (idx + 1 - args.resume_from) / elapsed.total_seconds()
            remaining = (len(df) - idx - 1) / videos_per_sec if videos_per_sec > 0 else 0
            
            print(f"\nProgress Update ({idx + 1}/{len(df)})")
            print(f"Speed: {videos_per_sec:.2f} videos/sec")
            print(f"Estimated time remaining: {remaining/3600:.2f} hours")
            print(f"GT Activity: {gt_activity}")
            print(f"Pred Activity: {pred_activity}")
            
            if not pd.isna(gt_context) and gt_context is not None:
                gt_label = vlm.numeric_to_context.get(int(gt_context), 'unknown')
                print(f"GT Context: {gt_context} ({gt_label}) [Original: {gt_context_original}]")
            else:
                print(f"GT Context: None [Original: {gt_context_original}]")
            
            if pred_context is not None:
                print(f"Pred Context: {pred_context} ({pred_context_label})")
            else:
                print(f"Pred Context: None")
                
            print(vlm.get_gpu_stats())
        
        # Save checkpoint
        if (idx + 1) % args.checkpoint_every == 0:
            checkpoint_df = pd.DataFrame({
                args.video_col: df[args.video_col].iloc[:idx+1].values,
                'ground_truth_activity': ground_truth_activities,
                'predicted_activity': predicted_activities,
                'ground_truth_context_original': ground_truth_context_originals,
                'ground_truth_context_numeric': ground_truth_contexts,
                'predicted_context_numeric': predicted_contexts,
                'predicted_context_label': predicted_context_labels
            })
            checkpoint_path = output_dir / f'checkpoint_{idx+1}.csv'
            checkpoint_df.to_csv(checkpoint_path, index=False)
            print(f"\nCheckpoint saved: {checkpoint_path}")
    
    # Save final predictions
    results_df = pd.DataFrame({
        args.video_col: df[args.video_col].values,
        'ground_truth_activity': ground_truth_activities,
        'predicted_activity': predicted_activities,
        'ground_truth_context_original': ground_truth_context_originals,
        'ground_truth_context_numeric': ground_truth_contexts,
        'predicted_context_numeric': predicted_contexts,
        'predicted_context_label': predicted_context_labels
    })
    
    results_path = output_dir / 'predictions_final.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n Final predictions saved: {results_path}")
    
    
    results = evaluator.evaluate_all(
        predicted_activities, ground_truth_activities,
        predicted_contexts, ground_truth_contexts
    )
    
    # Print metrics
    print( "ACTIVITY PREDICTION METRICS ")
    print(f"Valid predictions: {results['activity_metrics']['bleu']['count']}")
    print(f"BLEU Score: {results['activity_metrics']['bleu']['mean_bleu']:.4f}")
    print(f"ROUGE-1: {results['activity_metrics']['rouge']['rouge-1']:.4f}")
    print(f"Word Overlap: {results['activity_metrics']['word_overlap']['mean_overlap']:.4f}")
    
    print( "\n CONTEXT CLASSIFICATION METRICS ")
    print(f"Accuracy: {results['context_metrics']['accuracy']:.4f}")
    print(f"Macro F1: {results['context_metrics']['f1_scores']['macro_f1']:.4f}")
    
    print( "\n PER-CLASS PERFORMANCE ")
    print(results['context_metrics']['classification_report'])
    
    if results['context_metrics']['confusion_matrix'] is not None:
        print( "\n CONFUSION MATRIX ")
        print(results['context_metrics']['confusion_matrix'])
    
    # Save results
    results_to_save = {
        'model_info': model_info,
        'activity_metrics': {
            'bleu': results['activity_metrics']['bleu'],
            'rouge': results['activity_metrics']['rouge'],
            'exact_match': float(results['activity_metrics']['exact_match']),
            'word_overlap': results['activity_metrics']['word_overlap']
        },
        'context_metrics': {
            'accuracy': float(results['context_metrics']['accuracy']),
            'f1_scores': {k: float(v) for k, v in results['context_metrics']['f1_scores'].items()},
            'classification_report': results['context_metrics']['classification_report'],
            'confusion_matrix': results['context_metrics']['confusion_matrix'].tolist() if results['context_metrics']['confusion_matrix'] is not None else None
        },
        'metadata': {
            'total_videos': len(df),
            'model_name': args.model,
            'prompt_version': args.prompt_version,
            'timestamp': datetime.now().isoformat(),
            'rows_processed': f"{args.start_row} to {end_idx}",
            'csv_path': args.csv_path
        }
    }
    
    results_json_path = output_dir / 'evaluation_results.json'
    with open(results_json_path, 'w') as f:
        json.dump(results_to_save, f, indent=4)
    
    print(f"\n Evaluation results saved: {results_json_path}")
    
    total_time = datetime.now() - start_time
    print(f"Total processing time: {total_time}")
    print(f"Average time per video: {total_time.total_seconds() / len(df):.2f} seconds")
    
    vlm.cleanup()


if __name__ == "__main__":
    main()
