#!/usr/bin/env python3
"""
Compare Results Across Models

Usage:
    python compare_results.py results_dir1 results_dir2 results_dir3 [--output comparison.csv]
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(result_dir):
    """Load evaluation results from a directory"""
    result_path = Path(result_dir) / 'evaluation_results.json'
    
    if not result_path.exists():
        raise FileNotFoundError(f"No results found in {result_dir}")
    
    with open(result_path, 'r') as f:
        return json.load(f)


def extract_key_metrics(results):
    """Extract key metrics from results"""
    return {
        'model': results['model_info']['name'],
        'accuracy': results['context_metrics']['accuracy'],
        'macro_f1': results['context_metrics']['f1_scores']['macro_f1'],
        'weighted_f1': results['context_metrics']['f1_scores']['weighted_f1'],
        'bleu': results['activity_metrics']['bleu']['mean_bleu'],
        'rouge_1': results['activity_metrics']['rouge']['rouge-1'],
        'rouge_l': results['activity_metrics']['rouge']['rouge-l'],
        'word_overlap': results['activity_metrics']['word_overlap']['mean_overlap'],
        'exact_match': results['activity_metrics']['exact_match'],
        'total_videos': results['metadata']['total_videos']
    }


def create_comparison_table(result_dirs):
    """Create comparison table"""
    all_metrics = []
    
    for result_dir in result_dirs:
        try:
            results = load_results(result_dir)
            metrics = extract_key_metrics(results)
            metrics['result_dir'] = result_dir
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Warning: Could not load results from {result_dir}: {e}")
    
    if not all_metrics:
        raise ValueError("No valid results found")
    
    df = pd.DataFrame(all_metrics)
    return df


def print_comparison(df):
    """Print formatted comparison"""
    
    # Context classification metrics
    print( "CONTEXT CLASSIFICATION ")
    context_cols = ['model', 'accuracy', 'macro_f1', 'weighted_f1']
    print(df[context_cols].to_string(index=False))
    
    # Activity prediction metrics
    print("\n ACTIVITY PREDICTION ")
    activity_cols = ['model', 'bleu', 'rouge_1', 'rouge_l', 'word_overlap', 'exact_match']
    print(df[activity_cols].to_string(index=False))
    
    # Best models
    print("\n BEST MODELS ")
    print(f"Highest Accuracy: {df.loc[df['accuracy'].idxmax(), 'model']} ({df['accuracy'].max():.4f})")
    print(f"Highest Macro F1: {df.loc[df['macro_f1'].idxmax(), 'model']} ({df['macro_f1'].max():.4f})")
    print(f"Highest BLEU: {df.loc[df['bleu'].idxmax(), 'model']} ({df['bleu'].max():.4f})")
    print(f"Highest ROUGE-L: {df.loc[df['rouge_l'].idxmax(), 'model']} ({df['rouge_l'].max():.4f})")
    print()


def plot_comparison(df, output_dir=None):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Context classification metrics
    context_metrics = df[['model', 'accuracy', 'macro_f1', 'weighted_f1']].set_index('model')
    context_metrics.plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Context Classification Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].set_ylim([0, 1])
    
    # Activity prediction metrics
    activity_metrics = df[['model', 'bleu', 'rouge_1', 'rouge_l', 'word_overlap']].set_index('model')
    activity_metrics.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Activity Prediction Metrics')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].set_ylim([0, 1])
    
    # Accuracy comparison
    df_sorted = df.sort_values('accuracy', ascending=False)
    axes[1, 0].barh(df_sorted['model'], df_sorted['accuracy'])
    axes[1, 0].set_xlabel('Accuracy')
    axes[1, 0].set_title('Context Classification Accuracy')
    axes[1, 0].set_xlim([0, 1])
    
    # F1 vs Accuracy scatter
    axes[1, 1].scatter(df['accuracy'], df['macro_f1'], s=100)
    for idx, row in df.iterrows():
        axes[1, 1].annotate(row['model'], (row['accuracy'], row['macro_f1']), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 1].set_xlabel('Accuracy')
    axes[1, 1].set_ylabel('Macro F1')
    axes[1, 1].set_title('Accuracy vs Macro F1')
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'comparison_plots.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare VLM Model Results')
    parser.add_argument('result_dirs', nargs='+', help='Directories containing evaluation results')
    parser.add_argument('--output', type=str, default='comparison.csv', help='Output CSV file')
    parser.add_argument('--plot', action='store_true', help='Create comparison plots')
    parser.add_argument('--plot_dir', type=str, default=None, help='Directory to save plots')
    
    args = parser.parse_args()
    

    df = create_comparison_table(args.result_dirs)
    print_comparison(df)
    
    df.to_csv(args.output, index=False)
    print(f"Comparison saved to: {args.output}")
    if args.plot:
        plot_comparison(df, args.plot_dir)


if __name__ == "__main__":
    main()
