"""
Prepare Data for V-JEPA Fine-tuning
"""

import argparse
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
import os


def normalize_constraint_status(value):
    """Normalize constraint status to y/n/partial"""
    if pd.isna(value) or value == '':
        return None
    
    value_str = str(value).strip().lower()
    
    # Map yes variations
    if value_str in ['yes']:
        return 'y'
    # Map no variations  
    elif value_str in ['no']:
        return 'n'
    # Map partial variations
    elif value_str in ['partial']:
        return 'partial'
    
    return value_str


def is_valid_path(path_value):
    """Check if a value is a valid path string"""
    if pd.isna(path_value):
        return False
    if isinstance(path_value, float):
        return False
    if path_value == '':
        return False
    if str(path_value).strip() == '':
        return False
    return True


def prepare_data(csv_path, video_col, constrained_col, type_col, val_split=0.15, test_split=0.15, random_seed=42):
    """
    Prepare data for V-JEPA fine-tuning with train/val/test splits
    
    Args:
        csv_path: Path to CSV with labels
        video_col: Column name for video paths
        constrained_col: Column name for constraint status (y/n/partial)
        type_col: Column name for constraint type
        val_split: Validation split ratio (default 0.15 = 15%)
        test_split: Test split ratio (default 0.15 = 15%)
        random_seed: Random seed for reproducibility
        
    Returns:
        dict with train/val/test splits
    """
    print(f"\nCSV: {csv_path}")
    print(f"Video column: {video_col}")
    print(f"Constrained column: {constrained_col}")
    print(f"Type column: {type_col}")
    print(f"\nSplit ratios:")
    print(f"  Train: {1 - val_split - test_split:.1%}")
    print(f"  Val:   {val_split:.1%}")
    print(f"  Test:  {test_split:.1%}\n")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f" Total rows: {len(df)}")
    
    # Check if video column exists
    if video_col not in df.columns:
        print(f"\nERROR: Column '{video_col}' not found in CSV")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Normalize constraint status
    df['constrained'] = df[constrained_col].apply(normalize_constraint_status)
    
    # Filter out rows with invalid video paths
    print("\nFiltering valid video paths")
    df['valid_path'] = df[video_col].apply(is_valid_path)
    
    invalid_count = (~df['valid_path']).sum()
    if invalid_count > 0:
        print(f"Found {invalid_count} rows with invalid/missing video paths")
        print(f"Examples of invalid values:")
        invalid_samples = df[~df['valid_path']][video_col].head(5)
        for i, val in enumerate(invalid_samples, 1):
            print(f"    {i}. {repr(val)} (type: {type(val).__name__})")
    
    # Keep only rows with valid paths and labels
    df_clean = df[df['valid_path'] & df['constrained'].notna()].copy()
    print(f"Rows with valid paths and labels: {len(df_clean)}")
    
    if len(df_clean) == 0:
        print("\nERROR: No valid data after filtering")
        return None
    
    # Check video files exist
    print("\nChecking video files exist")
    missing = []
    valid_rows = []
    
    for idx, row in df_clean.iterrows():
        video_path = str(row[video_col]).strip()
        if os.path.exists(video_path):
            valid_rows.append(idx)
        else:
            missing.append(video_path)
    
    if missing:
        print(f"WARNING: {len(missing)} video files not found")
        print(f"First 5 missing:")
        for vid in missing[:5]:
            print(f"{vid}")
        df_clean = df_clean.loc[valid_rows]
        print(f"Proceeding with {len(df_clean)} videos that exist")
    else:
        print(f"All {len(df_clean)} videos found")
    
    if len(df_clean) == 0:
        print("\nERROR: No videos found")
        return None
    
    
    print("\nConstraint Status Distribution:")
    status_counts = df_clean['constrained'].value_counts()
    for status, count in status_counts.items():
        pct = count / len(df_clean) * 100
        print(f"  {status:10s}: {count:5d} ({pct:5.1f}%)")
    
    # Constraint types (only for constrained videos)
    constrained_df = df_clean[df_clean['constrained'].isin(['y', 'partial'])]
    
    if len(constrained_df) > 0 and type_col in df_clean.columns:
        print(f"\nConstraint Type Distribution ({len(constrained_df)} constrained videos):")
        type_counts = constrained_df[type_col].value_counts().head(10)
        for ctype, count in type_counts.items():
            if pd.notna(ctype) and str(ctype).strip() != '':
                print(f"  {str(ctype)[:40]:42s}: {count:5d}")
    
    # Create dataset
    data = []
    for idx, row in df_clean.iterrows():
        video_path = str(row[video_col]).strip()
        constrained = row['constrained']
        constraint_type = row.get(type_col, '') if type_col in df_clean.columns else ''
        
        # Clean constraint type
        if pd.isna(constraint_type):
            constraint_type = 'none'
        elif constrained == 'n':
            constraint_type = 'none'  # Force none for not constrained
        
        data.append({
            'video_path': video_path,
            'constrained': constrained,
            'constraint_type': str(constraint_type).strip()
        })
    
    # Three-way split: First split off test set, then split remaining into train/val
    stratify_labels = [d['constrained'] for d in data]
    
    # Step 1: Split off test set
    train_val_data, test_data = train_test_split(
        data,
        test_size=test_split,
        random_state=random_seed,
        stratify=stratify_labels
    )
    
    # Step 2: Split train_val into train and val
    adjusted_val_split = val_split / (1 - test_split)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=adjusted_val_split,
        random_state=random_seed,
        stratify=[d['constrained'] for d in train_val_data]
    )
    
    print(f"\nTrain set: {len(train_data)} videos ({len(train_data)/len(data)*100:.1f}%)")
    print(f"Val set:   {len(val_data)} videos ({len(val_data)/len(data)*100:.1f}%)")
    print(f"Test set:  {len(test_data)} videos ({len(test_data)/len(data)*100:.1f}%)")
    
    # Show split distributions
    def print_distribution(data_split, split_name):
        status_dist = pd.Series([d['constrained'] for d in data_split]).value_counts()
        print(f"\n{split_name} distribution:")
        for status, count in status_dist.items():
            pct = count / len(data_split) * 100
            print(f"  {status:10s}: {count:5d} ({pct:5.1f}%)")
    
    print_distribution(train_data, "Train")
    print_distribution(val_data, "Val")
    print_distribution(test_data, "Test")
    
    # Create output dictionary
    output = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'label_mapping': {
            'constrained': sorted(df_clean['constrained'].unique()),
            'constraint_type': sorted(df_clean[type_col].dropna().unique()) if type_col in df_clean.columns else []
        },
        'config': {
            'csv_path': csv_path,
            'video_col': video_col,
            'constrained_col': constrained_col,
            'type_col': type_col,
            'val_split': val_split,
            'test_split': test_split,
            'random_seed': random_seed,
            'total_videos': len(data),
            'train_videos': len(train_data),
            'val_videos': len(val_data),
            'test_videos': len(test_data)
        }
    }
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Prepare data for V-JEPA constraint prediction fine-tuning (with test set)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python prepare_data.py --csv data.csv
  python prepare_data.py --csv data.csv --val_split 0.15 --test_split 0.15
  python prepare_data.py --csv data.csv --video_col BidsProcessed --constrained_col Child_constrained
        """
    )
    
    parser.add_argument('--csv', type=str, required=True, help='CSV file with labels')
    parser.add_argument('--video_col', type=str, default='BidsProcessed', help='Video path column')
    parser.add_argument('--constrained_col', type=str, default='Child_constrained', help='Constraint status column')
    parser.add_argument('--type_col', type=str, default='Constraint_type', help='Constraint type column')
    parser.add_argument('--val_split', type=float, default=0.15, help='Validation split ratio (default: 0.15 = 15%)')
    parser.add_argument('--test_split', type=float, default=0.15, help='Test split ratio (default: 0.15 = 15%)')
    parser.add_argument('--output', type=str, default='prepared_data.pkl', help='Output file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Validate splits
    if args.val_split + args.test_split >= 1.0:
        print("ERROR: val_split + test_split must be < 1.0")
        return 1
    
    # Prepare data
    data = prepare_data(
        args.csv,
        args.video_col,
        args.constrained_col,
        args.type_col,
        args.val_split,
        args.test_split,
        args.seed
    )
    
    if data is None:
        print("\nData preparation failed! Please check errors above.")
        return 1
    
    # Save  
    output_path = Path(args.output)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nData saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    print(f"  Total videos: {data['config']['total_videos']}")
    print(f"  Train:        {data['config']['train_videos']} ({data['config']['train_videos']/data['config']['total_videos']*100:.1f}%)")
    print(f"  Val:          {data['config']['val_videos']} ({data['config']['val_videos']/data['config']['total_videos']*100:.1f}%)")
    print(f"  Test:         {data['config']['test_videos']} ({data['config']['test_videos']/data['config']['total_videos']*100:.1f}%)")
    
    print("\n1. Fine-tune V-JEPA2:")
    print(f"python finetune_vjepa2_fixed.py --data {args.output}")
    print("\n2. Evaluate on test set:")
    print(f"python evaluate_test_set.py --data {args.output} --split test --model ./finetuned_vjepa")
    
    return 0


if __name__ == "__main__":
    exit(main())
