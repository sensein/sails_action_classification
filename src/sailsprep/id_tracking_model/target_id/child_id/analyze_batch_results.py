#!/usr/bin/env python3
"""
Analyze Batch Child Identification Results

This script analyzes the batch processing results and creates summary reports.

Usage:
    python analyze_batch_results.py
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
LOG_DIR = Path("/orcd/data/satra/002/projects/SAILS/feature_processing/pipeline_outputs/face_body_poor_one_child_1or2_adult/child_classifications/logs")

def analyze_batch_results() -> None:
    """Analyze all batch processing results"""

    # Find all analysis JSON files
    analysis_files = list(LOG_DIR.glob("*_analysis.json"))

    if not analysis_files:
        print("No analysis files found!")
        return

    print(f"Found {len(analysis_files)} analysis files")

    # Collect data from all files
    results = []

    for file_path in analysis_files:
        try:
            with open(file_path) as f:
                data = json.load(f)

            # Extract key metrics
            result = {
                'filename': data['video_info']['filename'],
                'confidence': data['child_identification']['confidence'],
                'num_segments': data['child_identification']['num_segments'],
                'total_duration': data['child_identification']['total_duration_seconds'],
                'selected_tracks': data['child_identification']['selected_track_ids'],
                'processing_time': data['video_info']['processing_time_seconds'],
                'fps': data['video_info']['fps'],
                'total_frames': data['video_info']['total_frames'],

                # Analysis details
                'total_nodes': data['detailed_analysis']['total_nodes'],
                'total_edges': data['detailed_analysis']['total_edges'],

                # Age estimation results
                'age_probs': [node['age_prob'] for node in data['detailed_analysis']['nodes']
                             if node['age_prob'] is not None],
                'evidence_flags': [flag for node in data['detailed_analysis']['nodes']
                                  for flag in node['evidence_flags']],

                # Extract tracking data for merge analysis
                'tracking_data': data.get('tracking_results', {}),
                'detailed_nodes': data['detailed_analysis']['nodes'],
                'detailed_edges': data['detailed_analysis']['edges']
            }

            results.append(result)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not results:
        print("No valid results found!")
        return

    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    print("\n=== BATCH PROCESSING SUMMARY ===")
    print(f"Total videos processed: {len(results)}")
    print(f"Average confidence: {df['confidence'].mean():.4f}")
    print(f"Confidence range: {df['confidence'].min():.4f} - {df['confidence'].max():.4f}")
    print(f"Average processing time: {df['processing_time'].mean():.2f}s")
    print(f"Average child duration: {df['total_duration'].mean():.1f}s")

    # Confidence distribution
    print("\nConfidence distribution:")
    print(f"  > 0.9: {sum(df['confidence'] > 0.9)} videos ({sum(df['confidence'] > 0.9)/len(df)*100:.1f}%)")
    print(f"  > 0.8: {sum(df['confidence'] > 0.8)} videos ({sum(df['confidence'] > 0.8)/len(df)*100:.1f}%)")
    print(f"  > 0.7: {sum(df['confidence'] > 0.7)} videos ({sum(df['confidence'] > 0.7)/len(df)*100:.1f}%)")

    # Segment analysis
    print("\nSegment analysis:")
    print(f"  Single segment: {sum(df['num_segments'] == 1)} videos")
    print(f"  Multiple segments: {sum(df['num_segments'] > 1)} videos")
    print(f"  Average segments per video: {df['num_segments'].mean():.2f}")

    # Track analysis
    print("\nTrack analysis:")
    print(f"  Average tracks per video: {df['total_nodes'].mean():.1f}")
    print(f"  Videos with 1 track: {sum(df['total_nodes'] == 1)} ({sum(df['total_nodes'] == 1)/len(df)*100:.1f}%)")
    print(f"  Videos with >1 track: {sum(df['total_nodes'] > 1)} ({sum(df['total_nodes'] > 1)/len(df)*100:.1f}%)")

    # Age estimation analysis
    all_age_probs: list[float] = []
    for age_probs in df['age_probs']:
        all_age_probs.extend(age_probs)

    if all_age_probs:
        print("\nAge estimation analysis:")
        print(f"  Total age estimates: {len(all_age_probs)}")
        print(f"  Average child probability: {np.mean(all_age_probs):.4f}")
        print(f"  Child probability range: {min(all_age_probs):.4f} - {max(all_age_probs):.4f}")

    # Flag analysis
    all_flags: list[str] = []
    for flags in df['evidence_flags']:
        all_flags.extend(flags)

    if all_flags:
        flag_counts = Counter(all_flags)
        print("\nEvidence flags frequency:")
        for flag, count in flag_counts.most_common():
            print(f"  {flag}: {count} occurrences")

    # Save summary CSV
    output_csv = LOG_DIR / "batch_summary.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSummary saved to: {output_csv}")

    # Analyze ID merging and tracking quality
    tracking_analysis = analyze_tracking_quality(df)

    # Create visualizations
    create_visualizations(df, LOG_DIR)

    # Create tracking quality visualizations
    create_tracking_visualizations(df, tracking_analysis, LOG_DIR)

def calculate_bbox_similarity(
    bbox1: tuple[float, float, float, float],
    bbox2: tuple[float, float, float, float],
) -> dict[str, float]:
    """Calculate IoU and other similarity metrics between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection area
    x1_int = max(x1_1, x1_2)
    y1_int = max(y1_1, y1_2)
    x2_int = min(x2_1, x2_2)
    y2_int = min(y2_1, y2_2)

    if x2_int <= x1_int or y2_int <= y1_int:
        intersection = 0.0
    else:
        intersection = (x2_int - x1_int) * (y2_int - y1_int)

    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    # IoU
    iou = intersection / union if union > 0 else 0

    # Center distance
    center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
    center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
    center_dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    # Size similarity
    size1 = np.sqrt(area1)
    size2 = np.sqrt(area2)
    size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0

    return {
        'iou': iou,
        'center_distance': center_dist,
        'size_ratio': size_ratio
    }

def analyze_tracking_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze tracking quality including potential merges and splits"""
    print("\n=== TRACKING QUALITY ANALYSIS ===")

    tracking_metrics: dict[str, Any] = {
        'total_videos': len(df),
        'potential_merges': [],
        'potential_splits': [],
        'track_fragmentations': [],
        'merge_probabilities': [],
        'quality_scores': []
    }

    for _idx, row in df.iterrows():
        filename = row['filename']
        nodes = row['detailed_nodes']

        # Analyze potential merges (high similarity between different track IDs)
        merge_candidates = []
        for i, node1 in enumerate(nodes):
            for _j, node2 in enumerate(nodes[i+1:], i+1):
                if node1['track_id'] != node2['track_id']:
                    # Calculate similarity metrics
                    age_diff = abs((node1.get('age_prob', 0.5) or 0.5) - (node2.get('age_prob', 0.5) or 0.5))

                    # Check for similar evidence flags
                    flags1 = set(node1.get('evidence_flags', []))
                    flags2 = set(node2.get('evidence_flags', []))
                    flag_overlap = len(flags1.intersection(flags2)) / max(len(flags1.union(flags2)), 1)

                    # Calculate merge probability
                    merge_prob = (1 - age_diff) * 0.4 + flag_overlap * 0.6

                    if merge_prob > 0.7:  # Threshold for potential merge
                        merge_candidates.append({
                            'track1': node1['track_id'],
                            'track2': node2['track_id'],
                            'merge_probability': merge_prob,
                            'age_similarity': 1 - age_diff,
                            'flag_overlap': flag_overlap,
                            'video': filename
                        })

        # Analyze track fragmentation
        track_durations: dict[Any, list[dict[str, Any]]] = {}

        for node in nodes:
            track_id = node['track_id']
            start_frame = node.get('start_frame', 0)
            end_frame = node.get('end_frame', 0)
            duration = end_frame - start_frame

            if track_id not in track_durations:
                track_durations[track_id] = []
            track_durations[track_id].append({
                'start': start_frame,
                'end': end_frame,
                'duration': duration
            })

        # Check for fragmentation (same track ID with gaps)
        fragmentation_score = 0
        for track_id, segments in track_durations.items():
            if len(segments) > 1:
                segments = sorted(segments, key=lambda x: x['start'])
                gaps = []
                for k in range(len(segments) - 1):
                    gap = segments[k+1]['start'] - segments[k]['end']
                    gaps.append(gap)

                avg_gap = np.mean(gaps) if gaps else 0
                fragmentation_score += len(segments) - 1  # Number of fragments - 1

                if avg_gap < 30:  # Short gaps might indicate tracking issues
                    tracking_metrics['potential_splits'].append({
                        'track_id': track_id,
                        'segments': len(segments),
                        'avg_gap': avg_gap,
                        'video': filename
                    })

        # Calculate overall tracking quality score
        num_tracks = len({node['track_id'] for node in nodes})
        avg_confidence = row['confidence']
        merge_penalty = len(merge_candidates) * 0.1
        fragmentation_penalty = fragmentation_score * 0.05

        quality_score = avg_confidence - merge_penalty - fragmentation_penalty
        quality_score = max(0, min(1, quality_score))  # Clamp between 0 and 1

        tracking_metrics['potential_merges'].extend(merge_candidates)
        tracking_metrics['track_fragmentations'].append({
            'video': filename,
            'num_tracks': num_tracks,
            'fragmentation_score': fragmentation_score,
            'quality_score': quality_score
        })
        tracking_metrics['quality_scores'].append(quality_score)

    # Summary statistics
    print(f"Videos analyzed: {tracking_metrics['total_videos']}")
    print(f"Potential merges detected: {len(tracking_metrics['potential_merges'])}")
    print(f"Potential splits detected: {len(tracking_metrics['potential_splits'])}")

    if tracking_metrics['quality_scores']:
        print(f"Average tracking quality: {np.mean(tracking_metrics['quality_scores']):.3f}")
        print(f"Quality score range: {min(tracking_metrics['quality_scores']):.3f} - {max(tracking_metrics['quality_scores']):.3f}")

    # Merge probability analysis
    if tracking_metrics['potential_merges']:
        merge_probs = [m['merge_probability'] for m in tracking_metrics['potential_merges']]
        print(f"Merge probabilities - Mean: {np.mean(merge_probs):.3f}, Max: {max(merge_probs):.3f}")

        high_prob_merges = [m for m in tracking_metrics['potential_merges'] if m['merge_probability'] > 0.8]
        print(f"High probability merges (>0.8): {len(high_prob_merges)}")

    return tracking_metrics

def create_tracking_visualizations(df: pd.DataFrame, tracking_analysis: dict[str, Any], output_dir: Path) -> None:
    """Create detailed tracking quality visualizations"""

    # Set up the plotting style
    plt.style.use('default')

    # Create a comprehensive tracking analysis figure
    plt.figure(figsize=(20, 16))

    # 1. Tracking quality distribution
    plt.subplot(3, 3, 1)
    quality_scores = tracking_analysis['quality_scores']
    if quality_scores:
        plt.hist(quality_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(quality_scores), color='red', linestyle='--',
                   label=f'Mean: {np.mean(quality_scores):.3f}')
        plt.xlabel('Tracking Quality Score')
        plt.ylabel('Number of Videos')
        plt.title('Tracking Quality Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 2. Merge probability distribution
    ax2 = plt.subplot(3, 3, 2)
    merge_probs = [m['merge_probability'] for m in tracking_analysis['potential_merges']]
    if merge_probs:
        plt.hist(merge_probs, bins=15, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Merge Probability')
        plt.ylabel('Number of Potential Merges')
        plt.title('Merge Probability Distribution')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No potential merges detected', ha='center', va='center', transform=ax2.transAxes)
        plt.title('Merge Probability Distribution')

    # 3. Track fragmentation analysis
    plt.subplot(3, 3, 3)
    frag_scores = [f['fragmentation_score'] for f in tracking_analysis['track_fragmentations']]
    if frag_scores:
        frag_counts = Counter(frag_scores)
        plt.bar(list(frag_counts.keys()), list(frag_counts.values()), alpha=0.7, color='lightcoral')
        plt.xlabel('Fragmentation Score')
        plt.ylabel('Number of Videos')
        plt.title('Track Fragmentation Distribution')
        plt.grid(True, alpha=0.3)

    # 4. Number of tracks vs quality score
    ax4 = plt.subplot(3, 3, 4)
    num_tracks = [f['num_tracks'] for f in tracking_analysis['track_fragmentations']]
    quality_scores_aligned = [f['quality_score'] for f in tracking_analysis['track_fragmentations']]
    if num_tracks and quality_scores_aligned:
        plt.scatter(num_tracks, quality_scores_aligned, alpha=0.6, color='green')
        plt.xlabel('Number of Tracks')
        plt.ylabel('Quality Score')
        plt.title('Tracks vs Quality Score')
        plt.grid(True, alpha=0.3)

        # Add correlation coefficient
        if len(num_tracks) > 1:
            corr = np.corrcoef(num_tracks, quality_scores_aligned)[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax4.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 5. Age similarity in potential merges
    ax5 = plt.subplot(3, 3, 5)
    age_sims = [m['age_similarity'] for m in tracking_analysis['potential_merges']]
    if age_sims:
        plt.hist(age_sims, bins=15, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('Age Similarity')
        plt.ylabel('Number of Potential Merges')
        plt.title('Age Similarity in Potential Merges')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No potential merges detected', ha='center', va='center', transform=ax5.transAxes)
        plt.title('Age Similarity in Potential Merges')

    # 6. Flag overlap in potential merges
    ax6 = plt.subplot(3, 3, 6)
    flag_overlaps = [m['flag_overlap'] for m in tracking_analysis['potential_merges']]
    if flag_overlaps:
        plt.hist(flag_overlaps, bins=15, alpha=0.7, color='gold', edgecolor='black')
        plt.xlabel('Evidence Flag Overlap')
        plt.ylabel('Number of Potential Merges')
        plt.title('Flag Overlap in Potential Merges')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No potential merges detected', ha='center', va='center', transform=ax6.transAxes)
        plt.title('Flag Overlap in Potential Merges')

    # 7. Split analysis - gap distribution
    ax7 = plt.subplot(3, 3, 7)
    split_gaps = [s['avg_gap'] for s in tracking_analysis['potential_splits']]
    if split_gaps:
        plt.hist(split_gaps, bins=15, alpha=0.7, color='cyan', edgecolor='black')
        plt.xlabel('Average Frame Gap')
        plt.ylabel('Number of Potential Splits')
        plt.title('Frame Gap Distribution in Splits')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No potential splits detected', ha='center', va='center', transform=ax7.transAxes)
        plt.title('Frame Gap Distribution in Splits')

    # 8. Quality vs Confidence correlation
    ax8 = plt.subplot(3, 3, 8)
    confidences = df['confidence'].tolist()
    if quality_scores and len(confidences) == len(quality_scores):
        plt.scatter(confidences, quality_scores, alpha=0.6, color='red')
        plt.xlabel('Child Identification Confidence')
        plt.ylabel('Tracking Quality Score')
        plt.title('Confidence vs Tracking Quality')
        plt.grid(True, alpha=0.3)

        # Add correlation and trend line
        if len(confidences) > 1:
            corr = np.corrcoef(confidences, quality_scores)[0, 1]
            z = np.polyfit(confidences, quality_scores, 1)
            p = np.poly1d(z)
            plt.plot(confidences, p(confidences), "r--", alpha=0.8)
            plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax8.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 9. Summary statistics text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    # Create summary text
    summary_text = f"""TRACKING ANALYSIS SUMMARY

Total Videos: {tracking_analysis['total_videos']}
Potential Merges: {len(tracking_analysis['potential_merges'])}
Potential Splits: {len(tracking_analysis['potential_splits'])}

Quality Metrics:
• Mean Quality: {np.mean(quality_scores):.3f}
• Std Quality: {np.std(quality_scores):.3f}
• Videos >0.8 Quality: {sum(1 for q in quality_scores if q > 0.8)}

Merge Analysis:
• High Prob Merges (>0.8): {len([m for m in tracking_analysis['potential_merges'] if m['merge_probability'] > 0.8])}
• Mean Merge Prob: {np.mean(merge_probs):.3f}

Fragmentation:
• Videos with Fragments: {sum(1 for f in tracking_analysis['track_fragmentations'] if f['fragmentation_score'] > 0)}
• Mean Frag Score: {np.mean(frag_scores):.3f}
"""

    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    # Save the comprehensive plot
    plot_path = output_dir / "tracking_analysis_comprehensive.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive tracking analysis saved to: {plot_path}")
    plt.show()

    # Save detailed tracking analysis data
    tracking_csv = output_dir / "tracking_analysis_detailed.csv"

    # Create detailed DataFrame for export
    detailed_data = []
    for _video_idx, row in df.iterrows():
        filename = row['filename']

        # Add merge data
        video_merges = [m for m in tracking_analysis['potential_merges'] if m['video'] == filename]
        for merge in video_merges:
            detailed_data.append({
                'video': filename,
                'analysis_type': 'potential_merge',
                'track1': merge['track1'],
                'track2': merge['track2'],
                'merge_probability': merge['merge_probability'],
                'age_similarity': merge['age_similarity'],
                'flag_overlap': merge['flag_overlap']
            })

        # Add split data
        video_splits = [s for s in tracking_analysis['potential_splits'] if s['video'] == filename]
        for split in video_splits:
            detailed_data.append({
                'video': filename,
                'analysis_type': 'potential_split',
                'track_id': split['track_id'],
                'segments': split['segments'],
                'avg_gap': split['avg_gap']
            })

        # Add quality data
        frag_data = next(f for f in tracking_analysis['track_fragmentations'] if f['video'] == filename)
        detailed_data.append({
            'video': filename,
            'analysis_type': 'quality_metrics',
            'num_tracks': frag_data['num_tracks'],
            'fragmentation_score': frag_data['fragmentation_score'],
            'quality_score': frag_data['quality_score']
        })

    if detailed_data:
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(tracking_csv, index=False)
        print(f"Detailed tracking analysis saved to: {tracking_csv}")

def create_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Create visualization plots"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Confidence distribution
    axes[0, 0].hist(df['confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Child Identification Confidence Distribution')
    axes[0, 0].set_xlabel('Confidence Score')
    axes[0, 0].set_ylabel('Number of Videos')
    axes[0, 0].grid(True, alpha=0.3)

    # Processing time vs video length
    video_length = df['total_frames'] / df['fps']
    axes[0, 1].scatter(video_length, df['processing_time'], alpha=0.6, color='orange')
    axes[0, 1].set_title('Processing Time vs Video Length')
    axes[0, 1].set_xlabel('Video Length (seconds)')
    axes[0, 1].set_ylabel('Processing Time (seconds)')
    axes[0, 1].grid(True, alpha=0.3)

    # Number of tracks per video
    track_counts = df['total_nodes'].value_counts().sort_index()
    axes[1, 0].bar(track_counts.index, track_counts.values, alpha=0.7, color='lightgreen')
    axes[1, 0].set_title('Number of Tracks per Video')
    axes[1, 0].set_xlabel('Number of Tracks')
    axes[1, 0].set_ylabel('Number of Videos')
    axes[1, 0].grid(True, alpha=0.3)

    # Child duration distribution
    axes[1, 1].hist(df['total_duration'], bins=15, alpha=0.7, color='pink', edgecolor='black')
    axes[1, 1].set_title('Child Duration Distribution')
    axes[1, 1].set_xlabel('Child Duration (seconds)')
    axes[1, 1].set_ylabel('Number of Videos')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "batch_analysis_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_path}")

    plt.show()

if __name__ == "__main__":
    analyze_batch_results()