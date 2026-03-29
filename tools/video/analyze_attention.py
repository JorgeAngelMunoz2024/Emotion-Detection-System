#!/usr/bin/env python3
"""
Example: Extract and Visualize Attention Focus from Video
Demonstrates how to use the video processor to analyze facial attention patterns.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse


def analyze_attention_patterns(analysis_file: str):
    """
    Analyze attention patterns from video processing output.
    
    Args:
        analysis_file: Path to the JSON analysis file
    """
    print(f"\n{'='*70}")
    print(f"Analyzing Attention Patterns: {analysis_file}")
    print(f"{'='*70}\n")
    
    # Load analysis data
    with open(analysis_file, 'r') as f:
        data = json.load(f)
    
    summary = data['summary']
    frames = data['frame_data']
    
    # Print summary
    print("Video Summary:")
    print(f"  Total frames: {summary['total_frames']}")
    print(f"  Frames with faces: {summary['frames_with_faces']}")
    print(f"  Dominant emotion: {summary['dominant_emotion']}")
    print(f"  Processing time: {summary['processing_time']:.1f}s\n")
    
    # Extract attention data by emotion
    attention_by_emotion = defaultdict(lambda: defaultdict(list))
    emotion_counts = defaultdict(int)
    
    for frame in frames:
        if 'attention_focus' in frame and frame['face_detected']:
            emotion = frame['emotion']
            emotion_counts[emotion] += 1
            
            for region, score in frame['attention_focus'].items():
                attention_by_emotion[emotion][region].append(score)
    
    # Calculate average attention per region for each emotion
    print("Average Attention by Region for Each Emotion:")
    print("-" * 70)
    
    regions = ['left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow', 'mouth_outer', 'nose']
    
    for emotion in sorted(emotion_counts.keys()):
        if emotion_counts[emotion] < 5:  # Skip emotions with too few frames
            continue
        
        print(f"\n{emotion.upper()} ({emotion_counts[emotion]} frames):")
        
        region_avgs = {}
        for region in regions:
            if region in attention_by_emotion[emotion]:
                avg_attention = np.mean(attention_by_emotion[emotion][region])
                region_avgs[region] = avg_attention
                print(f"  {region:20s}: {avg_attention:.3f}")
        
        # Find most attended region
        if region_avgs:
            max_region = max(region_avgs.items(), key=lambda x: x[1])
            print(f"  → Focus: {max_region[0]} ({max_region[1]:.3f})")
    
    # Visualize attention patterns
    visualize_attention(attention_by_emotion, emotion_counts, regions)


def visualize_attention(attention_by_emotion, emotion_counts, regions):
    """Create visualization of attention patterns."""
    
    # Filter emotions with sufficient data
    emotions_to_plot = [e for e in emotion_counts.keys() if emotion_counts[e] >= 5]
    
    if not emotions_to_plot:
        print("\nNot enough data for visualization.")
        return
    
    # Create heatmap data
    heatmap_data = np.zeros((len(emotions_to_plot), len(regions)))
    
    for i, emotion in enumerate(emotions_to_plot):
        for j, region in enumerate(regions):
            if region in attention_by_emotion[emotion]:
                heatmap_data[i, j] = np.mean(attention_by_emotion[emotion][region])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap
    im = ax1.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(np.arange(len(regions)))
    ax1.set_yticks(np.arange(len(emotions_to_plot)))
    ax1.set_xticklabels(regions, rotation=45, ha='right')
    ax1.set_yticklabels(emotions_to_plot)
    ax1.set_title('Attention Heatmap by Emotion and Region')
    ax1.set_xlabel('Facial Region')
    ax1.set_ylabel('Emotion')
    
    # Add values to heatmap
    for i in range(len(emotions_to_plot)):
        for j in range(len(regions)):
            text = ax1.text(j, i, f'{heatmap_data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax1, label='Attention Score')
    
    # Bar chart for each emotion
    x = np.arange(len(regions))
    width = 0.8 / len(emotions_to_plot)
    
    for i, emotion in enumerate(emotions_to_plot):
        offset = width * i - (width * len(emotions_to_plot) / 2)
        values = [np.mean(attention_by_emotion[emotion].get(r, [0])) for r in regions]
        ax2.bar(x + offset, values, width, label=emotion)
    
    ax2.set_xlabel('Facial Region')
    ax2.set_ylabel('Average Attention Score')
    ax2.set_title('Attention Distribution Across Regions')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'attention_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_file}")
    
    # Show plot
    plt.show()


def compare_emotions_attention(analysis_file: str, emotion1: str, emotion2: str):
    """
    Compare attention patterns between two emotions.
    
    Args:
        analysis_file: Path to JSON analysis file
        emotion1: First emotion to compare
        emotion2: Second emotion to compare
    """
    with open(analysis_file, 'r') as f:
        data = json.load(f)
    
    frames = data['frame_data']
    regions = ['left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow', 'mouth_outer', 'nose']
    
    # Collect attention data
    attention1 = defaultdict(list)
    attention2 = defaultdict(list)
    
    for frame in frames:
        if 'attention_focus' in frame and frame['face_detected']:
            if frame['emotion'] == emotion1:
                for region, score in frame['attention_focus'].items():
                    attention1[region].append(score)
            elif frame['emotion'] == emotion2:
                for region, score in frame['attention_focus'].items():
                    attention2[region].append(score)
    
    # Calculate averages
    print(f"\n{'='*70}")
    print(f"Comparing Attention: {emotion1.upper()} vs {emotion2.upper()}")
    print(f"{'='*70}\n")
    
    print(f"{'Region':<20} {emotion1:>12} {emotion2:>12} {'Difference':>12}")
    print("-" * 70)
    
    for region in regions:
        avg1 = np.mean(attention1[region]) if attention1[region] else 0
        avg2 = np.mean(attention2[region]) if attention2[region] else 0
        diff = avg1 - avg2
        
        arrow = "→" if abs(diff) < 0.1 else ("↑" if diff > 0 else "↓")
        print(f"{region:<20} {avg1:>12.3f} {avg2:>12.3f} {diff:>11.3f} {arrow}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(regions))
    width = 0.35
    
    avgs1 = [np.mean(attention1[r]) if attention1[r] else 0 for r in regions]
    avgs2 = [np.mean(attention2[r]) if attention2[r] else 0 for r in regions]
    
    ax.bar(x - width/2, avgs1, width, label=emotion1, alpha=0.8)
    ax.bar(x + width/2, avgs2, width, label=emotion2, alpha=0.8)
    
    ax.set_xlabel('Facial Region')
    ax.set_ylabel('Average Attention Score')
    ax.set_title(f'Attention Comparison: {emotion1.title()} vs {emotion2.title()}')
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_file = f'attention_comparison_{emotion1}_{emotion2}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison visualization saved to: {output_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize attention patterns from video analysis'
    )
    parser.add_argument('analysis_file', type=str,
                       help='Path to video analysis JSON file')
    parser.add_argument('--compare', nargs=2, metavar=('EMOTION1', 'EMOTION2'),
                       help='Compare attention between two emotions')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_emotions_attention(args.analysis_file, args.compare[0], args.compare[1])
    else:
        analyze_attention_patterns(args.analysis_file)


if __name__ == "__main__":
    main()
