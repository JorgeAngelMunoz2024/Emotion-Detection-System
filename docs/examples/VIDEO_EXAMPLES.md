# Video Recording & Attention Analysis - Complete Examples

This document provides complete, copy-paste ready examples for using the video recording and attention analysis features.

## Quick Start Examples

### Example 1: Record a Quick Video Session

```bash
# Start the recorder
python video_recorder_with_landmarks.py

# Interactive session:
# 1. Position your face in the frame
# 2. Press 'R' to start recording
# 3. Express different emotions (happy, sad, surprised, etc.)
# 4. Press 'R' again to stop recording
# 5. Press 'Q' to quit

# Your video is saved in: recordings/recording_YYYYMMDD_HHMMSS.mp4
```

### Example 2: Process an Existing Video

```bash
# Basic processing
python process_video.py my_video.mp4

# With custom output name
python process_video.py my_video.mp4 --output analyzed_video.mp4

# View the results:
# - analyzed_video.mp4 (video with overlays)
# - analyzed_video_analysis.json (detailed data)
```

### Example 3: Analyze Attention Patterns

```bash
# First, process a video to get analysis data
python process_video.py video.mp4 --output processed.mp4

# Then analyze the attention patterns
python analyze_attention.py processed_analysis.json

# This creates:
# - attention_analysis.png (heatmap visualization)
# - Console output with statistics
```

## Advanced Examples

### Example 4: High-Quality Recording with GPU

```bash
# Record with MediaPipe model and GPU acceleration
python video_recorder_with_landmarks.py \
    --use-mediapipe \
    --model-path checkpoints/best_model.pth \
    --device cuda \
    --fps 60 \
    --output-dir my_recordings

# Benefits:
# - 60 FPS recording
# - GPU acceleration (faster)
# - MediaPipe landmarks (more accurate)
# - Custom output directory
```

### Example 5: Batch Process Multiple Videos

```bash
#!/bin/bash
# Create a batch processing script

# Create output directory
mkdir -p processed_videos

# Process all videos in a folder
for video in raw_videos/*.mp4; do
    filename=$(basename "$video" .mp4)
    echo "Processing: $filename"
    
    python process_video.py "$video" \
        --output "processed_videos/${filename}_processed.mp4" \
        --model-path checkpoints/best_model.pth \
        --device cuda \
        --no-display
done

echo "Batch processing complete!"
```

### Example 6: Compare Emotions

```bash
# Process a video with mixed emotions
python process_video.py emotion_test.mp4 \
    --output emotion_analyzed.mp4

# Compare attention patterns between happy and sad
python analyze_attention.py emotion_analyzed_analysis.json \
    --compare happy sad

# This shows:
# - Which facial regions differ between emotions
# - Bar chart comparison
# - Statistical differences
```

## Workflow Examples

### Workflow 1: Research Paper Analysis

```bash
# Step 1: Record experiment videos
python video_recorder_with_landmarks.py \
    --use-mediapipe \
    --device cuda \
    --output-dir experiment_data

# During recording:
# - Have subjects express each emotion for 10 seconds
# - Press 'R' to start/stop for each emotion
# - Press 'S' to capture key frames

# Step 2: Process all recordings
for video in experiment_data/*.mp4; do
    python process_video.py "$video" \
        --output "analyzed/$(basename $video)" \
        --use-mediapipe \
        --device cuda \
        --no-display
done

# Step 3: Analyze each emotion's attention patterns
for json_file in analyzed/*_analysis.json; do
    python analyze_attention.py "$json_file"
done

# Step 4: Compare specific emotions
python analyze_attention.py analyzed/recording1_analysis.json \
    --compare happy neutral
```

### Workflow 2: Model Debugging

```bash
# Test model on controlled expressions
python video_recorder_with_landmarks.py \
    --model-path checkpoints/my_model.pth \
    --device cuda

# While recording:
# 1. Toggle attention with 'A' to see focus areas
# 2. Toggle landmarks with 'L' to see facial points
# 3. Observe which regions light up for each emotion
# 4. Save screenshots of unexpected behavior with 'S'

# If model seems confused:
# - Check if attention focuses on correct regions
# - Verify landmarks are properly detected
# - Look for lighting or angle issues
```

### Workflow 3: Dataset Creation

```bash
# Record labeled emotion videos
python video_recorder_with_landmarks.py \
    --output-dir dataset/raw_videos

# Session plan:
# - Record 30 seconds of each emotion
# - Use 'R' to start/stop between emotions
# - Label files: happy_001.mp4, sad_001.mp4, etc.

# Process all videos with analysis
for video in dataset/raw_videos/*.mp4; do
    emotion=$(basename "$video" | cut -d'_' -f1)
    
    python process_video.py "$video" \
        --output "dataset/processed/$emotion/$(basename $video)" \
        --device cuda \
        --no-display
done

# Generate statistics per emotion
for emotion in happy sad angry neutral surprise fear disgust; do
    echo "Analyzing $emotion..."
    find dataset/processed/$emotion -name "*_analysis.json" \
        -exec python analyze_attention.py {} \;
done
```

## Python API Examples

### Example 7: Using the Recorder in Your Code

```python
from video_recorder_with_landmarks import VideoRecorderWithLandmarks

# Create recorder instance
recorder = VideoRecorderWithLandmarks(
    model_path='checkpoints/best_model.pth',
    use_mediapipe_model=True,
    device='cuda',
    output_dir='my_recordings',
    show_landmarks=True,
    show_attention=True,
    fps=30
)

# Run the recorder
recorder.run()
```

### Example 8: Process Video Programmatically

```python
from process_video import VideoProcessor

# Create processor
processor = VideoProcessor(
    model_path='checkpoints/best_model.pth',
    use_mediapipe_model=True,
    device='cuda',
    show_landmarks=True,
    show_attention=True,
    save_analysis=True
)

# Process video
summary = processor.process_video(
    input_path='input.mp4',
    output_path='output.mp4',
    display=False  # No GUI
)

# Access results
print(f"Dominant emotion: {summary['dominant_emotion']}")
print(f"Frames with faces: {summary['frames_with_faces']}")
print(f"Emotion distribution: {summary['emotion_distribution']}")

# Access frame-level data
for frame_data in processor.frame_analysis:
    if frame_data['face_detected']:
        print(f"Frame {frame_data['frame_number']}: "
              f"{frame_data['emotion']} "
              f"({frame_data['confidence']:.2f})")
```

### Example 9: Custom Analysis Script

```python
import json
import numpy as np
from collections import defaultdict

def analyze_video_emotions(analysis_file):
    """Analyze emotion timeline from video."""
    with open(analysis_file, 'r') as f:
        data = json.load(f)
    
    frames = data['frame_data']
    
    # Extract emotion timeline
    timeline = []
    for frame in frames:
        if frame['face_detected']:
            timeline.append({
                'time': frame['timestamp'],
                'emotion': frame['emotion'],
                'confidence': frame['confidence']
            })
    
    # Find emotion transitions
    transitions = []
    prev_emotion = None
    for entry in timeline:
        if entry['emotion'] != prev_emotion:
            transitions.append({
                'time': entry['time'],
                'from': prev_emotion,
                'to': entry['emotion']
            })
            prev_emotion = entry['emotion']
    
    print(f"Total emotion transitions: {len(transitions)}")
    for trans in transitions[:10]:  # Show first 10
        print(f"{trans['time']:.2f}s: {trans['from']} → {trans['to']}")
    
    return timeline, transitions

# Usage
timeline, transitions = analyze_video_emotions('video_analysis.json')
```

## Tips & Tricks

### Recording Tips

1. **Good Lighting**: Ensure face is well-lit
2. **Stable Camera**: Use tripod or stable surface
3. **Face Position**: Keep face centered and frontal
4. **Expression Duration**: Hold each emotion for 2-3 seconds
5. **Natural Transitions**: Move smoothly between emotions

### Processing Tips

1. **Use GPU**: Add `--device cuda` for 3-5x speedup
2. **Batch Processing**: Use `--no-display` for unattended runs
3. **Quality Check**: Review first few frames before full batch
4. **Storage**: 1 minute video ≈ 50-100MB processed

### Analysis Tips

1. **Focus on Dominant Emotions**: Ignore emotions with <5 frames
2. **Compare Similar Emotions**: happy vs surprise, sad vs fear
3. **Look for Patterns**: Which regions consistently get attention?
4. **Validate Model**: Does attention match intuition?

## Troubleshooting Examples

### Issue: No Face Detected

```bash
# Test face detection separately
python -c "
import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        print('✓ Face detected!')
    else:
        print('✗ No face detected - check lighting/position')
else:
    print('✗ Cannot read camera')

cap.release()
face_mesh.close()
"
```

### Issue: Low FPS During Recording

```bash
# Try without attention visualization
python video_recorder_with_landmarks.py --no-attention

# Or reduce FPS
python video_recorder_with_landmarks.py --fps 15

# Or disable landmarks
python video_recorder_with_landmarks.py --no-landmarks
```

### Issue: Video Processing Too Slow

```bash
# Use GPU
python process_video.py video.mp4 --device cuda

# Disable display
python process_video.py video.mp4 --no-display

# Disable features
python process_video.py video.mp4 --no-attention --no-landmarks
```

## Integration Examples

### Example 10: Integrate with Existing Webcam Detector

```python
# Modify webcam_detector.py to add recording
from video_recorder_with_landmarks import VideoRecorderWithLandmarks

# Add recording capability to existing detector
# Just replace WebcamEmotionDetector with VideoRecorderWithLandmarks
detector = VideoRecorderWithLandmarks(
    model_path='checkpoints/best_model.pth',
    use_mediapipe_model=True,
    device='cuda'
)
detector.run()
```

### Example 11: Export Attention Data for External Analysis

```python
import json
import pandas as pd

def export_to_csv(analysis_file, output_csv):
    """Convert JSON analysis to CSV for Excel/R/etc."""
    with open(analysis_file, 'r') as f:
        data = json.load(f)
    
    rows = []
    for frame in data['frame_data']:
        if frame['face_detected'] and 'attention_focus' in frame:
            row = {
                'frame': frame['frame_number'],
                'time': frame['timestamp'],
                'emotion': frame['emotion'],
                'confidence': frame['confidence'],
                **frame['attention_focus'],
                **frame['probabilities']
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✓ Exported to {output_csv}")

# Usage
export_to_csv('video_analysis.json', 'analysis.csv')
```

## Performance Benchmarks

### Recording Performance

| Configuration | FPS (CPU) | FPS (GPU) |
|--------------|-----------|-----------|
| Basic (no model) | 60+ | 60+ |
| With landmarks only | 30-40 | 60+ |
| + Attention | 20-30 | 45-60 |
| + MediaPipe model | 15-25 | 40-55 |

### Processing Performance

| Video Length | CPU Time | GPU Time |
|--------------|----------|----------|
| 30 seconds | ~60s | ~20s |
| 2 minutes | ~240s | ~60s |
| 5 minutes | ~600s | ~150s |

*Note: Times for 1920x1080 @ 30 FPS with all features enabled*

## Next Steps

After recording and analyzing videos:

1. **Review attention patterns** - Do they make sense?
2. **Identify model biases** - Is model focusing correctly?
3. **Create training data** - Use recordings for fine-tuning
4. **Document findings** - Export analysis to papers/reports
5. **Improve model** - Use insights to guide architecture changes

## Additional Resources

- Full documentation: `README_VIDEO_RECORDING.md`
- Model architecture: `ARCHITECTURE_SUMMARY.md`
- MediaPipe integration: `MEDIAPIPE_INTEGRATION.md`
- Quick commands: `QUICK_REFERENCE.md`
