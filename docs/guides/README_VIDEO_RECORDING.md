# Video Recording with Facial Landmarks and Attention Focus

This guide explains how to record videos with facial landmark detection and attention visualization, enabling you to see where the model focuses its attention when detecting emotions.

## Overview

The video recording system provides two main capabilities:

1. **Real-time Recording** (`video_recorder_with_landmarks.py`): Record webcam video with live facial landmarks and attention heatmaps
2. **Video Processing** (`process_video.py`): Process pre-recorded videos with facial analysis

## Key Features

### 🎯 Facial Landmark Detection
- **468 facial landmarks** extracted using MediaPipe Face Mesh
- Real-time tracking of key facial regions:
  - Eyes (left/right)
  - Eyebrows (left/right)
  - Mouth
  - Nose
- **Smart highlighting**: Different regions are emphasized based on detected emotions
  - Mouth highlighted for: happy, sad, disgust
  - Eyebrows highlighted for: surprise, fear, angry
  - Eyes highlighted for: surprise, fear

### 🔥 Attention Visualization
- **Attention heatmaps** show where the CNN model focuses
- Color-coded overlay (red = high attention, blue = low attention)
- Region-specific attention scores for facial features
- Helps understand model decision-making

### 📊 Emotion Detection
- Real-time emotion classification (7 emotions)
- Confidence scores for all emotion classes
- Visual probability bars
- MediaPipe-enhanced detection option for improved accuracy

### 📹 Recording Features
- MP4 video output
- Configurable FPS (default: 30)
- Toggle features on/off during recording:
  - Landmarks overlay (press 'L')
  - Attention heatmap (press 'A')
- Screenshot capture (press 'S')
- Frame-by-frame analysis saved to JSON

## Installation

Ensure all dependencies are installed:

```bash
# Basic requirements
pip install opencv-python torch numpy mediapipe

# Or use the project requirements
pip install -r requirements.txt
```

## Usage

### 1. Real-time Recording from Webcam

**Basic Usage:**
```bash
python video_recorder_with_landmarks.py
```

**With Model Checkpoint:**
```bash
python video_recorder_with_landmarks.py --model-path checkpoints/best_model.pth
```

**With MediaPipe-Enhanced Model:**
```bash
python video_recorder_with_landmarks.py --use-mediapipe --model-path checkpoints/mediapipe_model.pth
```

**Full Options:**
```bash
python video_recorder_with_landmarks.py \
    --model-path checkpoints/best_model.pth \
    --use-mediapipe \
    --output-dir recordings \
    --fps 30 \
    --device cuda \
    --no-landmarks \
    --no-attention
```

**Interactive Controls:**
- `R` - Start/Stop recording
- `L` - Toggle landmarks overlay
- `A` - Toggle attention heatmap
- `S` - Save screenshot
- `Q` - Quit

**Output:**
- Recordings saved to `recordings/recording_YYYYMMDD_HHMMSS.mp4`
- Screenshots saved to `recordings/screenshot_YYYYMMDD_HHMMSS.jpg`

### 2. Processing Pre-recorded Videos

**Basic Usage:**
```bash
python process_video.py input_video.mp4
```

**Specify Output Path:**
```bash
python process_video.py input_video.mp4 --output output_video.mp4
```

**With Analysis:**
```bash
python process_video.py input_video.mp4 \
    --model-path checkpoints/best_model.pth \
    --use-mediapipe \
    --device cuda
```

**Batch Processing (no display):**
```bash
python process_video.py input_video.mp4 \
    --output processed.mp4 \
    --no-display \
    --device cuda
```

**Full Options:**
```bash
python process_video.py input_video.mp4 \
    --output output_video.mp4 \
    --model-path checkpoints/best_model.pth \
    --use-mediapipe \
    --device cuda \
    --no-landmarks \
    --no-attention \
    --no-display \
    --no-analysis
```

**Output Files:**
- Processed video: `output_video.mp4`
- Analysis JSON: `output_video_analysis.json` (contains frame-by-frame data)

## Analysis Output Format

When processing videos, a detailed JSON analysis file is generated:

```json
{
  "summary": {
    "input_file": "input_video.mp4",
    "output_file": "output_video.mp4",
    "processing_time": 45.2,
    "total_frames": 900,
    "fps": 30.0,
    "frames_with_faces": 875,
    "emotion_distribution": {
      "happy": 450,
      "neutral": 300,
      "surprise": 100,
      "sad": 50,
      "angry": 0,
      "disgust": 0,
      "fear": 0
    },
    "dominant_emotion": "happy"
  },
  "frame_data": [
    {
      "frame_number": 1,
      "timestamp": 0.033,
      "face_detected": true,
      "emotion": "happy",
      "confidence": 0.89,
      "probabilities": {
        "angry": 0.01,
        "disgust": 0.01,
        "fear": 0.02,
        "happy": 0.89,
        "sad": 0.02,
        "surprise": 0.03,
        "neutral": 0.02
      },
      "face_bbox": [120, 80, 300, 350],
      "attention_focus": {
        "left_eye": 0.65,
        "right_eye": 0.68,
        "left_eyebrow": 0.45,
        "right_eyebrow": 0.47,
        "mouth_outer": 0.92,
        "nose": 0.55
      }
    }
  ]
}
```

## Understanding Attention Focus

### What are Attention Maps?
Attention maps visualize which parts of the face the CNN model considers most important for emotion detection. High attention (red) indicates regions that strongly influence the prediction.

### Attention Scores by Region
Each facial region gets an attention score (0.0 to 1.0):
- **0.0-0.3**: Low attention (blue) - region has minimal influence
- **0.3-0.6**: Medium attention (green/yellow) - moderate influence
- **0.6-1.0**: High attention (red) - strong influence on prediction

### Typical Attention Patterns by Emotion:
- **Happy**: High attention on mouth (smile) and eyes (crow's feet)
- **Sad**: Focus on mouth corners (downturned) and eyebrows
- **Angry**: Strong attention on eyebrows (furrowed) and eyes (narrowed)
- **Surprise**: Eyes (wide open) and eyebrows (raised) get highest attention
- **Fear**: Similar to surprise but with mouth emphasis
- **Disgust**: Nose bridge and upper lip area
- **Neutral**: Balanced attention across all features

## Use Cases

### 1. Model Debugging
- Verify the model looks at correct facial regions
- Identify potential biases in attention patterns
- Compare attention between different model architectures

### 2. Research & Analysis
- Study emotion expression patterns
- Analyze temporal emotion changes
- Extract attention statistics for papers/reports

### 3. Dataset Creation
- Record labeled emotion videos
- Create training data with ground truth
- Build temporal emotion datasets

### 4. Real-time Applications
- Interactive emotion-aware systems
- Video conferencing emotion analysis
- Mental health monitoring tools

### 5. Education & Demonstration
- Show how AI "sees" emotions
- Explain model decision-making
- Teaching computer vision concepts

## Technical Details

### Facial Landmark Regions
MediaPipe provides 468 landmarks organized into key regions:

```python
landmark_regions = {
    'left_eye': [133-155],        # 22 points
    'right_eye': [362-384],       # 22 points
    'left_eyebrow': [70, 63, ...], # 10 points
    'right_eyebrow': [300, 293, ...], # 10 points
    'mouth_outer': [61, 146, ...], # 20 points
    'nose': [1, 2, 98, 327]       # 4 key points
}
```

### Attention Extraction
Attention weights are extracted from the CNN's spatial attention module:

```python
# From the SpatialAttentionCNN module
attention_map = model.spatial_attention(features)  # Shape: (B, 1, H, W)
```

The attention map is then:
1. Normalized to [0, 1]
2. Resized to face dimensions
3. Color-mapped (COLORMAP_JET)
4. Blended with original frame (60% frame, 40% heatmap)

### Performance
- **Real-time Recording**: ~25-30 FPS on CPU, ~60+ FPS on GPU
- **Video Processing**: ~15-20 FPS on CPU, ~40-50 FPS on GPU
- **Memory Usage**: ~500MB (CPU), ~2GB (GPU)

## Troubleshooting

### No Face Detected
- Ensure adequate lighting
- Face should be clearly visible and frontal
- Minimum face size: 100x100 pixels
- Try adjusting camera angle

### Low FPS
- Use GPU: `--device cuda`
- Reduce resolution in camera settings
- Disable attention visualization: `--no-attention`
- Close other applications

### Landmarks Not Showing
- Press 'L' to toggle landmarks
- Check `--no-landmarks` flag is not set
- Verify MediaPipe installation: `pip install mediapipe`

### Attention Map Not Visible
- Press 'A' to toggle attention
- Model must have spatial attention module
- Some lightweight models may not support attention extraction

### Recording File Errors
- Ensure output directory exists
- Check disk space
- Verify write permissions
- Use different video codec if needed

## Advanced Configuration

### Custom Landmark Regions
Edit `landmark_regions` dictionary to define custom facial regions:

```python
self.landmark_regions = {
    'custom_region': [1, 2, 3, 4, 5],  # Landmark indices
    # Add more regions...
}
```

### Attention Colormap
Change the colormap in `overlay_attention_heatmap()`:

```python
heatmap = cv2.applyColorMap(attention_norm, cv2.COLORMAP_HOT)  # Try HOT, BONE, etc.
```

### Video Codec
Modify codec for different formats:

```python
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # .avi
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
```

## Integration with Existing Models

Both scripts support the project's emotion detection models:

### Hybrid Model (CNN + Transformer)
```bash
python video_recorder_with_landmarks.py \
    --model-path checkpoints/hybrid_model.pth
```

### MediaPipe-Enhanced Model
```bash
python video_recorder_with_landmarks.py \
    --use-mediapipe \
    --model-path checkpoints/mediapipe_model.pth
```

### Spatial CNN Only
```bash
python video_recorder_with_landmarks.py \
    --model-path checkpoints/spatial_cnn.pth
```

## Examples

### Example 1: Quick Recording Session
```bash
# Start recorder with default settings
python video_recorder_with_landmarks.py

# In the viewer:
# 1. Press 'R' to start recording
# 2. Express different emotions
# 3. Press 'R' to stop
# 4. Press 'S' to save screenshots of interesting moments
# 5. Press 'Q' to quit
```

### Example 2: Batch Video Analysis
```bash
# Process multiple videos
for video in videos/*.mp4; do
    python process_video.py "$video" \
        --output "processed/$(basename $video)" \
        --no-display \
        --device cuda
done
```

### Example 3: High-Quality Recording
```bash
# Record with GPU acceleration and MediaPipe model
python video_recorder_with_landmarks.py \
    --use-mediapipe \
    --model-path checkpoints/best_mediapipe.pth \
    --fps 60 \
    --device cuda \
    --output-dir high_quality_recordings
```

## Citation

If you use this video recording system in your research, please cite:

```bibtex
@software{emotion_video_recorder,
  title={Video Recording with Facial Landmarks and Attention Focus},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/emotion-detection}
}
```

## License

Same as the main project license.

## Support

For issues or questions:
1. Check this README
2. Review the main project README
3. Open an issue on GitHub
4. Contact the project maintainers
