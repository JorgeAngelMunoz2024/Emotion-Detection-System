# Real-time Emotion Detection - Hybrid CNN + Transformer

A PyTorch implementation of real-time emotion detection from webcam video, combining CNN spatial features with Transformer temporal modeling.

## 🎯 Overview

This project detects emotions from webcam video in real-time using a hybrid architecture:

1. **Spatial CNN**: Extracts features from facial regions (eyes, mouth, nose, cheeks) with attention
2. **Temporal Transformer**: Models emotion transitions over time from frame sequences  
3. **Hybrid Fusion**: Combines spatial + temporal predictions for robust emotion recognition

**7 Emotions**: angry, disgust, fear, happy, sad, surprise, neutral

## 🚀 Quick Start

### Test Models (No Training Needed)
```bash
python models/emotion_detector.py
```

### Run Webcam Detection
```bash
# Hybrid mode (CNN + Transformer)
python webcam_detector.py

# Spatial only (faster)
python webcam_detector.py --spatial-only

# With trained model
python webcam_detector.py --model-path checkpoints/best_model.pth
```

### 🎥 NEW: Record Videos with Facial Landmarks & Attention
```bash
# Real-time recording with landmarks and attention visualization
python video_recorder_with_landmarks.py

# Process existing videos
python process_video.py input_video.mp4

# Analyze attention patterns
python analyze_attention.py video_analysis.json
```

**Features:**
- **468 facial landmarks** from MediaPipe Face Mesh
- **Attention heatmaps** showing where the model focuses
- **Smart region highlighting** based on detected emotions
- **Frame-by-frame analysis** saved to JSON
- **Interactive controls**: Toggle landmarks (L), attention (A), recording (R)

See `README_VIDEO_RECORDING.md` for full documentation.

### Controls
- **Q**: Quit
- **S**: Save screenshot
- **R**: Start/Stop recording (video recorder only)
- **L**: Toggle landmarks (video recorder only)
- **A**: Toggle attention (video recorder only)

## 📁 Project Structure

```
MLProject/
├── models/
│   ├── emotion_detector.py     # 🆕 Main models (Spatial CNN, Temporal Transformer, Hybrid)
│   ├── mediapipe_detector.py   # 🆕 MediaPipe-enhanced emotion detection
│   ├── transformer.py          # Original Scale-Interaction Transformer
│   └── cnn.py                  # Original CNN models
├── webcam_detector.py          # 🆕 Real-time webcam emotion detection
├── video_recorder_with_landmarks.py  # 🎥 NEW: Record with landmarks & attention
├── process_video.py            # 🎥 NEW: Process existing videos
├── analyze_attention.py        # 🎥 NEW: Analyze attention patterns
├── train.py                    # Training script
├── config.py                   # Configuration management
├── utils.py                    # Utility functions
├── requirements.txt            # Dependencies (PyTorch, OpenCV, scipy, mediapipe)
├── Dockerfile                  # Multi-stage Docker setup
├── docker-compose.yml          # Docker services
├── README_EMOTION.md           # 🆕 Detailed emotion detection docs
├── README_VIDEO_RECORDING.md   # 🎥 NEW: Video recording documentation
├── VIDEO_EXAMPLES.md           # 🎥 NEW: Usage examples
├── ARCHITECTURE_SUMMARY.md     # 🆕 Architecture explanation
├── QUICK_REFERENCE.md          # Quick command reference
└── README.md                   # This file
```

## 🏗️ Architecture Details

### Spatial CNN with Attention
- **Backbone**: MobileNetV2 (fastest) or ResNet18/34/50
- **Pretrained**: ✅ **Recommended** (ImageNet weights for faster convergence)
- **Attention**: Focuses on facial regions (eyes, mouth, nose, cheeks)
- **Output**: Immediate emotion from current frame

### Temporal Transformer
- **Input**: Sequence of CNN features (16 frames default)
- **Architecture**: 2-layer transformer with 4 attention heads
- **Output**: Overall emotion state from temporal patterns
- **Purpose**: Understands emotion transitions over time

### Hybrid Fusion
- **Methods**: Weighted, concat, or addition
- **Combines**: Instant spatial features + temporal context
- **Result**: Robust emotion predictions

## ⚡ Performance

| Mode | Speed (CPU) | Speed (GPU) | Use Case |
|------|-------------|-------------|----------|
| Spatial CNN | 30-60 FPS | 200+ FPS | Real-time, instant emotions |
| Hybrid | 15-30 FPS | 100+ FPS | Robust, temporal context |

## 💡 Pretrained Backbone - YES!

**Recommendation: Use `pretrained=True`**

**Benefits:**
- ✅ 5-10x faster training convergence
- ✅ 5-15% better accuracy out of the box
- ✅ Works with less training data
- ✅ ImageNet features transfer well to faces

**When to skip:**
- Extremely domain-specific imagery (thermal, medical)
- Massive dataset (> 1M images)
- Research on training from scratch

**For emotion detection: Always start with pretrained!**

## Docker Setup

The project includes Docker support for easy deployment:

### CPU Development (Local Workstation)

```bash
# Build and run with CPU
docker-compose up beauty-score-cpu

# Or build manually
docker build --target cpu -t beauty-score-cpu .
docker run -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints beauty-score-cpu
```

### GPU Deployment (VM with NVIDIA GPU)

```bash
# Build and run with GPU
docker-compose up emotion-detector-gpu

# Ensure NVIDIA Container Toolkit is installed on the VM:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Interactive Jupyter Development

```bash
# Start Jupyter notebook server
docker-compose up jupyter

# Access at http://localhost:8888
```

## 🎓 Training

### 1. Get Dataset
Popular emotion datasets:
- **FER2013**: 35,887 images, 7 emotions (Kaggle)
- **RAF-DB**: 30,000 images, 7 emotions
- **AffectNet**: 450,000 images, 8 emotions

### 2. Train Model
```bash
# Update train.py with your dataset
python train.py

# Or use Docker
docker-compose up beauty-score-cpu  # CPU
docker-compose up beauty-score-gpu  # GPU on VM
```

### 3. Test with Webcam
```bash
python webcam_detector.py --model-path checkpoints/best_model.pth
```

## 📚 Documentation

- **`README_EMOTION.md`** - Complete emotion detection guide
- **`ARCHITECTURE_SUMMARY.md`** - Architecture explanation & design decisions
- **`SETUP_SUMMARY.md`** - Original inspiration: Scale-Interaction Transformer for beauty score prediction

## 🎯 Inspiration

This project was inspired by the Scale-Interaction Transformer (SIT) architecture for facial analysis. We adapted the multi-scale feature extraction and transformer-based temporal modeling concepts for real-time emotion detection from video.

## 🎯 Key Features

- ✅ Real-time webcam emotion detection
- ✅ Spatial CNN with facial region attention
- ✅ Temporal Transformer for emotion transitions
- ✅ Hybrid architecture combining both
- ✅ Pretrained backbone support (highly recommended)
- ✅ Docker containerization for easy deployment
- ✅ CPU and GPU support
- ✅ Face detection and tracking
- ✅ Confidence scores and probability visualization
- ✅ 7 emotion classes

## 🔧 Configuration

Edit `models/emotion_detector.py` for custom settings:

```python
# Spatial CNN
model = SpatialAttentionCNN(
    num_emotions=7,
    backbone='mobilenet_v2',  # or 'resnet18', 'resnet34', 'resnet50'
    pretrained=True,           # ✅ Recommended
    use_attention=True,
    dropout=0.5
)

# Hybrid Model
hybrid = HybridEmotionRecognizer(
    num_emotions=7,
    cnn_backbone='mobilenet_v2',
    pretrained=True,
    temporal_layers=2,
    temporal_heads=4,
    fusion_method='weighted'
)
```

## 🐛 Troubleshooting

### Webcam Not Working
```bash
# Check webcam device
ls /dev/video*

# Test OpenCV
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Slow FPS
- Use `--spatial-only` flag
- Reduce `--sequence-length` (default: 16)
- Use GPU: `--device cuda`
- Use lighter backbone: `mobilenet_v2`

### Face Detection Issues
- Ensure good lighting
- Face camera directly
- Minimum face size: 100x100 pixels

## 📦 Requirements

- Python 3.10+
- PyTorch 2.0+
- OpenCV (webcam + face detection)
- SciPy
- See `requirements.txt` for all dependencies

## 📄 License

MIT
