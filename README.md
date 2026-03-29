# Real-time Emotion Detection - Hybrid CNN + Transformer

A PyTorch implementation of real-time emotion detection from webcam video, combining CNN spatial features with Transformer temporal modeling.

## 🎯 Overview

This project detects emotions from webcam video and audio in real-time using a multimodal architecture:

1. **Spatial CNN**: Extracts features from facial regions (eyes, mouth, nose, cheeks) with attention
2. **Temporal Transformer**: Models emotion transitions over time from frame sequences  
3. **Audio Sentiment**: NLP-based sentiment analysis from speech transcription
4. **Personalized Acoustic Profiling**: Person-specific vocal baseline tracking with adaptive normalization
5. **Multimodal Fusion**: Combines visual + audio + personalized acoustic predictions

**7 Emotions**: angry, disgust, fear, happy, sad, surprise, neutral

## 🚀 Quick Start

### Test Models (No Training Needed)
```bash
python models/emotion_detector.py
```

### Run Webcam Detection
```bash
# Hybrid mode (CNN + Transformer)
python scripts/webcam_detector.py

# Spatial only (faster)
python scripts/webcam_detector.py --spatial-only

# With trained model
python scripts/webcam_detector.py --model-path checkpoints/best_model.pth
```

### 🎥 Record Videos with Facial Landmarks & Attention
```bash
# Real-time recording with landmarks and attention visualization
python tools/video/video_recorder_with_landmarks.py

# Process existing videos
python tools/video/process_video.py input_video.mp4

# Analyze attention patterns
python tools/video/analyze_attention.py video_analysis.json

# Quick help
./tools/video/video_help.sh
```

**Video Features:**
- **468 facial landmarks** from MediaPipe Face Mesh
- **Attention heatmaps** showing where the model focuses
- **Smart region highlighting** based on detected emotions
- **Frame-by-frame analysis** saved to JSON
- **Interactive controls**: Toggle landmarks (L), attention (A), recording (R)

### 🎤 Audio & Personalized Acoustic Analysis
```bash
# Audio-only sentiment analysis from speech
python tools/audio/audio_sentiment_analyzer.py

# Multimodal fusion (Visual + Audio + Personalized Acoustic)
python tools/audio/multimodal_fusion.py --device cuda --person-id user123

# Test audio fusion logic
make test-audio

# Test personalized acoustic profiling
make test-personalized
```

**Audio Features:**
- **Speech-to-Text**: Converts speech to text (Google/Sphinx)
- **Sentiment Analysis**: Transformer-based (DistilBERT) or TextBlob
- **Emotion Keywords**: Detects happy, sad, angry, fear from text
- **Positive/Negative Phrases**: Identifies "I love", "I hate", etc.
- **Personalized Acoustic Profiling** 🆕: Person-specific vocal baseline tracking
  - Individual pitch/energy/speaking rate baselines
  - Z-score normalization per person
  - Continuous adaptive learning
  - Multi-user profile management
- **Multimodal Fusion**: Combines visual + audio + personalized acoustic

### Controls
- **Q**: Quit
- **S**: Save screenshot
- **R**: Start/Stop recording (video recorder only)
- **L**: Toggle landmarks (video recorder only)
- **A**: Toggle attention (video recorder only)

## 📁 Project Structure

```
MLProject/
├── config/                     # Configuration files
│   ├── config.py              # Model and training configuration
│   └── requirements.txt       # Python dependencies
├── models/                     # Neural network models
│   ├── emotion_detector.py    # Main emotion detection models
│   ├── mediapipe_detector.py  # MediaPipe-enhanced detection
│   ├── audio_emotion_fusion.py # Multimodal fusion with audio
│   ├── personalized_acoustic_profiling.py # Person-specific acoustics 🆕
│   ├── transformer.py         # Transformer architecture
│   └── cnn.py                 # CNN architectures
├── scripts/                    # Executable scripts
│   ├── train.py               # Training script
│   ├── webcam_detector.py     # Real-time webcam detection
│   ├── utils.py               # Utility functions
│   └── quick_start.sh         # Interactive menu
├── tools/                      # Development tools
│   ├── video/                 # Video recording & analysis
│       ├── video_recorder_with_landmarks.py
│       ├── process_video.py
│       ├── analyze_attention.py
│       └── video_help.sh
│   └── audio/                 # Audio sentiment analysis 🎤 NEW!
│       ├── audio_sentiment_analyzer.py
│       ├── multimodal_fusion.py
│       └── README.md
├── tests/                      # Test files
│   ├── test_mediapipe_hybrid.py
│   ├── test_models.py
│   ├── test_audio_fusion_logic.py  # Audio fusion tests 🆕
│   └── test_personalized_acoustic_integration.py  # Personalized profiling tests 🆕
├── docs/                       # Documentation
│   ├── README.md              # Main documentation
│   ├── architecture/          # Architecture documentation
│   ├── guides/                # User guides
│   └── examples/              # Usage examples
├── checkpoints/               # Saved model checkpoints
├── data/                      # Datasets
├── logs/                      # Training logs
└── runs/                      # Experiment runs
```

## 📚 Documentation

- **Main Guide**: [docs/README.md](docs/README.md)
- **Quick Reference**: [docs/guides/QUICK_REFERENCE.md](docs/guides/QUICK_REFERENCE.md)
- **Video Recording**: [docs/guides/README_VIDEO_RECORDING.md](docs/guides/README_VIDEO_RECORDING.md)
- **Architecture**: [docs/architecture/ARCHITECTURE_SUMMARY.md](docs/architecture/ARCHITECTURE_SUMMARY.md)
- **Examples**: [docs/examples/VIDEO_EXAMPLES.md](docs/examples/VIDEO_EXAMPLES.md)

## 🚀 Installation

```bash
# Install dependencies
pip install -r config/requirements.txt

# Or use Docker
docker-compose up emotion-detector-cpu
```

## 🎓 Usage Examples

### Training
```bash
python scripts/train.py
python scripts/train.py --use-mediapipe  # Train with landmarks
```

### Real-time Detection
```bash
python scripts/webcam_detector.py --device cuda
```

### Video Recording & Analysis
```bash
# Record with all features
python tools/video/video_recorder_with_landmarks.py --device cuda

# Process existing video
python tools/video/process_video.py input.mp4 --output analyzed.mp4

# Analyze attention patterns
python tools/video/analyze_attention.py analyzed_analysis.json
```

### Testing
```bash
python tests/test_mediapipe_hybrid.py
python tests/test_models.py
```

## 🔧 Development

```bash
# Run quick start menu
./scripts/quick_start.sh

# Make commands
make test                # Run model tests
make test-audio          # Run audio fusion tests
make test-personalized   # Run personalized acoustic tests
make jupyter             # Start Jupyter
make clean               # Clean artifacts
```

## 📊 Features

- **Multi-modal Architecture**: CNN + Transformer + MediaPipe landmarks + Audio + Personalized Acoustics
- **Real-time Performance**: 30+ FPS on CPU, 60+ FPS on GPU
- **Attention Visualization**: See where the model focuses
- **Facial Landmarks**: 468-point face mesh tracking
- **Audio Sentiment Analysis**: Speech-to-text with NLP-based emotion detection
- **Personalized Acoustic Profiling** 🆕: Individual vocal baseline tracking with adaptive normalization
- **Multi-user Support**: Separate profiles for each person
- **Profile Persistence**: Save and load user-specific acoustic profiles
- **Video Recording**: Save and analyze emotion videos
- **Comprehensive Analysis**: Frame-by-frame emotion and attention data

## 📖 Key Documentation Files

| Document | Description |
|----------|-------------|
| [QUICK_REFERENCE.md](docs/guides/QUICK_REFERENCE.md) | Command cheatsheet |
| [README_VIDEO_RECORDING.md](docs/guides/README_VIDEO_RECORDING.md) | Video recording guide |
| [VIDEO_EXAMPLES.md](docs/examples/VIDEO_EXAMPLES.md) | Usage examples |
| [ARCHITECTURE_SUMMARY.md](docs/architecture/ARCHITECTURE_SUMMARY.md) | Architecture details |
| [MEDIAPIPE_INTEGRATION.md](docs/architecture/MEDIAPIPE_INTEGRATION.md) | MediaPipe integration |

## 🤝 Contributing

See documentation in `docs/` for development guidelines.

## 📝 License

See LICENSE file for details.

---

**Quick Help**: Run `./tools/video/video_help.sh` for video recording commands
