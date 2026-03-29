# Multi Modal Emotion Recognition System

A real-time multimodal emotion recognition system that fuses facial segmentation analysis with speech emotion recognition (SER) to detect 7 emotions from webcam video and microphone audio, running in a Docker container.

## Overview

This system combines two modalities for robust emotion detection:

1. **Visual — BiSeNet Face Segmentation**: Parses face regions (lips, eyes, eyebrows, nose, cheeks) using a pretrained BiSeNet model on CelebAMask-HQ, then extracts FACS-inspired geometric features for rule-based emotion classification
2. **Audio — Speech Emotion Recognition**: Uses the emotion2vec+ model (via FunASR) to classify emotions directly from raw audio waveforms
3. **Confidence-Adaptive Fusion**: Dynamically weights visual and audio predictions based on per-modality confidence, with agreement boosting

**7 Emotions**: angry, disgust, fear, happy, sad, surprise, neutral

## Key Features

- **BiSeNet face parsing** with JIT-traced inference and async segmentation worker thread
- **Progressive region visualization**: lips, eyes, eyebrows, nose, cheeks with buffer zones
- **FACS-based emotion classifier** with 18+ geometric features extracted from segmented face regions
- **Personalized calibration system**: Guided 7-emotion calibration flow that computes per-user decision thresholds using neutral baseline
- **emotion2vec+ SER**: Pretrained speech emotion model running inference on 2-second audio windows
- **Feature smoothing** (8-frame buffer) and **emotion smoothing** (15-frame weighted history with hysteresis) for stable output
- **Motion compensation** via phase correlation to skip redundant segmentation on static frames
- **Training data recording**: H.264 video (via ffmpeg re-encode) + WAV audio + HTML visualization dashboard with Chart.js
- **ALSA/JACK noise suppression**: ctypes-level error handler + fd-level stderr redirection for clean container logs
- **Docker containerized** with PulseAudio passthrough, X11 forwarding, and webcam device access

## Quick Start

### Run with Docker (recommended)

```bash
# Build and start the demo
docker-compose up emotion-demo

# Or rebuild from scratch
docker-compose build emotion-demo && docker-compose up emotion-demo
```

The GUI window will appear via X11 forwarding. Click **"Start Audio"** to enable the microphone, and **"Calibrate"** to personalize emotion thresholds.

### Prerequisites

- Docker and docker-compose
- Webcam (`/dev/video0`)
- PulseAudio running on host (for audio capture)
- X11 display (for GUI)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Tkinter GUI                        │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐ │
│  │ Webcam   │  │ Emotion  │  │ Controls:         │ │
│  │ Feed +   │  │ Display  │  │ Calibrate, Record │ │
│  │ Mask     │  │ + Chart  │  │ Audio, Sensitivity│ │
│  └──────────┘  └──────────┘  └───────────────────┘ │
└─────────────────┬───────────────────┬───────────────┘
                  │                   │
    ┌─────────────▼──────┐  ┌────────▼────────┐
    │  VideoProcessor    │  │ AudioProcessor  │
    │  ┌──────────────┐  │  │ ┌─────────────┐ │
    │  │ BiSeNet      │  │  │ │ PyAudio     │ │
    │  │ Segmentation │  │  │ │ Stream      │ │
    │  │ (async)      │  │  │ │ (16kHz)     │ │
    │  └──────┬───────┘  │  │ └──────┬──────┘ │
    │  ┌──────▼───────┐  │  │ ┌──────▼──────┐ │
    │  │ Feature      │  │  │ │ emotion2vec+│ │
    │  │ Extraction   │  │  │ │ SER Model   │ │
    │  │ (18 FACS)    │  │  │ │ (2s window) │ │
    │  └──────┬───────┘  │  │ └──────┬──────┘ │
    │  ┌──────▼───────┐  │  │        │        │
    │  │ Rule-based / │  │  └────────┼────────┘
    │  │ Calibrated   │  │           │
    │  │ Classifier   │  │           │
    │  └──────┬───────┘  │           │
    └─────────┼──────────┘           │
              │                      │
    ┌─────────▼──────────────────────▼─────────┐
    │      Confidence-Adaptive Fusion          │
    │  Visual weight: 0.3 (calibrated) / 0.2   │
    │  + agreement boost + hysteresis           │
    └──────────────────────────────────────────┘
```

## Project Structure

```
├── config/
│   ├── asound.conf              # ALSA→PulseAudio routing config
│   ├── config.py                # Model and training configuration
│   └── requirements.txt         # Python dependencies
├── models/
│   ├── bisenet.py               # BiSeNet face parsing (ResNet18, 19 classes)
│   ├── lip_segmentation_detector.py  # Segmentation, features, classification, calibration
│   ├── speech_emotion_recognition.py # emotion2vec+ SER wrapper
│   ├── emotion_detector.py      # CNN/Transformer emotion models
│   ├── mediapipe_detector.py    # MediaPipe face mesh integration
│   ├── audio_emotion_fusion.py  # Multimodal fusion logic
│   └── personalized_acoustic_profiling.py
├── tools/
│   └── demo/
│       └── multimodal_demo.py   # Main GUI application (~2200 lines)
├── scripts/                     # Training, utilities, download scripts
├── tests/                       # Unit and integration tests
├── docs/                        # Architecture docs and guides
├── backbones/                   # Model weights (git-ignored)
│   ├── emotion2vec_models/      # emotion2vec+ (~1.1GB)
│   └── face_parsing_models/     # BiSeNet weights (~52MB)
├── Dockerfile                   # Multi-stage build (base/cpu/gpu)
├── docker-compose.yml           # Service definitions
└── Makefile                     # Build shortcuts
```

## Calibration

The calibration system personalizes emotion detection thresholds:

1. Click **"Calibrate"** in the GUI
2. Follow the guided flow through 7 emotions (neutral → happy → sad → angry → surprise → fear → disgust)
3. Each emotion: 3-second countdown + 3-second capture window
4. The system computes midpoint decision boundaries between your neutral baseline and each emotion expression
5. After calibration, the segmentation classifier uses your personalized thresholds

## Recording

Click **"Start Recording"** to capture a training session:

- **Video**: Raw mp4v → ffmpeg re-encode to H.264 (libx264 + yuv420p + faststart) for browser playback
- **Audio**: WAV file (16kHz mono 16-bit PCM)
- **Visualization**: HTML dashboard with Chart.js showing emotion timeline, confidence, and audio waveform

Recordings are saved to `data/training_recordings/session_YYYYMMDD_HHMMSS/`.

## Docker Configuration

The container requires access to:
- **Webcam**: `/dev/video0`, `/dev/video1`
- **Audio**: `/dev/snd` + PulseAudio socket (`/run/user/1000/pulse`)
- **Display**: X11 socket (`/tmp/.X11-unix`) + `$DISPLAY`

Key environment variables:
- `PULSE_SERVER=unix:/run/user/1000/pulse/native`
- `JACK_NO_START_SERVER=1` (suppress JACK noise)
- `MODELSCOPE_CACHE=/app/backbones/emotion2vec_models`

## Testing

```bash
# Run all tests
python -m pytest tests/

# Specific test suites
python -m pytest tests/test_speech_emotion_recognition.py
python -m pytest tests/test_audio_fusion_logic.py
python -m pytest tests/test_emotion_detector_comprehensive.py
```
