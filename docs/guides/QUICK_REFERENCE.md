# Quick Reference - Emotion Detection

## 🚀 Instant Start

### Test Without Training
```bash
# Test bi-modal models (CNN + Transformer)
python models/emotion_detector.py

# Test tri-modal models with MediaPipe
python test_mediapipe_hybrid.py

# Run webcam (uses pretrained CNN, random emotion predictions)
python webcam_detector.py
```

### Commands Cheat Sheet
```bash
# Webcam detection
python webcam_detector.py                    # Hybrid mode (bi-modal)
python webcam_detector.py --spatial-only     # Faster, no temporal
python webcam_detector.py --use-mediapipe    # Tri-modal with landmarks
python webcam_detector.py --device cuda      # Use GPU
python webcam_detector.py --model-path X.pth # Use trained model

# Video recording with landmarks & attention
python video_recorder_with_landmarks.py      # Record with landmarks
python video_recorder_with_landmarks.py \
  --use-mediapipe --device cuda              # MediaPipe + GPU
python process_video.py input.mp4            # Process existing video
python process_video.py input.mp4 \
  --output out.mp4 --use-mediapipe           # Full analysis

# Docker
./quick_start.sh                             # Interactive menu
docker-compose up emotion-detector-cpu       # CPU training
docker-compose up emotion-detector-gpu       # GPU training
make test                                    # Run tests
make jupyter                                 # Start Jupyter

# Training
python train.py                              # Train model
python train.py --resume checkpoint.pth      # Resume training
python train.py --use-mediapipe              # Train with landmarks
```

## 📊 Model Comparison

| Model | Speed (CPU) | Speed (GPU) | Best For |
|-------|-------------|-------------|----------|
| **Spatial CNN Only** | 30-60 FPS | 200+ FPS | Real-time, instant reactions |
| **Hybrid (Bi-modal)** | 15-30 FPS | 100+ FPS | ✅ Best balance |
| **Hybrid (Tri-modal)** | 10-25 FPS | 80+ FPS | Highest accuracy |

### Backbone Comparison

| Backbone | Speed (CPU) | Speed (GPU) | Best For |
|----------|-------------|-------------|----------|
| **MobileNetV2** | 30-60 FPS | 200+ FPS | ✅ Real-time, best choice |
| **ResNet18** | 20-40 FPS | 150+ FPS | Balance speed/accuracy |
| **ResNet34** | 15-30 FPS | 100+ FPS | Better accuracy |
| **ResNet50** | 10-20 FPS | 80+ FPS | Best accuracy, research |

## 🎯 Pretrained Backbone?

| Use Pretrained | When | Why |
|----------------|------|-----|
| ✅ **YES** (Recommended) | Almost always | 5-10x faster training, better accuracy, less data needed |
| ❌ NO | Massive dataset (>1M) or very domain-specific | Full control, domain-specific features |

**For emotion detection: Always use `pretrained=True`**

## 🏗️ Architecture Modes

### 1. Spatial CNN Only
```python
model = SpatialAttentionCNN(
    num_emotions=7,
    backbone='mobilenet_v2',
    pretrained=True  # ✅
)
```
- **Speed**: Fast (30-60 FPS CPU)
- **Use**: Instant emotions, real-time
- **Limitation**: No temporal context

### 2. Hybrid (CNN + Transformer) - Bi-modal
```python
model = HybridEmotionRecognizer(
    num_emotions=7,
    cnn_backbone='mobilenet_v2',
    pretrained=True,  # ✅
    temporal_layers=2,
    temporal_heads=4,
    use_mediapipe=False  # Bi-modal
)
```
- **Speed**: Medium (15-30 FPS CPU)
- **Use**: Robust emotions, video analysis
- **Advantage**: Temporal context

### 3. Hybrid + MediaPipe - Tri-modal ✨
```python
model = HybridEmotionRecognizer(
    num_emotions=7,
    cnn_backbone='mobilenet_v2',
    pretrained=True,  # ✅
    temporal_layers=2,
    temporal_heads=4,
    use_mediapipe=True  # Tri-modal
)
```
- **Speed**: Slower (10-25 FPS CPU)
- **Use**: High-accuracy emotion detection
- **Advantage**: Appearance + temporal + geometry

## 📁 Key Files

| File | Purpose |
|------|---------|
| `models/emotion_detector.py` | 🆕 Main models |
| `webcam_detector.py` | 🆕 Real-time detection |
| `train.py` | Training script |
| `README_EMOTION.md` | Complete docs |
| `ARCHITECTURE_SUMMARY.md` | Architecture explanation |

## 🎓 Workflow

### 1️⃣ Test Setup (5 min)
```bash
python models/emotion_detector.py
python webcam_detector.py
```

### 2️⃣ Get Data (30 min)
Download FER2013 from Kaggle:
```bash
kaggle datasets download -d msambare/fer2013
```

### 3️⃣ Train (Hours)
```bash
# Update train.py with dataset path
python train.py

# Or Docker
docker-compose up emotion-detector-cpu
```

### 4️⃣ Deploy (5 min)
```bash
python webcam_detector.py --model-path checkpoints/best_model.pth
```

## 🎨 Emotions Detected

1. 😠 Angry
2. 🤢 Disgust  
3. 😨 Fear
4. 😊 Happy
5. 😢 Sad
6. 😲 Surprise
7. 😐 Neutral

## ⚙️ Configuration Quick Edit

Edit `models/emotion_detector.py` line ~30:
```python
backbone='mobilenet_v2',  # Change to 'resnet18', 'resnet34', 'resnet50'
pretrained=True,          # Keep True!
use_attention=True,       # Keep True for facial regions
dropout=0.5,              # Adjust if overfitting
```

Edit `webcam_detector.py` line ~30:
```python
sequence_length=16,       # Frames for temporal (8-32)
fusion_method='weighted', # Or 'concat', 'add'
```

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| Webcam not found | `ls /dev/video*` check device |
| Slow FPS | Use `--spatial-only` or `mobilenet_v2` |
| No face detected | Better lighting, face camera |
| Import errors | `pip install -r requirements.txt` |
| CUDA error | Check `nvidia-smi`, use `--device cpu` |

## 📈 Typical Performance

### Without Training (Pretrained CNN)
- Face detection: ✅ Works
- Model runs: ✅ Works
- Emotions: ❌ Random (needs training)

### After Training on FER2013
- Accuracy: 60-70% (typical for FER2013)
- Real-time: ✅ 30+ FPS on CPU
- Robust: ✅ With hybrid model

## 🎥 Video Recording & Analysis (NEW!)

### Real-time Recording with Landmarks
```bash
# Basic recording
python video_recorder_with_landmarks.py

# With trained model and GPU
python video_recorder_with_landmarks.py \
  --model-path checkpoints/best_model.pth \
  --use-mediapipe \
  --device cuda \
  --fps 60

# Controls during recording:
# R - Start/Stop Recording
# L - Toggle Landmarks
# A - Toggle Attention Heatmap
# S - Save Screenshot
# Q - Quit
```

### Process Existing Videos
```bash
# Process video with analysis
python process_video.py input_video.mp4 \
  --output processed_video.mp4 \
  --model-path checkpoints/best_model.pth \
  --device cuda

# Batch processing (no display)
python process_video.py input.mp4 \
  --output output.mp4 \
  --no-display \
  --device cuda

# Output:
# - processed_video.mp4 (with overlays)
# - processed_video_analysis.json (frame-by-frame data)
```

### What You Get
- **468 facial landmarks** from MediaPipe
- **Attention heatmaps** showing CNN focus areas
- **Smart region highlighting** based on emotion
  - Mouth: happy, sad, disgust
  - Eyebrows: angry, fear, surprise
  - Eyes: fear, surprise
- **Frame-by-frame JSON** with emotion & attention data
- **Statistics**: emotion distribution, attention scores

### Use Cases
- 🔬 Model debugging & validation
- 📊 Research & data analysis
- 🎓 Teaching & demonstrations
- 🎬 Creating training datasets
- 🔍 Understanding model decisions

**See `README_VIDEO_RECORDING.md` for full guide**

## 🔥 Pro Tips

1. **Always start with** `pretrained=True`
2. **Use** `mobilenet_v2` for fastest inference
3. **Train spatial CNN first**, then add temporal
4. **Good lighting** crucial for face detection
5. **16 frames** good default for temporal window
6. **Weighted fusion** usually best
7. **Test on webcam** before full training
8. **FER2013** most popular dataset
9. **Record videos** to analyze model attention 🎥 NEW!
10. **Use GPU** for video processing (`--device cuda`)

## 📞 Quick Help

```bash
# Model not loading?
python models/emotion_detector.py

# Webcam issues?
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# Video recording issues?
python -c "import mediapipe; print('MediaPipe OK')"

# Docker not working?
docker-compose down
docker-compose build --no-cache

# Check GPU
nvidia-smi

# Python packages
pip install -r requirements.txt
```

## 🎯 Remember

✅ Use `pretrained=True`  
✅ Start with `mobilenet_v2`  
✅ Test webcam before training  
✅ Record videos to visualize attention 🎥  
✅ Use GPU for faster video processing  
✅ FER2013 is easiest dataset  
✅ Spatial-only mode for speed  
✅ Hybrid mode for accuracy  

---

**TL;DR**: Run `python webcam_detector.py`, train on FER2013, use `pretrained=True` and `mobilenet_v2`! 🚀
