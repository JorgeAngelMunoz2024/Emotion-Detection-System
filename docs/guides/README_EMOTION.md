# Real-time Emotion Detection - Hybrid CNN + Transformer + MediaPipe

## 🎯 Project Overview

Real-time emotion detection from webcam video using a tri-modal hybrid architecture:

1. **Spatial CNN**: Extracts appearance features from facial regions (eyes, mouth, nose, cheeks) with attention mechanism
2. **Temporal Transformer**: Models emotion transitions over time from frame sequences
3. **MediaPipe Landmarks** (optional): Extracts geometric features from 468 facial landmarks
4. **Hybrid Fusion**: Combines spatial + temporal + geometric predictions for robust emotion recognition

**7 Emotion Classes**: angry, disgust, fear, happy, sad, surprise, neutral

---

## 🏗️ Architecture

### Spatial CNN with Attention
- **Backbone**: MobileNetV2 (fastest), ResNet18/34/50
- **Pretrained**: ✅ Recommended (ImageNet weights for faster convergence)
- **Attention Module**: Focuses on facial regions
- **Output**: Immediate emotion state from current frame (appearance features)

### Temporal Transformer
- **Input**: Sequence of CNN features (16 frames by default)
- **Architecture**: 2-layer transformer with 4 attention heads
- **Positional Encoding**: Tracks temporal position
- **Output**: Overall emotion state from temporal patterns

### MediaPipe Face Mesh (Optional Enhancement)
- **Input**: 468 facial landmarks (x, y, z coordinates)
- **Regions**: Eyes (66 pts), Eyebrows (40 pts), Mouth (40 pts), Nose (21 pts)
- **Features**: Geometric measurements of facial structure and movement
- **Output**: Emotion state from facial geometry
- **Benefit**: Complements CNN appearance features with precise landmark positions

### Hybrid Fusion
- **Bi-modal**: CNN + Transformer (default)
- **Tri-modal**: CNN + Transformer + MediaPipe (use_mediapipe=True)
- **Methods**: Weighted, concat, or addition
- **Benefits**: Combines instant reactions + temporal context + facial geometry
- **Real-time**: Buffers frames and landmarks for continuous prediction

---

## 📦 Installation

### Option 1: Docker (Recommended)
```bash
cd MLProject
./quick_start.sh
# Choose option 1 for CPU
```

### Option 2: Local Installation
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Test Models
```bash
# Test bi-modal model (CNN + Transformer)
python models/emotion_detector.py

# Test tri-modal model with MediaPipe
python test_mediapipe_hybrid.py
```

### Run Webcam Detection
```bash
# Hybrid mode (CNN + Transformer)
python webcam_detector.py

# Spatial CNN only (faster, no temporal)
python webcam_detector.py --spatial-only

# With MediaPipe landmarks (tri-modal)
python webcam_detector.py --use-mediapipe

# With trained model
python webcam_detector.py --model-path checkpoints/best_model.pth

# GPU inference
python webcam_detector.py --device cuda
```

### Controls
- **Q**: Quit
- **S**: Save screenshot

---

## 🎓 Training

### Prepare Dataset
Popular emotion datasets:
- **FER2013**: 35,887 images, 7 emotions
- **AffectNet**: 450,000 images, 8 emotions
- **RAF-DB**: 30,000 images, 7 emotions

### Train Model
```bash
# Update train.py with your dataset
python train.py

# Or use Docker
docker-compose up emotion-detector-cpu
```

---

## 💡 Pretrained Backbone - YES or NO?

### ✅ YES - Use Pretrained (Recommended)

**Advantages:**
- **Faster convergence**: 5-10x faster training
- **Better features**: ImageNet learned robust visual features
- **Less data needed**: Works well with smaller datasets
- **Higher accuracy**: Usually 5-15% better performance

**When to use:**
- Limited training data (< 50k images)
- Want faster experiments
- Real-time inference important
- First time training

### ❌ NO - Train from Scratch

**Advantages:**
- **Domain-specific features**: Learns only emotion-relevant features
- **Full control**: No ImageNet bias
- **Smaller model**: Can design custom architecture

**When to use:**
- Massive dataset (> 500k images)
- Very domain-specific (medical, infrared, etc.)
- Research purposes
- Unlimited compute resources

### 🎯 **Recommendation**: Use `pretrained=True`
For emotion detection, pretrained ImageNet weights provide excellent face feature extractors!

---

## 📊 Model Performance

### Spatial CNN Only
- **Speed**: ~30-60 FPS (CPU), ~200+ FPS (GPU)
- **Use case**: Real-time, instant reactions
- **Limitation**: No temporal context

### Hybrid CNN + Transformer (Bi-modal)
- **Speed**: ~15-30 FPS (CPU), ~100+ FPS (GPU)
- **Use case**: Robust emotion recognition
- **Advantage**: Understands emotion transitions

### Hybrid + MediaPipe (Tri-modal)
- **Speed**: ~10-25 FPS (CPU), ~80+ FPS (GPU)
- **Use case**: High-accuracy emotion recognition
- **Advantage**: Combines appearance + temporal + geometry
- **Overhead**: ~5-10 FPS reduction for landmark extraction

---

## 🎨 MediaPipe Integration

### Why Add MediaPipe?

**CNN** captures appearance (what face looks like):
- Texture, color, lighting
- Overall facial expression patterns
- Learned visual features

**MediaPipe** captures geometry (facial structure):
- Precise landmark positions (468 points)
- Mouth openness, eyebrow position, eye shape
- Geometric measurements of facial movements

**Together** = More robust emotion detection!

### MediaPipe Face Mesh Details

**Total Landmarks**: 468 points (x, y, z coordinates)

**Emotion-Relevant Regions**:
- **Eyes**: 66 landmarks (33 left + 33 right)
  - Captures: openness, squinting, wideness
  - Emotions: surprise (wide), disgust (squint)
  
- **Eyebrows**: 40 landmarks (20 left + 20 right)
  - Captures: raised, furrowed, position
  - Emotions: surprise (raised), anger (furrowed)
  
- **Mouth**: 40 landmarks
  - Captures: smile, frown, openness, corners
  - Emotions: happy (smile), sad (frown), surprise (open)
  
- **Nose**: 21 landmarks
  - Captures: nostril flare, wrinkles
  - Emotions: disgust (wrinkle), anger (flare)

### How to Use MediaPipe

```python
# Create model with MediaPipe enabled
hybrid_model = HybridEmotionRecognizer(
    num_emotions=7,
    cnn_backbone='mobilenet_v2',
    pretrained=True,
    temporal_layers=2,
    temporal_heads=4,
    fusion_method='weighted',
    use_mediapipe=True  # ✨ Enable MediaPipe
)

# Prepare inputs
frames = torch.randn(1, 16, 3, 224, 224)  # Video sequence
landmarks = torch.randn(1, 16, 1404)      # Landmarks (468*3=1404)

# Predict with both inputs
predictions = hybrid_model.predict_emotion(frames, landmarks)
print(predictions['combined_prediction'])  # Tri-modal prediction
print(predictions['spatial_prediction'])   # CNN prediction
print(predictions['temporal_prediction'])  # Transformer prediction
print(predictions['landmark_prediction'])  # MediaPipe prediction
```

### Fusion Methods

**Weighted** (Default):
```python
output = w1*CNN + w2*Transformer + w3*MediaPipe
# Learns optimal weights during training
```

**Concat**:
```python
concat = [CNN, Transformer, MediaPipe]
output = FC(concat)
# More parameters, flexible fusion
```

**Add**:
```python
output = CNN + Transformer + MediaPipe
# Simple, fast, equal contribution
```

---

## 🔧 Configuration

Edit `models/emotion_detector.py`:

```python
# Spatial CNN
spatial_cnn = SpatialAttentionCNN(
    num_emotions=7,
    backbone='mobilenet_v2',  # or 'resnet18', 'resnet34', 'resnet50'
    pretrained=True,           # ✅ Recommended
    use_attention=True,
    dropout=0.5
)

# Temporal Transformer
temporal_transformer = TemporalTransformer(
    feature_dim=1280,  # MobileNetV2: 1280, ResNet50: 2048
    num_emotions=7,
    num_layers=2,
    num_heads=4,
    dropout=0.1,
    max_seq_length=32
)

# Hybrid Model (Bi-modal)
hybrid_model = HybridEmotionRecognizer(
    num_emotions=7,
    cnn_backbone='mobilenet_v2',
    pretrained=True,
    temporal_layers=2,
    temporal_heads=4,
    fusion_method='weighted',  # 'weighted', 'concat', 'add'
    use_mediapipe=False,       # Disable MediaPipe (default)
    dropout=0.3
)

# Hybrid Model with MediaPipe (Tri-modal)
hybrid_mediapipe = HybridEmotionRecognizer(
    num_emotions=7,
    cnn_backbone='mobilenet_v2',
    pretrained=True,
    temporal_layers=2,
    temporal_heads=4,
    fusion_method='weighted',
    use_mediapipe=True,        # ✨ Enable MediaPipe landmarks
    dropout=0.3
)
```

---

## 📁 Project Structure

```
MLProject/
├── models/
│   ├── emotion_detector.py     # Main models: Spatial CNN, Temporal Transformer, Hybrid
│   ├── transformer.py          # (Old) Scale-Interaction Transformer
│   └── cnn.py                  # (Old) Original CNN models
├── webcam_detector.py          # Real-time webcam emotion detection
├── train.py                    # Training script
├── requirements.txt            # Dependencies
├── Dockerfile                  # Docker setup
├── docker-compose.yml          # Docker services
└── README_EMOTION.md           # This file
```

---

## 🎥 Webcam Detection Features

- ✅ Real-time face detection (Haar Cascade)
- ✅ Emotion classification with confidence scores
- ✅ Probability bars for all emotions
- ✅ Temporal buffering for hybrid mode
- ✅ FPS counter
- ✅ Color-coded bounding boxes per emotion
- ✅ Screenshot capture
- ✅ CPU and GPU support

---

## 🐛 Troubleshooting

### Webcam Not Working
```bash
# Check webcam device
ls /dev/video*

# Test with OpenCV
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Slow Inference
- Use `--spatial-only` for faster FPS
- Reduce `--sequence-length` (default: 16)
- Use GPU: `--device cuda`
- Use lighter backbone: `mobilenet_v2`

### Face Detection Issues
- Ensure good lighting
- Face camera directly
- Adjust `scaleFactor` and `minNeighbors` in code
- Try different face detector (Dlib, MTCNN)

---

## 📚 Datasets

### Download Emotion Datasets

**FER2013** (Most popular)
```bash
# Available on Kaggle
kaggle datasets download -d msambare/fer2013
```

**RAF-DB**
```bash
# Request access: http://www.whdeng.cn/raf/model1.html
```

**AffectNet**
```bash
# Request access: http://mohammadmahoor.com/affectnet/
```

---

## 🔬 Experiments to Try

1. **Compare backbones**: MobileNetV2 vs ResNet18 vs ResNet50
2. **Pretrained vs scratch**: Train with/without ImageNet weights
3. **Spatial vs Hybrid**: Compare instant vs temporal predictions
4. **Fusion methods**: Test weighted, concat, add
5. **Sequence length**: Try 8, 16, 32 frames
6. **Data augmentation**: Rotation, flip, brightness, contrast

---

## 🚀 Transfer to VM

```bash
# On local machine
cd ~/Desktop/Fall25/MachineLearning
tar -czf emotion-detector.tar.gz MLProject/
scp emotion-detector.tar.gz user@vm:~/

# On VM
tar -xzf emotion-detector.tar.gz
cd MLProject/
docker-compose up emotion-detector-gpu
```

---

## 📈 Next Steps

1. **Collect/Download Dataset**: FER2013, RAF-DB, or AffectNet
2. **Update train.py**: Implement dataset loading
3. **Train Spatial CNN**: Start with pretrained backbone
4. **Evaluate**: Test on validation set
5. **Train Hybrid**: Add temporal transformer
6. **Deploy**: Use webcam_detector.py for real-time testing
7. **Optimize**: Quantization, pruning for faster inference

---

## ⚡ Key Takeaways

### Pretrained Backbone
✅ **YES** - Use `pretrained=True` for:
- Faster training
- Better accuracy
- Less data required
- Production deployment

### Architecture Choice
- **Spatial CNN**: Fast, instant reactions
- **Hybrid**: Robust, temporal context
- **Choice**: Start with Spatial, add Temporal if needed

### Hardware
- **CPU**: MobileNetV2, spatial-only, 30 FPS
- **GPU**: Any backbone, hybrid, 100+ FPS

---

## 📝 Citation

If you use this code, consider citing relevant papers:
- FER2013 dataset
- MobileNetV2 / ResNet architecture papers
- Transformer architecture (Vaswani et al., 2017)

---

## 🤝 Contributing

This is a educational project. Feel free to:
- Add new emotion classes
- Implement RCNN for region-based detection
- Add facial landmark detection
- Improve temporal modeling
- Add audio emotion recognition

---

## 📞 Support

For issues:
1. Check webcam is working
2. Verify PyTorch installation
3. Test models: `python models/emotion_detector.py`
4. Check GPU: `nvidia-smi` (if using GPU)

---

**Remember**: Start with `pretrained=True` and `mobilenet_v2` for fastest results! 🚀
