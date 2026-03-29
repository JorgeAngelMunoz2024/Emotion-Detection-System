# Multimodal Emotion Recognition Architecture

## Overview

Complete **7-modality** emotion detection system with **personalized acoustic profiling** combining visual, temporal, geometric, linguistic, and person-specific acoustic features.

**Key Innovation**: Person-specific acoustic adaptation recognizes that emotional expression varies between individuals. What sounds "excited" for one person may be normal speaking tone for another.

---

## Input Modalities

### 1. **Visual Spatial Features (CNN)**
- **Input**: `(Batch, Time, 3, 224, 224)` - RGB video frames
- **Encoder**: MobileNetV2 / ResNet with spatial attention
- **Output**: Spatial feature vectors + emotion logits
- **Captures**: Facial expressions, visual appearance, static emotional cues

### 2. **Visual Temporal Features (Transformer)**
- **Input**: `(Batch, Time, 512)` - Sequence of CNN features
- **Encoder**: Multi-head self-attention transformer
- **Output**: Temporally-aware emotion logits
- **Captures**: Expression dynamics, temporal patterns, movement

### 3. **Facial Landmarks (MediaPipe)**
- **Input**: `(Batch, Time, 468×3)` - 468 facial keypoints (x,y,z)
- **Encoder**: Geometric feature extractor
- **Output**: Landmark-based emotion logits
- **Captures**: Facial geometry, muscle movements, facial action units

### 4. **Audio Sentiment (NLP)**
- **Input**: 
  - Sentiment scores: `(Batch, Time_audio, 3)` [pos, neg, neutral]
  - Emotion keywords: `(Batch, Time_audio, 7)`
- **Encoder**: Sentiment + emotion keyword fusion
- **Output**: Linguistic emotion logits
- **Captures**: Word choice, semantic emotional content, sentiment polarity

**Pipeline**:
```
Speech → Speech-to-Text → NLP Analysis (DistilBERT) → Sentiment Scores
                        ↓
                 Emotion Keyword Detection → Emotion Scores
```

### 5. **Acoustic Features (Spectrogram/Prosody)**
- **Input**: `(Batch, Time_audio, samples)` - Raw audio waveform
- **Encoder**: 
  - Mel-spectrogram CNN (2D convolutions)
  - Prosody encoder (pitch, energy, speaking rate)
- **Output**: Acoustic emotion logits
- **Captures**: Vocal tone, pitch, prosody, voice characteristics, emotional intonation

**Pipeline**:
```
Waveform → Mel-Spectrogram → CNN → Temporal Attention → Features
         ↓
         Prosody Extraction (pitch/energy/zcr) → Encoder → Features
         ↓
         Feature Fusion → Acoustic Logits
```

**Unsupervised Learning**: No transcription needed - purely acoustic analysis

### 6. **Personalized Acoustic Features (Person-Specific)**
- **Input**: 
  - Raw audio waveform: `(Batch, Time_audio, samples)`
  - Person ID: `str` (unique identifier)
- **Encoder**:
  - Base acoustic encoder (spectrograms + prosody)
  - Person-specific deviation computation
  - Individual adaptation layers per user
- **Output**: Personalized acoustic emotion logits
- **Captures**: Person-relative vocal patterns, individual baseline deviations, user-specific emotional expression

**Pipeline**:
```
Waveform → Feature Extraction → Pitch, Energy, Speaking Rate
         ↓
         Person Profile Lookup → Baseline Statistics
         ↓
         Deviation Computation → (current - baseline) / baseline
         ↓
         Base Acoustic Encoder → Universal Features
         ↓
         Person-Specific Adaptation Layer → Personalized Features
         ↓
         Personalized Classifier → Emotion Logits
```

**Key Innovation**:
- **Learns individual baselines**: Each person's typical pitch, energy, speaking rate
- **Computes relative deviations**: +20% pitch means different things for different people
- **Continuous adaptation**: Baselines evolve over time using exponential moving average
- **Privacy-preserving**: Profiles stored locally, no centralized database

**Example**:
```
Person A (Naturally High Energy):
  Baseline: pitch=250Hz, energy=0.8
  Current: pitch=280Hz → deviation=+12% → "happy"

Person B (Naturally Low Energy):
  Baseline: pitch=150Hz, energy=0.3
  Current: pitch=165Hz → deviation=+10% → "happy"

→ Same emotion detected despite different absolute values!
```

### 7. **Audio-Visual Joint (Cross-Modal Transformer)**
- **Input**: Visual features + Audio (NLP) features
- **Encoder**: Cross-attention transformer
- **Output**: Joint audio-visual emotion logits
- **Captures**: Synchronized patterns, audio-visual correlations

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT STREAMS                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Video Frames       Audio Waveform + Person ID      Facial Landmarks   │
│  (B,T,3,224,224)    (B,Ta,samples) + "alice_001"    (B,T,468×3)       │
│         │                    │                             │            │
│         ▼                    ▼                             ▼            │
│    ┌────────┐         ┌──────────┐                  ┌──────────┐      │
│    │  CNN   │         │ Speech-  │                  │ MediaPipe│      │
│    │Backbone│         │ to-Text  │                  │ Encoder  │      │
│    └────┬───┘         └────┬─────┘                  └────┬─────┘      │
│         │                  │                             │             │
│         │            ┌─────┴─────┐                       │             │
│         │            │           │                       │             │
│         │            ▼           ▼                       │             │
│         │      ┌─────────┐ ┌──────────────────┐         │             │
│         │      │   NLP   │ │  Personalized    │         │             │
│         │      │Sentiment│ │    Acoustic      │         │             │
│         │      │Analyzer │ │    Profiling     │         │             │
│         │      └────┬────┘ │                  │         │             │
│         │           │      │ • Profile Lookup │         │             │
│         │           │      │ • Deviation Calc │         │             │
│         │           │      │ • Adaptation     │         │             │
│         │           │      └──────┬───────────┘         │             │
│         │           │             │                     │             │
└─────────┼───────────┼─────────────┼─────────────────────┼─────────────┘
          │           │             │                     │
          ▼           ▼             ▼                     ▼
     ┌────────┐  ┌────────┐  ┌──────────────┐      ┌─────────┐
     │Spatial │  │ Audio  │  │ Personalized │      │Landmark │
     │Features│  │Features│  │  Acoustic    │      │Features │
     │        │  │        │  │         │      │         │
     │Logits₁ │  │Logits₄ │  │Logits₅  │      │Logits₃  │
     └───┬────┘  └────┬───┘  └────┬────┘      └────┬────┘
         │            │           │                 │
         ▼            │           │                 │
    ┌─────────┐       │           │                 │
    │Temporal │       │           │                 │
    │Transform│       │           │                 │
    │         │       │           │                 │
    │Logits₂  │       │           │                 │
    └────┬────┘       │           │                 │
         │            │           │                 │
         ├────────────┴───┐       │                 │
         │                │       │                 │
         ▼                ▼       │                 │
    ┌────────────────────────┐   │                 │
    │  Audio-Visual Cross    │   │                 │
    │  Attention Transformer │   │                 │
    │                        │   │                 │
    │      Logits₆           │   │                 │
    └───────────┬────────────┘   │                 │
                │                │                 │
                ▼                ▼                 ▼
         ┌──────────────────────────────────────────┐
         │          MULTIMODAL FUSION                │
         │  (Weighted / Concat / Attention)          │
         │                                           │
         │  Logits₁ + Logits₂ + Logits₃ + Logits₄  │
         │         + Logits₅ + Logits₆              │
         └──────────────────┬────────────────────────┘
                            ▼
                   ┌─────────────────┐
                   │ Final Emotion   │
                   │   Prediction    │
                   │                 │
                   │ [happy, sad,    │
                   │  angry, ...]    │
                   └─────────────────┘
```

---

## Fusion Strategies

### 1. Weighted Fusion (Learnable)
```python
final = w₁×spatial + w₂×temporal + w₃×landmarks + 
        w₄×audio_nlp + w₅×acoustic + w₆×audiovisual

where Σwᵢ = 1 (softmax normalized)
```

### 2. Concatenation Fusion
```python
concat = [spatial | temporal | landmarks | audio_nlp | acoustic | audiovisual]
final = FC_layers(concat)
```

### 3. Attention Fusion
```python
αᵢ = attention(modalityᵢ)  # Dynamic weights per sample
final = Σ αᵢ × modalityᵢ
```

---

## Data Flow Example

**Input**: 
- 16 video frames at 30 FPS (0.5s)
- 0.5s audio waveform (8000 samples at 16kHz)
- 4 NLP audio segments (0.125s each)

**Processing**:

```
Video (16 frames)
    ↓
CNN → 16×512 features → Spatial logits (7)
    ↓
Transformer → Temporal logits (7)

Landmarks (16 frames)
    ↓
MediaPipe → 16×(468×3) → Landmark logits (7)

Audio Waveform (8000 samples)
    ↓
Mel-Spectrogram (128×time) → CNN → Acoustic Features
    ↓
Prosody (pitch/energy/zcr) → Prosody Features
    ↓
Fusion → Acoustic logits (7)

Audio Segments (4 segments)
    ↓
Speech-to-Text → "I'm feeling great today!"
    ↓
NLP Sentiment → [0.9 pos, 0.05 neg, 0.05 neutral]
Emotion Keywords → [0.7 happy, 0.1 sad, ...]
    ↓
Audio NLP logits (7)

Visual (16×512) + Audio NLP (4×256)
    ↓
Cross-Attention → AudioVisual logits (7)

═══════════════════════════════════════════
FUSION (6 modality predictions)
═══════════════════════════════════════════
Weighted combination → Final: "happy" (0.87)
```

---

## Key Features

### Complementary Information
- **Visual**: Expression appearance
- **Temporal**: Expression dynamics
- **Landmarks**: Geometric precision
- **Audio NLP**: Semantic content
- **Acoustic**: Vocal tone (unsupervised)
- **Cross-Modal**: Synchronized patterns

### Robustness
- Missing modalities handled gracefully
- Each modality provides independent evidence
- Fusion combines strengths, mitigates weaknesses

### Two Audio Modalities
1. **NLP Audio** (supervised): Requires transcription
   - Understands "I'm sad" vs "I'm happy"
   - Semantic emotional content
   
2. **Acoustic Audio** (unsupervised): No transcription needed
   - Detects angry tone in voice
   - Pitch, prosody, vocal characteristics
   - Works across languages

---

## Implementation Files

```
models/
├── audio_emotion_fusion.py      # Main multimodal architecture
├── audio_acoustic_encoder.py    # Spectrogram + prosody encoder
├── emotion_detector.py          # CNN + Transformer base
└── mediapipe_detector.py        # Facial landmark extractor

tools/audio/
├── audio_sentiment_analyzer.py  # Speech-to-text + NLP
└── multimodal_fusion.py        # Application-level fusion
```

---

## Usage Example

```python
from models.audio_emotion_fusion import MultimodalEmotionRecognizer

# Create 6-modality model
model = MultimodalEmotionRecognizer(
    num_emotions=7,
    use_mediapipe=True,
    use_audio=True,        # NLP sentiment
    use_acoustic=True,     # Spectrogram/prosody
    fusion_method='weighted'
)

# Prepare inputs
frames = torch.randn(2, 16, 3, 224, 224)          # Video
audio_waveform = torch.randn(2, 4, 8000)          # Raw audio
audio_sentiment = torch.randn(2, 4, 3).softmax(-1) # NLP sentiment
audio_emotions = torch.randn(2, 4, 7).softmax(-1)  # NLP emotions
landmarks = torch.randn(2, 16, 468*3)             # Face landmarks

# Forward pass
result = model(
    frames, 
    audio_sentiment, 
    audio_emotions, 
    audio_waveform,
    landmarks
)

# Result contains:
# - logits: Final fused prediction
# - spatial_logits: CNN prediction
# - temporal_logits: Transformer prediction
# - landmark_logits: MediaPipe prediction
# - audio_logits: NLP sentiment prediction
# - acoustic_logits: Spectrogram/prosody prediction
# - audiovisual_logits: Cross-modal prediction
# - fusion_weights: Modality importance (if weighted fusion)
```

---

## Benefits of 6-Modality System

1. **Redundancy**: If one modality fails (e.g., poor audio), others compensate
2. **Accuracy**: Multiple sources of evidence improve confidence
3. **Language-Agnostic**: Acoustic features work without understanding words
4. **Completeness**: Captures both what is said (NLP) and how it's said (acoustic)
5. **Robustness**: Handles noisy environments, occlusions, poor lighting
6. **Interpretability**: See which modalities contribute most to decision

---

## Future Enhancements

- [ ] Add body language (pose estimation)
- [ ] Include physiological signals (heart rate, GSR)
- [ ] Multi-language NLP support
- [ ] Real-time adaptive fusion weights
- [ ] Uncertainty estimation per modality
