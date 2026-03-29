# Tri-Modal Architecture Diagram

```
INPUT STAGE
===========
Video Frame Sequence              MediaPipe Face Mesh
(B, T, 3, 224, 224)              (B, T, 468, 3)
        ↓                                ↓
        ↓                         Flatten coordinates
        ↓                         (B, T, 1404)
        ↓                                ↓


FEATURE EXTRACTION STAGE
========================

┌─────────────────────────────────────────────────────────────┐
│                    SPATIAL PATHWAY (CNN)                     │
├─────────────────────────────────────────────────────────────┤
│  For each frame t in sequence:                              │
│    Frame[t] → MobileNetV2 Backbone → Features (1280-dim)   │
│                          ↓                                   │
│                Spatial Attention Module                      │
│              (Focus on eyes, mouth, nose)                    │
│                          ↓                                   │
│                    Classifier → 7 emotions                   │
│                                                              │
│  Output: Spatial predictions (B, T, 7)                      │
│  Average: Spatial logits (B, 7)                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│               TEMPORAL PATHWAY (Transformer)                 │
├─────────────────────────────────────────────────────────────┤
│  Feature Sequence (B, T, 1280)                              │
│          ↓                                                   │
│  Positional Encoding                                         │
│          ↓                                                   │
│  Transformer Encoder (2 layers, 4 heads)                    │
│    - Multi-head self-attention                              │
│    - Feed-forward network                                   │
│          ↓                                                   │
│  Global Average Pooling                                      │
│          ↓                                                   │
│  Temporal Classifier → 7 emotions                           │
│                                                              │
│  Output: Temporal logits (B, 7)                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              GEOMETRIC PATHWAY (MediaPipe)                   │
├─────────────────────────────────────────────────────────────┤
│  For each frame t in sequence:                              │
│    Landmarks[t] (1404) → Landmark Encoder                   │
│                              ↓                               │
│    ┌────────────────────────────────────┐                  │
│    │  Region-Specific Encoders:         │                  │
│    │    - Eyes (66×3) → 64-dim          │                  │
│    │    - Eyebrows (40×3) → 32-dim      │                  │
│    │    - Mouth (40×3) → 64-dim         │                  │
│    │  Region Fusion → 256-dim           │                  │
│    └────────────────────────────────────┘                  │
│                              ↓                               │
│                 Landmark Classifier → 7 emotions             │
│                                                              │
│  Output: Landmark predictions (B, T, 7)                     │
│  Average: Landmark logits (B, 7)                            │
└─────────────────────────────────────────────────────────────┘


FUSION STAGE
============

     Spatial Logits (B, 7)
              ↓
     Temporal Logits (B, 7)      ┌──────────────────────┐
              ↓                   │  FUSION METHODS:     │
     Landmark Logits (B, 7)      │                      │
              ↓                   │  1. WEIGHTED:        │
     ┌────────────────┐          │     w1*S + w2*T +    │
     │ FUSION LAYER   │          │     w3*L             │
     │                │          │                      │
     │  - Weighted    │←─────────│  2. CONCAT:          │
     │  - Concat      │          │     FC([S,T,L])      │
     │  - Add         │          │                      │
     └────────────────┘          │  3. ADD:             │
              ↓                   │     S + T + L        │
     Combined Logits (B, 7)      └──────────────────────┘
              ↓
         Softmax
              ↓
     Probabilities (B, 7)
              ↓
         ArgMax
              ↓
     FINAL PREDICTION


OUTPUT STAGE
============
{
  'combined_prediction': tensor([3]),      # happy
  'combined_probabilities': tensor([[...]),
  'spatial_prediction': tensor([3]),
  'spatial_probabilities': tensor([[...]),
  'temporal_prediction': tensor([3]),
  'temporal_probabilities': tensor([[...]),
  'landmark_prediction': tensor([4]),      # sad
  'landmark_probabilities': tensor([[...])
}


INFORMATION FLOW
================

Frame → What face LOOKS like
        (appearance, texture, lighting)
              ↓
        CNN Features
              ↓
        How emotions CHANGE over time
              ↓
        Transformer Features
              
Landmarks → Where facial points ARE
            (geometric positions)
              ↓
        MediaPipe Features

ALL TOGETHER → Robust emotion prediction
```

## Key Dimensions

### Input Shapes
- **Frames**: (Batch, Time, 3, Height, Width) = (B, T, 3, 224, 224)
- **Landmarks**: (Batch, Time, 1404) = (B, T, 468×3)

### Feature Shapes
- **CNN Features**: (B, T, 1280) for MobileNetV2
- **Transformer Output**: (B, 7)
- **Landmark Features**: (B, T, 256) → (B, 7)

### Output Shape
- **Logits**: (B, 7) - one score per emotion
- **Probabilities**: (B, 7) - sums to 1.0
- **Prediction**: (B,) - integer class index

## Emotion Class Mapping
```
0 → angry
1 → disgust
2 → fear
3 → happy
4 → sad
5 → surprise
6 → neutral
```

## Parameter Count Comparison

| Model Configuration | Parameters | Speed (CPU) |
|---------------------|------------|-------------|
| CNN Only            | ~2.3M      | 30-60 FPS   |
| CNN + Transformer   | ~3.5M      | 15-30 FPS   |
| CNN + Trans + MP    | ~3.8M      | 10-25 FPS   |

Legend: MP = MediaPipe
