"""
Models package for emotion detection.
Contains spatial CNN, temporal transformer, hybrid architectures,
and pretrained models for both visual and speech emotion recognition.
"""

from .transformer import ScaleInteractionTransformer
from .transformer import MultiScaleFeatureModule, TransformerBlock, RegressionHead
from .cnn import EmotionDetectorCNN, HybridEmotionRecognition, SpatialAttentionModule

# Pretrained model wrappers
try:
    from .fer_cnn import FEREmotionCNN, FER_EMOTION_LABELS
except ImportError:
    FEREmotionCNN = None
    FER_EMOTION_LABELS = None

try:
    from .speech_emotion_recognition import (
        SpeechEmotionRecognizer,
        SpeechEmotionRecognizerTorch,
        EMOTION_LABELS_7
    )
except ImportError:
    SpeechEmotionRecognizer = None
    SpeechEmotionRecognizerTorch = None
    EMOTION_LABELS_7 = None

__all__ = [
    # Transformer
    'ScaleInteractionTransformer',
    'MultiScaleFeatureModule',
    'TransformerBlock',
    'RegressionHead',
    # CNN
    'EmotionDetectorCNN',
    'HybridEmotionRecognition',
    'SpatialAttentionModule',
    # Pretrained visual emotion
    'FEREmotionCNN',
    'FER_EMOTION_LABELS',
    # Pretrained speech emotion
    'SpeechEmotionRecognizer',
    'SpeechEmotionRecognizerTorch',
    'EMOTION_LABELS_7',
]
