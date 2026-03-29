"""
FER2013-compatible CNN for emotion detection.
This module provides both a PyTorch model and a Keras model loader
for using pretrained weights from FER2013 training.

The Kaggle pretrained model (face_model.h5) expects:
- Input: 48x48 grayscale images
- Output: 7 emotion classes (angry, disgust, fear, happy, sad, surprise, neutral)

Models are cached in the 'backbones' folder.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Project root and backbones directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKBONES_DIR = PROJECT_ROOT / 'backbones'
BACKBONES_DIR.mkdir(exist_ok=True)

# Try to import TensorFlow/Keras for loading .h5 weights
try:
    import tensorflow as tf
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")

# Try PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# FER2013 emotion labels (same order as Kaggle model)
FER_EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


class FEREmotionCNN:
    """
    Wrapper for FER2013 emotion detection CNN.
    Supports loading Keras .h5 weights from Kaggle pretrained model.
    """
    
    def __init__(self, weights_path: Optional[str] = None):
        """
        Initialize the FER emotion CNN.
        
        Args:
            weights_path: Path to pretrained .h5 weights file (face_model.h5)
        """
        self.model = None
        self.weights_path = weights_path
        
        if not KERAS_AVAILABLE:
            print("TensorFlow/Keras not available. CNN emotion detection disabled.")
            return
        
        # Check for weights in backbones folder first, then other locations
        if weights_path is None:
            possible_paths = [
                BACKBONES_DIR / 'face_model.h5',
                BACKBONES_DIR / 'fer_model.h5',
                PROJECT_ROOT / 'checkpoints' / 'face_model.h5',
                PROJECT_ROOT / 'checkpoints' / 'fer_model.h5',
                Path.home() / '.cache' / 'emotion_models' / 'face_model.h5',
            ]
            for p in possible_paths:
                if p.exists():
                    weights_path = str(p)
                    break
        
        if weights_path and Path(weights_path).exists():
            self._load_pretrained(weights_path)
        else:
            self._create_model()
            print("No pretrained weights found. Using randomly initialized model.")
            print("Download face_model.h5 from:")
            print("https://www.kaggle.com/datasets/abhisheksingh016/machine-model-for-emotion-detection")
            print(f"Place it in: {BACKBONES_DIR / 'face_model.h5'}")
    
    def _create_model(self):
        """Create the CNN architecture matching FER2013 training."""
        if not KERAS_AVAILABLE:
            return
        
        # Standard FER2013 CNN architecture
        self.model = keras.Sequential([
            # Block 1
            keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Conv2D(64, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            
            # Block 2
            keras.layers.Conv2D(128, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Conv2D(128, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            
            # Block 3
            keras.layers.Conv2D(256, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Conv2D(256, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            
            # Fully connected
            keras.layers.Flatten(),
            keras.layers.Dense(512),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(7, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def _load_pretrained(self, weights_path: str):
        """Load pretrained Keras model from .h5 file."""
        if not KERAS_AVAILABLE:
            return
        
        try:
            # Load the full model (architecture + weights)
            self.model = keras.models.load_model(weights_path)
            print(f"Loaded pretrained FER model from: {weights_path}")
            
            # Print model summary
            input_shape = self.model.input_shape
            output_shape = self.model.output_shape
            print(f"  Input shape: {input_shape}")
            print(f"  Output shape: {output_shape}")
            print(f"  Parameters: {self.model.count_params():,}")
            
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Creating default architecture...")
            self._create_model()
    
    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for the model.
        
        Args:
            face_img: Face image (any size, BGR or grayscale)
            
        Returns:
            Preprocessed image ready for model (1, 48, 48, 1)
        """
        import cv2
        
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Resize to 48x48
        resized = cv2.resize(gray, (48, 48))
        
        # Normalize to 0-1
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        processed = normalized.reshape(1, 48, 48, 1)
        
        return processed
    
    def predict(self, face_img: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion from face image.
        
        Args:
            face_img: Face image (any size, BGR or grayscale)
            
        Returns:
            Tuple of (emotion_label, confidence, probabilities)
        """
        if self.model is None:
            # Return neutral if model not available
            probs = np.zeros(7)
            probs[6] = 1.0  # neutral
            return 'neutral', 1.0, probs
        
        # Preprocess
        processed = self.preprocess(face_img)
        
        # Predict
        predictions = self.model.predict(processed, verbose=0)
        probs = predictions[0]
        
        # Get top emotion
        emotion_idx = np.argmax(probs)
        confidence = probs[emotion_idx]
        emotion = FER_EMOTION_LABELS[emotion_idx]
        
        return emotion, float(confidence), probs
    
    def predict_batch(self, face_imgs: list) -> list:
        """
        Predict emotions for multiple face images.
        
        Args:
            face_imgs: List of face images
            
        Returns:
            List of (emotion_label, confidence, probabilities) tuples
        """
        if self.model is None or len(face_imgs) == 0:
            return [('neutral', 1.0, np.array([0,0,0,0,0,0,1]))] * len(face_imgs)
        
        # Preprocess all images
        batch = np.vstack([self.preprocess(img) for img in face_imgs])
        
        # Predict
        predictions = self.model.predict(batch, verbose=0)
        
        results = []
        for probs in predictions:
            emotion_idx = np.argmax(probs)
            confidence = probs[emotion_idx]
            emotion = FER_EMOTION_LABELS[emotion_idx]
            results.append((emotion, float(confidence), probs))
        
        return results


# PyTorch version for consistency with rest of codebase
class FEREmotionCNNTorch(nn.Module):
    """
    PyTorch version of FER2013 CNN architecture.
    Can convert weights from Keras model if needed.
    """
    
    def __init__(self):
        super().__init__()
        
        # Block 1
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.25)
        
        # Fully connected
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 7)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.drop1(self.pool1(x))
        
        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.drop2(self.pool2(x))
        
        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.drop3(self.pool3(x))
        
        # Flatten and FC
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)


def download_kaggle_model(output_path: Optional[str] = None) -> str:
    """
    Instructions for downloading the Kaggle pretrained model.
    
    Returns:
        Path where the model should be saved
    """
    if output_path is None:
        output_path = str(BACKBONES_DIR / 'face_model.h5')
    
    print("=" * 60)
    print("To download the pretrained FER2013 emotion model:")
    print("=" * 60)
    print()
    print("1. Go to: https://www.kaggle.com/datasets/abhisheksingh016/machine-model-for-emotion-detection")
    print("2. Click 'Download' to get the face_model.h5 file")
    print(f"3. Move the file to: {output_path}")
    print()
    print("Or use kaggle CLI:")
    print("  kaggle datasets download -d abhisheksingh016/machine-model-for-emotion-detection")
    print()
    print("=" * 60)
    
    return output_path


if __name__ == "__main__":
    # Test the model
    print("FER2013 Emotion CNN Test")
    print("=" * 40)
    
    # Show download instructions
    download_kaggle_model()
    
    # Try to load model
    model = FEREmotionCNN()
    
    if model.model is not None:
        # Test with dummy image
        import cv2
        dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        emotion, conf, probs = model.predict(dummy_img)
        print(f"\nTest prediction: {emotion} ({conf:.2%})")
        print(f"All probabilities: {dict(zip(FER_EMOTION_LABELS, probs))}")
