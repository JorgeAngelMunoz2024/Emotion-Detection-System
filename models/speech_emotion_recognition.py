"""
Speech Emotion Recognition (SER) with emotion2vec+

Uses emotion2vec+ (Alibaba DAMO Academy) for speech emotion recognition:
- emotion2vec+ large: ~300M params, 9 emotion classes
- State-of-the-art performance on multiple SER benchmarks
- Universal, robust emotion recognition across languages/environments

Supported emotions: angry, disgust, fear, happy, neutral, sad, surprise
(emotion2vec also outputs 'other' and 'unknown' which map to neutral)

Models are cached in the 'backbones' folder to avoid re-downloading.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import os
import tempfile
import soundfile as sf

# Project root and backbones directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKBONES_DIR = PROJECT_ROOT / 'backbones'
BACKBONES_DIR.mkdir(exist_ok=True)

# Try to import FunASR for emotion2vec
FUNASR_AVAILABLE = False
try:
    from funasr import AutoModel as FunASRAutoModel
    FUNASR_AVAILABLE = True
except ImportError:
    print("Warning: funasr not available. Install with: pip install -U funasr modelscope")

# Try torchaudio for audio processing
try:
    import torchaudio
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


# Emotion label mappings
EMOTION_LABELS_7 = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# emotion2vec+ label mapping (9 classes -> 7 standard)
# emotion2vec+ labels: 0:angry, 1:disgusted, 2:fearful, 3:happy, 4:neutral, 5:other, 6:sad, 7:surprised, 8:unknown
EMOTION2VEC_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', 'unknown']

# Map emotion2vec labels to our standard 7 emotions
LABEL_MAPPING = {
    'angry': 'angry',
    'disgusted': 'disgust',
    'fearful': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'other': 'neutral',      # Map 'other' to neutral
    'sad': 'sad',
    'surprised': 'surprise',
    'unknown': 'neutral',    # Map 'unknown' to neutral
}


class SpeechEmotionRecognizer:
    """
    Pretrained Speech Emotion Recognition using emotion2vec+.
    
    emotion2vec+ is a state-of-the-art SER model from Alibaba DAMO Academy
    trained on 40k+ hours of speech emotion data.
    
    Models available:
    - 'iic/emotion2vec_plus_large' (~300M params, best accuracy)
    - 'iic/emotion2vec_plus_base' (~90M params, faster)
    - 'iic/emotion2vec_plus_seed' (smallest, academic data only)
    """
    
    # Default model - base is good balance of speed and accuracy
    DEFAULT_MODEL = 'iic/emotion2vec_plus_base'
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        sample_rate: int = 16000
    ):
        """
        Initialize the speech emotion recognizer.
        
        Args:
            model_name: ModelScope model name. If None, uses default.
            device: 'cuda', 'cpu', or None for auto-detect
            sample_rate: Expected audio sample rate (16kHz for emotion2vec)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.sample_rate = sample_rate
        self.model = None
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Emotion labels for emotion2vec+
        self.model_labels = EMOTION2VEC_LABELS
        
        # Set up local cache directory
        self.cache_dir = BACKBONES_DIR / 'emotion2vec_models'
        self.cache_dir.mkdir(exist_ok=True)
        
        if not FUNASR_AVAILABLE:
            print("FunASR not available. SER will return neutral.")
            return
        
        self._load_model()
    
    def _load_model(self):
        """Load the emotion2vec+ model."""
        print(f"Loading SER model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Cache dir: {self.cache_dir}")
        
        try:
            # FunASR will handle caching automatically
            # disable_update=True prevents the unnecessary funasr version check that adds startup time
            self.model = FunASRAutoModel(
                model=self.model_name,
                device=self.device,
                disable_update=True
            )
            print(f"  ✓ Loaded emotion2vec+ model")
            print(f"  Labels: {self.model_labels}")
            
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            print("  SER will return neutral predictions.")
            self.model = None
    
    def preprocess_audio(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Preprocess audio for the model.
        
        Args:
            audio: Audio waveform as numpy array
            sr: Sample rate of input audio (resamples if different from 16kHz)
            
        Returns:
            Preprocessed audio array
        """
        sr = sr or self.sample_rate
        
        # Convert to float32 in range [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Resample if needed (emotion2vec expects 16kHz)
        if sr != self.sample_rate and TORCHAUDIO_AVAILABLE:
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            resampler = T.Resample(sr, self.sample_rate)
            audio_tensor = resampler(audio_tensor)
            audio = audio_tensor.squeeze(0).numpy()
        
        return audio
    
    def predict(self, audio: np.ndarray, sr: int = None) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion from audio.
        
        Args:
            audio: Audio waveform (numpy array, float32 or int16)
            sr: Sample rate of input audio
            
        Returns:
            Tuple of (emotion_label, confidence, probabilities)
            - emotion_label: One of the 7 standard emotions
            - confidence: Confidence score (0-1)
            - probabilities: Array of 7 probabilities for each emotion
        """
        # Default return if model not loaded
        default_probs = np.zeros(7)
        default_probs[6] = 1.0  # neutral
        
        if self.model is None:
            return 'neutral', 1.0, default_probs
        
        # Preprocess audio
        audio = self.preprocess_audio(audio, sr)
        
        # Calculate RMS for debug
        rms = np.sqrt(np.mean(audio ** 2))
        print(f"[SER DEBUG] Audio len: {len(audio)}, min: {audio.min():.4f}, max: {audio.max():.4f}, std: {audio.std():.4f}, rms: {rms:.4f}")
        
        # Minimum audio length (0.5 seconds)
        min_samples = int(0.5 * self.sample_rate)
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))
        
        try:
            # emotion2vec requires audio file or path - save to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
                sf.write(temp_path, audio, self.sample_rate)
            
            try:
                # Run emotion2vec+ inference
                result = self.model.generate(
                    temp_path,
                    granularity="utterance",
                    extract_embedding=False
                )
                
                print(f"[SER DEBUG] Raw result: {result}")
                
                # Parse result - emotion2vec returns list of dicts
                if result and len(result) > 0:
                    res = result[0]
                    
                    # Get scores - emotion2vec returns 'scores' as list
                    if 'scores' in res:
                        scores = res['scores']
                        # Scores correspond to emotion2vec labels (9 classes)
                        model_probs = {}
                        for i, label in enumerate(self.model_labels):
                            if i < len(scores):
                                model_probs[label] = float(scores[i])
                        
                        print(f"[SER DEBUG] Model probs: {model_probs}")
                        
                        # Map to standard 7 emotions
                        standard_probs = np.zeros(7)
                        for model_label, prob in model_probs.items():
                            std_label = LABEL_MAPPING.get(model_label, 'neutral')
                            if std_label in EMOTION_LABELS_7:
                                idx = EMOTION_LABELS_7.index(std_label)
                                standard_probs[idx] += prob
                        
                        # Normalize
                        if standard_probs.sum() > 0:
                            standard_probs = standard_probs / standard_probs.sum()
                        else:
                            standard_probs[6] = 1.0  # neutral
                        
                        print(f"[SER DEBUG] Standard probs: {list(zip(EMOTION_LABELS_7, standard_probs))}")
                        
                        # Get top emotion
                        emotion_idx = np.argmax(standard_probs)
                        emotion = EMOTION_LABELS_7[emotion_idx]
                        confidence = standard_probs[emotion_idx]
                        
                        return emotion, float(confidence), standard_probs
                    
                    # Alternative: check for 'labels' field
                    elif 'labels' in res:
                        top_label = res['labels'][0] if res['labels'] else 'neutral'
                        std_label = LABEL_MAPPING.get(top_label.lower(), 'neutral')
                        emotion_idx = EMOTION_LABELS_7.index(std_label)
                        standard_probs = np.zeros(7)
                        standard_probs[emotion_idx] = 1.0
                        return std_label, 1.0, standard_probs
                
                return 'neutral', 1.0, default_probs
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            print(f"SER prediction error: {e}")
            import traceback
            traceback.print_exc()
            return 'neutral', 1.0, default_probs
    
    def predict_from_file(self, audio_path: str) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion from audio file.
        
        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            
        Returns:
            Tuple of (emotion_label, confidence, probabilities)
        """
        default_probs = np.zeros(7)
        default_probs[6] = 1.0  # neutral
        
        if self.model is None:
            return 'neutral', 1.0, default_probs
        
        try:
            # emotion2vec can directly process file paths
            result = self.model.generate(
                audio_path,
                granularity="utterance", 
                extract_embedding=False
            )
            
            if result and len(result) > 0:
                res = result[0]
                if 'scores' in res:
                    scores = res['scores']
                    model_probs = {label: float(scores[i]) for i, label in enumerate(self.model_labels) if i < len(scores)}
                    
                    # Map to standard 7 emotions
                    standard_probs = np.zeros(7)
                    for model_label, prob in model_probs.items():
                        std_label = LABEL_MAPPING.get(model_label, 'neutral')
                        if std_label in EMOTION_LABELS_7:
                            idx = EMOTION_LABELS_7.index(std_label)
                            standard_probs[idx] += prob
                    
                    if standard_probs.sum() > 0:
                        standard_probs = standard_probs / standard_probs.sum()
                    else:
                        standard_probs[6] = 1.0
                    
                    emotion_idx = np.argmax(standard_probs)
                    emotion = EMOTION_LABELS_7[emotion_idx]
                    confidence = standard_probs[emotion_idx]
                    
                    return emotion, float(confidence), standard_probs
            
            return 'neutral', 1.0, default_probs
            
        except Exception as e:
            print(f"SER prediction error: {e}")
            return 'neutral', 1.0, default_probs
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'sample_rate': self.sample_rate,
            'model_labels': self.model_labels,
            'standard_labels': EMOTION_LABELS_7,
            'loaded': self.model is not None
        }


class SpeechEmotionRecognizerTorch(nn.Module):
    """
    PyTorch wrapper for SER - uses emotion2vec internally.
    Note: emotion2vec doesn't support direct PyTorch integration for training,
    but this wrapper provides a compatible interface for feature extraction.
    """
    
    def __init__(
        self,
        model_name: str = SpeechEmotionRecognizer.DEFAULT_MODEL,
        freeze_encoder: bool = True,
        num_emotions: int = 7
    ):
        """
        Args:
            model_name: ModelScope model name
            freeze_encoder: Not used for emotion2vec (model is always frozen)
            num_emotions: Number of output emotion classes
        """
        super().__init__()
        
        self.num_emotions = num_emotions
        self.sample_rate = 16000
        
        # Create the emotion2vec recognizer
        self.recognizer = SpeechEmotionRecognizer(model_name=model_name)
        self.is_pretrained = self.recognizer.model is not None
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns emotion logits.
        
        Args:
            audio: Audio waveform (B, samples) or (B, 1, samples)
            
        Returns:
            Emotion logits (B, num_emotions)
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        batch_size = audio.shape[0]
        logits = torch.zeros(batch_size, self.num_emotions, device=audio.device)
        
        # Process each audio in batch
        for i in range(batch_size):
            audio_np = audio[i].cpu().numpy()
            _, _, probs = self.recognizer.predict(audio_np)
            # Convert probs to logits (inverse softmax approximation)
            logits[i] = torch.from_numpy(np.log(probs + 1e-10)).to(audio.device)
        
        return logits
    
    def predict(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with probabilities.
        
        Args:
            audio: Audio waveform (B, samples)
            
        Returns:
            (predictions, probabilities)
        """
        logits = self.forward(audio)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return preds, probs


def test_speech_emotion_recognition():
    """Test the speech emotion recognition module."""
    print("=" * 70)
    print("Testing Speech Emotion Recognition (SER) with emotion2vec+")
    print("=" * 70)
    
    # Test SpeechEmotionRecognizer
    print("\n1. Testing SpeechEmotionRecognizer...")
    ser = SpeechEmotionRecognizer()
    
    print(f"\nModel info: {ser.get_model_info()}")
    
    # Test with random audio (2 seconds)
    duration = 2.0
    sample_rate = 16000
    audio = np.random.randn(int(duration * sample_rate)).astype(np.float32) * 0.1
    
    print(f"\nTest audio: {len(audio)} samples ({duration}s at {sample_rate}Hz)")
    
    emotion, confidence, probs = ser.predict(audio)
    print(f"\nPrediction:")
    print(f"  Emotion: {emotion}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  All probabilities:")
    for i, label in enumerate(EMOTION_LABELS_7):
        print(f"    {label}: {probs[i]:.2%}")
    
    print("\n✓ SER test complete!")


if __name__ == "__main__":
    test_speech_emotion_recognition()
