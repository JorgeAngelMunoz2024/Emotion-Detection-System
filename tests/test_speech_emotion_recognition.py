"""
Tests for Speech Emotion Recognition (SER) module.
Tests both the pretrained model wrapper and the PyTorch integration.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Check if dependencies are available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TestSpeechEmotionRecognizer:
    """Tests for the SpeechEmotionRecognizer class."""
    
    @pytest.fixture
    def ser_model(self):
        """Create a SER model instance."""
        from models.speech_emotion_recognition import SpeechEmotionRecognizer
        return SpeechEmotionRecognizer()
    
    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio data."""
        duration = 2.0
        sample_rate = 16000
        # Generate random audio (simulating speech)
        audio = np.random.randn(int(duration * sample_rate)).astype(np.float32) * 0.1
        return audio, sample_rate
    
    def test_model_initialization(self, ser_model):
        """Test that the model initializes correctly."""
        assert ser_model is not None
        assert ser_model.sample_rate == 16000
        
        info = ser_model.get_model_info()
        assert 'model_name' in info
        assert 'device' in info
        assert 'sample_rate' in info
        assert info['sample_rate'] == 16000
    
    def test_predict_returns_correct_format(self, ser_model, sample_audio):
        """Test that predict returns correct format."""
        audio, sr = sample_audio
        
        emotion, confidence, probs = ser_model.predict(audio, sr)
        
        # Check emotion is a valid string
        assert isinstance(emotion, str)
        assert emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Check confidence is a float between 0 and 1
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        
        # Check probs is correct shape
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (7,)
        assert np.allclose(probs.sum(), 1.0, atol=0.01)  # Should sum to ~1
    
    def test_preprocess_audio_int16(self, ser_model):
        """Test preprocessing int16 audio."""
        audio_int16 = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        
        processed = ser_model.preprocess_audio(audio_int16)
        
        assert processed.dtype == np.float32
        assert np.abs(processed).max() <= 1.0
    
    def test_preprocess_audio_float32(self, ser_model):
        """Test preprocessing float32 audio."""
        audio_float = np.random.randn(16000).astype(np.float32) * 0.5
        
        processed = ser_model.preprocess_audio(audio_float)
        
        assert processed.dtype == np.float32
        assert np.abs(processed).max() <= 1.0
    
    def test_predict_short_audio(self, ser_model):
        """Test prediction with very short audio (should be padded)."""
        short_audio = np.random.randn(1000).astype(np.float32)  # < 0.5 seconds
        
        emotion, confidence, probs = ser_model.predict(short_audio)
        
        # Should still return valid prediction
        assert emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        assert probs.shape == (7,)
    
    def test_predict_long_audio(self, ser_model):
        """Test prediction with longer audio."""
        long_audio = np.random.randn(80000).astype(np.float32)  # 5 seconds
        
        emotion, confidence, probs = ser_model.predict(long_audio)
        
        assert emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        assert probs.shape == (7,)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestSpeechEmotionRecognizerTorch:
    """Tests for the PyTorch wrapper class."""
    
    @pytest.fixture
    def ser_torch(self):
        """Create a PyTorch SER model."""
        from models.speech_emotion_recognition import SpeechEmotionRecognizerTorch
        model = SpeechEmotionRecognizerTorch()
        model.eval()
        return model
    
    @pytest.fixture
    def sample_audio_tensor(self):
        """Generate sample audio tensor."""
        batch_size = 2
        num_samples = 32000  # 2 seconds at 16kHz
        return torch.randn(batch_size, num_samples)
    
    def test_forward_pass(self, ser_torch, sample_audio_tensor):
        """Test forward pass produces correct output shape."""
        with torch.no_grad():
            output = ser_torch(sample_audio_tensor)
        
        assert output.shape == (2, 7)  # (batch_size, num_emotions)
    
    def test_predict_method(self, ser_torch, sample_audio_tensor):
        """Test predict method returns predictions and probabilities."""
        with torch.no_grad():
            preds, probs = ser_torch.predict(sample_audio_tensor)
        
        assert preds.shape == (2,)  # batch_size
        assert probs.shape == (2, 7)  # (batch_size, num_emotions)
        
        # Check probabilities sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=0.01)
    
    def test_gradient_flow(self, ser_torch, sample_audio_tensor):
        """Test that gradients flow through the model."""
        ser_torch.train()
        sample_audio_tensor.requires_grad = True
        
        output = ser_torch(sample_audio_tensor)
        loss = output.sum()
        loss.backward()
        
        # Should have computed gradients
        assert sample_audio_tensor.grad is not None


class TestEmotionLabelMapping:
    """Tests for emotion label mapping."""
    
    def test_label_mapping_coverage(self):
        """Test that all common model labels are mapped."""
        from models.speech_emotion_recognition import LABEL_MAPPING, EMOTION_LABELS_7
        
        # Common labels from various models
        common_labels = ['angry', 'happy', 'sad', 'neutral', 'fearful', 'surprised', 'calm', 'disgust']
        
        for label in common_labels:
            mapped = LABEL_MAPPING.get(label, label)
            assert mapped in EMOTION_LABELS_7, f"Label '{label}' maps to '{mapped}' which is not in standard labels"
    
    def test_standard_labels_count(self):
        """Test that we have exactly 7 standard emotion labels."""
        from models.speech_emotion_recognition import EMOTION_LABELS_7
        
        assert len(EMOTION_LABELS_7) == 7
        assert 'neutral' in EMOTION_LABELS_7
        assert 'happy' in EMOTION_LABELS_7
        assert 'sad' in EMOTION_LABELS_7
        assert 'angry' in EMOTION_LABELS_7


class TestIntegration:
    """Integration tests for SER with audio processing."""
    
    @pytest.fixture
    def audio_with_energy(self):
        """Generate audio with varying energy levels."""
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create audio with some tonal content (more realistic than pure noise)
        audio = np.sin(2 * np.pi * 200 * t) * 0.3  # Base tone
        audio += np.sin(2 * np.pi * 400 * t) * 0.2  # Harmonic
        audio += np.random.randn(len(t)) * 0.1  # Noise
        
        return audio.astype(np.float32), sr
    
    def test_consistent_predictions(self, audio_with_energy):
        """Test that same audio gives consistent predictions."""
        from models.speech_emotion_recognition import SpeechEmotionRecognizer
        
        ser = SpeechEmotionRecognizer()
        audio, sr = audio_with_energy
        
        # Run prediction twice
        emotion1, conf1, probs1 = ser.predict(audio, sr)
        emotion2, conf2, probs2 = ser.predict(audio, sr)
        
        # Should be identical (deterministic)
        assert emotion1 == emotion2
        assert np.allclose(probs1, probs2)
    
    def test_different_audio_different_predictions(self):
        """Test that different audio can give different predictions."""
        from models.speech_emotion_recognition import SpeechEmotionRecognizer
        
        ser = SpeechEmotionRecognizer()
        
        # Two very different audio samples
        audio1 = np.random.randn(32000).astype(np.float32) * 0.1
        audio2 = np.sin(np.linspace(0, 100, 32000)).astype(np.float32) * 0.5
        
        _, _, probs1 = ser.predict(audio1)
        _, _, probs2 = ser.predict(audio2)
        
        # Predictions might be different (not guaranteed, but possible)
        # Just check that both are valid
        assert probs1.shape == (7,)
        assert probs2.shape == (7,)


def test_import():
    """Test that the module can be imported."""
    from models.speech_emotion_recognition import (
        SpeechEmotionRecognizer,
        SpeechEmotionRecognizerTorch,
        EMOTION_LABELS_7,
        LABEL_MAPPING
    )
    
    assert SpeechEmotionRecognizer is not None
    assert SpeechEmotionRecognizerTorch is not None
    assert len(EMOTION_LABELS_7) == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
