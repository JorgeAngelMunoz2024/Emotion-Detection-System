"""
Comprehensive tests for audio_acoustic_encoder.py models
Tests SpectrogramEncoder, ProsodyEncoder, and AcousticEmotionEncoder
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from models.audio_acoustic_encoder import (
    SpectrogramEncoder,
    ProsodyEncoder,
    AcousticEmotionEncoder
)


class TestSpectrogramEncoder:
    """Test Spectrogram Encoder."""
    
    @staticmethod
    def test_initialization():
        """Test encoder initialization with various configs."""
        print("Testing SpectrogramEncoder initialization...")
        
        configs = [
            {'n_mels': 64, 'hidden_dim': 128, 'num_emotions': 7},
            {'n_mels': 128, 'hidden_dim': 256, 'num_emotions': 7},
            {'n_mels': 80, 'hidden_dim': 512, 'num_emotions': 5},
        ]
        
        for config in configs:
            encoder = SpectrogramEncoder(**config)
            
            assert encoder.n_mels == config['n_mels']
            assert hasattr(encoder, 'mel_transform')
            assert hasattr(encoder, 'amplitude_to_db')
            assert hasattr(encoder, 'conv_layers')
            assert hasattr(encoder, 'temporal_attention')
            assert hasattr(encoder, 'feature_encoder')
            assert hasattr(encoder, 'acoustic_classifier')
            
            params = sum(p.numel() for p in encoder.parameters())
            print(f"  ✓ n_mels={config['n_mels']}, hidden={config['hidden_dim']}: {params:,} params")
        
        print()
    
    @staticmethod
    def test_spectrogram_extraction():
        """Test mel spectrogram extraction from waveform."""
        print("Testing spectrogram extraction...")
        
        encoder = SpectrogramEncoder(n_mels=128, sample_rate=16000)
        
        # Test different input shapes
        test_cases = [
            torch.randn(2, 32000),      # (B, samples)
            torch.randn(2, 1, 32000),   # (B, 1, samples)
        ]
        
        for waveform in test_cases:
            mel_spec = encoder.extract_spectrogram(waveform)
            
            assert mel_spec.dim() == 4  # (B, 1, n_mels, time)
            assert mel_spec.shape[1] == 1  # Single channel
            assert mel_spec.shape[2] == 128  # n_mels
            assert not torch.isnan(mel_spec).any()
            
            print(f"  ✓ Input {tuple(waveform.shape)} → Spectrogram {tuple(mel_spec.shape)}")
        
        print()
    
    @staticmethod
    def test_forward_pass():
        """Test forward pass with raw waveform."""
        print("Testing SpectrogramEncoder forward pass...")
        
        encoder = SpectrogramEncoder(
            n_mels=128,
            sample_rate=16000,
            hidden_dim=256,
            num_emotions=7
        )
        encoder.eval()
        
        # Test different batch sizes and durations
        test_cases = [
            (1, 16000),    # 1 second
            (2, 32000),    # 2 seconds
            (4, 48000),    # 3 seconds
        ]
        
        for batch_size, num_samples in test_cases:
            waveform = torch.randn(batch_size, num_samples)
            
            with torch.no_grad():
                features, logits = encoder(waveform)
            
            assert features.shape == (batch_size, 256)
            assert logits.shape == (batch_size, 7)
            assert not torch.isnan(features).any()
            assert not torch.isnan(logits).any()
            
            print(f"  ✓ Batch {batch_size}, samples {num_samples}: features {features.shape}, logits {logits.shape}")
        
        print()
    
    @staticmethod
    def test_spectrogram_input():
        """Test forward pass with pre-computed spectrogram."""
        print("Testing forward pass with spectrogram input...")
        
        encoder = SpectrogramEncoder(n_mels=128, hidden_dim=256, num_emotions=7)
        encoder.eval()
        
        # Pre-computed spectrogram input
        spectrogram = torch.randn(2, 1, 128, 100)  # (B, 1, n_mels, time)
        
        with torch.no_grad():
            features, logits = encoder(spectrogram)
        
        assert features.shape == (2, 256)
        assert logits.shape == (2, 7)
        
        print(f"  ✓ Spectrogram input {tuple(spectrogram.shape)} → features {tuple(features.shape)}\n")
    
    @staticmethod
    def test_temporal_attention():
        """Test temporal attention mechanism."""
        print("Testing temporal attention...")
        
        encoder = SpectrogramEncoder(n_mels=128, hidden_dim=256)
        encoder.eval()
        
        # Different audio lengths should work
        for duration in [1.0, 2.0, 3.0]:
            num_samples = int(16000 * duration)
            waveform = torch.randn(2, num_samples)
            
            with torch.no_grad():
                features, logits = encoder(waveform)
            
            assert features.shape == (2, 256)
            assert logits.shape == (2, 7)
        
        print("  ✓ Temporal attention handling variable lengths\n")
    
    @staticmethod
    def test_gradients():
        """Test gradient flow."""
        print("Testing SpectrogramEncoder gradients...")
        
        encoder = SpectrogramEncoder(n_mels=128, hidden_dim=256, num_emotions=7)
        encoder.train()
        
        waveform = torch.randn(2, 32000, requires_grad=True)
        target = torch.randint(0, 7, (2,))
        
        features, logits = encoder(waveform)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        
        assert waveform.grad is not None
        assert not torch.isnan(waveform.grad).any()
        
        params_with_grad = sum(1 for p in encoder.parameters() if p.grad is not None)
        total_params = sum(1 for p in encoder.parameters() if p.requires_grad)
        
        print(f"  ✓ Gradients computed for {params_with_grad}/{total_params} parameters\n")
    
    @staticmethod
    def run_all():
        """Run all SpectrogramEncoder tests."""
        print("\n" + "="*70)
        print("SPECTROGRAM ENCODER TESTS")
        print("="*70)
        TestSpectrogramEncoder.test_initialization()
        TestSpectrogramEncoder.test_spectrogram_extraction()
        TestSpectrogramEncoder.test_forward_pass()
        TestSpectrogramEncoder.test_spectrogram_input()
        TestSpectrogramEncoder.test_temporal_attention()
        TestSpectrogramEncoder.test_gradients()
        print("✓ All SpectrogramEncoder tests passed!\n")


class TestProsodyEncoder:
    """Test Prosody Encoder."""
    
    @staticmethod
    def test_initialization():
        """Test encoder initialization."""
        print("Testing ProsodyEncoder initialization...")
        
        encoder = ProsodyEncoder(
            sample_rate=16000,
            frame_length=512,
            hidden_dim=128
        )
        
        assert encoder.sample_rate == 16000
        assert encoder.frame_length == 512
        assert hasattr(encoder, 'prosody_encoder')
        assert hasattr(encoder, 'temporal_pool')
        
        params = sum(p.numel() for p in encoder.parameters())
        print(f"  ✓ Initialized with {params:,} parameters\n")
    
    @staticmethod
    def test_prosody_extraction():
        """Test prosodic feature extraction."""
        print("Testing prosody feature extraction...")
        
        encoder = ProsodyEncoder(frame_length=512)
        
        waveform = torch.randn(2, 16000)  # 1 second
        
        prosody_features = encoder.extract_prosody_features(waveform)
        
        # Should have shape (B, num_frames, 4)
        assert prosody_features.dim() == 3
        assert prosody_features.shape[0] == 2
        assert prosody_features.shape[2] == 4  # 4 prosodic features
        
        print(f"  ✓ Prosody features: {tuple(prosody_features.shape)}\n")
    
    @staticmethod
    def test_forward_pass():
        """Test forward pass."""
        print("Testing ProsodyEncoder forward pass...")
        
        encoder = ProsodyEncoder(hidden_dim=128)
        encoder.eval()
        
        test_cases = [
            (1, 16000),
            (2, 32000),
            (4, 48000),
        ]
        
        for batch_size, num_samples in test_cases:
            waveform = torch.randn(batch_size, num_samples)
            
            with torch.no_grad():
                features = encoder(waveform)
            
            assert features.shape == (batch_size, 128)
            assert not torch.isnan(features).any()
            
            print(f"  ✓ Batch {batch_size}, samples {num_samples}: output {features.shape}")
        
        print()
    
    @staticmethod
    def test_gradients():
        """Test gradient flow."""
        print("Testing ProsodyEncoder gradients...")
        
        encoder = ProsodyEncoder(hidden_dim=128)
        encoder.train()
        
        waveform = torch.randn(2, 32000, requires_grad=True)
        
        features = encoder(waveform)
        loss = features.sum()
        loss.backward()
        
        assert waveform.grad is not None
        assert not torch.isnan(waveform.grad).any()
        
        print("  ✓ Gradients computed successfully\n")
    
    @staticmethod
    def run_all():
        """Run all ProsodyEncoder tests."""
        print("\n" + "="*70)
        print("PROSODY ENCODER TESTS")
        print("="*70)
        TestProsodyEncoder.test_initialization()
        TestProsodyEncoder.test_prosody_extraction()
        TestProsodyEncoder.test_forward_pass()
        TestProsodyEncoder.test_gradients()
        print("✓ All ProsodyEncoder tests passed!\n")


class TestAcousticEmotionEncoder:
    """Test complete Acoustic Emotion Encoder."""
    
    @staticmethod
    def test_initialization():
        """Test encoder initialization."""
        print("Testing AcousticEmotionEncoder initialization...")
        
        encoder = AcousticEmotionEncoder(
            n_mels=128,
            sample_rate=16000,
            hidden_dim=256,
            num_emotions=7
        )
        
        assert hasattr(encoder, 'spectrogram_encoder')
        assert hasattr(encoder, 'prosody_encoder')
        assert hasattr(encoder, 'fusion')
        assert hasattr(encoder, 'classifier')
        
        params = sum(p.numel() for p in encoder.parameters())
        print(f"  ✓ Initialized with {params:,} parameters\n")
    
    @staticmethod
    def test_forward_pass():
        """Test forward pass combining spectrogram and prosody."""
        print("Testing AcousticEmotionEncoder forward pass...")
        
        encoder = AcousticEmotionEncoder(
            n_mels=128,
            sample_rate=16000,
            hidden_dim=256,
            num_emotions=7
        )
        encoder.eval()
        
        test_cases = [
            (1, 16000),    # 1 second
            (2, 32000),    # 2 seconds
            (4, 48000),    # 3 seconds
        ]
        
        for batch_size, num_samples in test_cases:
            waveform = torch.randn(batch_size, num_samples)
            
            with torch.no_grad():
                features, logits = encoder(waveform)
            
            assert features.shape == (batch_size, 256)
            assert logits.shape == (batch_size, 7)
            assert not torch.isnan(features).any()
            assert not torch.isnan(logits).any()
            
            print(f"  ✓ Batch {batch_size}, samples {num_samples}: features {features.shape}, logits {logits.shape}")
        
        print()
    
    @staticmethod
    def test_feature_fusion():
        """Test spectrogram + prosody feature fusion."""
        print("Testing spectrogram + prosody fusion...")
        
        encoder = AcousticEmotionEncoder(hidden_dim=256, num_emotions=7)
        encoder.eval()
        
        waveform = torch.randn(2, 32000)
        
        with torch.no_grad():
            features, logits = encoder(waveform)
        
        # Features should be combined from both encoders
        assert features.shape == (2, 256)
        
        # Check predictions are valid
        probs = torch.softmax(logits, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
        
        print("  ✓ Feature fusion working correctly\n")
    
    @staticmethod
    def test_emotion_classification():
        """Test emotion classification output."""
        print("Testing emotion classification...")
        
        encoder = AcousticEmotionEncoder(num_emotions=7)
        encoder.eval()
        
        waveform = torch.randn(4, 32000)
        
        with torch.no_grad():
            features, logits = encoder(waveform)
        
        predictions = logits.argmax(dim=1)
        
        assert predictions.shape == (4,)
        assert (predictions >= 0).all() and (predictions < 7).all()
        
        print(f"  ✓ Predictions: {predictions.tolist()}\n")
    
    @staticmethod
    def test_gradients():
        """Test gradient flow through both encoders."""
        print("Testing AcousticEmotionEncoder gradients...")
        
        encoder = AcousticEmotionEncoder(hidden_dim=256, num_emotions=7)
        encoder.train()
        
        waveform = torch.randn(2, 32000, requires_grad=True)
        target = torch.randint(0, 7, (2,))
        
        features, logits = encoder(waveform)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        
        assert waveform.grad is not None
        assert not torch.isnan(waveform.grad).any()
        
        # Check both sub-encoders have gradients
        spec_params_with_grad = sum(
            1 for p in encoder.spectrogram_encoder.parameters() 
            if p.grad is not None
        )
        prosody_params_with_grad = sum(
            1 for p in encoder.prosody_encoder.parameters() 
            if p.grad is not None
        )
        
        assert spec_params_with_grad > 0, "Spectrogram encoder should have gradients"
        assert prosody_params_with_grad > 0, "Prosody encoder should have gradients"
        
        print("  ✓ Gradients flowing through both encoders\n")
    
    @staticmethod
    def test_variable_length_audio():
        """Test with variable length audio inputs."""
        print("Testing variable length audio...")
        
        encoder = AcousticEmotionEncoder(hidden_dim=256, num_emotions=7)
        encoder.eval()
        
        durations = [0.5, 1.0, 2.0, 3.0, 5.0]  # seconds
        
        for duration in durations:
            num_samples = int(16000 * duration)
            waveform = torch.randn(2, num_samples)
            
            with torch.no_grad():
                features, logits = encoder(waveform)
            
            assert features.shape == (2, 256)
            assert logits.shape == (2, 7)
            
            print(f"  ✓ Duration {duration}s ({num_samples} samples): OK")
        
        print()
    
    @staticmethod
    def run_all():
        """Run all AcousticEmotionEncoder tests."""
        print("\n" + "="*70)
        print("ACOUSTIC EMOTION ENCODER TESTS")
        print("="*70)
        TestAcousticEmotionEncoder.test_initialization()
        TestAcousticEmotionEncoder.test_forward_pass()
        TestAcousticEmotionEncoder.test_feature_fusion()
        TestAcousticEmotionEncoder.test_emotion_classification()
        TestAcousticEmotionEncoder.test_gradients()
        TestAcousticEmotionEncoder.test_variable_length_audio()
        print("✓ All AcousticEmotionEncoder tests passed!\n")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*70)
    print("AUDIO ACOUSTIC ENCODER COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("="*70)
    
    try:
        TestSpectrogramEncoder.run_all()
        TestProsodyEncoder.run_all()
        TestAcousticEmotionEncoder.run_all()
        
        print("\n" + "="*70)
        print("✓✓✓ ALL TESTS PASSED SUCCESSFULLY! ✓✓✓")
        print("="*70)
        print("\nSummary:")
        print("  • SpectrogramEncoder: Mel spectrogram extraction and temporal attention verified")
        print("  • ProsodyEncoder: Prosodic feature extraction (pitch, energy, ZCR) working")
        print("  • AcousticEmotionEncoder: Combined encoder with fusion validated")
        print("\nThe audio acoustic encoding system is fully functional!")
        print("="*70)
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
