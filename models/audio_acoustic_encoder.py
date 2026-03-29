"""
Audio Acoustic Feature Encoder (Trainable from scratch)

Processes raw audio waveforms to extract acoustic features (tone, pitch, prosody)
Uses spectrograms for emotion detection from vocal characteristics.

NOTE: This model is designed for TRAINING from scratch on your dataset.
For pretrained speech emotion recognition, use:
    from models.speech_emotion_recognition import SpeechEmotionRecognizer
    
The SpeechEmotionRecognizer uses pretrained Wav2Vec2 models fine-tuned on 
emotion datasets (RAVDESS, IEMOCAP) and provides better out-of-box performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from typing import Tuple, Optional


class SpectrogramEncoder(nn.Module):
    """
    Encodes mel-spectrograms into acoustic features.
    Captures tone, pitch, prosody, and vocal emotional characteristics.
    
    This is a trainable encoder - initialize with random weights and train on your data.
    """
    def __init__(
        self,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 16000,
        hidden_dim: int = 256,
        num_emotions: int = 7,
        dropout: float = 0.3
    ):
        """
        Args:
            n_mels: Number of mel filterbanks
            n_fft: FFT window size
            hop_length: Hop length for STFT
            sample_rate: Audio sample rate
            hidden_dim: Hidden dimension for encoder
            num_emotions: Number of emotion classes
            dropout: Dropout rate
        """
        super(SpectrogramEncoder, self).__init__()
        
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        
        # Convert to dB scale
        self.amplitude_to_db = T.AmplitudeToDB()
        
        # CNN for spectrogram processing (2D convolutions)
        self.conv_layers = nn.Sequential(
            # Layer 1: (B, 1, n_mels, time) -> (B, 32, n_mels/2, time/2)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            # Layer 2: -> (B, 64, n_mels/4, time/4)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            # Layer 3: -> (B, 128, n_mels/8, time/8)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            # Layer 4: -> (B, 256, n_mels/16, time/16)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        
        # Calculate flattened size after convolutions
        # With 4 stride-2 convs: n_mels/16 * time/16 * 256
        self.flatten_dim = (n_mels // 16) * 256
        
        # Temporal attention for time dimension
        self.temporal_attention = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Emotion classifier from acoustic features
        self.acoustic_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_emotions)
        )
    
    def extract_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract mel-spectrogram from raw audio waveform.
        
        Args:
            waveform: Raw audio (B, samples) or (B, 1, samples)
            
        Returns:
            Mel-spectrogram (B, 1, n_mels, time)
        """
        # Ensure correct shape
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # Add channel dim
        elif waveform.dim() == 3 and waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)  # Mono
        
        # Extract mel spectrogram
        mel_spec = self.mel_transform(waveform.squeeze(1))  # (B, n_mels, time)
        
        # Convert to dB
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # Add channel dimension
        mel_spec_db = mel_spec_db.unsqueeze(1)  # (B, 1, n_mels, time)
        
        return mel_spec_db
    
    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process audio waveform to extract acoustic emotion features.
        
        Args:
            waveform: Raw audio waveform (B, samples) or spectrogram (B, 1, n_mels, time)
            
        Returns:
            (acoustic_features, acoustic_logits)
            - acoustic_features: Encoded features (B, hidden_dim)
            - acoustic_logits: Emotion predictions (B, num_emotions)
        """
        B = waveform.shape[0]
        
        # Check if input is already a spectrogram
        if waveform.dim() == 4:
            mel_spec = waveform
        else:
            # Extract spectrogram from waveform
            mel_spec = self.extract_spectrogram(waveform)
        
        # CNN feature extraction
        conv_features = self.conv_layers(mel_spec)  # (B, 256, n_mels/16, time/16)
        
        # Reshape for temporal attention: (B, time/16, n_mels/16 * 256)
        B, C, H, W = conv_features.shape
        conv_features = conv_features.permute(0, 3, 1, 2)  # (B, time/16, 256, n_mels/16)
        conv_features = conv_features.reshape(B, W, -1)  # (B, time/16, flatten_dim)
        
        # Apply temporal attention
        attention_scores = self.temporal_attention(conv_features)  # (B, time/16, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (B, time/16, 1)
        
        # Weighted sum over time
        attended_features = (conv_features * attention_weights).sum(dim=1)  # (B, flatten_dim)
        
        # Encode to hidden dimension
        acoustic_features = self.feature_encoder(attended_features)  # (B, hidden_dim)
        
        # Classify emotion from acoustics
        acoustic_logits = self.acoustic_classifier(acoustic_features)  # (B, num_emotions)
        
        return acoustic_features, acoustic_logits


class ProsodyEncoder(nn.Module):
    """
    Encodes prosodic features (pitch, energy, speaking rate).
    Complements spectrogram-based encoding with handcrafted acoustic features.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 512,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        """
        Args:
            sample_rate: Audio sample rate
            frame_length: Frame length for feature extraction
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super(ProsodyEncoder, self).__init__()
        
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        
        # Input: [pitch, energy, zcr, spectral_centroid] = 4 features per frame
        self.prosody_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Temporal aggregation
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
    
    def extract_prosody_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract prosodic features from waveform.
        
        Args:
            waveform: Raw audio (B, samples)
            
        Returns:
            Prosody features (B, num_frames, 4)
        """
        # This is a placeholder - in production you'd use librosa or similar
        # For now, we'll extract simple frame-level statistics
        
        B, samples = waveform.shape
        frame_length = self.frame_length
        num_frames = samples // frame_length
        
        # Reshape into frames
        frames = waveform[:, :num_frames * frame_length].reshape(B, num_frames, frame_length)
        
        # Extract simple features (placeholder)
        # In production: use librosa.feature for accurate prosody
        energy = frames.pow(2).mean(dim=2)  # (B, num_frames)
        zcr = (frames[:, :, 1:] * frames[:, :, :-1] < 0).float().mean(dim=2)  # (B, num_frames)
        
        # Simple pitch proxy (mean and std of signal in frame)
        pitch_proxy = frames.mean(dim=2)  # (B, num_frames)
        spectral_centroid_proxy = frames.std(dim=2)  # (B, num_frames)
        
        # Stack features
        prosody = torch.stack([pitch_proxy, energy, zcr, spectral_centroid_proxy], dim=2)  # (B, num_frames, 4)
        
        return prosody
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Encode prosody features from waveform.
        
        Args:
            waveform: Raw audio (B, samples)
            
        Returns:
            Prosody features (B, hidden_dim)
        """
        # Extract prosody
        prosody = self.extract_prosody_features(waveform)  # (B, num_frames, 4)
        
        B, T, F = prosody.shape
        
        # Encode each frame
        prosody_flat = prosody.reshape(B * T, F)
        encoded_flat = self.prosody_encoder(prosody_flat)  # (B*T, hidden_dim)
        encoded = encoded_flat.reshape(B, T, -1)  # (B, T, hidden_dim)
        
        # Temporal pooling
        encoded = encoded.permute(0, 2, 1)  # (B, hidden_dim, T)
        pooled = self.temporal_pool(encoded).squeeze(-1)  # (B, hidden_dim)
        
        return pooled


class AcousticEmotionEncoder(nn.Module):
    """
    Complete acoustic emotion encoder combining spectrogram and prosody.
    Unsupervised learning from raw audio to capture vocal tone and characteristics.
    """
    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        hidden_dim: int = 256,
        num_emotions: int = 7,
        dropout: float = 0.3
    ):
        """
        Args:
            n_mels: Number of mel filterbanks
            sample_rate: Audio sample rate
            hidden_dim: Hidden dimension
            num_emotions: Number of emotion classes
            dropout: Dropout rate
        """
        super(AcousticEmotionEncoder, self).__init__()
        
        self.spectrogram_encoder = SpectrogramEncoder(
            n_mels=n_mels,
            sample_rate=sample_rate,
            hidden_dim=hidden_dim,
            num_emotions=num_emotions,
            dropout=dropout
        )
        
        self.prosody_encoder = ProsodyEncoder(
            sample_rate=sample_rate,
            hidden_dim=hidden_dim // 2,
            dropout=dropout
        )
        
        # Combine spectrogram + prosody features
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Final emotion classifier from combined acoustics
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_emotions)
        )
    
    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract emotion from acoustic characteristics.
        
        Args:
            waveform: Raw audio waveform (B, samples)
            
        Returns:
            (acoustic_features, acoustic_logits)
            - acoustic_features: Combined features (B, hidden_dim)
            - acoustic_logits: Emotion predictions (B, num_emotions)
        """
        # Extract spectrogram-based features
        spec_features, spec_logits = self.spectrogram_encoder(waveform)  # (B, hidden_dim), (B, 7)
        
        # Extract prosody features
        prosody_features = self.prosody_encoder(waveform)  # (B, hidden_dim/2)
        
        # Combine both
        combined = torch.cat([spec_features, prosody_features], dim=1)  # (B, hidden_dim*1.5)
        acoustic_features = self.fusion(combined)  # (B, hidden_dim)
        
        # Classify from combined acoustics
        acoustic_logits = self.classifier(acoustic_features)  # (B, num_emotions)
        
        return acoustic_features, acoustic_logits


def test_acoustic_encoder():
    """Test the acoustic emotion encoder."""
    print("=" * 70)
    print("Testing Acoustic Emotion Encoder")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Create encoder
    encoder = AcousticEmotionEncoder(
        n_mels=128,
        sample_rate=16000,
        hidden_dim=256,
        num_emotions=7
    ).to(device)
    
    # Test input: 2 seconds of audio at 16kHz
    batch_size = 2
    duration = 2.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    
    waveform = torch.randn(batch_size, num_samples).to(device)
    
    print(f"Input waveform: {waveform.shape}")
    print(f"Duration: {duration}s at {sample_rate}Hz\n")
    
    # Forward pass
    encoder.eval()
    with torch.no_grad():
        acoustic_features, acoustic_logits = encoder(waveform)
    
    print("Output:")
    print(f"  Acoustic features: {acoustic_features.shape}")
    print(f"  Acoustic logits: {acoustic_logits.shape}")
    print(f"  Predicted emotions: {acoustic_logits.argmax(dim=1)}")
    
    # Test gradients
    encoder.train()
    waveform.requires_grad = True
    features, logits = encoder(waveform)
    loss = logits.sum()
    loss.backward()
    
    print(f"\n✓ Gradients computed successfully")
    print(f"✓ All tests passed!")


if __name__ == "__main__":
    test_acoustic_encoder()
