"""
Personalized Acoustic Profiling for Emotion Recognition
Uses unsupervised learning to build person-specific acoustic profiles.
Adapts to individual vocal characteristics, tone patterns, and emotional expression styles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import pickle
from pathlib import Path


class PersonalizedAcousticProfile:
    """
    Stores and manages person-specific acoustic characteristics.
    Learns individual baseline tone, pitch range, energy patterns, etc.
    """
    def __init__(self, person_id: str):
        """
        Args:
            person_id: Unique identifier for the person
        """
        self.person_id = person_id
        
        # Baseline acoustic statistics
        self.baseline_pitch = None
        self.baseline_energy = None
        self.baseline_speaking_rate = None
        
        # Statistical distributions per emotion
        self.emotion_acoustics = {
            'happy': {'pitch': [], 'energy': [], 'spectral': []},
            'sad': {'pitch': [], 'energy': [], 'spectral': []},
            'angry': {'pitch': [], 'energy': [], 'spectral': []},
            'fear': {'pitch': [], 'energy': [], 'spectral': []},
            'disgust': {'pitch': [], 'energy': [], 'spectral': []},
            'surprise': {'pitch': [], 'energy': [], 'spectral': []},
            'neutral': {'pitch': [], 'energy': [], 'spectral': []},
        }
        
        # Adaptation parameters
        self.num_samples = 0
        self.adaptation_rate = 0.1  # How quickly to adapt
        
        # Personalized feature embeddings
        self.feature_centroid = None
        self.feature_covariance = None
    
    def update_baseline(self, acoustic_features: Dict[str, float]):
        """
        Update baseline acoustic statistics using exponential moving average.
        
        Args:
            acoustic_features: Dict with 'pitch', 'energy', 'speaking_rate'
        """
        if self.baseline_pitch is None:
            # Initialize
            self.baseline_pitch = acoustic_features['pitch']
            self.baseline_energy = acoustic_features['energy']
            self.baseline_speaking_rate = acoustic_features.get('speaking_rate', 1.0)
        else:
            # Exponential moving average
            alpha = self.adaptation_rate
            self.baseline_pitch = (1 - alpha) * self.baseline_pitch + alpha * acoustic_features['pitch']
            self.baseline_energy = (1 - alpha) * self.baseline_energy + alpha * acoustic_features['energy']
            self.baseline_speaking_rate = (1 - alpha) * self.baseline_speaking_rate + alpha * acoustic_features.get('speaking_rate', self.baseline_speaking_rate)
        
        self.num_samples += 1
    
    def add_emotion_sample(self, emotion: str, acoustic_features: Dict[str, np.ndarray]):
        """
        Add a labeled sample to build emotion-specific distributions.
        
        Args:
            emotion: Emotion label (e.g., 'happy', 'sad')
            acoustic_features: Dict with feature arrays
        """
        if emotion in self.emotion_acoustics:
            self.emotion_acoustics[emotion]['pitch'].append(acoustic_features['pitch'])
            self.emotion_acoustics[emotion]['energy'].append(acoustic_features['energy'])
            if 'spectral' in acoustic_features:
                self.emotion_acoustics[emotion]['spectral'].append(acoustic_features['spectral'])
    
    def get_emotion_statistics(self, emotion: str) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Get mean and std for an emotion's acoustic features.
        
        Args:
            emotion: Emotion label
            
        Returns:
            Dict with (mean, std) for each feature, or None if insufficient data
        """
        if emotion not in self.emotion_acoustics:
            return None
        
        data = self.emotion_acoustics[emotion]
        if len(data['pitch']) < 5:  # Need at least 5 samples
            return None
        
        stats = {}
        for feature_name in ['pitch', 'energy', 'spectral']:
            if len(data[feature_name]) > 0:
                values = np.array(data[feature_name])
                stats[feature_name] = (np.mean(values), np.std(values))
        
        return stats
    
    def compute_deviation(self, acoustic_features: Dict[str, float]) -> Dict[str, float]:
        """
        Compute how much current features deviate from baseline.
        This is person-specific normalization.
        
        Args:
            acoustic_features: Current acoustic features
            
        Returns:
            Deviation scores (z-scores relative to person's baseline)
        """
        if self.baseline_pitch is None:
            return {'pitch': 0.0, 'energy': 0.0, 'speaking_rate': 0.0}
        
        # Z-score normalization relative to person's baseline
        pitch_dev = (acoustic_features['pitch'] - self.baseline_pitch) / (self.baseline_pitch + 1e-6)
        energy_dev = (acoustic_features['energy'] - self.baseline_energy) / (self.baseline_energy + 1e-6)
        
        return {
            'pitch': pitch_dev,
            'energy': energy_dev,
            'speaking_rate': (acoustic_features.get('speaking_rate', 1.0) - self.baseline_speaking_rate) / (self.baseline_speaking_rate + 1e-6)
        }
    
    def save(self, filepath: str):
        """Save profile to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str) -> 'PersonalizedAcousticProfile':
        """Load profile from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class PersonalizedAcousticEncoder(nn.Module):
    """
    Acoustic encoder that adapts to individual users.
    Uses person-specific normalization and feature extraction.
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
        super(PersonalizedAcousticEncoder, self).__init__()
        
        from models.audio_acoustic_encoder import SpectrogramEncoder
        
        # Base spectrogram encoder (shared)
        self.spectrogram_encoder = SpectrogramEncoder(
            n_mels=n_mels,
            sample_rate=sample_rate,
            hidden_dim=hidden_dim,
            num_emotions=num_emotions,
            dropout=dropout
        )
        
        # Person-specific adaptation layers
        self.person_adaptation = nn.ModuleDict()
        
        # Deviation encoder (encodes difference from baseline)
        self.deviation_encoder = nn.Sequential(
            nn.Linear(3, 64),  # pitch_dev, energy_dev, speaking_rate_dev
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Combine base features + person-specific deviations
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Personalized emotion classifier
        self.personalized_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_emotions)
        )
    
    def add_person(self, person_id: str):
        """
        Add a new person to the system.
        Creates person-specific adaptation layers.
        
        Args:
            person_id: Unique identifier for the person
        """
        if person_id not in self.person_adaptation:
            # Create person-specific adaptation layers
            self.person_adaptation[person_id] = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256)
            )
    
    def forward(
        self, 
        waveform: torch.Tensor, 
        person_id: Optional[str] = None,
        deviations: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional person-specific adaptation.
        
        Args:
            waveform: Raw audio waveform (B, samples)
            person_id: Person identifier for personalization
            deviations: Person-specific deviation features (B, 3)
            
        Returns:
            (features, logits)
        """
        # Extract base acoustic features
        base_features, base_logits = self.spectrogram_encoder(waveform)
        
        # If deviations provided, incorporate person-specific info
        if deviations is not None:
            deviation_features = self.deviation_encoder(deviations)
            combined = torch.cat([base_features, deviation_features], dim=1)
            personalized_features = self.feature_fusion(combined)
        else:
            personalized_features = base_features
        
        # Apply person-specific adaptation if available
        if person_id is not None and person_id in self.person_adaptation:
            personalized_features = self.person_adaptation[person_id](personalized_features)
        
        # Personalized classification
        personalized_logits = self.personalized_classifier(personalized_features)
        
        return personalized_features, personalized_logits


class PersonalizedAcousticProfileManager:
    """
    Manages multiple person profiles and provides personalized inference.
    """
    def __init__(self, profiles_dir: str = "acoustic_profiles"):
        """
        Args:
            profiles_dir: Directory to store person profiles
        """
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True, parents=True)
        
        # In-memory cache of profiles
        self.profiles: Dict[str, PersonalizedAcousticProfile] = {}
    
    def get_or_create_profile(self, person_id: str) -> PersonalizedAcousticProfile:
        """
        Get existing profile or create new one.
        
        Args:
            person_id: Unique identifier
            
        Returns:
            PersonalizedAcousticProfile
        """
        if person_id in self.profiles:
            return self.profiles[person_id]
        
        # Try to load from disk
        profile_path = self.profiles_dir / f"{person_id}.pkl"
        if profile_path.exists():
            profile = PersonalizedAcousticProfile.load(str(profile_path))
            self.profiles[person_id] = profile
            return profile
        
        # Create new profile
        profile = PersonalizedAcousticProfile(person_id)
        self.profiles[person_id] = profile
        return profile
    
    def update_profile(
        self, 
        person_id: str, 
        acoustic_features: Dict[str, float],
        emotion: Optional[str] = None
    ):
        """
        Update a person's profile with new observation.
        
        Args:
            person_id: Person identifier
            acoustic_features: Extracted acoustic features
            emotion: Optional emotion label for supervised adaptation
        """
        profile = self.get_or_create_profile(person_id)
        
        # Update baseline
        profile.update_baseline(acoustic_features)
        
        # If labeled, add to emotion-specific distribution
        if emotion is not None:
            profile.add_emotion_sample(emotion, acoustic_features)
        
        # Save updated profile
        self.save_profile(person_id)
    
    def get_personalized_features(
        self, 
        person_id: str, 
        current_acoustics: Dict[str, float]
    ) -> torch.Tensor:
        """
        Compute person-specific normalized features.
        
        Args:
            person_id: Person identifier
            current_acoustics: Current acoustic measurements
            
        Returns:
            Deviation tensor (pitch_dev, energy_dev, rate_dev)
        """
        profile = self.get_or_create_profile(person_id)
        deviations = profile.compute_deviation(current_acoustics)
        
        # Convert to tensor
        deviation_tensor = torch.tensor([
            deviations['pitch_dev'],
            deviations['energy_dev'],
            deviations['speaking_rate_dev']
        ], dtype=torch.float32)
        
        return deviation_tensor
    
    def save_profile(self, person_id: str):
        """Save profile to disk."""
        if person_id in self.profiles:
            profile_path = self.profiles_dir / f"{person_id}.pkl"
            self.profiles[person_id].save(str(profile_path))
    
    def save_all_profiles(self):
        """Save all cached profiles to disk."""
        for person_id in self.profiles:
            self.save_profile(person_id)
    
    def list_profiles(self) -> List[str]:
        """List all available person profiles."""
        profiles = []
        for file in self.profiles_dir.glob("*.pkl"):
            profiles.append(file.stem)
        return profiles
    
    def get_profile_stats(self, person_id: str) -> Dict:
        """
        Get statistics about a person's profile.
        
        Args:
            person_id: Person identifier
            
        Returns:
            Dictionary with profile statistics
        """
        profile = self.get_or_create_profile(person_id)
        
        stats = {
            'person_id': person_id,
            'num_samples': profile.num_samples,
            'baseline_pitch': profile.baseline_pitch,
            'baseline_energy': profile.baseline_energy,
            'baseline_speaking_rate': profile.baseline_speaking_rate,
            'emotions_sampled': {}
        }
        
        # Count samples per emotion
        for emotion, data in profile.emotion_acoustics.items():
            stats['emotions_sampled'][emotion] = len(data['pitch'])
        
        return stats


def extract_acoustic_features(waveform: np.ndarray, sample_rate: int = 16000) -> Dict[str, float]:
    """
    Extract simple acoustic features from waveform.
    In production, use librosa for more accurate features.
    
    Args:
        waveform: Audio waveform (samples,)
        sample_rate: Sample rate
        
    Returns:
        Dict with acoustic features
    """
    # Simple placeholder - in production use librosa
    frame_length = 512
    num_frames = len(waveform) // frame_length
    frames = waveform[:num_frames * frame_length].reshape(num_frames, frame_length)
    
    # Energy
    energy = np.mean(frames ** 2)
    
    # Pitch proxy (autocorrelation-based)
    pitch_proxy = np.mean(np.abs(waveform))
    
    # Speaking rate (zero-crossing rate)
    zcr = np.mean(np.abs(np.diff(np.sign(waveform))))
    speaking_rate = zcr * sample_rate / 2
    
    return {
        'pitch': float(pitch_proxy),
        'energy': float(energy),
        'speaking_rate': float(speaking_rate)
    }


# Example usage
def example_personalized_system():
    """Example of personalized acoustic emotion recognition."""
    print("=" * 70)
    print("Personalized Acoustic Emotion Recognition Example")
    print("=" * 70)
    
    # Create profile manager
    manager = PersonalizedAcousticProfileManager("./profiles")
    
    # Create personalized encoder
    encoder = PersonalizedAcousticEncoder(
        n_mels=128,
        sample_rate=16000,
        hidden_dim=256,
        num_emotions=7
    )
    
    # Simulate two different people
    person_a = "alice_smith"
    person_b = "bob_jones"
    
    # Add people to encoder
    encoder.add_person(person_a)
    encoder.add_person(person_b)
    
    print(f"\nInitialized profiles for: {person_a}, {person_b}")
    
    # Simulate audio samples
    waveform_a = np.random.randn(16000 * 2)  # 2 seconds
    waveform_b = np.random.randn(16000 * 2)
    
    # Extract features
    features_a = extract_acoustic_features(waveform_a)
    features_b = extract_acoustic_features(waveform_b)
    
    print(f"\n{person_a} baseline: pitch={features_a['pitch']:.3f}, energy={features_a['energy']:.6f}")
    print(f"{person_b} baseline: pitch={features_b['pitch']:.3f}, energy={features_b['energy']:.6f}")
    
    # Update profiles
    manager.update_profile(person_a, features_a, emotion='happy')
    manager.update_profile(person_b, features_b, emotion='neutral')
    
    # Get personalized features
    deviations_a = manager.get_personalized_features(person_a, features_a)
    deviations_b = manager.get_personalized_features(person_b, features_b)
    
    print(f"\n{person_a} deviations: {deviations_a}")
    print(f"{person_b} deviations: {deviations_b}")
    
    # Inference with personalization
    encoder.eval()
    with torch.no_grad():
        waveform_tensor_a = torch.tensor(waveform_a, dtype=torch.float32).unsqueeze(0)
        waveform_tensor_b = torch.tensor(waveform_b, dtype=torch.float32).unsqueeze(0)
        
        features_a, logits_a = encoder(waveform_tensor_a, person_id=person_a, deviations=deviations_a.unsqueeze(0))
        features_b, logits_b = encoder(waveform_tensor_b, person_id=person_b, deviations=deviations_b.unsqueeze(0))
    
    print(f"\n{person_a} personalized prediction: {logits_a.argmax().item()}")
    print(f"{person_b} personalized prediction: {logits_b.argmax().item()}")
    
    # Get profile statistics
    stats_a = manager.get_profile_stats(person_a)
    stats_b = manager.get_profile_stats(person_b)
    
    print(f"\n{person_a} profile: {stats_a['num_samples']} samples collected")
    print(f"{person_b} profile: {stats_b['num_samples']} samples collected")
    
    print("\n✓ Personalized system demonstration complete!")


if __name__ == "__main__":
    example_personalized_system()
