"""
Comprehensive tests for personalized_acoustic_profiling.py
Tests PersonalizedAcousticProfile, PersonalizedAcousticEncoder, and ProfileManager
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
import warnings
warnings.filterwarnings('ignore')

from models.personalized_acoustic_profiling import (
    PersonalizedAcousticProfile,
    PersonalizedAcousticEncoder,
    PersonalizedAcousticProfileManager,
    extract_acoustic_features
)


class TestPersonalizedAcousticProfile:
    """Test PersonalizedAcousticProfile."""
    
    @staticmethod
    def test_initialization():
        """Test profile initialization."""
        print("Testing PersonalizedAcousticProfile initialization...")
        
        profile = PersonalizedAcousticProfile(person_id="test_user")
        
        assert profile.person_id == "test_user"
        assert profile.baseline_pitch is None
        assert profile.baseline_energy is None
        assert profile.baseline_speaking_rate is None
        assert profile.num_samples == 0
        assert len(profile.emotion_acoustics) == 7
        
        print("  ✓ Profile initialized correctly\n")
    
    @staticmethod
    def test_update_baseline():
        """Test baseline update with EMA."""
        print("Testing baseline update...")
        
        profile = PersonalizedAcousticProfile(person_id="test_user")
        
        # First update
        features1 = {'pitch': 200.0, 'energy': 0.5, 'speaking_rate': 150.0}
        profile.update_baseline(features1)
        
        assert profile.baseline_pitch == 200.0
        assert profile.baseline_energy == 0.5
        assert profile.num_samples == 1
        
        # Second update - should use EMA
        features2 = {'pitch': 220.0, 'energy': 0.6, 'speaking_rate': 160.0}
        profile.update_baseline(features2)
        
        # EMA: new_value = (1-alpha) * old + alpha * new
        expected_pitch = (1 - 0.1) * 200.0 + 0.1 * 220.0
        assert abs(profile.baseline_pitch - expected_pitch) < 0.01
        assert profile.num_samples == 2
        
        print("  ✓ Baseline updates with EMA working\n")
    
    @staticmethod
    def test_emotion_samples():
        """Test emotion-specific sample collection."""
        print("Testing emotion sample collection...")
        
        profile = PersonalizedAcousticProfile(person_id="test_user")
        
        # Add samples for different emotions
        for i in range(10):
            profile.add_emotion_sample('happy', {
                'pitch': 200.0 + np.random.randn() * 10,
                'energy': 0.7 + np.random.randn() * 0.1,
                'spectral': np.random.randn()
            })
            profile.add_emotion_sample('sad', {
                'pitch': 150.0 + np.random.randn() * 10,
                'energy': 0.3 + np.random.randn() * 0.1,
                'spectral': np.random.randn()
            })
        
        assert len(profile.emotion_acoustics['happy']['pitch']) == 10
        assert len(profile.emotion_acoustics['sad']['pitch']) == 10
        assert len(profile.emotion_acoustics['angry']['pitch']) == 0
        
        print("  ✓ Emotion samples collected correctly\n")
    
    @staticmethod
    def test_emotion_statistics():
        """Test emotion statistics computation."""
        print("Testing emotion statistics...")
        
        profile = PersonalizedAcousticProfile(person_id="test_user")
        
        # Need at least 5 samples
        for i in range(10):
            profile.add_emotion_sample('happy', {
                'pitch': 200.0 + i,
                'energy': 0.7 + i * 0.01
            })
        
        stats = profile.get_emotion_statistics('happy')
        
        assert stats is not None
        assert 'pitch' in stats
        assert 'energy' in stats
        
        pitch_mean, pitch_std = stats['pitch']
        assert abs(pitch_mean - 204.5) < 0.1  # Mean of 200-209
        
        # Insufficient samples should return None
        stats_angry = profile.get_emotion_statistics('angry')
        assert stats_angry is None
        
        print("  ✓ Emotion statistics computed correctly\n")
    
    @staticmethod
    def test_compute_deviation():
        """Test deviation computation."""
        print("Testing deviation computation...")
        
        profile = PersonalizedAcousticProfile(person_id="test_user")
        
        # Set baseline
        profile.update_baseline({'pitch': 200.0, 'energy': 0.5, 'speaking_rate': 150.0})
        
        # Test deviations
        test_features = {'pitch': 220.0, 'energy': 0.6}
        deviations = profile.compute_deviation(test_features)
        
        # Deviation = (current - baseline) / baseline
        expected_pitch_dev = (220.0 - 200.0) / 200.0
        expected_energy_dev = (0.6 - 0.5) / 0.5
        
        assert abs(deviations['pitch'] - expected_pitch_dev) < 0.001
        assert abs(deviations['energy'] - expected_energy_dev) < 0.001
        
        print("  ✓ Deviations computed correctly\n")
    
    @staticmethod
    def test_save_and_load():
        """Test profile persistence."""
        print("Testing profile save/load...")
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create and populate profile
            profile = PersonalizedAcousticProfile(person_id="test_user")
            profile.update_baseline({'pitch': 200.0, 'energy': 0.5, 'speaking_rate': 150.0})
            profile.update_baseline({'pitch': 210.0, 'energy': 0.55, 'speaking_rate': 155.0})
            
            # Save
            filepath = f"{temp_dir}/test_user.pkl"
            profile.save(filepath)
            
            # Load
            loaded_profile = PersonalizedAcousticProfile.load(filepath)
            
            assert loaded_profile.person_id == profile.person_id
            assert loaded_profile.baseline_pitch == profile.baseline_pitch
            assert loaded_profile.baseline_energy == profile.baseline_energy
            assert loaded_profile.num_samples == profile.num_samples
            
            print("  ✓ Profile saved and loaded correctly\n")
            
        finally:
            shutil.rmtree(temp_dir)
    
    @staticmethod
    def run_all():
        """Run all PersonalizedAcousticProfile tests."""
        print("\n" + "="*70)
        print("PERSONALIZED ACOUSTIC PROFILE TESTS")
        print("="*70)
        TestPersonalizedAcousticProfile.test_initialization()
        TestPersonalizedAcousticProfile.test_update_baseline()
        TestPersonalizedAcousticProfile.test_emotion_samples()
        TestPersonalizedAcousticProfile.test_emotion_statistics()
        TestPersonalizedAcousticProfile.test_compute_deviation()
        TestPersonalizedAcousticProfile.test_save_and_load()
        print("✓ All PersonalizedAcousticProfile tests passed!\n")


class TestPersonalizedAcousticEncoder:
    """Test PersonalizedAcousticEncoder."""
    
    @staticmethod
    def test_initialization():
        """Test encoder initialization."""
        print("Testing PersonalizedAcousticEncoder initialization...")
        
        encoder = PersonalizedAcousticEncoder(
            n_mels=128,
            sample_rate=16000,
            hidden_dim=256,
            num_emotions=7
        )
        
        assert hasattr(encoder, 'spectrogram_encoder')
        assert hasattr(encoder, 'person_adaptation')
        assert hasattr(encoder, 'deviation_encoder')
        assert hasattr(encoder, 'feature_fusion')
        assert hasattr(encoder, 'personalized_classifier')
        
        params = sum(p.numel() for p in encoder.parameters())
        print(f"  ✓ Initialized with {params:,} parameters\n")
    
    @staticmethod
    def test_add_person():
        """Test adding new person."""
        print("Testing person addition...")
        
        encoder = PersonalizedAcousticEncoder(hidden_dim=256, num_emotions=7)
        
        # Add multiple people
        encoder.add_person("alice")
        encoder.add_person("bob")
        encoder.add_person("charlie")
        
        assert "alice" in encoder.person_adaptation
        assert "bob" in encoder.person_adaptation
        assert "charlie" in encoder.person_adaptation
        
        # Adding same person twice should not create duplicates
        encoder.add_person("alice")
        assert len(encoder.person_adaptation) == 3
        
        print("  ✓ Persons added correctly\n")
    
    @staticmethod
    def test_forward_without_personalization():
        """Test forward pass without personalization."""
        print("Testing forward pass without personalization...")
        
        encoder = PersonalizedAcousticEncoder(hidden_dim=256, num_emotions=7)
        encoder.eval()
        
        waveform = torch.randn(2, 32000)
        
        with torch.no_grad():
            features, logits = encoder(waveform)
        
        assert features.shape == (2, 256)
        assert logits.shape == (2, 7)
        assert not torch.isnan(features).any()
        
        print(f"  ✓ Output: features {features.shape}, logits {logits.shape}\n")
    
    @staticmethod
    def test_forward_with_deviations():
        """Test forward pass with deviation features."""
        print("Testing forward pass with deviations...")
        
        encoder = PersonalizedAcousticEncoder(hidden_dim=256, num_emotions=7)
        encoder.eval()
        
        waveform = torch.randn(2, 32000)
        deviations = torch.randn(2, 3)  # [pitch_dev, energy_dev, rate_dev]
        
        with torch.no_grad():
            features, logits = encoder(waveform, deviations=deviations)
        
        assert features.shape == (2, 256)
        assert logits.shape == (2, 7)
        
        print(f"  ✓ Deviations incorporated: {features.shape}\n")
    
    @staticmethod
    def test_forward_with_person_id():
        """Test forward pass with person-specific adaptation."""
        print("Testing forward pass with person ID...")
        
        encoder = PersonalizedAcousticEncoder(hidden_dim=256, num_emotions=7)
        encoder.add_person("alice")
        encoder.eval()
        
        waveform = torch.randn(2, 32000)
        deviations = torch.randn(2, 3)
        
        with torch.no_grad():
            features, logits = encoder(waveform, person_id="alice", deviations=deviations)
        
        assert features.shape == (2, 256)
        assert logits.shape == (2, 7)
        
        print(f"  ✓ Person-specific adaptation applied\n")
    
    @staticmethod
    def test_different_persons_different_outputs():
        """Test that different persons get different adaptations."""
        print("Testing different persons produce different outputs...")
        
        encoder = PersonalizedAcousticEncoder(hidden_dim=256, num_emotions=7)
        encoder.add_person("alice")
        encoder.add_person("bob")
        
        # Initialize person-specific layers differently
        with torch.no_grad():
            for name, layer in encoder.person_adaptation["alice"].named_parameters():
                layer.add_(torch.randn_like(layer) * 0.1)
        
        encoder.eval()
        
        waveform = torch.randn(2, 32000)
        deviations = torch.randn(2, 3)
        
        with torch.no_grad():
            features_alice, logits_alice = encoder(waveform, person_id="alice", deviations=deviations)
            features_bob, logits_bob = encoder(waveform, person_id="bob", deviations=deviations)
        
        # Outputs should differ due to person-specific layers
        diff = (features_alice - features_bob).abs().mean().item()
        assert diff > 0.01, "Different persons should produce different outputs"
        
        print(f"  ✓ Feature difference between persons: {diff:.4f}\n")
    
    @staticmethod
    def test_gradients():
        """Test gradient flow through personalized encoder."""
        print("Testing PersonalizedAcousticEncoder gradients...")
        
        encoder = PersonalizedAcousticEncoder(hidden_dim=256, num_emotions=7)
        encoder.add_person("alice")
        encoder.train()
        
        waveform = torch.randn(2, 32000, requires_grad=True)
        deviations = torch.randn(2, 3, requires_grad=True)
        target = torch.randint(0, 7, (2,))
        
        features, logits = encoder(waveform, person_id="alice", deviations=deviations)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        
        assert waveform.grad is not None
        assert deviations.grad is not None
        
        # Check person-specific layers have gradients
        person_params_with_grad = sum(
            1 for p in encoder.person_adaptation["alice"].parameters()
            if p.grad is not None
        )
        assert person_params_with_grad > 0
        
        print("  ✓ Gradients flowing through all components\n")
    
    @staticmethod
    def run_all():
        """Run all PersonalizedAcousticEncoder tests."""
        print("\n" + "="*70)
        print("PERSONALIZED ACOUSTIC ENCODER TESTS")
        print("="*70)
        TestPersonalizedAcousticEncoder.test_initialization()
        TestPersonalizedAcousticEncoder.test_add_person()
        TestPersonalizedAcousticEncoder.test_forward_without_personalization()
        TestPersonalizedAcousticEncoder.test_forward_with_deviations()
        TestPersonalizedAcousticEncoder.test_forward_with_person_id()
        TestPersonalizedAcousticEncoder.test_different_persons_different_outputs()
        TestPersonalizedAcousticEncoder.test_gradients()
        print("✓ All PersonalizedAcousticEncoder tests passed!\n")


class TestPersonalizedAcousticProfileManager:
    """Test Profile Manager."""
    
    @staticmethod
    def test_initialization():
        """Test manager initialization."""
        print("Testing ProfileManager initialization...")
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            manager = PersonalizedAcousticProfileManager(profiles_dir=temp_dir)
            
            assert manager.profiles_dir.exists()
            assert len(manager.profiles) == 0
            
            print(f"  ✓ Manager initialized at {temp_dir}\n")
            
        finally:
            shutil.rmtree(temp_dir)
    
    @staticmethod
    def test_get_or_create_profile():
        """Test profile creation and retrieval."""
        print("Testing get_or_create_profile...")
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            manager = PersonalizedAcousticProfileManager(profiles_dir=temp_dir)
            
            # Create new profile
            profile1 = manager.get_or_create_profile("alice")
            assert profile1.person_id == "alice"
            
            # Get same profile
            profile2 = manager.get_or_create_profile("alice")
            assert profile1 is profile2  # Same object
            
            # Create another profile
            profile3 = manager.get_or_create_profile("bob")
            assert profile3.person_id == "bob"
            assert profile3 is not profile1
            
            print("  ✓ Profile creation and retrieval working\n")
            
        finally:
            shutil.rmtree(temp_dir)
    
    @staticmethod
    def test_update_profile():
        """Test profile updates."""
        print("Testing profile updates...")
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            manager = PersonalizedAcousticProfileManager(profiles_dir=temp_dir)
            
            # Update with acoustic features
            features = {'pitch': 200.0, 'energy': 0.5, 'speaking_rate': 150.0}
            manager.update_profile("alice", features, emotion='happy')
            
            profile = manager.get_or_create_profile("alice")
            assert profile.num_samples == 1
            assert profile.baseline_pitch == 200.0
            assert len(profile.emotion_acoustics['happy']['pitch']) == 1
            
            print("  ✓ Profile updates working\n")
            
        finally:
            shutil.rmtree(temp_dir)
    
    @staticmethod
    def test_save_and_load_profiles():
        """Test profile persistence through manager."""
        print("Testing profile persistence through manager...")
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create and populate profiles
            manager1 = PersonalizedAcousticProfileManager(profiles_dir=temp_dir)
            
            for i in range(5):
                manager1.update_profile("alice", {
                    'pitch': 200.0 + i,
                    'energy': 0.5 + i * 0.01,
                    'speaking_rate': 150.0
                })
            
            manager1.save_all_profiles()
            
            # Load profiles in new manager
            manager2 = PersonalizedAcousticProfileManager(profiles_dir=temp_dir)
            profile = manager2.get_or_create_profile("alice")
            
            assert profile.num_samples == 5
            assert profile.baseline_pitch is not None
            
            print("  ✓ Profiles persisted and loaded correctly\n")
            
        finally:
            shutil.rmtree(temp_dir)
    
    @staticmethod
    def test_list_profiles():
        """Test listing available profiles."""
        print("Testing profile listing...")
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            manager = PersonalizedAcousticProfileManager(profiles_dir=temp_dir)
            
            # Create profiles
            manager.update_profile("alice", {'pitch': 200.0, 'energy': 0.5, 'speaking_rate': 150.0})
            manager.update_profile("bob", {'pitch': 180.0, 'energy': 0.4, 'speaking_rate': 140.0})
            manager.update_profile("charlie", {'pitch': 220.0, 'energy': 0.6, 'speaking_rate': 160.0})
            
            manager.save_all_profiles()
            
            profiles = manager.list_profiles()
            
            assert "alice" in profiles
            assert "bob" in profiles
            assert "charlie" in profiles
            
            print(f"  ✓ Listed profiles: {profiles}\n")
            
        finally:
            shutil.rmtree(temp_dir)
    
    @staticmethod
    def test_get_profile_stats():
        """Test profile statistics."""
        print("Testing profile statistics...")
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            manager = PersonalizedAcousticProfileManager(profiles_dir=temp_dir)
            
            # Create and populate profile
            for i in range(10):
                manager.update_profile("alice", {
                    'pitch': 200.0 + i,
                    'energy': 0.5 + i * 0.01,
                    'speaking_rate': 150.0
                }, emotion='happy' if i % 2 == 0 else 'sad')
            
            stats = manager.get_profile_stats("alice")
            
            assert stats['person_id'] == 'alice'
            assert stats['num_samples'] == 10
            assert stats['baseline_pitch'] is not None
            assert 'emotions_sampled' in stats
            assert stats['emotions_sampled']['happy'] == 5
            assert stats['emotions_sampled']['sad'] == 5
            
            print(f"  ✓ Profile stats: {stats['num_samples']} samples\n")
            
        finally:
            shutil.rmtree(temp_dir)
    
    @staticmethod
    def run_all():
        """Run all ProfileManager tests."""
        print("\n" + "="*70)
        print("PROFILE MANAGER TESTS")
        print("="*70)
        TestPersonalizedAcousticProfileManager.test_initialization()
        TestPersonalizedAcousticProfileManager.test_get_or_create_profile()
        TestPersonalizedAcousticProfileManager.test_update_profile()
        TestPersonalizedAcousticProfileManager.test_save_and_load_profiles()
        TestPersonalizedAcousticProfileManager.test_list_profiles()
        TestPersonalizedAcousticProfileManager.test_get_profile_stats()
        print("✓ All ProfileManager tests passed!\n")


class TestAcousticFeatureExtraction:
    """Test acoustic feature extraction utility."""
    
    @staticmethod
    def test_extract_features():
        """Test acoustic feature extraction from waveform."""
        print("Testing acoustic feature extraction...")
        
        # Create dummy waveform
        sample_rate = 16000
        duration = 2.0
        waveform = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        features = extract_acoustic_features(waveform, sample_rate)
        
        assert 'pitch' in features
        assert 'energy' in features
        assert 'speaking_rate' in features
        
        assert isinstance(features['pitch'], float)
        assert isinstance(features['energy'], float)
        assert isinstance(features['speaking_rate'], float)
        
        print(f"  ✓ Extracted features: pitch={features['pitch']:.4f}, energy={features['energy']:.6f}\n")
    
    @staticmethod
    def run_all():
        """Run all feature extraction tests."""
        print("\n" + "="*70)
        print("ACOUSTIC FEATURE EXTRACTION TESTS")
        print("="*70)
        TestAcousticFeatureExtraction.test_extract_features()
        print("✓ All feature extraction tests passed!\n")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*70)
    print("PERSONALIZED ACOUSTIC PROFILING COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("="*70)
    
    try:
        TestPersonalizedAcousticProfile.run_all()
        TestPersonalizedAcousticEncoder.run_all()
        TestPersonalizedAcousticProfileManager.run_all()
        TestAcousticFeatureExtraction.run_all()
        
        print("\n" + "="*70)
        print("✓✓✓ ALL TESTS PASSED SUCCESSFULLY! ✓✓✓")
        print("="*70)
        print("\nSummary:")
        print("  • PersonalizedAcousticProfile: Baseline EMA and emotion stats working")
        print("  • PersonalizedAcousticEncoder: Person-specific adaptations verified")
        print("  • ProfileManager: Persistence and multi-person management validated")
        print("  • Feature extraction: Acoustic feature computation working")
        print("\nThe personalized acoustic profiling system is fully functional!")
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
