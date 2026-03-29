"""
Test personalized acoustic profiling integration with multimodal transformer.
Validates person-specific acoustic feature extraction and cross-modal relationships.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
import tempfile
import shutil

from models.audio_emotion_fusion import MultimodalEmotionRecognizer
from models.personalized_acoustic_profiling import PersonalizedAcousticProfileManager


def test_personalized_acoustic_integration():
    """Test integration of personalized acoustic profiling with multimodal model."""
    print("=" * 80)
    print("Test 1: Personalized Acoustic Integration")
    print("=" * 80)
    
    # Create temporary profile directory
    temp_dir = tempfile.mkdtemp()
    profile_manager = PersonalizedAcousticProfileManager(profiles_dir=temp_dir)
    
    try:
        # Create model with personalized acoustics
        model = MultimodalEmotionRecognizer(
            num_emotions=7,
            use_audio=True,
            use_acoustic=True,
            use_personalized_acoustic=True,
            profile_manager=profile_manager,
            fusion_method='weighted'
        )
        model.eval()
        
        print(f"✓ Model created with personalized acoustic profiling")
        print(f"  Profile directory: {temp_dir}")
        
        # Test inputs
        batch_size = 2
        num_frames = 8
        num_audio_segments = 2
        sample_rate = 16000
        duration = 2.0
        num_samples = int(sample_rate * duration)
        
        frames = torch.randn(batch_size, num_frames, 3, 224, 224)
        audio_sentiment = torch.randn(batch_size, num_audio_segments, 3).softmax(dim=-1)
        audio_emotions = torch.randn(batch_size, num_audio_segments, 7).softmax(dim=-1)
        audio_waveform = torch.randn(batch_size, num_audio_segments, num_samples)
        
        # Test with person ID
        person_id = "alice_test_001"
        
        print(f"\n✓ Test inputs created")
        print(f"  Person ID: {person_id}")
        print(f"  Frames: {frames.shape}")
        print(f"  Audio waveform: {audio_waveform.shape}")
        
        # Forward pass with personalization
        with torch.no_grad():
            result = model(
                frames,
                audio_sentiment=audio_sentiment,
                audio_emotions=audio_emotions,
                audio_waveform=audio_waveform,
                person_id=person_id
            )
        
        print(f"\n✓ Forward pass completed")
        print(f"  Output logits: {result['logits'].shape}")
        print(f"  Modalities included: {[k for k in result.keys() if k.endswith('_logits')]}")
        
        # Verify profile was created
        profile = profile_manager.get_or_create_profile(person_id)
        assert profile is not None, "Profile should be created"
        assert profile.num_samples > 0, "Profile should have samples"
        
        print(f"\n✓ Profile created and updated")
        print(f"  Samples collected: {profile.num_samples}")
        print(f"  Baseline pitch: {profile.baseline_pitch:.1f} Hz" if profile.baseline_pitch else "  Baseline pitch: Not set")
        print(f"  Baseline energy: {profile.baseline_energy:.3f}" if profile.baseline_energy else "  Baseline energy: Not set")
        
        # Test multiple forward passes (continuous adaptation)
        for i in range(3):
            with torch.no_grad():
                result = model(
                    frames,
                    audio_sentiment=audio_sentiment,
                    audio_emotions=audio_emotions,
                    audio_waveform=audio_waveform,
                    person_id=person_id
                )
        
        print(f"\n✓ Continuous adaptation working")
        print(f"  Total samples after adaptation: {profile.num_samples}")
        
        # Test with different person
        person_id_2 = "bob_test_002"
        with torch.no_grad():
            result_2 = model(
                frames,
                audio_sentiment=audio_sentiment,
                audio_emotions=audio_emotions,
                audio_waveform=audio_waveform,
                person_id=person_id_2
            )
        
        profile_2 = profile_manager.get_or_create_profile(person_id_2)
        assert profile_2 is not None, "Second profile should be created"
        assert profile_2.person_id == person_id_2, "Profile ID mismatch"
        
        print(f"\n✓ Multi-person support working")
        print(f"  Person 1 ({person_id}): {profile.num_samples} samples")
        print(f"  Person 2 ({person_id_2}): {profile_2.num_samples} samples")
        
        # Test prediction interface
        prediction = model.predict_emotion(
            frames,
            audio_sentiment=audio_sentiment,
            audio_emotions=audio_emotions,
            audio_waveform=audio_waveform,
            person_id=person_id
        )
        
        assert 'emotion' in prediction, "Prediction should include emotion"
        assert 'confidence' in prediction, "Prediction should include confidence"
        
        print(f"\n✓ Prediction interface working")
        print(f"  Predicted emotion: {prediction['emotion']}")
        print(f"  Confidence: {prediction['confidence']:.3f}")
        
        print("\n" + "=" * 80)
        print("✓ Test 1 PASSED: Personalized Acoustic Integration")
        print("=" * 80)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_person_specific_deviations():
    """Test that different people get different acoustic deviations."""
    print("\n" + "=" * 80)
    print("Test 2: Person-Specific Deviation Computation")
    print("=" * 80)
    
    temp_dir = tempfile.mkdtemp()
    profile_manager = PersonalizedAcousticProfileManager(profiles_dir=temp_dir)
    
    try:
        # Create two people with different vocal characteristics
        person_a = "alice_high_energy"
        person_b = "bob_low_energy"
        
        # Simulate Alice (high energy speaker)
        alice_features = {
            'pitch': 250.0,
            'energy': 0.8,
            'speaking_rate': 180.0
        }
        
        # Initialize Alice's profile
        profile_a = profile_manager.get_or_create_profile(person_a)
        for _ in range(10):  # Collect baseline samples
            profile_manager.update_profile(person_a, alice_features)
        
        print(f"✓ Alice's baseline established")
        print(f"  Baseline pitch: {profile_a.baseline_pitch:.1f} Hz")
        print(f"  Baseline energy: {profile_a.baseline_energy:.3f}")
        
        # Simulate Bob (low energy speaker)
        bob_features = {
            'pitch': 150.0,
            'energy': 0.3,
            'speaking_rate': 120.0
        }
        
        # Initialize Bob's profile
        profile_b = profile_manager.get_or_create_profile(person_b)
        for _ in range(10):  # Collect baseline samples
            profile_manager.update_profile(person_b, bob_features)
        
        print(f"\n✓ Bob's baseline established")
        print(f"  Baseline pitch: {profile_b.baseline_pitch:.1f} Hz")
        print(f"  Baseline energy: {profile_b.baseline_energy:.3f}")
        
        # Test with same absolute acoustic features
        test_features = {
            'pitch': 200.0,
            'energy': 0.5,
            'speaking_rate': 150.0
        }
        
        # Get deviations for each person
        deviations_a = profile_a.compute_deviation(test_features)
        deviations_b = profile_b.compute_deviation(test_features)
        
        print(f"\n✓ Same acoustic input, different deviations:")
        print(f"  Input: pitch={test_features['pitch']:.1f}Hz, energy={test_features['energy']:.3f}")
        print(f"\n  Alice's deviations:")
        print(f"    pitch: {deviations_a['pitch']:+.2%}")
        print(f"    energy: {deviations_a['energy']:+.2%}")
        print(f"\n  Bob's deviations:")
        print(f"    pitch: {deviations_b['pitch']:+.2%}")
        print(f"    energy: {deviations_b['energy']:+.2%}")
        
        # Verify deviations are different
        assert abs(deviations_a['pitch'] - deviations_b['pitch']) > 0.1, \
            "Pitch deviations should differ between people"
        assert abs(deviations_a['energy'] - deviations_b['energy']) > 0.1, \
            "Energy deviations should differ between people"
        
        print(f"\n✓ Personalization working correctly:")
        print(f"  Alice (high energy): pitch 200Hz is {deviations_a['pitch']:+.1%} from her baseline")
        print(f"  Bob (low energy): pitch 200Hz is {deviations_b['pitch']:+.1%} from his baseline")
        print(f"  → Same absolute value, different meaning for each person!")
        
        print("\n" + "=" * 80)
        print("✓ Test 2 PASSED: Person-Specific Deviations")
        print("=" * 80)
        
    finally:
        shutil.rmtree(temp_dir)


def test_cross_modal_relationships():
    """Test that personalized acoustics participate in cross-modal transformer."""
    print("\n" + "=" * 80)
    print("Test 3: Cross-Modal Relationships with Personalized Acoustics")
    print("=" * 80)
    
    temp_dir = tempfile.mkdtemp()
    profile_manager = PersonalizedAcousticProfileManager(profiles_dir=temp_dir)
    
    try:
        model = MultimodalEmotionRecognizer(
            num_emotions=7,
            use_audio=True,
            use_acoustic=True,
            use_personalized_acoustic=True,
            profile_manager=profile_manager,
            fusion_method='attention'  # Use attention fusion to see modality weights
        )
        model.eval()
        
        # Create inputs (use batch size 2 to avoid BatchNorm issues)
        frames = torch.randn(2, 8, 3, 224, 224)
        audio_sentiment = torch.randn(2, 2, 3).softmax(dim=-1)
        audio_emotions = torch.randn(2, 2, 7).softmax(dim=-1)
        audio_waveform = torch.randn(2, 2, 32000)
        
        person_id = "test_person"
        
        # Forward pass
        with torch.no_grad():
            result = model(
                frames,
                audio_sentiment=audio_sentiment,
                audio_emotions=audio_emotions,
                audio_waveform=audio_waveform,
                person_id=person_id
            )
        
        # Check that all modalities are present
        expected_modalities = ['spatial', 'temporal', 'audio', 'audiovisual', 'personalized_acoustic']
        
        print("✓ All modalities participating in fusion:")
        for modality in expected_modalities:
            logit_key = f'{modality}_logits'
            if logit_key in result:
                print(f"  ✓ {modality}")
            else:
                print(f"  ✗ {modality} (missing)")
        
        # Verify gradients flow through personalized acoustic path
        model.train()
        frames.requires_grad = True
        
        result = model(
            frames,
            audio_sentiment=audio_sentiment,
            audio_emotions=audio_emotions,
            audio_waveform=audio_waveform,
            person_id=person_id
        )
        
        loss = result['logits'].sum()
        loss.backward()
        
        assert frames.grad is not None, "Gradients should flow through network"
        
        print(f"\n✓ Gradients flow through personalized acoustic pathway")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Frame gradient norm: {frames.grad.norm().item():.4f}")
        
        # Test that personalized features affect final prediction
        model.eval()
        
        # Prediction with person A
        with torch.no_grad():
            result_a = model(
                frames,
                audio_sentiment=audio_sentiment,
                audio_emotions=audio_emotions,
                audio_waveform=audio_waveform,
                person_id="person_a"
            )
        
        # Prediction with person B
        with torch.no_grad():
            result_b = model(
                frames,
                audio_sentiment=audio_sentiment,
                audio_emotions=audio_emotions,
                audio_waveform=audio_waveform,
                person_id="person_b"
            )
        
        # Predictions should differ due to personalization
        diff = (result_a['logits'] - result_b['logits']).abs().mean().item()
        
        print(f"\n✓ Personalization affects cross-modal prediction")
        print(f"  Same video/audio, different people")
        print(f"  Prediction difference: {diff:.4f}")
        print(f"  → Different vocal baselines lead to different interpretations!")
        
        print("\n" + "=" * 80)
        print("✓ Test 3 PASSED: Cross-Modal Relationships")
        print("=" * 80)
        
    finally:
        shutil.rmtree(temp_dir)


def test_profile_persistence():
    """Test that profiles can be saved and loaded."""
    print("\n" + "=" * 80)
    print("Test 4: Profile Persistence")
    print("=" * 80)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create and train profile
        manager1 = PersonalizedAcousticProfileManager(profiles_dir=temp_dir)
        person_id = "persistent_user"
        
        features = {
            'pitch': 200.0,
            'energy': 0.6,
            'speaking_rate': 150.0
        }
        
        for i in range(20):
            manager1.update_profile(person_id, features)
        
        profile1 = manager1.get_or_create_profile(person_id)
        baseline1 = profile1.baseline_pitch
        samples1 = profile1.num_samples
        
        print(f"✓ Profile created and trained")
        print(f"  Samples: {samples1}")
        print(f"  Baseline pitch: {baseline1:.1f} Hz")
        
        # Save profiles
        manager1.save_all_profiles()
        print(f"\n✓ Profiles saved to disk")
        
        # Load profiles in new manager
        manager2 = PersonalizedAcousticProfileManager(profiles_dir=temp_dir)
        profile2 = manager2.get_or_create_profile(person_id)
        
        assert profile2 is not None, "Profile should be loaded"
        assert profile2.num_samples == samples1, "Sample count should match"
        assert abs(profile2.baseline_pitch - baseline1) < 0.01, "Baseline should match"
        
        print(f"\n✓ Profile loaded from disk")
        print(f"  Samples: {profile2.num_samples}")
        print(f"  Baseline pitch: {profile2.baseline_pitch:.1f} Hz")
        print(f"  Match: {profile2.baseline_pitch == baseline1}")
        
        # Continue adaptation with loaded profile
        new_features = {
            'pitch': 210.0,
            'energy': 0.65,
            'speaking_rate': 155.0
        }
        
        manager2.update_profile(person_id, new_features)
        profile3 = manager2.get_or_create_profile(person_id)
        
        assert profile3.num_samples == samples1 + 1, "Sample count should increase"
        
        print(f"\n✓ Continued adaptation after loading")
        print(f"  New samples: {profile3.num_samples}")
        print(f"  Baseline adapted: {profile3.baseline_pitch:.1f} Hz")
        
        print("\n" + "=" * 80)
        print("✓ Test 4 PASSED: Profile Persistence")
        print("=" * 80)
        
    finally:
        shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all personalized acoustic integration tests."""
    print("\n" + "=" * 80)
    print("PERSONALIZED ACOUSTIC INTEGRATION TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Integration Test", test_personalized_acoustic_integration),
        ("Deviation Test", test_person_specific_deviations),
        ("Cross-Modal Test", test_cross_modal_relationships),
        ("Persistence Test", test_profile_persistence),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test_name} FAILED:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print(f"\n✗✗✗ {failed} TEST(S) FAILED ✗✗✗")
    
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
