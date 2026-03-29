"""
Comprehensive tests for mediapipe_detector.py models
Tests FaceMeshFeatureExtractor, MediaPipeEmotionDetector
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

# Check if MediaPipe is available
try:
    from models.mediapipe_detector import (
        FaceMeshFeatureExtractor,
        MediaPipeEmotionDetector,
        MediaPipeFaceDetector,
        count_parameters
    )
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"⚠ MediaPipe not available: {e}")


class TestFaceMeshFeatureExtractor:
    """Test Face Mesh Feature Extractor."""
    
    @staticmethod
    def test_initialization():
        """Test extractor initialization."""
        print("Testing FaceMeshFeatureExtractor initialization...")
        
        extractor = FaceMeshFeatureExtractor(
            landmark_dim=468 * 3,
            hidden_dim=256
        )
        
        assert hasattr(extractor, 'landmark_encoder')
        assert hasattr(extractor, 'eye_encoder')
        assert hasattr(extractor, 'eyebrow_encoder')
        assert hasattr(extractor, 'mouth_encoder')
        assert hasattr(extractor, 'region_fusion')
        
        print(f"  ✓ Initialized with {count_parameters(extractor):,} parameters\n")
    
    @staticmethod
    def test_forward_pass():
        """Test forward pass with various batch sizes."""
        print("Testing FaceMeshFeatureExtractor forward pass...")
        
        extractor = FaceMeshFeatureExtractor(
            landmark_dim=468 * 3,
            hidden_dim=256
        )
        extractor.eval()
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            landmarks = torch.randn(batch_size, 468, 3)
            
            with torch.no_grad():
                global_features, region_features = extractor(landmarks)
            
            assert global_features.shape == (batch_size, 256), f"Expected ({batch_size}, 256), got {global_features.shape}"
            assert region_features.shape == (batch_size, 256), f"Expected ({batch_size}, 256), got {region_features.shape}"
            assert not torch.isnan(global_features).any()
            assert not torch.isnan(region_features).any()
        
        print(f"  ✓ Forward pass working for batch sizes: {batch_sizes}\n")
    
    @staticmethod
    def test_region_extraction():
        """Test region-specific landmark extraction."""
        print("Testing region-specific landmark extraction...")
        
        extractor = FaceMeshFeatureExtractor()
        
        # Create dummy landmarks
        landmarks = np.random.randn(468, 3).astype(np.float32)
        
        regions = extractor.extract_region_landmarks(landmarks)
        
        assert 'left_eye' in regions
        assert 'right_eye' in regions
        assert 'left_eyebrow' in regions
        assert 'right_eyebrow' in regions
        assert 'mouth' in regions
        
        print("  ✓ Region extraction working\n")
    
    @staticmethod
    def test_gradients():
        """Test gradient flow."""
        print("Testing FaceMeshFeatureExtractor gradients...")
        
        extractor = FaceMeshFeatureExtractor()
        extractor.train()
        
        landmarks = torch.randn(2, 468, 3, requires_grad=True)
        
        global_features, region_features = extractor(landmarks)
        loss = global_features.sum() + region_features.sum()
        loss.backward()
        
        assert landmarks.grad is not None
        assert not torch.isnan(landmarks.grad).any()
        
        print("  ✓ Gradients computed successfully\n")
    
    @staticmethod
    def run_all():
        """Run all FaceMeshFeatureExtractor tests."""
        print("\n" + "="*70)
        print("FACE MESH FEATURE EXTRACTOR TESTS")
        print("="*70)
        TestFaceMeshFeatureExtractor.test_initialization()
        TestFaceMeshFeatureExtractor.test_forward_pass()
        TestFaceMeshFeatureExtractor.test_region_extraction()
        TestFaceMeshFeatureExtractor.test_gradients()
        print("✓ All FaceMeshFeatureExtractor tests passed!\n")


class TestMediaPipeEmotionDetector:
    """Test MediaPipe Emotion Detector."""
    
    @staticmethod
    def test_initialization():
        """Test detector initialization with different fusion methods."""
        print("Testing MediaPipeEmotionDetector initialization...")
        
        fusion_methods = ['concat', 'attention', 'add']
        
        for method in fusion_methods:
            model = MediaPipeEmotionDetector(
                num_emotions=7,
                cnn_backbone='mobilenet_v2',
                pretrained=False,
                fusion_method=method,
                dropout=0.4
            )
            
            assert model.num_emotions == 7
            assert model.fusion_method == method
            assert hasattr(model, 'cnn')
            assert hasattr(model, 'landmark_extractor')
            assert hasattr(model, 'fusion')
            
            print(f"  ✓ Fusion method '{method}': {count_parameters(model):,} parameters")
        
        print()
    
    @staticmethod
    def test_forward_pass():
        """Test forward pass with image and landmarks."""
        print("Testing MediaPipeEmotionDetector forward pass...")
        
        model = MediaPipeEmotionDetector(
            num_emotions=7,
            cnn_backbone='mobilenet_v2',
            pretrained=False,
            fusion_method='concat'
        )
        model.eval()
        
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            image = torch.randn(batch_size, 3, 224, 224)
            landmarks = torch.randn(batch_size, 468, 3)
            
            with torch.no_grad():
                logits = model(image, landmarks)
            
            assert logits.shape == (batch_size, 7)
            assert not torch.isnan(logits).any()
            
            print(f"  ✓ Batch size {batch_size}: output {logits.shape}")
        
        print()
    
    @staticmethod
    def test_all_fusion_methods():
        """Test all fusion methods."""
        print("Testing all fusion methods...")
        
        fusion_methods = ['concat', 'attention', 'add']
        
        for method in fusion_methods:
            model = MediaPipeEmotionDetector(
                num_emotions=7,
                cnn_backbone='mobilenet_v2',
                pretrained=False,
                fusion_method=method
            )
            model.eval()
            
            image = torch.randn(2, 3, 224, 224)
            landmarks = torch.randn(2, 468, 3)
            
            with torch.no_grad():
                logits = model(image, landmarks)
            
            assert logits.shape == (2, 7)
            assert not torch.isnan(logits).any()
            
            print(f"  ✓ Fusion method '{method}' working")
        
        print()
    
    @staticmethod
    def test_return_features():
        """Test feature extraction mode."""
        print("Testing feature extraction mode...")
        
        model = MediaPipeEmotionDetector(
            num_emotions=7,
            cnn_backbone='mobilenet_v2',
            pretrained=False
        )
        model.eval()
        
        image = torch.randn(2, 3, 224, 224)
        landmarks = torch.randn(2, 468, 3)
        
        with torch.no_grad():
            logits, cnn_features, landmark_features = model(image, landmarks, return_features=True)
        
        assert logits.shape == (2, 7)
        assert cnn_features.shape[0] == 2
        assert landmark_features.shape == (2, 512)  # global + region features
        
        print("  ✓ Feature extraction working\n")
    
    @staticmethod
    def test_predict_emotion():
        """Test emotion prediction interface."""
        print("Testing predict_emotion interface...")
        
        model = MediaPipeEmotionDetector(
            num_emotions=7,
            cnn_backbone='mobilenet_v2',
            pretrained=False
        )
        model.eval()
        
        image = torch.randn(2, 3, 224, 224)
        landmarks = torch.randn(2, 468, 3)
        
        with torch.no_grad():
            predicted, probs = model.predict_emotion(image, landmarks)
        
        assert predicted.shape == (2,)
        assert probs.shape == (2, 7)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
        
        print("  ✓ Prediction interface working\n")
    
    @staticmethod
    def test_gradients():
        """Test gradient flow through both modalities."""
        print("Testing MediaPipeEmotionDetector gradients...")
        
        model = MediaPipeEmotionDetector(
            num_emotions=7,
            cnn_backbone='mobilenet_v2',
            pretrained=False
        )
        model.train()
        
        image = torch.randn(2, 3, 224, 224, requires_grad=True)
        landmarks = torch.randn(2, 468, 3, requires_grad=True)
        target = torch.randint(0, 7, (2,))
        
        logits = model(image, landmarks)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        
        assert image.grad is not None
        assert landmarks.grad is not None
        assert not torch.isnan(image.grad).any()
        assert not torch.isnan(landmarks.grad).any()
        
        print("  ✓ Gradients flowing through both modalities\n")
    
    @staticmethod
    def run_all():
        """Run all MediaPipeEmotionDetector tests."""
        print("\n" + "="*70)
        print("MEDIAPIPE EMOTION DETECTOR TESTS")
        print("="*70)
        TestMediaPipeEmotionDetector.test_initialization()
        TestMediaPipeEmotionDetector.test_forward_pass()
        TestMediaPipeEmotionDetector.test_all_fusion_methods()
        TestMediaPipeEmotionDetector.test_return_features()
        TestMediaPipeEmotionDetector.test_predict_emotion()
        TestMediaPipeEmotionDetector.test_gradients()
        print("✓ All MediaPipeEmotionDetector tests passed!\n")


class TestMediaPipeFaceDetector:
    """Test MediaPipe Face Detector wrapper."""
    
    @staticmethod
    def test_initialization():
        """Test detector initialization."""
        print("Testing MediaPipeFaceDetector initialization...")
        
        detector = MediaPipeFaceDetector(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        assert hasattr(detector, 'face_mesh')
        assert hasattr(detector, 'mp_drawing')
        
        detector.close()
        
        print("  ✓ Detector initialized and closed\n")
    
    @staticmethod
    def test_detect_landmarks():
        """Test landmark detection (with dummy image)."""
        print("Testing landmark detection...")
        
        detector = MediaPipeFaceDetector()
        
        # Create dummy image (random noise - won't detect face)
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        landmarks = detector.detect_landmarks(dummy_image)
        
        # With random noise, no face should be detected
        # This is expected behavior
        if landmarks is None:
            print("  ✓ No face detected (expected with random image)")
        else:
            assert landmarks.shape == (468, 3)
            print(f"  ✓ Detected landmarks shape: {landmarks.shape}")
        
        detector.close()
        print()
    
    @staticmethod
    def run_all():
        """Run all MediaPipeFaceDetector tests."""
        print("\n" + "="*70)
        print("MEDIAPIPE FACE DETECTOR TESTS")
        print("="*70)
        TestMediaPipeFaceDetector.test_initialization()
        TestMediaPipeFaceDetector.test_detect_landmarks()
        print("✓ All MediaPipeFaceDetector tests passed!\n")


def run_all_tests():
    """Run complete test suite."""
    if not MEDIAPIPE_AVAILABLE:
        print("\n" + "="*70)
        print("⚠ MEDIAPIPE NOT AVAILABLE - SKIPPING TESTS")
        print("="*70)
        print("\nInstall MediaPipe to run these tests:")
        print("  pip install mediapipe>=0.10.0")
        return True  # Don't fail if MediaPipe not installed
    
    print("\n" + "="*70)
    print("MEDIAPIPE DETECTOR COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("="*70)
    
    try:
        TestFaceMeshFeatureExtractor.run_all()
        TestMediaPipeEmotionDetector.run_all()
        TestMediaPipeFaceDetector.run_all()
        
        print("\n" + "="*70)
        print("✓✓✓ ALL TESTS PASSED SUCCESSFULLY! ✓✓✓")
        print("="*70)
        print("\nSummary:")
        print("  • FaceMeshFeatureExtractor: Region-specific feature extraction verified")
        print("  • MediaPipeEmotionDetector: All fusion methods working")
        print("  • MediaPipeFaceDetector: Landmark detection wrapper functional")
        print("\nMediaPipe integration is fully functional!")
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
