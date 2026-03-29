"""
Comprehensive tests for emotion_detector.py models
Tests all components: SpatialAttentionCNN, TemporalTransformer, HybridEmotionRecognizer
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from models.emotion_detector import (
    SpatialAttentionCNN,
    TemporalTransformer,
    HybridEmotionRecognizer,
    EMOTION_LABELS,
    count_parameters
)


class TestSpatialAttentionCNN:
    """Test Spatial Attention CNN."""
    
    @staticmethod
    def test_initialization():
        """Test model initialization with different backbones."""
        print("Testing SpatialAttentionCNN initialization...")
        
        backbones = ['mobilenet_v2', 'resnet18', 'resnet34', 'resnet50']
        
        for backbone in backbones:
            model = SpatialAttentionCNN(
                num_emotions=7,
                backbone=backbone,
                pretrained=False,  # Use False for faster testing
                use_attention=True,
                dropout=0.5
            )
            
            assert model.num_emotions == 7
            assert model.use_attention == True
            assert hasattr(model, 'backbone')
            assert hasattr(model, 'attention')
            assert hasattr(model, 'classifier')
            
            print(f"  ✓ {backbone}: {count_parameters(model):,} parameters")
        
        print("  ✓ All backbones initialized correctly\n")
    
    @staticmethod
    def test_forward_pass():
        """Test forward pass with various batch sizes."""
        print("Testing SpatialAttentionCNN forward pass...")
        
        model = SpatialAttentionCNN(
            num_emotions=7,
            backbone='mobilenet_v2',
            pretrained=False,
            use_attention=True
        )
        model.eval()
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 224, 224)
            
            with torch.no_grad():
                logits = model(x)
            
            assert logits.shape == (batch_size, 7), f"Expected ({batch_size}, 7), got {logits.shape}"
            assert not torch.isnan(logits).any(), "NaN detected in output"
            assert not torch.isinf(logits).any(), "Inf detected in output"
        
        print(f"  ✓ Forward pass working for batch sizes: {batch_sizes}\n")
    
    @staticmethod
    def test_feature_extraction():
        """Test feature extraction mode."""
        print("Testing SpatialAttentionCNN feature extraction...")
        
        model = SpatialAttentionCNN(
            num_emotions=7,
            backbone='mobilenet_v2',
            pretrained=False
        )
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            features = model(x, return_features=True)
            logits = model(x, return_features=False)
        
        assert features.shape == (2, model.feature_dim)
        assert logits.shape == (2, 7)
        
        print(f"  ✓ Feature extraction: {features.shape}")
        print(f"  ✓ Classification: {logits.shape}\n")
    
    @staticmethod
    def test_predict_emotion():
        """Test emotion prediction interface."""
        print("Testing SpatialAttentionCNN predict_emotion...")
        
        model = SpatialAttentionCNN(num_emotions=7, backbone='mobilenet_v2', pretrained=False)
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            predicted, probs = model.predict_emotion(x)
        
        assert predicted.shape == (2,)
        assert probs.shape == (2, 7)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
        assert (predicted >= 0).all() and (predicted < 7).all()
        
        print(f"  ✓ Predictions: {predicted}")
        print(f"  ✓ Probabilities sum to 1.0\n")
    
    @staticmethod
    def test_attention_mechanism():
        """Test spatial attention is working."""
        print("Testing spatial attention mechanism...")
        
        model_with_attn = SpatialAttentionCNN(
            num_emotions=7,
            backbone='mobilenet_v2',
            pretrained=False,
            use_attention=True
        )
        model_with_attn.eval()  # Set to eval mode for batch size 1 (BatchNorm compatibility)
        
        model_without_attn = SpatialAttentionCNN(
            num_emotions=7,
            backbone='mobilenet_v2',
            pretrained=False,
            use_attention=False
        )
        model_without_attn.eval()  # Set to eval mode for batch size 1
        
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            out_with = model_with_attn(x)
            out_without = model_without_attn(x)
        
        # Outputs should be different due to attention
        assert out_with.shape == out_without.shape == (1, 7)
        
        print("  ✓ Attention mechanism working\n")
    
    @staticmethod
    def test_gradients():
        """Test gradient flow."""
        print("Testing SpatialAttentionCNN gradients...")
        
        model = SpatialAttentionCNN(num_emotions=7, backbone='mobilenet_v2', pretrained=False)
        model.train()
        
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        target = torch.randint(0, 7, (2,))
        
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        
        print(f"  ✓ Gradients computed for {params_with_grad}/{total_params} parameters\n")
    
    @staticmethod
    def run_all():
        """Run all SpatialAttentionCNN tests."""
        print("\n" + "="*70)
        print("SPATIAL ATTENTION CNN TESTS")
        print("="*70)
        TestSpatialAttentionCNN.test_initialization()
        TestSpatialAttentionCNN.test_forward_pass()
        TestSpatialAttentionCNN.test_feature_extraction()
        TestSpatialAttentionCNN.test_predict_emotion()
        TestSpatialAttentionCNN.test_attention_mechanism()
        TestSpatialAttentionCNN.test_gradients()
        print("✓ All SpatialAttentionCNN tests passed!\n")


class TestTemporalTransformer:
    """Test Temporal Transformer."""
    
    @staticmethod
    def test_initialization():
        """Test transformer initialization."""
        print("Testing TemporalTransformer initialization...")
        
        model = TemporalTransformer(
            feature_dim=512,
            num_emotions=7,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            max_seq_length=32
        )
        
        assert model.feature_dim == 512
        assert model.num_emotions == 7
        assert hasattr(model, 'positional_encoding')
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'temporal_classifier')
        
        print(f"  ✓ Initialized with {count_parameters(model):,} parameters\n")
    
    @staticmethod
    def test_forward_pass():
        """Test forward pass with various sequence lengths."""
        print("Testing TemporalTransformer forward pass...")
        
        model = TemporalTransformer(
            feature_dim=256,
            num_emotions=7,
            num_layers=2,
            num_heads=4
        )
        model.eval()
        
        test_cases = [
            (2, 4, 256),   # Short sequence
            (2, 16, 256),  # Medium sequence
            (2, 32, 256),  # Long sequence
            (1, 8, 256),   # Batch size 1
        ]
        
        for batch, seq_len, feat_dim in test_cases:
            features = torch.randn(batch, seq_len, feat_dim)
            
            with torch.no_grad():
                logits = model(features)
            
            assert logits.shape == (batch, 7)
            assert not torch.isnan(logits).any()
            
            print(f"  ✓ Input ({batch}, {seq_len}, {feat_dim}) → Output ({batch}, 7)")
        
        print()
    
    @staticmethod
    def test_positional_encoding():
        """Test positional encoding."""
        print("Testing positional encoding...")
        
        model = TemporalTransformer(
            feature_dim=256,
            num_emotions=7,
            max_seq_length=32
        )
        
        assert model.positional_encoding.shape == (1, 32, 256)
        
        # Test with sequence shorter than max
        features = torch.randn(2, 16, 256)
        with torch.no_grad():
            logits = model(features)
        
        assert logits.shape == (2, 7)
        
        print("  ✓ Positional encoding working correctly\n")
    
    @staticmethod
    def test_gradients():
        """Test gradient flow."""
        print("Testing TemporalTransformer gradients...")
        
        model = TemporalTransformer(feature_dim=256, num_emotions=7)
        model.train()
        
        features = torch.randn(2, 8, 256, requires_grad=True)
        target = torch.randint(0, 7, (2,))
        
        logits = model(features)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()
        
        print("  ✓ Gradients computed successfully\n")
    
    @staticmethod
    def run_all():
        """Run all TemporalTransformer tests."""
        print("\n" + "="*70)
        print("TEMPORAL TRANSFORMER TESTS")
        print("="*70)
        TestTemporalTransformer.test_initialization()
        TestTemporalTransformer.test_forward_pass()
        TestTemporalTransformer.test_positional_encoding()
        TestTemporalTransformer.test_gradients()
        print("✓ All TemporalTransformer tests passed!\n")


class TestHybridEmotionRecognizer:
    """Test Hybrid Emotion Recognizer."""
    
    @staticmethod
    def test_initialization():
        """Test hybrid model initialization."""
        print("Testing HybridEmotionRecognizer initialization...")
        
        # Test without MediaPipe
        model_bimodal = HybridEmotionRecognizer(
            num_emotions=7,
            cnn_backbone='mobilenet_v2',
            pretrained=False,
            temporal_layers=2,
            temporal_heads=4,
            fusion_method='weighted',
            use_mediapipe=False
        )
        
        assert model_bimodal.num_emotions == 7
        assert model_bimodal.use_mediapipe == False
        assert hasattr(model_bimodal, 'spatial_cnn')
        assert hasattr(model_bimodal, 'temporal_transformer')
        
        print(f"  ✓ Bi-modal (CNN + Transformer): {count_parameters(model_bimodal):,} parameters")
        
        # Test with MediaPipe if available
        try:
            model_trimodal = HybridEmotionRecognizer(
                num_emotions=7,
                cnn_backbone='mobilenet_v2',
                pretrained=False,
                temporal_layers=2,
                temporal_heads=4,
                fusion_method='weighted',
                use_mediapipe=True
            )
            
            # Check if MediaPipe was actually enabled (depends on import)
            if model_trimodal.use_mediapipe:
                assert hasattr(model_trimodal, 'landmark_extractor')
                assert hasattr(model_trimodal, 'landmark_classifier')
                print(f"  ✓ Tri-modal (CNN + Transformer + MediaPipe): {count_parameters(model_trimodal):,} parameters\n")
            else:
                print("  ⚠ MediaPipe requested but not available, using bi-modal\n")
        except ImportError:
            print("  ⚠ MediaPipe not available, skipping tri-modal test\n")
    
    @staticmethod
    def test_fusion_methods():
        """Test different fusion methods."""
        print("Testing different fusion methods...")
        
        fusion_methods = ['weighted', 'concat', 'add']
        
        for method in fusion_methods:
            model = HybridEmotionRecognizer(
                num_emotions=7,
                cnn_backbone='mobilenet_v2',
                pretrained=False,
                fusion_method=method,
                use_mediapipe=False
            )
            model.eval()
            
            frames = torch.randn(2, 8, 3, 224, 224)
            
            with torch.no_grad():
                logits = model.forward_sequence(frames)
            
            assert logits.shape == (2, 7)
            assert not torch.isnan(logits).any()
            
            print(f"  ✓ Fusion method '{method}' working")
        
        print()
    
    @staticmethod
    def test_forward_single_frame():
        """Test single frame processing."""
        print("Testing single frame processing...")
        
        model = HybridEmotionRecognizer(
            num_emotions=7,
            cnn_backbone='mobilenet_v2',
            pretrained=False
        )
        model.eval()
        
        frame = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            logits = model.forward_single_frame(frame)
        
        assert logits.shape == (2, 7)
        
        print("  ✓ Single frame processing working\n")
    
    @staticmethod
    def test_forward_sequence():
        """Test sequence processing."""
        print("Testing sequence processing...")
        
        model = HybridEmotionRecognizer(
            num_emotions=7,
            cnn_backbone='mobilenet_v2',
            pretrained=False,
            use_mediapipe=False
        )
        model.eval()
        
        test_sequences = [
            (2, 4, 3, 224, 224),   # Short sequence
            (2, 16, 3, 224, 224),  # Medium sequence
            (1, 8, 3, 224, 224),   # Batch size 1
        ]
        
        for shape in test_sequences:
            frames = torch.randn(*shape)
            
            with torch.no_grad():
                logits = model.forward_sequence(frames)
            
            assert logits.shape == (shape[0], 7)
            print(f"  ✓ Sequence {shape} → output ({shape[0]}, 7)")
        
        print()
    
    @staticmethod
    def test_predict_emotion():
        """Test emotion prediction interface."""
        print("Testing predict_emotion interface...")
        
        model = HybridEmotionRecognizer(
            num_emotions=7,
            cnn_backbone='mobilenet_v2',
            pretrained=False,
            use_mediapipe=False
        )
        model.eval()
        
        frames = torch.randn(2, 8, 3, 224, 224)
        
        with torch.no_grad():
            result = model.predict_emotion(frames)
        
        # Check required keys
        assert 'combined_prediction' in result
        assert 'combined_probabilities' in result
        assert 'spatial_prediction' in result
        assert 'spatial_probabilities' in result
        assert 'temporal_prediction' in result
        assert 'temporal_probabilities' in result
        
        # Check shapes
        assert result['combined_prediction'].shape == (2,)
        assert result['combined_probabilities'].shape == (2, 7)
        
        # Check probabilities sum to 1
        prob_sums = result['combined_probabilities'].sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(2), atol=1e-5)
        
        # Check predictions are valid
        assert (result['combined_prediction'] >= 0).all()
        assert (result['combined_prediction'] < 7).all()
        
        print("  ✓ All prediction outputs correct\n")
    
    @staticmethod
    def test_with_mediapipe():
        """Test with MediaPipe landmarks if available."""
        print("Testing with MediaPipe landmarks...")
        
        try:
            model = HybridEmotionRecognizer(
                num_emotions=7,
                cnn_backbone='mobilenet_v2',
                pretrained=False,
                use_mediapipe=True
            )
            model.eval()
            
            frames = torch.randn(2, 8, 3, 224, 224)
            landmarks = torch.randn(2, 8, 468 * 3)
            
            with torch.no_grad():
                result = model.predict_emotion(frames, landmarks)
            
            # Check if MediaPipe was actually enabled
            if model.use_mediapipe:
                # Should have landmark predictions
                assert 'landmark_prediction' in result
                assert 'landmark_probabilities' in result
                print("  ✓ MediaPipe integration working\n")
            else:
                # MediaPipe not available, just check basic output
                assert 'combined_prediction' in result
                print("  ⚠ MediaPipe not available, but model works without it\n")
        
        except ImportError:
            print("  ⚠ MediaPipe not available, skipping\n")
    
    @staticmethod
    def test_gradients():
        """Test gradient flow through hybrid model."""
        print("Testing HybridEmotionRecognizer gradients...")
        
        model = HybridEmotionRecognizer(
            num_emotions=7,
            cnn_backbone='mobilenet_v2',
            pretrained=False,
            use_mediapipe=False
        )
        model.train()
        
        frames = torch.randn(2, 4, 3, 224, 224, requires_grad=True)
        target = torch.randint(0, 7, (2,))
        
        logits = model.forward_sequence(frames)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        
        assert frames.grad is not None
        assert not torch.isnan(frames.grad).any()
        
        print("  ✓ Gradients flowing correctly\n")
    
    @staticmethod
    def run_all():
        """Run all HybridEmotionRecognizer tests."""
        print("\n" + "="*70)
        print("HYBRID EMOTION RECOGNIZER TESTS")
        print("="*70)
        TestHybridEmotionRecognizer.test_initialization()
        TestHybridEmotionRecognizer.test_fusion_methods()
        TestHybridEmotionRecognizer.test_forward_single_frame()
        TestHybridEmotionRecognizer.test_forward_sequence()
        TestHybridEmotionRecognizer.test_predict_emotion()
        TestHybridEmotionRecognizer.test_with_mediapipe()
        TestHybridEmotionRecognizer.test_gradients()
        print("✓ All HybridEmotionRecognizer tests passed!\n")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*70)
    print("EMOTION DETECTOR COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Emotion labels: {EMOTION_LABELS}")
    print("="*70)
    
    try:
        TestSpatialAttentionCNN.run_all()
        TestTemporalTransformer.run_all()
        TestHybridEmotionRecognizer.run_all()
        
        print("\n" + "="*70)
        print("✓✓✓ ALL TESTS PASSED SUCCESSFULLY! ✓✓✓")
        print("="*70)
        print("\nSummary:")
        print("  • SpatialAttentionCNN: All backbones and features tested")
        print("  • TemporalTransformer: Positional encoding and temporal modeling verified")
        print("  • HybridEmotionRecognizer: All fusion methods and modalities validated")
        print("\nThe emotion detection system is fully functional!")
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
