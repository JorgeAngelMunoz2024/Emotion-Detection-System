"""
Real-time Emotion Detection System
Combines CNN spatial features with Transformer temporal modeling for video emotion recognition.

Architecture:
1. CNN with Spatial Attention: Extracts features from facial regions (eyes, mouth, nose, cheeks)
2. Temporal Transformer: Models emotion transitions over time
3. Hybrid Fusion: Combines spatial + temporal for robust emotion prediction

Use Case: Webcam-based real-time emotion detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, List, Tuple, Dict
import numpy as np

try:
    from .mediapipe_detector import FaceMeshFeatureExtractor
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# Emotion labels (7 basic emotions + neutral)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


class SpatialAttentionCNN(nn.Module):
    """
    CNN with spatial attention focusing on facial regions for emotion detection.
    Can use pretrained backbone (recommended) or train from scratch.
    """
    def __init__(
        self,
        num_emotions: int = 7,
        backbone: str = 'mobilenet_v2',
        pretrained: bool = True,  # Recommended: True for faster convergence
        use_attention: bool = True,
        dropout: float = 0.5
    ):
        """
        Args:
            num_emotions: Number of emotion classes (default: 7)
            backbone: 'mobilenet_v2' (fastest), 'resnet18', 'resnet34', 'resnet50'
            pretrained: Use ImageNet pretrained weights (highly recommended!)
            use_attention: Enable spatial attention on facial features
            dropout: Dropout rate for regularization
        """
        super(SpatialAttentionCNN, self).__init__()
        
        self.num_emotions = num_emotions
        self.use_attention = use_attention
        
        # Load backbone
        if backbone == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Spatial attention for facial regions
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, feature_dim),
                nn.Sigmoid()
            )
        
        # Emotion classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_emotions)
        )
        
        # Feature extractor for temporal model
        self.feature_dim = feature_dim
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial features from image."""
        features = self.backbone(x)
        
        if self.use_attention:
            attention_weights = self.attention(features)
            features = features * attention_weights
        
        return features
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, H, W)
            return_features: If True, return features instead of logits
            
        Returns:
            Emotion logits (B, num_emotions) or features (B, feature_dim)
        """
        features = self.extract_features(x)
        
        if return_features:
            return features
        
        logits = self.classifier(features)
        return logits
    
    def predict_emotion(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict emotion with probabilities.
        
        Returns:
            (predicted_class, probabilities)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        predicted = torch.argmax(probs, dim=1)
        return predicted, probs


class TemporalTransformer(nn.Module):
    """
    Transformer for temporal emotion modeling.
    Processes sequences of CNN features to understand emotion transitions.
    """
    def __init__(
        self,
        feature_dim: int,
        num_emotions: int = 7,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 32
    ):
        """
        Args:
            feature_dim: Dimension of input features from CNN
            num_emotions: Number of emotion classes
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_seq_length: Maximum sequence length for positional encoding
        """
        super(TemporalTransformer, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_emotions = num_emotions
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_length, feature_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Temporal pooling and classification
        self.temporal_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_emotions)
        )
        
    def forward(self, features_sequence: torch.Tensor) -> torch.Tensor:
        """
        Process temporal sequence of features.
        
        Args:
            features_sequence: Sequence of features (B, T, feature_dim)
            
        Returns:
            Emotion logits (B, num_emotions)
        """
        B, T, D = features_sequence.shape
        
        # Add positional encoding
        pos_enc = self.positional_encoding[:, :T, :]
        features_sequence = features_sequence + pos_enc
        
        # Transformer encoding
        temporal_features = self.transformer(features_sequence)
        
        # Global average pooling over time
        pooled_features = torch.mean(temporal_features, dim=1)
        
        # Classify
        logits = self.temporal_classifier(pooled_features)
        return logits


class HybridEmotionRecognizer(nn.Module):
    """
    Hybrid architecture combining spatial CNN, temporal Transformer, and optional MediaPipe landmarks.
    
    Workflow:
    1. CNN: Extract spatial features from each frame + predict immediate emotion
    2. Transformer: Model temporal dynamics across frame sequence
    3. MediaPipe (optional): Extract geometric features from facial landmarks
    4. Fusion: Combine all modalities for final output
    """
    def __init__(
        self,
        num_emotions: int = 7,
        cnn_backbone: str = 'mobilenet_v2',
        pretrained: bool = True,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        fusion_method: str = 'weighted',  # 'weighted', 'concat', 'add'
        use_mediapipe: bool = False,  # Enable MediaPipe landmark features
        dropout: float = 0.3
    ):
        """
        Args:
            num_emotions: Number of emotion classes
            cnn_backbone: CNN backbone architecture
            pretrained: Use pretrained CNN backbone (recommended)
            temporal_layers: Number of transformer layers
            temporal_heads: Number of transformer attention heads
            fusion_method: How to combine spatial and temporal predictions
            use_mediapipe: Enable MediaPipe facial landmark features
            dropout: Dropout rate
        """
        super(HybridEmotionRecognizer, self).__init__()
        
        self.num_emotions = num_emotions
        self.fusion_method = fusion_method
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        
        # Spatial CNN
        self.spatial_cnn = SpatialAttentionCNN(
            num_emotions=num_emotions,
            backbone=cnn_backbone,
            pretrained=pretrained,
            use_attention=True,
            dropout=dropout
        )
        
        # Temporal Transformer
        self.temporal_transformer = TemporalTransformer(
            feature_dim=self.spatial_cnn.feature_dim,
            num_emotions=num_emotions,
            num_layers=temporal_layers,
            num_heads=temporal_heads,
            dropout=dropout
        )
        
        # MediaPipe landmark feature extractor (optional)
        if self.use_mediapipe:
            self.landmark_extractor = FaceMeshFeatureExtractor(
                landmark_dim=468 * 3,  # 468 landmarks * (x, y, z)
                hidden_dim=256
            )
            # Landmark-based emotion classifier
            self.landmark_classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_emotions)
            )
        
        # Fusion layers
        num_modalities = 3 if self.use_mediapipe else 2
        if fusion_method == 'weighted':
            self.fusion_weight = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        elif fusion_method == 'concat':
            self.fusion_fc = nn.Sequential(
                nn.Linear(num_emotions * num_modalities, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_emotions)
            )
        elif fusion_method == 'add':
            pass  # Simple addition
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Process a single frame (for real-time inference).
        
        Args:
            frame: Single frame (B, 3, H, W)
            
        Returns:
            Spatial emotion logits (B, num_emotions)
        """
        return self.spatial_cnn(frame)
    
    def forward_sequence(
        self, 
        frame_sequence: torch.Tensor,
        landmarks_sequence: Optional[torch.Tensor] = None,
        return_spatial: bool = False
    ) -> torch.Tensor:
        """
        Process a sequence of frames with optional MediaPipe landmarks.
        
        Args:
            frame_sequence: Sequence of frames (B, T, 3, H, W)
            landmarks_sequence: Optional sequence of landmarks (B, T, 468*3) if use_mediapipe=True
            return_spatial: If True, also return individual modality predictions
            
        Returns:
            Combined emotion logits (B, num_emotions) or tuple if return_spatial=True
        """
        B, T, C, H, W = frame_sequence.shape
        
        # Validate landmark input
        if self.use_mediapipe and landmarks_sequence is None:
            raise ValueError("landmarks_sequence required when use_mediapipe=True")
        if not self.use_mediapipe and landmarks_sequence is not None:
            landmarks_sequence = None  # Ignore landmarks if not enabled
        
        # Extract spatial features for each frame
        features_list = []
        spatial_logits_list = []
        
        for t in range(T):
            frame = frame_sequence[:, t, :, :, :]
            features = self.spatial_cnn.extract_features(frame)
            features_list.append(features)
            
            spatial_logits = self.spatial_cnn.classifier(features)
            spatial_logits_list.append(spatial_logits)
        
        # Stack features and get average spatial prediction
        features_sequence = torch.stack(features_list, dim=1)  # (B, T, feature_dim)
        spatial_logits = torch.stack(spatial_logits_list, dim=1).mean(dim=1)  # (B, num_emotions)
        
        # Get temporal prediction
        temporal_logits = self.temporal_transformer(features_sequence)  # (B, num_emotions)
        
        # Get MediaPipe landmark prediction (if enabled)
        if self.use_mediapipe:
            # Process landmarks for each frame and average
            landmark_logits_list = []
            for t in range(T):
                landmarks = landmarks_sequence[:, t, :]  # (B, 468*3)
                landmark_features = self.landmark_extractor.landmark_encoder(landmarks)
                landmark_logits = self.landmark_classifier(landmark_features)
                landmark_logits_list.append(landmark_logits)
            landmark_logits = torch.stack(landmark_logits_list, dim=1).mean(dim=1)  # (B, num_emotions)
        
        # Fusion
        if self.use_mediapipe:
            # Tri-modal fusion: CNN + Transformer + MediaPipe
            if self.fusion_method == 'weighted':
                w = F.softmax(self.fusion_weight, dim=0)
                combined_logits = w[0] * spatial_logits + w[1] * temporal_logits + w[2] * landmark_logits
            elif self.fusion_method == 'concat':
                concat_logits = torch.cat([spatial_logits, temporal_logits, landmark_logits], dim=1)
                combined_logits = self.fusion_fc(concat_logits)
            elif self.fusion_method == 'add':
                combined_logits = spatial_logits + temporal_logits + landmark_logits
        else:
            # Bi-modal fusion: CNN + Transformer only
            if self.fusion_method == 'weighted':
                w = F.softmax(self.fusion_weight, dim=0)
                combined_logits = w[0] * spatial_logits + w[1] * temporal_logits
            elif self.fusion_method == 'concat':
                concat_logits = torch.cat([spatial_logits, temporal_logits], dim=1)
                combined_logits = self.fusion_fc(concat_logits)
            elif self.fusion_method == 'add':
                combined_logits = spatial_logits + temporal_logits
        
        if return_spatial:
            if self.use_mediapipe:
                return combined_logits, spatial_logits, temporal_logits, landmark_logits
            return combined_logits, spatial_logits, temporal_logits
        
        return combined_logits
    
    def predict_emotion(
        self, 
        frame_sequence: torch.Tensor,
        landmarks_sequence: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict emotion from frame sequence with detailed outputs.
        
        Args:
            frame_sequence: Sequence of frames (B, T, 3, H, W)
            landmarks_sequence: Optional sequence of landmarks (B, T, 468*3)
            
        Returns:
            Dictionary with predictions and probabilities
        """
        outputs = self.forward_sequence(
            frame_sequence, landmarks_sequence=landmarks_sequence, return_spatial=True
        )
        
        if self.use_mediapipe:
            combined_logits, spatial_logits, temporal_logits, landmark_logits = outputs
        else:
            combined_logits, spatial_logits, temporal_logits = outputs
        
        # Get probabilities
        combined_probs = F.softmax(combined_logits, dim=1)
        spatial_probs = F.softmax(spatial_logits, dim=1)
        temporal_probs = F.softmax(temporal_logits, dim=1)
        
        # Get predictions
        combined_pred = torch.argmax(combined_probs, dim=1)
        spatial_pred = torch.argmax(spatial_probs, dim=1)
        temporal_pred = torch.argmax(temporal_probs, dim=1)
        
        result = {
            'combined_prediction': combined_pred,
            'combined_probabilities': combined_probs,
            'spatial_prediction': spatial_pred,
            'spatial_probabilities': spatial_probs,
            'temporal_prediction': temporal_pred,
            'temporal_probabilities': temporal_probs
        }
        
        # Add MediaPipe predictions if enabled
        if self.use_mediapipe:
            landmark_probs = F.softmax(landmark_logits, dim=1)
            landmark_pred = torch.argmax(landmark_probs, dim=1)
            result.update({
                'landmark_prediction': landmark_pred,
                'landmark_probabilities': landmark_probs
            })
        
        return result


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Test Spatial CNN
    print("=" * 60)
    print("Testing Spatial Attention CNN")
    print("=" * 60)
    spatial_cnn = SpatialAttentionCNN(
        num_emotions=7,
        backbone='mobilenet_v2',
        pretrained=True,
        use_attention=True
    ).to(device)
    print(f"Parameters: {count_parameters(spatial_cnn):,}")
    
    test_frame = torch.randn(2, 3, 224, 224).to(device)
    spatial_output = spatial_cnn(test_frame)
    print(f"Input: {test_frame.shape} → Output: {spatial_output.shape}")
    predicted, probs = spatial_cnn.predict_emotion(test_frame)
    print(f"Predicted emotions: {predicted}")
    print(f"Probabilities shape: {probs.shape}\n")
    
    # Test Temporal Transformer
    print("=" * 60)
    print("Testing Temporal Transformer")
    print("=" * 60)
    temporal_transformer = TemporalTransformer(
        feature_dim=1280,  # MobileNetV2 feature dim
        num_emotions=7,
        num_layers=2,
        num_heads=4
    ).to(device)
    print(f"Parameters: {count_parameters(temporal_transformer):,}")
    
    test_sequence = torch.randn(2, 16, 1280).to(device)
    temporal_output = temporal_transformer(test_sequence)
    print(f"Input: {test_sequence.shape} → Output: {temporal_output.shape}\n")
    
    # Test Hybrid Model
    print("=" * 60)
    print("Testing Hybrid Emotion Recognizer")
    print("=" * 60)
    hybrid_model = HybridEmotionRecognizer(
        num_emotions=7,
        cnn_backbone='mobilenet_v2',
        pretrained=True,
        temporal_layers=2,
        temporal_heads=4,
        fusion_method='weighted',
        use_mediapipe=False  # Test without MediaPipe first
    ).to(device)
    print(f"Parameters: {count_parameters(hybrid_model):,}")
    
    test_frame_seq = torch.randn(2, 16, 3, 224, 224).to(device)
    hybrid_output = hybrid_model.forward_sequence(test_frame_seq)
    print(f"Input: {test_frame_seq.shape} → Output: {hybrid_output.shape}")
    
    # Test prediction
    predictions = hybrid_model.predict_emotion(test_frame_seq)
    print(f"\nPrediction results (bi-modal):")
    print(f"  Combined: {predictions['combined_prediction']}")
    print(f"  Spatial: {predictions['spatial_prediction']}")
    print(f"  Temporal: {predictions['temporal_prediction']}")
    
    # Test with MediaPipe if available
    if MEDIAPIPE_AVAILABLE:
        print("\n" + "=" * 60)
        print("Testing Hybrid Model with MediaPipe")
        print("=" * 60)
        hybrid_mediapipe = HybridEmotionRecognizer(
            num_emotions=7,
            cnn_backbone='mobilenet_v2',
            pretrained=True,
            temporal_layers=2,
            temporal_heads=4,
            fusion_method='weighted',
            use_mediapipe=True  # Enable MediaPipe
        ).to(device)
        print(f"Parameters: {count_parameters(hybrid_mediapipe):,}")
        
        # Create dummy landmarks (468 points * 3 coords)
        test_landmarks_seq = torch.randn(2, 16, 468 * 3).to(device)
        hybrid_mediapipe_output = hybrid_mediapipe.forward_sequence(
            test_frame_seq, landmarks_sequence=test_landmarks_seq
        )
        print(f"Input: {test_frame_seq.shape} + landmarks {test_landmarks_seq.shape} → Output: {hybrid_mediapipe_output.shape}")
        
        # Test prediction
        predictions_mediapipe = hybrid_mediapipe.predict_emotion(test_frame_seq, test_landmarks_seq)
        print(f"\nPrediction results (tri-modal):")
        print(f"  Combined: {predictions_mediapipe['combined_prediction']}")
        print(f"  Spatial: {predictions_mediapipe['spatial_prediction']}")
        print(f"  Temporal: {predictions_mediapipe['temporal_prediction']}")
        print(f"  Landmark: {predictions_mediapipe['landmark_prediction']}")
    else:
        print("\n⚠️  MediaPipe not available. Install with: pip install mediapipe>=0.10.0")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nRecommendation: Use pretrained=True for faster convergence!")
    print("Pretrained weights from ImageNet provide excellent feature extractors.")
    if MEDIAPIPE_AVAILABLE:
        print("\nMediaPipe Integration:")
        print("  - Adds 468 facial landmarks for geometric features")
        print("  - Complements CNN appearance features with landmark geometry")
        print("  - Use use_mediapipe=True for enhanced emotion detection")
