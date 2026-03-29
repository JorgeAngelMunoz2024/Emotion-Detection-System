"""
CNN-based model for emotion detection with spatial attention.
Focuses on facial regions (eyes, mouth, nose, cheeks) for feature extraction.
This will be combined with the transformer for temporal emotion analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, List, Tuple


class SpatialAttentionModule(nn.Module):
    """
    Spatial attention module to focus on important facial regions
    (eyes, mouth, nose, cheeks) for emotion detection.
    """
    def __init__(self, feature_dim: int):
        """
        Args:
            feature_dim: Dimension of input features
        """
        super(SpatialAttentionModule, self).__init__()
        
        # Attention weights generator
        self.attention_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 8),
            nn.ReLU(),
            nn.Linear(feature_dim // 8, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention to features.
        
        Args:
            x: Input features of shape (B, feature_dim)
            
        Returns:
            Attended features of shape (B, feature_dim)
        """
        # Generate attention weights
        attention_weights = self.attention_fc(x)
        
        # Apply attention
        attended_features = x * attention_weights
        
        return attended_features


class EmotionDetectorCNN(nn.Module):
    """
    CNN-based emotion detector with spatial attention on facial regions.
    Focuses on eyes, mouth, nose, and cheeks for emotion recognition.
    Can be used standalone or as part of a hybrid architecture.
    """
    def __init__(
        self, 
        num_emotions: int = 7,  # 7 basic emotions: angry, disgust, fear, happy, sad, surprise, neutral
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.5,
        hidden_dim: int = 512,
        use_attention: bool = True
    ):
        """
        Args:
            num_emotions: Number of emotion classes
            backbone: Backbone architecture ('resnet50', 'resnet34', 'mobilenet_v2')
            pretrained: Whether to use pretrained weights (recommended for faster convergence)
            dropout: Dropout rate
            hidden_dim: Hidden dimension for fully connected layers
            use_attention: Whether to use spatial attention on facial regions
        """
        super(EmotionDetectorCNN, self).__init__()
        
        self.backbone_name = backbone
        self.num_emotions = num_emotions
        self.use_attention = use_attention
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Spatial attention module for facial regions
        if use_attention:
            self.attention = SpatialAttentionModule(feature_dim)
        
        # Emotion classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_emotions)
        )
        
        # Feature extractor for hybrid model (returns features before classification)
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            return_features: If True, return intermediate features for hybrid model
            
        Returns:
            Emotion logits of shape (B, num_emotions) or features if return_features=True
        """
        # Extract features
        features = self.backbone(x)  # (B, feature_dim)
        
        # Apply spatial attention if enabled
        if self.use_attention:
            features = self.attention(features)
        
        # Return features for hybrid model
        if return_features:
            return self.feature_extractor(features)
        
        # Classify emotion
        emotion_logits = self.classifier(features)
        
        return emotion_logits
    
    def get_emotion_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get emotion probabilities using softmax.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            
        Returns:
            Emotion probabilities of shape (B, num_emotions)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class HybridEmotionRecognition(nn.Module):
    """
    Hybrid model combining CNN spatial features with Transformer temporal modeling.
    
    Architecture:
    1. CNN: Extracts spatial features from current frame and detects immediate emotions
    2. Transformer: Models temporal dynamics of emotion transitions over time
    3. Fusion: Combines spatial + temporal for overall emotion state prediction
    
    Use case: Real-time webcam emotion detection
    """
    def __init__(
        self,
        num_emotions: int = 7,
        cnn_backbone: str = 'mobilenet_v2',
        transformer_config: Optional[dict] = None,
        temporal_window: int = 16,  # Number of frames to consider
        fusion_method: str = 'concat',
        dropout: float = 0.3
    ):
        """
        Args:
            num_emotions: Number of emotion classes
            cnn_backbone: CNN backbone architecture
            transformer_config: Configuration for transformer (not used for emotion - we'll use custom temporal)
            temporal_window: Number of frames in temporal sequence
            fusion_method: How to fuse CNN and temporal features ('concat', 'add', 'attention')
            dropout: Dropout rate
        """
        super(HybridEmotionRecognition, self).__init__()
        
        from models.transformer import ScaleInteractionTransformer
        
        # CNN branch
        if cnn_backbone == 'resnet50':
            self.cnn_backbone = models.resnet50(pretrained=True)
            cnn_feature_dim = self.cnn_backbone.fc.in_features
            self.cnn_backbone.fc = nn.Identity()
        elif cnn_backbone == 'mobilenet_v2':
            self.cnn_backbone = models.mobilenet_v2(pretrained=True)
            cnn_feature_dim = self.cnn_backbone.classifier[1].in_features
            self.cnn_backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported CNN backbone: {cnn_backbone}")
        
        # Transformer branch
        if transformer_config is None:
            transformer_config = {
                'scales': [3, 10, 13],
                'd_proj': 128,
                'num_transformer_blocks': 2,
                'num_heads': 4,
                'dropout': 0.1
            }
        
        self.transformer = ScaleInteractionTransformer(**transformer_config)
        transformer_feature_dim = transformer_config['d_proj']
        
        self.fusion_method = fusion_method
        
        # Fusion layer
        if fusion_method == 'concat':
            fusion_dim = cnn_feature_dim + transformer_feature_dim
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1)
            )
        elif fusion_method == 'add':
            # Project both to same dimension
            self.cnn_proj = nn.Linear(cnn_feature_dim, 256)
            self.trans_proj = nn.Linear(transformer_feature_dim, 256)
            self.fusion = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1)
            )
        elif fusion_method == 'attention':
            # Cross-attention fusion
            self.cnn_proj = nn.Linear(cnn_feature_dim, 256)
            self.trans_proj = nn.Linear(transformer_feature_dim, 256)
            self.attention = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
            self.fusion = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1)
            )
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining CNN and transformer features.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            
        Returns:
            Predicted beauty scores of shape (B, 1)
        """
        # Extract CNN features
        cnn_features = self.cnn_backbone(x)  # (B, cnn_feature_dim)
        
        # Extract transformer features
        # Get the fused representation before final regression
        trans_features = self.transformer.feature_extractor(x)
        
        # Process through transformer
        F_base, F_mid, F_high = trans_features
        scale_features = []
        for feat in [F_base, F_mid, F_high]:
            pooled = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)
            scale_features.append(pooled)
        
        projected = []
        for feat, proj, bias in zip(scale_features, 
                                   self.transformer.projections, 
                                   self.transformer.proj_biases):
            proj_feat = proj(feat) + bias
            projected.append(proj_feat)
        
        S = torch.stack(projected, dim=1)
        S_trans = S
        for transformer_block in self.transformer.transformer_blocks:
            S_trans = transformer_block(S_trans)
        
        # Global average pooling
        trans_features = torch.mean(S_trans, dim=1)  # (B, d_proj)
        
        # Fuse features
        if self.fusion_method == 'concat':
            fused = torch.cat([cnn_features, trans_features], dim=1)
            output = self.fusion(fused)
        elif self.fusion_method == 'add':
            cnn_proj = self.cnn_proj(cnn_features)
            trans_proj = self.trans_proj(trans_features)
            fused = cnn_proj + trans_proj
            output = self.fusion(fused)
        elif self.fusion_method == 'attention':
            cnn_proj = self.cnn_proj(cnn_features).unsqueeze(1)  # (B, 1, 256)
            trans_proj = self.trans_proj(trans_features).unsqueeze(1)  # (B, 1, 256)
            attended, _ = self.attention(cnn_proj, trans_proj, trans_proj)
            fused = attended.squeeze(1)
            output = self.fusion(fused)
        
        return output


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test CNN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Test CNN model
    print("Testing CNN model...")
    cnn_model = CNNBeautyPredictor(backbone='mobilenet_v2').to(device)
    print(f"CNN parameters: {count_parameters(cnn_model):,}")
    
    test_input = torch.randn(4, 3, 224, 224).to(device)
    output = cnn_model(test_input)
    print(f"CNN output shape: {output.shape}\n")
    
    # Test Hybrid model
    print("Testing Hybrid model...")
    hybrid_model = HybridCNNTransformer(
        cnn_backbone='mobilenet_v2',
        fusion_method='concat'
    ).to(device)
    print(f"Hybrid parameters: {count_parameters(hybrid_model):,}")
    
    output = hybrid_model(test_input)
    print(f"Hybrid output shape: {output.shape}")
