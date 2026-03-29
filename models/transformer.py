"""
Scale-Interaction Transformer (SIT) for Beauty Score Prediction
Based on the architecture described in Algorithm 1 and 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Tuple


class MultiScaleFeatureModule(nn.Module):
    """
    Multi-Scale Feature Extraction Module using MobileNetV2 as backbone.
    Extracts features at 3 different scales from the backbone network.
    """
    def __init__(self, scales: List[int] = [3, 10, 13]):
        """
        Args:
            scales: List of layer indices to extract features from MobileNetV2
        """
        super(MultiScaleFeatureModule, self).__init__()
        
        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.features = mobilenet.features
        self.scales = scales
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract multi-scale features from input image.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Tuple of feature maps at different scales (F_base, F_mid, F_high)
        """
        scale_features = []
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.scales:
                scale_features.append(x)
                
        return tuple(scale_features)


class TransformerBlock(nn.Module):
    """
    Transformer Block with Multi-Head Self-Attention as described in Algorithm 1.
    Includes residual connections, layer normalization, and feed-forward network.
    """
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        # FFN with ReLU as specified in Algorithm 1
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass following Algorithm 1.
        
        Args:
            x: Input tensor of shape (B, n, d_model)
            
        Returns:
            Transformed features of shape (B, n, d_model)
        """
        # Line 1: LayerNorm
        h = self.norm1(x)
        
        # Line 2: Multi-Head Self-Attention
        h, _ = self.attention(h, h, h)
        
        # Line 3: Residual connection 1 with dropout
        x = x + self.dropout1(h)
        
        # Line 4: LayerNorm
        h = self.norm2(x)
        
        # Line 5: FFN
        f = self.ffn(h)
        
        # Line 6: Residual connection 2 with dropout
        x = x + self.dropout2(f)
        
        return x


class RegressionHead(nn.Module):
    """
    Regression head for predicting beauty scores.
    Uses global average pooling followed by a linear layer with dropout.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of input features
            dropout: Dropout rate
        """
        super(RegressionHead, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict beauty score from sequence representation.
        
        Args:
            x: Input tensor of shape (B, n, d_model)
            
        Returns:
            Predicted beauty scores of shape (B, 1)
        """
        # Global average pooling across sequence dimension (n=3)
        v = torch.mean(x, dim=1)  # (B, d_model)
        
        # Apply dropout and linear layer (Equation 11)
        v = self.dropout(v)
        y_hat = self.fc(v)
        
        return y_hat


class ScaleInteractionTransformer(nn.Module):
    """
    Complete Scale-Interaction Transformer (SIT) model for beauty score prediction.
    Combines multi-scale feature extraction, transformer blocks, and regression head.
    """
    def __init__(
        self, 
        scales: List[int] = [3, 10, 13],
        d_proj: int = 128,
        num_transformer_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            scales: Layer indices for multi-scale feature extraction
            d_proj: Projection dimension for transformer
            num_transformer_blocks: Number of transformer blocks (L=2 in paper)
            num_heads: Number of attention heads (4 in paper)
            dropout: Dropout rate
        """
        super(ScaleInteractionTransformer, self).__init__()
        
        # Multi-scale feature extraction
        self.feature_extractor = MultiScaleFeatureModule(scales)
        
        # Get feature dimensions from MobileNetV2 architecture
        # Scales [3, 10, 13] correspond to channels [24, 64, 96]
        # Layer mapping: 3->24ch, 10->64ch, 13->96ch
        self.feature_dims = [24, 64, 96]
        
        # Linear projection for each scale (Equation 9)
        self.projections = nn.ModuleList([
            nn.Linear(dim, d_proj) for dim in self.feature_dims
        ])
        
        # Learnable bias terms for each scale
        self.proj_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(d_proj)) for _ in self.feature_dims
        ])
        
        # Transformer blocks (L=2)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_proj, num_heads, dropout) 
            for _ in range(num_transformer_blocks)
        ])
        
        # Regression head
        self.regression_head = RegressionHead(d_proj, dropout)
        
        self.d_proj = d_proj
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the complete SIT model.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            
        Returns:
            Predicted beauty scores of shape (B, 1)
        """
        # Step 5: Extract multi-scale features using MobileNetV2
        F_base, F_mid, F_high = self.feature_extractor(x)
        
        # Step 6: Process each scale into feature vectors via global average pooling
        scale_features = []
        for feat in [F_base, F_mid, F_high]:
            # Global average pooling: (B, C, H, W) -> (B, C)
            pooled = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)
            scale_features.append(pooled)
        
        # Step 7: Linear projection (Equation 9)
        # Create sequence S of shape (B, 3, d_proj)
        projected = []
        for i, (feat, proj, bias) in enumerate(zip(scale_features, self.projections, self.proj_biases)):
            # S_proj = S * W_proj + b_proj
            proj_feat = proj(feat) + bias
            projected.append(proj_feat)
        
        # Stack to create sequence: (B, 3, d_proj)
        S = torch.stack(projected, dim=1)
        
        # Step 8: Process through transformer blocks
        S_trans = S
        for transformer in self.transformer_blocks:
            S_trans = transformer(S_trans)
        
        # Step 9-10: Global average pooling and regression
        y_hat = self.regression_head(S_trans)
        
        return y_hat


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = ScaleInteractionTransformer(
        scales=[3, 10, 13],
        d_proj=128,
        num_transformer_blocks=2,
        num_heads=4,
        dropout=0.1
    ).to(device)
    
    print(f"Total trainable parameters: {count_parameters(model):,}")
    
    # Test with random input
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Forward pass
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample predictions: {output.squeeze().detach().cpu().numpy()}")
