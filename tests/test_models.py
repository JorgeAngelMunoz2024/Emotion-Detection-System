"""
Test script to verify the model architecture works correctly.
Run this before starting full training to ensure everything is set up properly.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.transformer import ScaleInteractionTransformer, count_parameters
from models.cnn import EmotionDetectorCNN, HybridEmotionRecognition


def test_transformer():
    """Test the Scale-Interaction Transformer model."""
    print("=" * 60)
    print("Testing Scale-Interaction Transformer")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    model = ScaleInteractionTransformer(
        scales=[3, 10, 13],
        d_proj=128,
        num_transformer_blocks=2,
        num_heads=4,
        dropout=0.1
    ).to(device)
    
    # Model info
    n_params = count_parameters(model)
    print(f"Total parameters: {n_params:,}")
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print(f"\nInput shape: {test_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Sample predictions: {output.squeeze().cpu().numpy()}")
    
    # Verify output shape
    assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"
    
    print("\n✓ Transformer test passed!")


def test_cnn():
    """Test the CNN model."""
    print("\n" + "=" * 60)
    print("Testing CNN Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    model = EmotionDetectorCNN(
        num_emotions=7,
        backbone='mobilenet_v2',
        pretrained=True,
        dropout=0.5
    ).to(device)
    
    # Model info
    n_params = count_parameters(model)
    print(f"Total parameters: {n_params:,}")
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print(f"\nInput shape: {test_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Sample predictions: {output.cpu().numpy()}")
    
    # Verify output shape (num_emotions instead of 1)
    assert output.shape == (batch_size, 7), f"Expected shape ({batch_size}, 7), got {output.shape}"
    
    print("\n✓ CNN test passed!")


def test_hybrid():
    """Test the Hybrid model."""
    print("\n" + "=" * 60)
    print("Testing Hybrid Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    model = HybridEmotionRecognition(
        num_emotions=7,
        cnn_backbone='mobilenet_v2',
        fusion_method='concat',
        dropout=0.3
    ).to(device)
    
    # Model info
    n_params = count_parameters(model)
    print(f"Total parameters: {n_params:,}")
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print(f"\nInput shape: {test_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Sample predictions: {output.squeeze().cpu().numpy()}")
    
    # Verify output shape
    assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"
    
    print("\n✓ Hybrid test passed!")


def test_backward_pass():
    """Test that backpropagation works correctly."""
    print("\n" + "=" * 60)
    print("Testing Backward Pass")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    model = ScaleInteractionTransformer().to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    # Test data
    test_input = torch.randn(2, 3, 224, 224).to(device)
    test_target = torch.randn(2, 1).to(device)
    
    # Forward pass
    output = model(test_input)
    loss = criterion(output, test_target)
    
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check that gradients exist
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients computed!"
    
    print("✓ Backward pass test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE VERIFICATION")
    print("=" * 60)
    
    try:
        # Test all models
        test_transformer()
        test_cnn()
        test_hybrid()
        test_backward_pass()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour setup is ready for training.")
        print("Run 'python train.py' or './quick_start.sh' to begin.")
        return 0
            
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ ERROR OCCURRED")
        print("=" * 60)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
