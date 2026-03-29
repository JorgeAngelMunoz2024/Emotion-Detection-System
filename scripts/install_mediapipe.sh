#!/bin/bash
# Install compatible MediaPipe version with solutions module using uv

echo "========================================"
echo "Installing MediaPipe with uv"
echo "========================================"

# Install curl if needed (required for uv installer)
if ! command -v curl &> /dev/null; then
    echo "Installing curl..."
    apt-get update -qq && apt-get install -y -qq curl wget ca-certificates
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv (fast Python package installer)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    # Source the cargo env to get uv in PATH
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
    
    # Verify installation
    if ! command -v uv &> /dev/null; then
        echo "✗ uv installation failed, trying alternative method..."
        # Try pip install as fallback
        pip install uv
        if command -v uv &> /dev/null; then
            echo "✓ uv installed via pip"
            USE_UV=true
        else
            echo "✗ uv cannot be installed, falling back to pip"
            USE_UV=false
        fi
    else
        echo "✓ uv installed successfully"
        USE_UV=true
    fi
else
    echo "✓ uv already installed"
    USE_UV=true
fi

if [ "$USE_UV" = true ]; then
    # Use uv for better dependency resolution
    echo ""
    echo "Installing packages with uv (smart dependency resolver)..."
    
    # Strategy: Try newest MediaPipe first (might have better protobuf compatibility)
    echo "Attempting MediaPipe 0.10.33 (newest) with uv resolver..."
    uv pip install --system mediapipe==0.10.33 fer
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "Trying MediaPipe 0.10.14..."
        uv pip install --system mediapipe==0.10.14 fer
        
        if [ $? -ne 0 ]; then
            echo ""
            echo "Last resort: Installing with protobuf 4.x (may break TensorFlow temporarily)..."
            uv pip install --system --force-reinstall 'protobuf>=4.25.3,<5'
            uv pip install --system --no-deps mediapipe==0.10.14
            uv pip install --system fer
        fi
    fi
    
    # Check if MediaPipe installed
    if python3 -c "import mediapipe" 2>/dev/null; then
        echo "✓ MediaPipe installed successfully"
    else
        echo "✗ MediaPipe installation failed with uv, falling back to FER-only mode"
    fi
    
else
    # Fallback to pip
    echo ""
    echo "Using pip (uv not available)..."
    
    # First, restore compatible protobuf for TensorFlow
    echo "Ensuring TensorFlow-compatible protobuf..."
    uv pip install --upgrade 'protobuf>=6.31.1,<8.0.0'
    
    # Try MediaPipe 0.10.14
    echo ""
    echo "Attempting to install MediaPipe 0.10.14..."
    uv pip uninstall -y mediapipe 2>/dev/null
    uv pip install mediapipe==0.10.14
    
    # If that fails, try other versions
    if [ $? -ne 0 ]; then
        echo "MediaPipe 0.10.14 failed, trying 0.10.21..."
        uv pip install mediapipe==0.10.21
    fi
    
    # Reinstall FER
    echo ""
    echo "Reinstalling FER..."
    uv pip install --upgrade fer
fi

# Test import
echo ""
echo "========================================"
echo "Testing Installation"
echo "========================================"
python3 << 'PYTHON_TEST'
import sys

# Test MediaPipe
print("Testing MediaPipe...")
try:
    import mediapipe as mp
    print(f"✓ MediaPipe {mp.__version__} installed")
    
    # Test solutions module
    has_solutions = False
    if hasattr(mp, 'solutions'):
        print("✓ mp.solutions available")
        has_solutions = True
        if hasattr(mp.solutions, 'face_mesh'):
            print("✓ mp.solutions.face_mesh available")
        else:
            print("✗ mp.solutions.face_mesh NOT available")
    else:
        # Try legacy import
        try:
            from mediapipe.python import solutions
            mp.solutions = solutions
            print("✓ Legacy mediapipe.python.solutions available")
            has_solutions = True
            if hasattr(mp.solutions, 'face_mesh'):
                print("✓ face_mesh available via legacy import")
        except ImportError:
            print("✗ No solutions module found")
    
    if not has_solutions:
        print("\n⚠️  WARNING: MediaPipe installed but solutions module unavailable")
        print("   Landmarks/mesh features will not work")
        
except ImportError as e:
    print(f"✗ MediaPipe import failed: {e}")
    print("\n⚠️  Will use FER-only mode (no landmarks/mesh)")

# Test FER
print("\nTesting FER...")
try:
    from fer import FER
    print("✓ FER module available")
    # Quick test (without MTCNN to avoid slow init)
    try:
        detector = FER(mtcnn=False)
        print("✓ FER detector can be created")
    except Exception as e:
        print(f"⚠️  FER detector creation issue: {e}")
except Exception as e:
    print(f"✗ FER failed: {e}")

# Test TensorFlow
print("\nTesting TensorFlow...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} available")
except Exception as e:
    print(f"⚠️  TensorFlow issue: {e}")

print("\n" + "="*50)
print("Summary:")
print("  - Run the demo with: python /app/tools/demo/multimodal_demo.py")
print("  - FER will provide emotion detection")
print("  - MediaPipe (if working) provides landmarks/mesh")
print("="*50)
PYTHON_TEST

echo ""
echo "========================================"
echo "MediaPipe installation complete"
echo "========================================"
