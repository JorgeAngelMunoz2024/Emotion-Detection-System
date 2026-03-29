#!/bin/bash
# Script to download the FER2013 pretrained model from Kaggle
# Run this BEFORE building the Docker container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"

echo "=================================="
echo "FER2013 Pretrained Model Download"
echo "=================================="
echo ""

# Create checkpoints directory if it doesn't exist
mkdir -p "$CHECKPOINTS_DIR"

MODEL_PATH="$CHECKPOINTS_DIR/face_model.h5"

if [ -f "$MODEL_PATH" ]; then
    echo "✓ Model already exists at: $MODEL_PATH"
    echo "  Size: $(ls -lh "$MODEL_PATH" | awk '{print $5}')"
    exit 0
fi

echo "The FER2013 pretrained model (face_model.h5) is not found."
echo ""
echo "To download it:"
echo ""
echo "Option 1: Manual Download"
echo "  1. Go to: https://www.kaggle.com/datasets/abhisheksingh016/machine-model-for-emotion-detection"
echo "  2. Click 'Download' button"
echo "  3. Extract and copy face_model.h5 to: $CHECKPOINTS_DIR/"
echo ""
echo "Option 2: Kaggle CLI (if you have kaggle credentials)"
echo "  1. Install kaggle: pip install kaggle"
echo "  2. Set up kaggle API credentials (~/.kaggle/kaggle.json)"
echo "  3. Run:"
echo "     cd $CHECKPOINTS_DIR"
echo "     kaggle datasets download -d abhisheksingh016/machine-model-for-emotion-detection"
echo "     unzip machine-model-for-emotion-detection.zip"
echo ""

# Check if kaggle is available
if command -v kaggle &> /dev/null; then
    echo "Kaggle CLI detected. Attempting automatic download..."
    echo ""
    
    if [ -f ~/.kaggle/kaggle.json ]; then
        cd "$CHECKPOINTS_DIR"
        kaggle datasets download -d abhisheksingh016/machine-model-for-emotion-detection
        
        if [ -f "machine-model-for-emotion-detection.zip" ]; then
            unzip -o machine-model-for-emotion-detection.zip
            rm -f machine-model-for-emotion-detection.zip
            echo ""
            echo "✓ Model downloaded successfully!"
            echo "  Location: $MODEL_PATH"
        else
            echo "Download may have failed. Please try manual download."
        fi
    else
        echo "Kaggle credentials not found at ~/.kaggle/kaggle.json"
        echo "Please set up Kaggle API credentials or download manually."
    fi
else
    echo "Kaggle CLI not found. Please install with: pip install kaggle"
    echo "Or download manually from the Kaggle website."
fi

echo ""
echo "=================================="
