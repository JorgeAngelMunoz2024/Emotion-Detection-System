#!/bin/bash
#
# Download BiSeNet Face Parsing Model
# ===================================
#
# Downloads pre-trained BiSeNet model for face parsing/segmentation.
# This model is trained on CelebAMask-HQ dataset and can segment 19 facial regions.
#
# Usage:
#   bash scripts/download_face_parsing_model.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_ROOT/backbones/face_parsing_models"

echo "========================================"
echo "BiSeNet Face Parsing Model Download"
echo "========================================"
echo ""

# Create model directory
echo "Creating model directory: $MODEL_DIR"
mkdir -p "$MODEL_DIR"

# Model URL (using zllrunning's face-parsing.PyTorch pre-trained weights)
# This is a well-known implementation with pre-trained weights on CelebAMask-HQ
MODEL_URL="https://github.com/zllrunning/face-parsing.PyTorch/raw/master/res/cp/79999_iter.pth"
MODEL_FILE="$MODEL_DIR/bisenet_celebamaskhq.pth"

echo "Model will be saved to: $MODEL_FILE"
echo ""

# Check if model already exists
if [ -f "$MODEL_FILE" ]; then
    echo "✓ Model already exists: $MODEL_FILE"
    echo ""
    read -p "Do you want to re-download? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        exit 0
    fi
    echo "Re-downloading model..."
fi

# Download model
echo "Downloading BiSeNet model from GitHub..."
echo "Source: $MODEL_URL"
echo ""

if command -v wget &> /dev/null; then
    wget -O "$MODEL_FILE" "$MODEL_URL" --progress=bar:force 2>&1
elif command -v curl &> /dev/null; then
    curl -L -o "$MODEL_FILE" "$MODEL_URL" --progress-bar
else
    echo "Error: Neither wget nor curl is installed."
    echo "Please install wget or curl and try again."
    exit 1
fi

# Verify download
if [ -f "$MODEL_FILE" ]; then
    FILE_SIZE=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat -c%s "$MODEL_FILE" 2>/dev/null)
    echo ""
    echo "✓ Download complete!"
    echo "  File: $MODEL_FILE"
    echo "  Size: $(numfmt --to=iec-i --suffix=B $FILE_SIZE 2>/dev/null || echo "$FILE_SIZE bytes")"
    echo ""
    
    # Create model info file
    cat > "$MODEL_DIR/MODEL_INFO.txt" << EOF
BiSeNet Face Parsing Model
=========================

Model: BiSeNet (Bilateral Segmentation Network)
Dataset: CelebAMask-HQ
Classes: 19 facial regions
Source: https://github.com/zllrunning/face-parsing.PyTorch

Facial Region Classes:
---------------------
0: background
1: skin
2: left eyebrow (l_brow)
3: right eyebrow (r_brow)
4: left eye (l_eye)
5: right eye (r_eye)
6: eye glasses (eye_g)
7: left ear (l_ear)
8: right ear (r_ear)
9: ear rings (ear_r)
10: nose
11: mouth (interior)
12: upper lip (u_lip)
13: lower lip (l_lip)
14: neck
15: neck_l
16: cloth
17: hair
18: hat

Usage:
------
from models.lip_segmentation_detector import LipSegmentationDetector

detector = LipSegmentationDetector(
    model_path='$MODEL_FILE',
    device='cpu'  # or 'cuda'
)

results = detector.process_frame(frame)

Downloaded: $(date)
EOF
    
    echo "✓ Model info saved to: $MODEL_DIR/MODEL_INFO.txt"
    echo ""
    echo "Setup complete!"
    echo ""
    echo "To use the model:"
    echo "  python tools/demo/lip_segmentation_demo.py --model-path $MODEL_FILE"
    echo ""
    
else
    echo ""
    echo "✗ Download failed!"
    echo "Please check your internet connection and try again."
    exit 1
fi

echo "Done!"
