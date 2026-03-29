#!/bin/bash
# Download emotion2vec+ model for Speech Emotion Recognition

set -e

BACKBONES_DIR="./backbones/emotion2vec_models"
mkdir -p "$BACKBONES_DIR"

echo "Downloading emotion2vec+ model..."
python3 << 'EOF'
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from funasr import AutoModel
    
    print("Downloading emotion2vec+ model from ModelScope...")
    print("This may take a few minutes (~1GB)...")
    
    # Download the model - FunASR will cache it locally
    model = AutoModel(
        model='iic/emotion2vec_plus_base',
        device='cpu',
        disable_update=True
    )
    
    print("✓ Model downloaded successfully!")
    print(f"Model cache location: {model.model_path if hasattr(model, 'model_path') else 'FunASR cache'}")
    
except Exception as e:
    print(f"✗ Failed to download model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

EOF

echo "Download complete!"
