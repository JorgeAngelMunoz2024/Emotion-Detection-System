#!/bin/bash
#
# Quick setup script for running the multimodal demo in Docker
# Fixes MediaPipe and device access issues
#

set -e

echo "========================================"
echo "Multimodal Demo Setup"
echo "========================================"
echo ""

# 1. Fix MediaPipe installation
echo "1. Checking MediaPipe installation..."
python3 -c "import mediapipe; print(f'MediaPipe version: {mediapipe.__version__}')" 2>/dev/null || {
    echo "   MediaPipe needs to be reinstalled"
    echo "   Installing MediaPipe 0.10.9..."
    pip install --force-reinstall mediapipe==0.10.9
}

# 2. Test MediaPipe
echo ""
echo "2. Testing MediaPipe functionality..."
python3 /app/scripts/test_mediapipe.py || {
    echo "   MediaPipe test failed, trying alternative version..."
    pip install --force-reinstall mediapipe==0.10.0
}

# 3. Check camera access
echo ""
echo "3. Checking camera access..."
if [ -e /dev/video0 ]; then
    echo "   ✓ Camera device found: /dev/video0"
    ls -l /dev/video0
else
    echo "   ✗ No camera device found"
    echo "   Make sure to run container with: --device=/dev/video0:/dev/video0"
fi

# 4. Check display
echo ""
echo "4. Checking display access..."
if [ -z "$DISPLAY" ]; then
    echo "   ✗ DISPLAY variable not set"
    echo "   Make sure to run container with: -e DISPLAY=\$DISPLAY"
else
    echo "   ✓ DISPLAY is set to: $DISPLAY"
fi

# 5. Check X11 socket
if [ -e /tmp/.X11-unix ]; then
    echo "   ✓ X11 socket mounted"
else
    echo "   ✗ X11 socket not mounted"
    echo "   Make sure to run container with: -v /tmp/.X11-unix:/tmp/.X11-unix:rw"
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To run the demo:"
echo "  python /app/tools/demo/multimodal_demo.py"
echo ""
