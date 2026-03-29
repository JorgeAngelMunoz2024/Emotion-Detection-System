#!/bin/bash
#
# Start Docker container with full hardware access for multimodal demo
# Handles: display (X11), camera (webcam), audio (microphone)
#

set -e

echo "========================================"
echo "Starting Multimodal Demo Container"
echo "========================================"
echo ""

# Enable X11 forwarding
echo "1. Enabling X11 forwarding..."
xhost +local:docker 2>/dev/null || {
    echo "   Warning: xhost not available, display may not work"
}

# Detect available cameras
echo ""
echo "2. Detecting cameras..."
CAMERAS_FOUND=0
for cam in /dev/video*; do
    if [ -c "$cam" ]; then
        echo "   Found: $cam"
        CAMERAS_FOUND=1
    fi
done

if [ $CAMERAS_FOUND -eq 0 ]; then
    echo "   ✗ No camera found"
    echo "   Demo may not have webcam access"
else
    echo "   ✓ Camera devices ready"
fi

# Detect audio devices
echo ""
echo "3. Detecting audio devices..."
if [ -d /dev/snd ]; then
    echo "   Found: /dev/snd"
    echo "   ✓ Audio devices ready"
else
    echo "   ✗ No audio devices found"
fi

# Start container
echo ""
echo "4. Starting container with hardware access..."
echo "   Display: $DISPLAY"
echo "   Camera: /dev/video0, /dev/video1"
echo "   Audio: /dev/snd"
echo ""
echo "Starting interactive shell..."
echo ""

docker-compose run --rm -e DISPLAY="$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix:rw --device=/dev/video0:/dev/video0 --device=/dev/video1:/dev/video1 --device=/dev/snd:/dev/snd emotion-detector-cpu bash
