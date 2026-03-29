#!/bin/bash
#
# Start container using docker run (works with older docker-compose)
#

set -e

echo "========================================"
echo "Starting Multimodal Demo Container"
echo "========================================"
echo ""

# Enable X11
echo "1. Enabling X11 forwarding..."
xhost +local:docker 2>/dev/null || echo "   Warning: xhost not available"

# Get the image name
IMAGE_NAME="mlproject_emotion-detector-cpu"

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "Error: Image $IMAGE_NAME not found"
    echo "Please build it first with: docker-compose build emotion-detector-cpu"
    exit 1
fi

echo "2. Starting container..."
echo "   Image: $IMAGE_NAME"
echo "   Display: $DISPLAY"
echo ""

docker run -it --rm \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$(pwd)":/app \
    --device=/dev/video0:/dev/video0 \
    --device=/dev/video1:/dev/video1 \
    --device=/dev/snd:/dev/snd \
    --network host \
    "$IMAGE_NAME" \
    bash

echo ""
echo "Container exited."
