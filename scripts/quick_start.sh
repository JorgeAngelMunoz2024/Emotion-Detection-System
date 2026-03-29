#!/bin/bash
# Quick start script for Real-time Emotion Detection project

set -e  # Exit on error

echo "========================================"
echo "Real-time Emotion Detection - Quick Start"
echo "========================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed."
    echo "Please install Docker:"
    echo "  sudo apt update && sudo apt install docker.io docker-compose -y && sudo systemctl start docker && sudo systemctl enable docker && sudo usermod -aG docker \$USER"
    echo ""
    echo "After installation, log out and log back in, then run this script again."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed."
    echo "Please install docker-compose:"
    echo "  sudo apt update && sudo apt install docker-compose -y"
    exit 1
fi

echo "Docker and docker-compose are installed ✓"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p data checkpoints logs
echo "Directories created ✓"
echo ""

# Prompt user for deployment target
echo "Choose deployment target:"
echo "1) CPU (local workstation)"
echo "2) GPU (VM with NVIDIA GPU)"
echo "3) Jupyter Notebook (interactive development)"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Building and starting CPU container..."
        docker-compose build emotion-detector-cpu
        docker-compose up emotion-detector-cpu
        ;;
    2)
        echo ""
        echo "Building and starting GPU container..."
        # Check for nvidia-docker
        if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            echo "Warning: NVIDIA Container Toolkit may not be properly installed."
            echo "See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            read -p "Continue anyway? [y/N]: " confirm
            if [[ $confirm != [yY] ]]; then
                exit 1
            fi
        fi
        docker-compose build emotion-detector-gpu
        docker-compose up emotion-detector-gpu
        ;;
    3)
        echo ""
        echo "Building and starting Jupyter Notebook..."
        docker-compose build jupyter
        docker-compose up -d jupyter
        echo ""
        echo "Jupyter Notebook is running!"
        echo "Access it at: http://localhost:8888"
        echo ""
        echo "To see the logs and token:"
        docker-compose logs jupyter | grep -A 5 "token"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Done!"
