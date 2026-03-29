# Multi-stage Dockerfile for Real-time Emotion Detection
# Supports both CPU and GPU environments

FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies including Tkinter for GUI
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    portaudio19-dev \
    python3-pyaudio \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    # Tkinter and X11 dependencies for GUI
    python3-tk \
    libgl1 \
    libegl1 \
    libxcb1 \
    libx11-xcb1 \
    # PulseAudio client for audio in Docker
    pulseaudio-utils \
    libpulse0 \
    alsa-utils \
    # ALSA PulseAudio plugin - critical for routing ALSA through PulseAudio
    libasound2-plugins \
    && rm -rf /var/lib/apt/lists/*

# Configure ALSA to use PulseAudio
COPY config/asound.conf /etc/asound.conf

# Install uv for faster package installation
RUN pip install --no-cache-dir uv

# Copy requirements first for better caching
COPY config/requirements.txt .

# Create necessary directories
RUN mkdir -p checkpoints data logs backbones/emotion2vec_models backbones/face_parsing_models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "train.py"]

# -------------------------------------------------------------------
# CPU variant (installs CPU-only PyTorch)
# -------------------------------------------------------------------
FROM base as cpu

# Install CPU-only PyTorch and other dependencies (CACHED if requirements.txt unchanged)
RUN uv pip install --system --no-cache-dir --upgrade pip && \
    uv pip install --system --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system --no-cache-dir -r requirements.txt

# Pre-download emotion2vec+ model (Audio SER) - CACHED once downloaded
RUN python3 << 'EOF'
import sys
print("Pre-downloading emotion2vec+ SER model...")
try:
    from funasr import AutoModel
    # Download to cache
    model = AutoModel(
        model='iic/emotion2vec_plus_base',
        device='cpu',
        disable_update=True
    )
    print("✓ emotion2vec+ model downloaded successfully!")
except Exception as e:
    print(f"✗ Warning: Could not pre-download model: {e}")
    print("  Model will be downloaded on first run instead.")
    sys.exit(0)
EOF

# Pre-download BiSeNet face parsing model (Lip Segmentation) - optional (~52MB)
RUN python3 -c "\
import urllib.request, os; \
dest='/app/backbones/face_parsing_models/bisenet_celebamaskhq.pth'; \
url='https://drive.google.com/uc?export=download&id=154JgKpzCPW82qINcVieuPH3fZ2e0P812&confirm=t'; \
urllib.request.urlretrieve(url, dest); \
size=os.path.getsize(dest); \
print(f'BiSeNet downloaded: {size/1024/1024:.1f} MB')" \
    || echo "BiSeNet model download skipped"

# Copy project files LAST (so changes don't invalidate package/model cache)
COPY . .

# -------------------------------------------------------------------
# GPU variant (installs CUDA-enabled PyTorch)
# -------------------------------------------------------------------
FROM base as gpu

# Install CUDA-enabled PyTorch and other dependencies (CACHED if requirements.txt unchanged)
RUN uv pip install --system --no-cache-dir --upgrade pip && \
    uv pip install --system --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    uv pip install --system --no-cache-dir -r requirements.txt

# Pre-download emotion2vec+ model (Audio SER) - CACHED once downloaded
RUN python3 << 'EOF'
import sys
print("Pre-downloading emotion2vec+ SER model...")
try:
    from funasr import AutoModel
    # Download to cache
    model = AutoModel(
        model='iic/emotion2vec_plus_base',
        device='cuda',
        disable_update=True
    )
    print("✓ emotion2vec+ model downloaded successfully!")
except Exception as e:
    print(f"✗ Warning: Could not pre-download model: {e}")
    print("  Model will be downloaded on first run instead.")
    sys.exit(0)
EOF

# Pre-download BiSeNet face parsing model (Lip Segmentation) - optional (~52MB)
RUN python3 -c "\
import urllib.request, os; \
dest='/app/backbones/face_parsing_models/bisenet_celebamaskhq.pth'; \
url='https://drive.google.com/uc?export=download&id=154JgKpzCPW82qINcVieuPH3fZ2e0P812&confirm=t'; \
urllib.request.urlretrieve(url, dest); \
size=os.path.getsize(dest); \
print(f'BiSeNet downloaded: {size/1024/1024:.1f} MB')" \
    || echo "BiSeNet model download skipped"

# Copy project files LAST (so changes don't invalidate package/model cache)
COPY . .
