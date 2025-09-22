#!/bin/bash

# Run with latest NVIDIA NGC PyTorch container that has EVERYTHING pre-installed
# Including Flash Attention 2/3, Transformer Engine, cuDNN, etc.

echo "🚀 Running with NVIDIA NGC PyTorch 25.01 (Everything pre-installed!)"

# Try to run with GPU, fallback to CPU if not available
if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
    echo "✅ GPU support detected"
    GPU_FLAG="--gpus all"
    DEVICE="cuda"
else
    echo "⚠️  No GPU support detected. For Garuda/Arch, install nvidia-container-toolkit:"
    echo "  yay -S nvidia-container-toolkit"
    echo "  sudo systemctl restart docker"
    echo ""
    echo "Running in CPU mode for now..."
    GPU_FLAG=""
    DEVICE="cpu"
fi

# Use basic CUDA container that actually works
docker pull nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install everything from scratch
docker run $GPU_FLAG \
    --rm \
    -it \
    --device /dev/snd \
    --group-add audio \
    --privileged \
    --net=host \
    -v $(pwd):/workspace \
    -w /workspace \
    nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 \
    bash -c "
        apt-get update && 
        apt-get install -y python3 python3-pip portaudio19-dev &&
        echo '=== Checking CUDA and cuDNN libraries ===' &&
        find /usr -name '*cudnn*' 2>/dev/null | head -10 &&
        echo '=== CUDA version ===' &&
        nvcc --version &&
        echo '=== LD_LIBRARY_PATH ===' &&
        echo \$LD_LIBRARY_PATH &&
        echo '=== Setting library paths ===' &&
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH &&
        ldconfig &&
        pip3 install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118 &&
        pip3 install 'accelerate>=0.26.0' faster-whisper parler-tts IndicTransToolkit webrtcvad pyaudio &&
        python3 real_time_translator.py --target hi --mic 0 --device $DEVICE
    "