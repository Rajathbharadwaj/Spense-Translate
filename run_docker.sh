#!/bin/bash

# Build and run the optimized Docker container with Flash Attention 2

echo "🐳 Building Docker container with NVIDIA NGC PyTorch + Flash Attention 2..."

# Build the Docker image
docker build -t spense-translator .

# Run the container with GPU support and audio access
docker run --gpus all \
    --rm \
    -it \
    --device /dev/snd \
    --group-add audio \
    --privileged \
    --net=host \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
    -v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
    spense-translator