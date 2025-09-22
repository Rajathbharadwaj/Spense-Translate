# Use NVIDIA NGC PyTorch container with Flash Attention 2 pre-installed
# Check latest at: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Flash Attention 2 (if not already in container)
RUN pip install flash-attn --no-build-isolation

# Install optimized inference libraries
RUN pip install \
    faster-whisper \
    ctranslate2 \
    parler-tts \
    IndicTransToolkit \
    soundfile \
    webrtcvad \
    pyaudio

# Copy application code
COPY . /app

# Set environment variables for optimal performance
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDA_MODULE_LOADING=LAZY

# Enable Flash Attention 2
ENV USE_FLASH_ATTENTION=1

# Run the application
CMD ["python", "real_time_translator.py", "--target", "hi", "--mic", "0"]