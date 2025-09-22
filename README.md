# Spense-Translate

Real-time speech translation system for VKYC (Video Know Your Customer) platforms. Translates English speech to Indian languages with native voice synthesis.

## Features

- Real-time English speech recognition using Whisper large-v3
- Translation to Indian languages using IndicTrans2
- Native voice synthesis using Indic Parler-TTS
- Support for Hindi, Tamil, Kannada, Telugu, Bengali, Malayalam, and Gujarati

## Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.10+
- CUDA 11.8 or higher
- Microphone for audio input

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Rajathbharadwaj/Spense-Translate.git
cd Spense-Translate
```

### 2. Create conda environment from file
```bash
conda env create -f environment.yml
conda activate indicf5
```

**OR** create environment manually:
```bash
conda create -n indicf5 python=3.10
conda activate indicf5
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Flash Attention 2
```bash
pip install flash-attn --no-build-isolation
```

## Usage

Run the real-time translator with your target language:

```bash
python real_time_translator.py --target hi --device cuda
```

### Supported Languages

- `hi` - Hindi
- `ta` - Tamil  
- `kn` - Kannada
- `te` - Telugu
- `bn` - Bengali
- `ml` - Malayalam
- `gu` - Gujarati

### Command Line Options

- `--target` - Target language code (required)
- `--device` - Device to use: 'cuda' or 'cpu' (default: cuda)
- `--chunk-duration` - Audio chunk duration in seconds (default: 5)
- `--sample-rate` - Audio sample rate in Hz (default: 16000)

### Example

For Hindi translation:
```bash
python real_time_translator.py --target hi --device cuda
```

For Tamil translation with custom chunk duration:
```bash
python real_time_translator.py --target ta --chunk-duration 3
```

## How It Works

1. **Audio Capture**: Captures audio from microphone in configurable chunks
2. **Speech Recognition**: Whisper large-v3 with Flash Attention 2 converts speech to text
3. **Translation**: IndicTrans2 translates English to target Indian language
4. **Voice Synthesis**: Indic Parler-TTS generates native speaker audio
5. **Playback**: Synthesized audio is played through speakers

## Performance

- Model loading: ~30 seconds
- Processing latency: <2 seconds per 5-second chunk
- Requires ~18GB VRAM for all models

## Troubleshooting

### CUDA Out of Memory
- Reduce chunk duration: `--chunk-duration 3`
- Use CPU mode: `--device cpu` (slower but works without GPU)

### Microphone Not Found
- Check microphone permissions
- Verify device with: `python -m sounddevice`

### Slow Performance
- Ensure CUDA is properly installed
- Check GPU availability: `nvidia-smi`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

## Note

Only `real_time_translator.py` is production-ready. Other files in the repository are experimental versions under development.

## License

MIT
