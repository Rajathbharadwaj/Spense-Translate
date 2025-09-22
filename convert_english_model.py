#!/usr/bin/env python3
"""
Convert OpenAI Whisper English model to CTranslate2 format
"""

import os
import sys

try:
    from faster_whisper import WhisperModel
    import ctranslate2
    from transformers import WhisperForConditionalGeneration, WhisperTokenizer
except ImportError:
    print("Please install required packages:")
    print("pip install transformers faster-whisper ctranslate2")
    sys.exit(1)

def convert_model():
    model_name = "openai/whisper-small.en"  # English-only model
    output_dir = "models/whisper-english-proper-ct2"
    
    print(f"Converting {model_name} to CTranslate2 format...")
    
    # Use the conversion command
    cmd = f"ct2-transformers-converter --model {model_name} --output_dir {output_dir} --quantization int8 --force"
    
    print(f"Running: {cmd}")
    result = os.system(cmd)
    
    if result == 0:
        print(f"✓ Model converted successfully to {output_dir}")
        
        # Update the symlink
        if os.path.exists("models/whisper-english-ct2"):
            os.remove("models/whisper-english-ct2")
        os.symlink("whisper-english-proper-ct2", "models/whisper-english-ct2")
        print("✓ Updated English model symlink")
    else:
        print(f"✗ Conversion failed with code {result}")
        return False
    
    return True

if __name__ == "__main__":
    convert_model()