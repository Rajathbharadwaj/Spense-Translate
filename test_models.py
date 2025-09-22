#!/usr/bin/env python3
"""Test converted CT2 models"""

import numpy as np
from faster_whisper import WhisperModel

# Create a short test audio (1 second of silence)
test_audio = np.zeros(16000, dtype=np.float32)

models_to_test = {
    'Hindi': 'models/whisper-hindi-ct2',
    'Kannada': 'models/whisper-kannada-ct2', 
    'Tamil': 'models/whisper-tamil-ct2',
    'English': 'models/whisper-english-ct2'
}

print("Testing converted models...")
print("=" * 50)

for name, model_path in models_to_test.items():
    print(f"\nTesting {name} model ({model_path}):")
    try:
        # Load model
        model = WhisperModel(model_path, device="cuda", compute_type="int8")
        
        # Get model info
        print(f"  ✓ Model loaded successfully")
        
        # Try transcription with appropriate language code
        lang_map = {'Hindi': 'hi', 'Kannada': 'kn', 'Tamil': 'ta', 'English': 'en'}
        segments, info = model.transcribe(
            test_audio, 
            language=lang_map[name],
            vad_filter=True,
            condition_on_previous_text=False
        )
        
        # Process segments
        text = " ".join([s.text for s in segments])
        print(f"  ✓ Transcription works (output: '{text[:50] if text else 'empty'}')")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "=" * 50)
print("All models tested!")