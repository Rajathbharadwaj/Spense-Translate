#!/usr/bin/env python3
"""Test Faster Whisper with Batched Inference"""

from faster_whisper import WhisperModel, BatchedInferencePipeline
import numpy as np
import time

# Create test audio (3 seconds of simulated speech-like noise)
print("Creating test audio...")
sample_rate = 16000
duration = 3  # seconds

# Generate a more speech-like test signal
t = np.linspace(0, duration, sample_rate * duration)
# Mix of frequencies that somewhat resembles speech
audio = (
    0.1 * np.sin(2 * np.pi * 200 * t) +  # Low frequency
    0.05 * np.sin(2 * np.pi * 400 * t) +  # Mid frequency  
    0.02 * np.sin(2 * np.pi * 800 * t)    # High frequency
) * np.random.rand(len(t)) * 0.5  # Add randomness

audio = audio.astype(np.float32)

print("\nTesting different models with BatchedInferencePipeline:")
print("=" * 60)

# Test configurations
test_configs = [
    {
        'name': 'English (base.en)',
        'model': 'models/whisper-english-base-ct2',
        'language': 'en',
        'compute_type': 'int8'
    },
    {
        'name': 'Hindi',
        'model': 'models/whisper-hindi-ct2', 
        'language': 'hi',
        'compute_type': 'int8'
    }
]

for config in test_configs:
    print(f"\n{config['name']} Model:")
    print("-" * 40)
    
    try:
        # Load model
        start_time = time.time()
        model = WhisperModel(
            config['model'],
            device="cuda",
            compute_type=config['compute_type']
        )
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f}s")
        
        # Create batched pipeline
        batched_model = BatchedInferencePipeline(model=model)
        print("✓ Batched pipeline created")
        
        # Run transcription
        start_time = time.time()
        segments, info = batched_model.transcribe(
            audio,
            batch_size=16,
            language=config['language'],
            vad_filter=True,
            condition_on_previous_text=False
        )
        
        # Process results
        segments = list(segments)  # Execute the generator
        transcribe_time = time.time() - start_time
        
        print(f"✓ Transcription completed in {transcribe_time:.2f}s")
        print(f"  Language detected: {info.language} (probability: {info.language_probability:.2f})")
        print(f"  Number of segments: {len(segments)}")
        
        if segments:
            for i, segment in enumerate(segments[:3]):  # Show first 3 segments
                print(f"  [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text[:50]}")
        else:
            print("  No speech detected (which is expected for noise)")
            
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("Testing with actual model names from Hugging Face:")
print("=" * 60)

# Test with standard models
standard_models = ["turbo", "base", "small"]

for model_name in standard_models:
    print(f"\nTrying {model_name} model:")
    try:
        model = WhisperModel(model_name, device="cuda", compute_type="float16")
        batched_model = BatchedInferencePipeline(model=model)
        
        segments, info = batched_model.transcribe(
            audio,
            batch_size=16,
            vad_filter=True
        )
        
        segments = list(segments)
        print(f"✓ {model_name} model works!")
        print(f"  Language detected: {info.language}")
        
    except Exception as e:
        print(f"✗ {model_name} failed: {str(e)[:100]}")

print("\nDone!")