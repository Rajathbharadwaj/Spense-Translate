#!/usr/bin/env python3
"""
List available microphones and test them
"""

import pyaudio
import numpy as np

def list_microphones():
    """List all available audio input devices"""
    audio = pyaudio.PyAudio()
    
    print("Available microphones:")
    print("=" * 50)
    
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        
        # Only show input devices
        if device_info['maxInputChannels'] > 0:
            is_default = " (DEFAULT)" if i == audio.get_default_input_device_info()['index'] else ""
            print(f"Device {i}: {device_info['name']}{is_default}")
            print(f"  - Channels: {device_info['maxInputChannels']}")
            print(f"  - Sample Rate: {device_info['defaultSampleRate']}")
            print()
    
    audio.terminate()
    return audio.get_default_input_device_info()['index']

def test_microphone(device_index=None):
    """Test microphone by recording a short sample"""
    audio = pyaudio.PyAudio()
    
    # Use default if no device specified
    if device_index is None:
        device_index = audio.get_default_input_device_info()['index']
    
    device_info = audio.get_device_info_by_index(device_index)
    print(f"Testing microphone: {device_info['name']}")
    print("Speak for 3 seconds...")
    
    # Record 3 seconds
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=1024
    )
    
    frames = []
    for _ in range(0, int(16000 / 1024 * 3)):  # 3 seconds
        data = stream.read(1024)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Analyze the recording
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    volume = np.sqrt(np.mean(audio_data**2))
    
    print(f"✓ Recording complete!")
    print(f"  Average volume: {volume:.2f}")
    
    if volume > 100:
        print("  🔊 Good signal detected!")
    elif volume > 10:
        print("  🔉 Weak signal - try speaking louder")
    else:
        print("  🔇 No signal - check microphone")

if __name__ == "__main__":
    import sys
    
    # List all microphones
    default_device = list_microphones()
    
    if len(sys.argv) > 1:
        # Test specific device
        device_id = int(sys.argv[1])
        test_microphone(device_id)
    else:
        # Test default device
        print(f"Testing default microphone (device {default_device}):")
        test_microphone(default_device)
        
    print("\nTo use a specific microphone with the transcriber:")
    print("python listen_and_transcribe.py --device-id <device_number>")