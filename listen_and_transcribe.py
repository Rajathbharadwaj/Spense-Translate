#!/usr/bin/env python3
"""
Simple audio listener with STT transcription
Listens to microphone and transcribes using faster-whisper models
"""

import pyaudio
import numpy as np
import time
import threading
from collections import deque
from faster_whisper import WhisperModel, BatchedInferencePipeline
import webrtcvad
import argparse

class AudioTranscriber:
    def __init__(self, model_path="models/whisper-english-base-ct2", language="en", 
                 device="cuda", chunk_duration=3, microphone_index=None):
        self.model_path = model_path
        self.language = language
        self.device = device
        self.chunk_duration = chunk_duration
        self.microphone_index = microphone_index
        
        # Audio settings
        self.model_sample_rate = 16000  # What the model expects
        self.mic_sample_rate = 44100    # HyperX QuadCast S native rate
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        
        # Audio buffer (use model sample rate for buffer size)
        self.audio_buffer = deque(maxlen=self.model_sample_rate * self.chunk_duration * 2)
        self.is_recording = False
        
        # Load STT model
        print(f"Loading model from {model_path}...")
        base_model = WhisperModel(
            model_path,
            device=device,
            compute_type="int8"
        )
        self.model = BatchedInferencePipeline(model=base_model)
        print("✓ Model loaded successfully!")
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
    def start_listening(self):
        """Start listening to microphone"""
        print(f"\n🎤 Starting to listen... (Press Ctrl+C to stop)")
        print(f"📊 Mic sample rate: {self.mic_sample_rate}Hz → Model: {self.model_sample_rate}Hz")
        print(f"⏱️  Chunk duration: {self.chunk_duration}s")
        print(f"🌐 Language: {self.language}")
        print(f"🔊 Speak clearly into your microphone...\n")
        
        # Open microphone stream (use mic's native sample rate)
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.mic_sample_rate,
            input=True,
            input_device_index=self.microphone_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        self.is_recording = True
        stream.start_stream()
        
        try:
            while stream.is_active():
                time.sleep(0.1)
                self._process_audio_buffer()
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        finally:
            stream.stop_stream()
            stream.close()
            self.audio.terminate()
            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        # Convert to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Resample from mic rate to model rate (44100 -> 16000)
        if self.mic_sample_rate != self.model_sample_rate:
            # Simple decimation (take every ~2.76th sample for 44100->16000)
            decimation_factor = self.mic_sample_rate // self.model_sample_rate
            resampled_data = audio_data[::decimation_factor]
        else:
            resampled_data = audio_data
        
        # Add to buffer
        self.audio_buffer.extend(resampled_data)
        
        return (None, pyaudio.paContinue)
    
    def _has_speech(self, audio_chunk):
        """Check if audio chunk contains speech using VAD"""
        # Convert to 16-bit PCM for VAD
        audio_int16 = (audio_chunk * 32767).astype(np.int16).tobytes()
        
        # VAD works with 10ms, 20ms, or 30ms frames
        frame_duration = 30  # ms
        frame_size = int(self.model_sample_rate * frame_duration / 1000)
        
        speech_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio_chunk) - frame_size, frame_size):
            frame = audio_chunk[i:i + frame_size]
            frame_bytes = (frame * 32767).astype(np.int16).tobytes()
            
            try:
                if self.vad.is_speech(frame_bytes, self.model_sample_rate):
                    speech_frames += 1
                total_frames += 1
            except:
                continue
        
        # Return True if at least 30% of frames contain speech
        if total_frames > 0:
            speech_ratio = speech_frames / total_frames
            return speech_ratio > 0.3
        return False
    
    def _process_audio_buffer(self):
        """Process accumulated audio buffer"""
        if len(self.audio_buffer) < self.model_sample_rate * self.chunk_duration:
            return
        
        # Get audio chunk
        audio_chunk = np.array(list(self.audio_buffer)[:self.model_sample_rate * self.chunk_duration])
        
        # Clear processed portion (keep 50% overlap)
        overlap_size = len(self.audio_buffer) // 2
        for _ in range(len(self.audio_buffer) - overlap_size):
            if self.audio_buffer:
                self.audio_buffer.popleft()
        
        # Check for speech
        if not self._has_speech(audio_chunk):
            return
        
        print("🔊 Speech detected, transcribing...")
        
        # Transcribe
        start_time = time.time()
        try:
            segments, info = self.model.transcribe(
                audio_chunk,
                language=self.language,
                batch_size=8,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.6,
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200
                },
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            
            # Collect text
            text_segments = []
            for segment in segments:
                seg_text = segment.text.strip()
                if len(seg_text) > 2:  # Filter very short segments
                    text_segments.append(seg_text)
            
            transcription_time = time.time() - start_time
            
            if text_segments:
                full_text = " ".join(text_segments)
                print(f"📝 [{time.strftime('%H:%M:%S')}] \"{full_text}\"")
                print(f"⚡ Processed in {transcription_time:.2f}s | Language: {info.language} ({info.language_probability:.2f})")
                print("-" * 60)
            else:
                print("🔇 No clear speech detected")
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Real-time audio transcription")
    parser.add_argument("--model", default="models/whisper-english-base-ct2", 
                       help="Path to STT model")
    parser.add_argument("--language", default="en", 
                       help="Language code (en, hi, kn, ta)")
    parser.add_argument("--duration", type=int, default=3,
                       help="Audio chunk duration in seconds")
    parser.add_argument("--device", default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--mic", type=int, default=None,
                       help="Microphone device index (use list_microphones.py to see options)")
    
    args = parser.parse_args()
    
    # Language to model mapping
    model_map = {
        "en": "models/whisper-english-base-ct2",
        "hi": "models/whisper-hindi-ct2", 
        "kn": "models/whisper-kannada-ct2",
        "ta": "models/whisper-tamil-ct2"
    }
    
    if args.language in model_map:
        model_path = model_map[args.language]
    else:
        model_path = args.model
    
    print(f"🚀 Starting Audio Transcriber")
    print(f"🤖 Model: {model_path}")
    print(f"🌐 Language: {args.language}")
    print(f"💻 Device: {args.device}")
    
    transcriber = AudioTranscriber(
        model_path=model_path,
        language=args.language,
        device=args.device,
        chunk_duration=args.duration,
        microphone_index=args.mic
    )
    
    transcriber.start_listening()

if __name__ == "__main__":
    main()