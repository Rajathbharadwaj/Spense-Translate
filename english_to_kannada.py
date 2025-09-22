#!/usr/bin/env python3
"""
English Speech to Kannada Text Converter
Listens to English speech and converts to Kannada text via translation
"""

import pyaudio
import numpy as np
import time
import argparse
from collections import deque
from faster_whisper import WhisperModel, BatchedInferencePipeline
import webrtcvad

# For translation, we'll use a simple approach first
# You can replace this with IndicTrans2 later
class SimpleTranslator:
    def __init__(self):
        # This is a placeholder - you'd use IndicTrans2 here
        self.translations = {
            "hello": "ನಮಸ್ಕಾರ",
            "how are you": "ನೀವು ಹೇಗಿದ್ದೀರಿ",
            "thank you": "ಧನ್ಯವಾದ", 
            "good morning": "ಶುಭೋದಯ",
            "good evening": "ಶುಭ ಸಂಜೆ",
            "what is your name": "ನಿಮ್ಮ ಹೆಸರು ಏನು",
            "my name is": "ನನ್ನ ಹೆಸರು",
            "yes": "ಹೌದು",
            "no": "ಇಲ್ಲ",
            "please": "ದಯವಿಟ್ಟು",
            "sorry": "ಕ್ಷಮಿಸಿ",
            "water": "ನೀರು",
            "food": "ಆಹಾರ",
            "home": "ಮನೆ",
            "work": "ಕೆಲಸ"
        }
    
    def translate(self, english_text):
        """Simple word-by-word translation with fallback"""
        words = english_text.lower().strip().split()
        kannada_words = []
        
        # Try to find exact phrase first
        if english_text.lower().strip() in self.translations:
            return self.translations[english_text.lower().strip()]
        
        # Try word by word
        for word in words:
            clean_word = word.strip('.,!?').lower()
            if clean_word in self.translations:
                kannada_words.append(self.translations[clean_word])
            else:
                kannada_words.append(f"[{word}]")  # Untranslated words in brackets
        
        return " ".join(kannada_words) if kannada_words else f"[Translation needed: {english_text}]"

class EnglishToKannadaConverter:
    def __init__(self, kannada_model_path="models/whisper-kannada-ct2", 
                 device="cuda", chunk_duration=3, microphone_index=None):
        self.kannada_model_path = kannada_model_path
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
        self.vad = webrtcvad.Vad(2)
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.model_sample_rate * self.chunk_duration * 2)
        self.is_recording = False
        
        # Load Kannada STT model (to transcribe English speech into Kannada script)
        print(f"Loading Kannada STT model from {kannada_model_path}...")
        base_model = WhisperModel(
            kannada_model_path,
            device=device,
            compute_type="int8"
        )
        self.stt_model = BatchedInferencePipeline(model=base_model)
        print("✓ Kannada STT model loaded!")
        
        # Initialize translator
        print("Loading translator...")
        self.translator = SimpleTranslator()
        print("✓ Translator loaded!")
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
    def start_listening(self):
        """Start listening and converting"""
        print(f"\n🎤 Starting English → Kannada converter...")
        print(f"📊 Mic: {self.mic_sample_rate}Hz → Model: {self.model_sample_rate}Hz")
        print(f"⏱️  Chunk duration: {self.chunk_duration}s")
        print(f"🗣️  Speak English → Kannada STT model will convert to Kannada script!")
        print(f"🔊 Speak clearly into your microphone...\n")
        
        # Open microphone stream
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
            print("\n🛑 Stopping converter...")
        finally:
            stream.stop_stream()
            stream.close()
            self.audio.terminate()
            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback with resampling"""
        # Convert to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Resample from 44100 to 16000
        if self.mic_sample_rate != self.model_sample_rate:
            decimation_factor = self.mic_sample_rate // self.model_sample_rate
            resampled_data = audio_data[::decimation_factor]
        else:
            resampled_data = audio_data
        
        # Add to buffer
        self.audio_buffer.extend(resampled_data)
        
        return (None, pyaudio.paContinue)
    
    def _has_speech(self, audio_chunk):
        """VAD check"""
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
        
        if total_frames > 0:
            speech_ratio = speech_frames / total_frames
            return speech_ratio > 0.3
        return False
    
    def _process_audio_buffer(self):
        """Process audio and convert English to Kannada"""
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
        
        print("🔊 English speech detected, processing with Kannada model...")
        
        # Transcribe English speech using Kannada model (should output Kannada script)
        start_time = time.time()
        try:
            segments, info = self.stt_model.transcribe(
                audio_chunk,
                language="kn",  # Use Kannada language code
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
            
            # Collect Kannada text directly from the model
            kannada_segments = []
            for segment in segments:
                seg_text = segment.text.strip()
                if len(seg_text) > 2:
                    kannada_segments.append(seg_text)
            
            if kannada_segments:
                kannada_text = " ".join(kannada_segments)
                
                processing_time = time.time() - start_time
                
                print(f"🗣️  English Speech Input")
                print(f"🇮🇳 Kannada Output: \"{kannada_text}\"")
                print(f"⚡ Processed in {processing_time:.2f}s")
                print(f"🌍 Detected Language: {info.language} (confidence: {info.language_probability:.2f})")
                print("-" * 60)
            else:
                print("🔇 No speech detected by Kannada model")
                
        except Exception as e:
            print(f"❌ Processing error: {e}")

def main():
    parser = argparse.ArgumentParser(description="English Speech to Kannada Text Converter")
    parser.add_argument("--model", default="models/whisper-kannada-ct2", 
                       help="Path to Kannada STT model")
    parser.add_argument("--duration", type=int, default=3,
                       help="Audio chunk duration in seconds")
    parser.add_argument("--device", default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--mic", type=int, default=0,
                       help="Microphone device index (0 for HyperX QuadCast S)")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting English → Kannada Converter")
    print(f"🤖 Kannada STT Model: {args.model}")
    print(f"🎤 Microphone: Device {args.mic}")
    print(f"💻 Device: {args.device}")
    print()
    print("🧪 Experimental: Using Kannada STT model on English speech")
    print("📝 This will transliterate English sounds into Kannada script")
    print("   Try speaking simple English words clearly!")
    print()
    
    converter = EnglishToKannadaConverter(
        kannada_model_path=args.model,
        device=args.device,
        chunk_duration=args.duration,
        microphone_index=args.mic
    )
    
    converter.start_listening()

if __name__ == "__main__":
    main()