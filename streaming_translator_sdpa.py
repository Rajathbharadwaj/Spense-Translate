#!/usr/bin/env python3
"""
Streaming Real-time Speech Translation Pipeline - Optimized for Best UX
English Speech → English Text → Target Language Text → Target Language Speech

Optimizations for User Experience:
- Overlapping audio chunks (2s chunks, 0.5s overlap)
- Pipeline parallelization  
- Streaming TTS playback
- Lower perceived latency
"""

import pyaudio
import numpy as np
import time
import argparse
import threading
import queue
from collections import deque
import torch
from threading import Thread

class StreamingTranslator:
    def __init__(self, target_language="hi", device="cuda", chunk_duration=2.0, overlap=0.5, microphone_index=None):
        self.target_language = target_language
        self.device = device
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.microphone_index = microphone_index
        
        # Audio settings
        self.model_sample_rate = 16000
        self.mic_sample_rate = 44100
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        
        # Streaming buffers and queues
        self.audio_buffer = deque(maxlen=int(self.model_sample_rate * self.chunk_duration * 3))
        self.processing_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # Language mapping
        self.lang_map = {
            "hi": {"name": "Hindi", "trans_code": "hin_Deva", "speaker": "Rohit"},
            "kn": {"name": "Kannada", "trans_code": "kan_Knda", "speaker": "Suresh"}, 
            "ta": {"name": "Tamil", "trans_code": "tam_Taml", "speaker": "Jaya"}
        }
        
        # Threading control
        self.is_running = False
        self.processing_thread = None
        self.playback_thread = None
        
        # Initialize models
        self._load_models()
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
    
    def _load_models(self):
        """Load all required models with optimizations"""
        print("🔄 Loading optimized streaming models...")
        
        # Load ASR with Flash Attention 2
        self._load_asr()
        
        # Load Translation Model  
        self._load_translation()
        
        # Load TTS Model
        self._load_tts()
        
        print("✅ All models loaded successfully!")
    
    def _load_asr(self):
        """Load Whisper with Flash Attention 2 (exact same approach as working real_time_translator.py)"""
        print("📝 Loading streaming Whisper ASR with Flash Attention 2...")
        import time
        start = time.time()
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            
            torch.set_float32_matmul_precision("high")
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Use the EXACT same approach as real_time_translator.py
            device_map = "cuda:0" if self.device == "cuda" else "cpu"
            print(f"  Loading Whisper model... (this should take <30s)")
            asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                "openai/whisper-large-v3",
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                device_map=device_map,  # This is the key - same as working version
                attn_implementation="flash_attention_2"  # Use Flash Attention 2 like working version
            )
            
            processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
            
            self.asr_model = pipeline(
                "automatic-speech-recognition",
                model=asr_model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
            )
            
            # Quick warmup for Flash Attention 2 (CRITICAL for performance!)
            print("🔥 Warming up Flash Attention 2 model...")
            dummy_audio = torch.randn(16000).numpy()  # 1 second of dummy audio
            _ = self.asr_model(dummy_audio, generate_kwargs={"language": "english"})
            
            print(f"✓ Streaming Whisper ASR loaded with Flash Attention 2 ({time.time()-start:.1f}s)")
            
        except Exception as e:
            print(f"❌ Failed to load ASR: {e}")
            raise
    
    def _load_translation(self):
        """Load IndicTrans2 for translation"""
        print("🌍 Loading IndicTrans2...")
        import time
        start = time.time()
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            from IndicTransToolkit.processor import IndicProcessor
            
            # Load IndicTrans2 with proper preprocessing (use cache)
            self.trans_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True, local_files_only=False)
            self.trans_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True, local_files_only=False).to(self.device)
            
            # Initialize IndicTrans2 processor for proper text preprocessing
            self.ip = IndicProcessor(inference=True)
            
            print(f"✓ IndicTrans2 loaded with preprocessing ({time.time()-start:.1f}s)")
            
        except Exception as e:
            print(f"❌ Failed to load translation: {e}")
            raise
    
    def _load_tts(self):
        """Load Indic Parler-TTS"""
        print("🔊 Loading Indic Parler-TTS...")
        import time
        start = time.time()
        try:
            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
            
            # Load Indic Parler-TTS
            self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
                "ai4bharat/indic-parler-tts",
                torch_dtype=torch.bfloat16,
                attn_implementation="eager"
            ).to(self.device)
            
            self.tts_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
            self.tts_description_tokenizer = AutoTokenizer.from_pretrained(
                self.tts_model.config.text_encoder._name_or_path
            )
            
            # Voice descriptions with recommended speakers
            self.voice_descriptions = {
                "hi": "Rohit speaks with a clear Hindi voice, moderate pace and excellent recording quality with no background noise.",
                "kn": "Suresh speaks with a clear Kannada voice, moderate pace and excellent recording quality with no background noise.", 
                "ta": "Jaya speaks with a clear Tamil voice, moderate pace and excellent recording quality with no background noise."
            }
            
            print(f"✓ Indic Parler-TTS loaded ({time.time()-start:.1f}s)")
            
        except Exception as e:
            print(f"❌ Failed to load TTS: {e}")
            raise
    
    def transcribe_audio(self, audio_chunk):
        """Transcribe audio using optimized Whisper"""
        try:
            print(f"    [DEBUG] Whisper transcribe called, audio shape: {audio_chunk.shape}")
            generate_kwargs = {
                "language": "english",
                "task": "transcribe", 
                "condition_on_prev_tokens": False,
                "compression_ratio_threshold": 1.35,
                "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                "logprob_threshold": -0.3,
                "no_speech_threshold": 0.4,
                "return_timestamps": False
            }
            
            print(f"    [DEBUG] Calling ASR pipeline...")
            result = self.asr_model(audio_chunk, generate_kwargs=generate_kwargs)
            print(f"    [DEBUG] ASR result: {result['text'][:50]}...")
            return result["text"].strip()
            
        except Exception as e:
            print(f"❌ ASR error: {e}")
            return ""
    
    def translate_text(self, text):
        """Translate English to target language using IndicTrans2"""
        try:
            target_code = self.lang_map[self.target_language]["trans_code"]
            
            # Preprocess using IndicProcessor
            batch = self.ip.preprocess_batch([text], src_lang="eng_Latn", tgt_lang=target_code)
            
            # Tokenize
            inputs = self.trans_tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = self.trans_model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )
            
            # Decode
            with self.trans_tokenizer.as_target_tokenizer():
                generated_tokens = self.trans_tokenizer.batch_decode(
                    generated_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            
            # Postprocess
            translations = self.ip.postprocess_batch(generated_tokens, lang=target_code)
            
            return translations[0] if translations else text
            
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return text
    
    def synthesize_speech(self, text):
        """Generate speech using Indic Parler-TTS"""
        try:
            description = self.voice_descriptions.get(self.target_language, self.voice_descriptions["hi"])
            
            description_input_ids = self.tts_description_tokenizer(description, return_tensors="pt").to(self.device)
            prompt_input_ids = self.tts_tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generation = self.tts_model.generate(
                    input_ids=description_input_ids.input_ids,
                    attention_mask=description_input_ids.attention_mask,
                    prompt_input_ids=prompt_input_ids.input_ids,
                    prompt_attention_mask=prompt_input_ids.attention_mask,
                    do_sample=True,
                    temperature=1.0
                )
            
            audio_arr = generation.cpu().float().numpy().squeeze()
            return audio_arr, self.tts_model.config.sampling_rate
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return None
    
    def _play_audio(self, audio_data, sample_rate):
        """Play generated audio"""
        try:
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            output_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True
            )
            
            output_stream.write(audio_int16.tobytes())
            output_stream.stop_stream()
            output_stream.close()
            
        except Exception as e:
            print(f"❌ Playback error: {e}")
    
    def _processing_worker(self):
        """Background processing thread for pipeline"""
        print("  [DEBUG] Processing worker started")
        while self.is_running:
            try:
                if not self.processing_queue.empty():
                    audio_chunk = self.processing_queue.get(timeout=0.1)
                    print(f"  [DEBUG] Processing chunk...")
                    
                    start_time = time.time()
                    
                    # Step 1: ASR
                    print(f"  [DEBUG] Starting ASR...")
                    english_text = self.transcribe_audio(audio_chunk)
                    if not english_text or len(english_text) < 3:
                        continue
                    
                    # Step 2: Translation
                    translated_text = self.translate_text(english_text)
                    
                    # Step 3: TTS
                    audio_result = self.synthesize_speech(translated_text)
                    
                    total_time = time.time() - start_time
                    
                    # Queue for playback
                    self.output_queue.put({
                        'english': english_text,
                        'translated': translated_text,
                        'audio': audio_result,
                        'timing': total_time
                    })
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Processing error: {e}")
                time.sleep(0.1)
    
    def _playback_worker(self):
        """Background playback thread"""
        while self.is_running:
            try:
                if not self.output_queue.empty():
                    result = self.output_queue.get(timeout=0.1)
                    
                    print(f"🇬🇧 English: \"{result['english']}\"")
                    print(f"🌍 {self.lang_map[self.target_language]['name']}: \"{result['translated']}\"")
                    print(f"⚡ Processing: {result['timing']:.2f}s")
                    
                    if result['audio']:
                        audio_data, sample_rate = result['audio']
                        self._play_audio(audio_data, sample_rate)
                        print(f"🔊 Audio played ({len(audio_data)/sample_rate:.2f}s)")
                    
                    print("-" * 50)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Playback error: {e}")
                time.sleep(0.1)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback with streaming chunks"""
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Resample to 16kHz
        decimation_factor = self.mic_sample_rate // self.model_sample_rate
        resampled_data = audio_data[::decimation_factor]
        
        # Add to buffer
        self.audio_buffer.extend(resampled_data)
        
        # Check if we have enough for a chunk
        chunk_samples = int(self.model_sample_rate * self.chunk_duration)
        if len(self.audio_buffer) >= chunk_samples:
            # Extract chunk with overlap
            audio_chunk = np.array(list(self.audio_buffer)[:chunk_samples])
            
            # Simple VAD check
            if np.max(np.abs(audio_chunk)) > 0.01:  # Basic threshold
                # Queue for processing
                if not self.processing_queue.full():
                    self.processing_queue.put(audio_chunk.copy())
            
            # Remove processed samples (keeping overlap)
            overlap_samples = int(self.model_sample_rate * self.overlap)
            remove_samples = max(1, chunk_samples - overlap_samples)  # Ensure we remove at least 1
            for _ in range(remove_samples):
                if self.audio_buffer:
                    self.audio_buffer.popleft()
        
        return (None, pyaudio.paContinue)
    
    def start_streaming(self):
        """Start streaming translation"""
        lang_name = self.lang_map[self.target_language]["name"]
        print(f"\n🎤 Streaming English → {lang_name} Translation")
        print(f"📊 Chunk: {self.chunk_duration}s, Overlap: {self.overlap}s")
        print("🔴 Starting... Press Ctrl+C to stop")
        
        self.is_running = True
        
        # Start worker threads
        self.processing_thread = Thread(target=self._processing_worker, daemon=True)
        self.playback_thread = Thread(target=self._playback_worker, daemon=True)
        
        self.processing_thread.start()
        self.playback_thread.start()
        
        # Start audio stream
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.mic_sample_rate,
            input=True,
            input_device_index=self.microphone_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        stream.start_stream()
        
        try:
            while stream.is_active() and self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping streaming translator...")
        finally:
            self.is_running = False
            stream.stop_stream()
            stream.close()
            self.audio.terminate()

def main():
    parser = argparse.ArgumentParser(description="Streaming Real-time Speech Translation")
    parser.add_argument("--target", choices=["hi", "kn", "ta"], default="hi",
                       help="Target language (hi=Hindi, kn=Kannada, ta=Tamil)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                       help="Device to use for inference")
    parser.add_argument("--mic", type=int, default=None,
                       help="Microphone device index")
    parser.add_argument("--chunk", type=float, default=2.0,
                       help="Audio chunk duration in seconds")
    parser.add_argument("--overlap", type=float, default=0.5,
                       help="Chunk overlap in seconds")
    
    args = parser.parse_args()
    
    translator = StreamingTranslator(
        target_language=args.target,
        device=args.device,
        chunk_duration=args.chunk,
        overlap=args.overlap,
        microphone_index=args.mic
    )
    
    translator.start_streaming()

if __name__ == "__main__":
    main()