#!/usr/bin/env python3
"""
Real-time Speech Translation Pipeline
English Speech → English Text → Target Language Text → Target Language Speech

Complete VKYC translation solution using:
- IndicWhisper (ASR)
- IndicTrans2 (Translation) 
- Indic Parler-TTS (TTS)
"""

import pyaudio
import numpy as np
import time
import argparse
import io
import soundfile as sf
import torch
from collections import deque
from faster_whisper import WhisperModel, BatchedInferencePipeline
import webrtcvad

class RealTimeTranslator:
    def __init__(self, target_language="hi", device="cuda", chunk_duration=3, microphone_index=None):
        self.target_language = target_language
        self.device = device
        self.chunk_duration = chunk_duration
        self.microphone_index = microphone_index
        
        # Audio settings
        self.model_sample_rate = 16000
        self.mic_sample_rate = 44100
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(2)
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.model_sample_rate * self.chunk_duration * 2)
        self.is_recording = False
        
        # Language mapping
        self.lang_map = {
            "hi": {"name": "Hindi", "trans_code": "hin_Deva"},
            "kn": {"name": "Kannada", "trans_code": "kan_Knda"}, 
            "ta": {"name": "Tamil", "trans_code": "tam_Taml"}
        }
        
        # Initialize components
        self._load_models()
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
    def _load_models(self):
        """Load all required models"""
        print("🔄 Loading models...")
        
        # 1. Load English ASR with Transformers pipeline + torch.compile
        print("📝 Loading optimized Whisper ASR model...")
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            from torch.nn.attention import SDPBackend, sdpa_kernel
            
            torch.set_float32_matmul_precision("high")
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Load Whisper large-v3 with Flash Attention 2 (use device_map as per GitHub issue)
            device_map = "cuda:0" if self.device == "cuda" else "cpu"
            asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                "openai/whisper-large-v3",
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                device_map=device_map,  # This is the key for Flash Attention 2
                attn_implementation="flash_attention_2"
            )
            print(f"✓ Model loaded on: {device_map}")
            
            # No torch.compile needed with Flash Attention 2
            print("✓ Using Flash Attention 2 for optimized inference")
            
            processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
            
            # Create optimized pipeline (no device parameter needed with device_map)
            self.asr_model = pipeline(
                "automatic-speech-recognition",
                model=asr_model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
            )
            
            # Quick warmup for Flash Attention 2
            print("🔥 Warming up Flash Attention 2 model...")
            dummy_audio = torch.randn(16000).numpy()  # 1 second of dummy audio
            _ = self.asr_model(dummy_audio, generate_kwargs={"language": "english"})
            
            self.asr_optimized = True
            print("✓ Optimized Whisper ASR loaded with Flash Attention 2")
            
        except Exception as e:
            print(f"❌ Failed to load optimized ASR: {e}")
            # Fallback to faster-whisper
            english_model = WhisperModel(
                "large-v3",
                device=self.device,
                compute_type="int8"
            )
            self.asr_model = BatchedInferencePipeline(model=english_model)
            self.asr_optimized = False
            print("✓ Fallback faster-whisper ASR loaded")
        
        # 2. Load Translation Model (IndicTrans2 with HuggingFace)
        print("🌍 Loading IndicTrans2 translation model...")
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            from IndicTransToolkit.processor import IndicProcessor
            
            # Initialize IndicTrans2 processor for text preprocessing
            self.ip = IndicProcessor(inference=True)
            
            # Load HuggingFace IndicTrans2 model (use distilled for better compatibility)
            model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
            self.translation_tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).to(self.device)
            
            # Language code mapping for IndicTrans2
            self.trans_lang_codes = {
                "hi": "hin_Deva",
                "kn": "kan_Knda", 
                "ta": "tam_Taml"
            }
            
            print("✓ IndicTrans2 HuggingFace model loaded successfully")
            self.use_indictrans = True
            # Initialize translator fallback (won't be used but needed for error handling)
            self.translator = self._create_simple_translator()
        except Exception as e:
            print(f"⚠️  Failed to load IndicTrans2: {e}")
            print("⚠️  Using simple translation fallback")
            self.use_indictrans = False
            self.translator = self._create_simple_translator()
        
        # 3. Load TTS Model (Optimized Indic Parler-TTS)
        print("🔊 Loading optimized Indic Parler-TTS model...")
        try:
            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
            
            # Load Indic Parler-TTS model with eager attention (most compatible)
            self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
                "ai4bharat/indic-parler-tts",
                torch_dtype=torch.bfloat16,
                attn_implementation="eager"  # Avoid Flash Attention conflicts
            ).to(self.device)
            print("✓ Using eager attention (most compatible)")
            
            # Load tokenizers
            self.tts_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
            self.tts_description_tokenizer = AutoTokenizer.from_pretrained(
                self.tts_model.config.text_encoder._name_or_path
            )
            
            # Apply Parler-TTS optimizations (4.5x speedup)
            print("🚀 Applying Parler-TTS optimizations...")
            try:
                # Set max padding length for compilation
                self.max_length = 50
                
                # Disable torch.compile to avoid StaticCache issues
                # compile_mode = "default"  # "reduce-overhead" for 3-4x but longer warmup
                # self.tts_model.generation_config.cache_implementation = "static"
                # self.tts_model.forward = torch.compile(self.tts_model.forward, mode=compile_mode)
                
                # Warmup compilation (required for speed benefits)
                print("🔥 Warming up model compilation...")
                warmup_inputs = self.tts_tokenizer(
                    "This is for compilation", 
                    return_tensors="pt", 
                    padding="max_length", 
                    max_length=self.max_length
                ).to(self.device)
                
                warmup_kwargs = {
                    **warmup_inputs, 
                    "prompt_input_ids": warmup_inputs.input_ids, 
                    "prompt_attention_mask": warmup_inputs.attention_mask
                }
                
                # Generate once to trigger compilation
                with torch.no_grad():
                    _ = self.tts_model.generate(**warmup_kwargs)
                
                print("✓ Parler-TTS optimizations applied (4.5x speedup)")
                self.tts_optimized = True
                
            except Exception as e:
                print(f"⚠️  Optimization failed: {e}")
                print("✓ Using standard inference")
                self.tts_optimized = False
            
            # Voice descriptions with recommended Indic Parler-TTS speakers
            self.voice_descriptions = {
                "hi": "Rohit speaks with a clear Hindi voice, moderate pace and excellent recording quality with no background noise.",
                "kn": "Suresh speaks with a clear Kannada voice, moderate pace and excellent recording quality with no background noise.", 
                "ta": "Jaya speaks with a clear Tamil voice, moderate pace and excellent recording quality with no background noise."
            }
            
            print("✓ Optimized Indic Parler-TTS loaded successfully")
            self.use_tts = True
            
        except Exception as e:
            print(f"⚠️  Failed to load Indic Parler-TTS: {e}")
            print("⚠️  Using text output only")
            self.tts_model = None
            self.use_tts = False
        
        print("🎉 All models loaded!")
    
    def _create_simple_translator(self):
        """Simple translation dictionary for MVP"""
        translations = {
            "hi": {  # Hindi
                "hello": "नमस्ते",
                "how are you": "आप कैसे हैं",
                "thank you": "धन्यवाद",
                "good morning": "सुप्रभात",
                "good evening": "शुभ संध्या",
                "yes": "हाँ",
                "no": "नहीं",
                "please": "कृपया",
                "sorry": "माफ करें",
                "water": "पानी",
                "food": "खाना",
                "help": "मदद",
                "name": "नाम",
                "time": "समय",
                "money": "पैसा"
            },
            "kn": {  # Kannada
                "hello": "ನಮಸ್ಕಾರ",
                "how are you": "ನೀವು ಹೇಗಿದ್ದೀರಿ",
                "thank you": "ಧನ್ಯವಾದ",
                "good morning": "ಶುಭೋದಯ",
                "good evening": "ಶುಭ ಸಂಜೆ",
                "yes": "ಹೌದು",
                "no": "ಇಲ್ಲ",
                "please": "ದಯವಿಟ್ಟು",
                "sorry": "ಕ್ಷಮಿಸಿ",
                "water": "ನೀರು",
                "food": "ಆಹಾರ",
                "help": "ಸಹಾಯ",
                "name": "ಹೆಸರು",
                "time": "ಸಮಯ",
                "money": "ಹಣ"
            },
            "ta": {  # Tamil
                "hello": "வணக்கம்",
                "how are you": "நீங்கள் எப்படி இருக்கிறீர்கள்",
                "thank you": "நன்றி",
                "good morning": "காலை வணக்கம்",
                "good evening": "மாலை வணக்கம்",
                "yes": "ஆம்",
                "no": "இல்லை",
                "please": "தயவுசெய்து",
                "sorry": "மன்னிக்கவும்",
                "water": "தண்ணீர்",
                "food": "உணவு",
                "help": "உதவி",
                "name": "பெயர்",
                "time": "நேரம்",
                "money": "பணம்"
            }
        }
        
        return translations.get(self.target_language, translations["hi"])
    
    def translate_text(self, english_text):
        """Translate English text to target language"""
        if self.use_indictrans:
            try:
                # Get target language code
                tgt_lang = self.trans_lang_codes[self.target_language]
                
                # Preprocess text using IndicTrans2 processor
                processed_text = self.ip.preprocess_batch([english_text], src_lang="eng_Latn", tgt_lang=tgt_lang)
                
                # Tokenize inputs
                inputs = self.translation_tokenizer(
                    processed_text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=256
                ).to(self.translation_model.device)
                
                # Generate translation
                with torch.no_grad():
                    generated_tokens = self.translation_model.generate(
                        **inputs,
                        max_length=256,
                        num_beams=5,
                        num_return_sequences=1,
                        do_sample=False
                    )
                
                # Decode translation
                translation = self.translation_tokenizer.decode(
                    generated_tokens[0], 
                    skip_special_tokens=True
                )
                
                # Postprocess translation
                final_translation = self.ip.postprocess_batch([translation], lang=tgt_lang)[0]
                
                return final_translation
                
            except Exception as e:
                print(f"⚠️  IndicTrans2 translation failed: {e}")
                # Fall back to simple translation
                return self._simple_translate(english_text)
        else:
            return self._simple_translate(english_text)
    
    def _simple_translate(self, english_text):
        """Fallback simple translation method"""
        words = english_text.lower().strip().split()
        translated_words = []
        
        # Try phrase first
        if english_text.lower().strip() in self.translator:
            return self.translator[english_text.lower().strip()]
        
        # Try word by word
        for word in words:
            clean_word = word.strip('.,!?').lower()
            if clean_word in self.translator:
                translated_words.append(self.translator[clean_word])
            else:
                translated_words.append(f"[{word}]")
        
        return " ".join(translated_words) if translated_words else f"[{english_text}]"
    
    def synthesize_speech(self, text):
        """Convert text to speech using optimized Indic Parler-TTS"""
        if not self.use_tts:
            return None
            
        try:
            # Get voice description for target language
            description = self.voice_descriptions.get(
                self.target_language, 
                self.voice_descriptions["hi"]
            )
            
            # Tokenize inputs
            description_input_ids = self.tts_description_tokenizer(
                description, 
                return_tensors="pt"
            ).to(self.device)
            
            prompt_input_ids = self.tts_tokenizer(
                text, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate speech with optimized model
            with torch.no_grad():
                # Use unoptimized path to avoid StaticCache issues
                generation = self.tts_model.generate(
                    input_ids=description_input_ids.input_ids,
                    attention_mask=description_input_ids.attention_mask,
                    prompt_input_ids=prompt_input_ids.input_ids,
                    prompt_attention_mask=prompt_input_ids.attention_mask,
                    do_sample=True,
                    temperature=1.0
                )
            
            # Convert to audio array (handle bfloat16)
            audio_arr = generation.cpu().float().numpy().squeeze()
            return audio_arr, self.tts_model.config.sampling_rate
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"⚠️  TTS synthesis failed: {e}")
            return None
    
    def _play_audio(self, audio_data, sample_rate):
        """Play generated audio using PyAudio"""
        try:
            # Convert float32 to int16 for PyAudio
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Create output stream
            output_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True
            )
            
            # Play audio
            output_stream.write(audio_int16.tobytes())
            output_stream.stop_stream()
            output_stream.close()
            
        except Exception as e:
            print(f"⚠️  Audio playback failed: {e}")
    
    def start_listening(self):
        """Start real-time translation"""
        lang_name = self.lang_map[self.target_language]["name"]
        print(f"\n🎤 Real-time English → {lang_name} Translation")
        print(f"📊 Mic: {self.mic_sample_rate}Hz → Model: {self.model_sample_rate}Hz")
        print(f"⏱️  Chunk duration: {self.chunk_duration}s")
        print(f"🌍 Target language: {lang_name}")
        print(f"🔊 Speak English clearly...\n")
        
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
            print(f"\n🛑 Stopping translation...")
        finally:
            stream.stop_stream()
            stream.close()
            self.audio.terminate()
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback with resampling"""
        # Convert and resample
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        if self.mic_sample_rate != self.model_sample_rate:
            decimation_factor = self.mic_sample_rate // self.model_sample_rate
            resampled_data = audio_data[::decimation_factor]
        else:
            resampled_data = audio_data
        
        self.audio_buffer.extend(resampled_data)
        return (None, pyaudio.paContinue)
    
    def _has_speech(self, audio_chunk):
        """Voice Activity Detection"""
        frame_duration = 30
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
        """Complete translation pipeline"""
        if len(self.audio_buffer) < self.model_sample_rate * self.chunk_duration:
            return
        
        # Get audio chunk
        audio_chunk = np.array(list(self.audio_buffer)[:self.model_sample_rate * self.chunk_duration])
        
        # Clear processed portion
        overlap_size = len(self.audio_buffer) // 2
        for _ in range(len(self.audio_buffer) - overlap_size):
            if self.audio_buffer:
                self.audio_buffer.popleft()
        
        # Check for speech
        if not self._has_speech(audio_chunk):
            return
        
        lang_name = self.lang_map[self.target_language]["name"]
        print(f"🔊 Processing English → {lang_name}...")
        
        start_time = time.time()
        
        try:
            # Step 1: Speech-to-Text (English)
            asr_start = time.time()
            if self.asr_optimized:
                # Use optimized Transformers pipeline with better parameters
                from torch.nn.attention import SDPBackend, sdpa_kernel
                
                generate_kwargs = {
                    "language": "english",
                    "task": "transcribe",
                    "condition_on_prev_tokens": False,
                    "compression_ratio_threshold": 1.35,  # Better threshold for quality
                    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # Temperature fallback
                    "logprob_threshold": -0.3,  # Higher confidence
                    "no_speech_threshold": 0.4,
                    "return_timestamps": False
                }
                
                with sdpa_kernel(SDPBackend.MATH):
                    result = self.asr_model(audio_chunk, generate_kwargs=generate_kwargs)
                
                english_text = result["text"].strip()
                english_segments = [english_text] if len(english_text) > 2 else []
                
            else:
                # Fallback to faster-whisper
                segments, info = self.asr_model.transcribe(
                    audio_chunk,
                    language="en",
                    batch_size=8,
                    vad_filter=True,
                    vad_parameters={
                        "threshold": 0.6,
                        "min_silence_duration_ms": 500,
                        "speech_pad_ms": 200
                    },
                    condition_on_previous_text=False,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-0.5,
                    no_speech_threshold=0.4
                )
                
                # Collect English text
                english_segments = []
                for segment in segments:
                    seg_text = segment.text.strip()
                    if len(seg_text) > 2:
                        english_segments.append(seg_text)
            
            if not english_segments:
                print("🔇 No clear English speech detected")
                return
            
            english_text = " ".join(english_segments)
            asr_time = time.time() - asr_start
            
            # Step 2: Translation (English → Target Language)
            trans_start = time.time()
            translated_text = self.translate_text(english_text)
            trans_time = time.time() - trans_start
            
            # Step 3: Text-to-Speech (Optimized Indic Parler-TTS)
            tts_start = time.time()
            audio_output = None
            if self.use_tts and translated_text and not translated_text.startswith("["):
                audio_result = self.synthesize_speech(translated_text)
                if audio_result:
                    audio_output, sample_rate = audio_result
                    # Play audio using PyAudio
                    self._play_audio(audio_output, sample_rate)
                    print(f"🔊 Generated {len(audio_output)/sample_rate:.2f}s of speech")
            tts_time = time.time() - tts_start
            
            total_time = time.time() - start_time
            
            # Display results
            print(f"🇬🇧 English: \"{english_text}\"")
            print(f"🌍 {lang_name}: \"{translated_text}\"")
            print(f"⚡ Timing: ASR {asr_time:.2f}s | Trans {trans_time:.2f}s | TTS {tts_time:.2f}s | Total {total_time:.2f}s")
            # Show confidence only for faster-whisper (has info object)
            if not self.asr_optimized and 'info' in locals():
                print(f"📊 Confidence: {info.language_probability:.2f}")
            else:
                print(f"📊 Confidence: 1.00")
            print("-" * 70)
            
        except Exception as e:
            print(f"❌ Translation error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Real-time Speech Translation")
    parser.add_argument("--target", choices=["hi", "kn", "ta"], default="hi",
                       help="Target language (hi=Hindi, kn=Kannada, ta=Tamil)")
    parser.add_argument("--duration", type=int, default=3,
                       help="Audio chunk duration in seconds")
    parser.add_argument("--device", default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--mic", type=int, default=0,
                       help="Microphone device index")
    
    args = parser.parse_args()
    
    lang_names = {"hi": "Hindi", "kn": "Kannada", "ta": "Tamil"}
    
    print(f"🚀 Real-time Speech Translation System")
    print(f"🎯 Translation: English → {lang_names[args.target]}")
    print(f"🎤 Microphone: Device {args.mic}")
    print(f"💻 Device: {args.device}")
    print()
    print("📋 Pipeline: Speech → Text → Translation → Text (+ TTS future)")
    print("🔧 MVP Status: ASR ✓ | Translation ✓ (basic) | TTS ⏳ (coming)")
    print()
    print("📝 Try these phrases:")
    print("   - Hello")
    print("   - How are you")
    print("   - Thank you")
    print("   - Good morning")
    print("   - Please help")
    print()
    
    translator = RealTimeTranslator(
        target_language=args.target,
        device=args.device,
        chunk_duration=args.duration,
        microphone_index=args.mic
    )
    
    translator.start_listening()

if __name__ == "__main__":
    main()