#!/usr/bin/env python3
"""
Enhanced STT with debugging to fix connection issues
"""

import asyncio
import json
import numpy as np
from typing import Optional, Dict, Any
import torch
import torchaudio
from collections import deque
from datetime import datetime
import os
import logging
import time
import sys

# FastAPI and WebRTC
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder
import av
import webrtcvad

# Enhanced logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more info
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress some verbose logs
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.INFO)

# =====================================
# Auto-detect available STT backends
# =====================================

STT_BACKEND = None

# Check for converted CTranslate2 model
if os.path.exists("models/whisper-hindi-ct2/model.bin"):
    try:
        from faster_whisper import WhisperModel
        STT_BACKEND = "faster_whisper"
        logger.info("✓ Found converted CTranslate2 model - using faster-whisper")
    except ImportError:
        pass

if not STT_BACKEND:
    try:
        import whisper
        STT_BACKEND = "openai_whisper"
        logger.info("✓ Using OpenAI Whisper")
    except ImportError:
        logger.error("No STT backend available!")
        sys.exit(1)

# =====================================
# Configuration
# =====================================

class Config:
    SAMPLE_RATE = 16000
    BUFFER_DURATION_S = 3.0  # Longer chunks for better context
    VAD_AGGRESSIVENESS = 2  # More aggressive to filter noise
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    if STT_BACKEND == "faster_whisper":
        # Language-specific CT2 models
        MODELS = {
            'hindi': 'models/whisper-hindi-ct2',
            'kannada': 'models/whisper-kannada-ct2',
            'tamil': 'models/whisper-tamil-ct2',
            'english': 'models/whisper-english-base-ct2'  # English-specific model
        }
        COMPUTE_TYPE = "int8"
    else:
        # Fallback to OpenAI Whisper models
        MODELS = {
            'hindi': 'small',
            'kannada': 'small',
            'tamil': 'small',
            'english': 'base.en'
        }

# =====================================
# Audio Processing
# =====================================

class AudioBuffer:
    def __init__(self, sample_rate: int = 16000, buffer_duration: float = 1.0):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_duration)
        self.buffer = deque(maxlen=self.buffer_size * 2)
        self.processing = False
        
    def add_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        self.buffer.extend(audio_chunk.flatten())
        
        if len(self.buffer) >= self.buffer_size and not self.processing:
            audio_array = np.array(list(self.buffer)[:self.buffer_size])
            # Keep 50% overlap
            for _ in range(self.buffer_size // 2):
                if self.buffer:
                    self.buffer.popleft()
            self.processing = True
            return audio_array
        return None
    
    def reset_processing(self):
        self.processing = False

# =====================================
# Unified STT
# =====================================

class UnifiedSTT:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.backend = STT_BACKEND
        self.models = {}  # Cache for language-specific models
        
        logger.info(f"Initializing {self.backend} on {device}...")
        
        # Pre-load models for all languages
        for language in ['hindi', 'kannada', 'tamil', 'english']:
            self._load_model(language)
        
        logger.info(f"✓ All language models loaded successfully!")
    
    def _load_model(self, language: str):
        """Load language-specific model if not already cached"""
        if language in self.models:
            return self.models[language]
        
        logger.info(f"Loading {language} model...")
        
        if self.backend == "faster_whisper":
            from faster_whisper import WhisperModel, BatchedInferencePipeline
            model_path = Config.MODELS.get(language, Config.MODELS['hindi'])
            
            # Check if model exists
            if not os.path.exists(model_path):
                logger.warning(f"Model {model_path} not found, using Hindi model as fallback")
                model_path = Config.MODELS['hindi']
            
            # Load base model
            base_model = WhisperModel(
                model_path,
                device=self.device,
                compute_type=Config.COMPUTE_TYPE,
                num_workers=2
            )
            
            # Wrap with BatchedInferencePipeline for better performance
            self.models[language] = BatchedInferencePipeline(model=base_model)
            logger.info(f"✓ Loaded {language} model from {model_path} with batched inference")
        else:
            import whisper
            model_size = Config.MODELS.get(language, 'small')
            self.models[language] = whisper.load_model(model_size, device=self.device)
            logger.info(f"✓ Loaded {language} Whisper model: {model_size}")
        
        return self.models[language]
    
    def _warmup(self):
        dummy_audio = np.zeros(16000, dtype=np.float32)
        self.transcribe(dummy_audio, language='en')
        logger.info("Model warmed up!")
    
    def transcribe(self, audio: np.ndarray, language: str = 'hi') -> Dict[str, Any]:
        start_time = time.time()
        
        # Map language codes to model languages
        language_map = {
            'hi': 'hindi',
            'kn': 'kannada', 
            'ta': 'tamil',
            'en': 'english'
        }
        model_language = language_map.get(language, 'hindi')
        
        # Get the appropriate model
        model = self.models.get(model_language)
        if not model:
            logger.warning(f"No model for {model_language}, loading it now...")
            model = self._load_model(model_language)
        
        # Normalize audio
        if audio.max() > 1.0:
            audio = audio / 32768.0
        
        try:
            if self.backend == "faster_whisper":
                logger.debug(f"Transcribing with language code: {language}, model: {model_language}")
                # BatchedInferencePipeline for faster processing
                segments, info = model.transcribe(
                    audio,
                    language=language,
                    beam_size=5,
                    temperature=0,
                    batch_size=8,  # Process in batches for speed
                    vad_filter=True,
                    vad_parameters={
                        "threshold": 0.6,  # Higher threshold to be more selective
                        "min_silence_duration_ms": 1000,  # Longer silence required
                        "speech_pad_ms": 300  # More padding around speech
                    },
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False  # Critical: Prevents repetitive hallucinations
                )
                text = " ".join([seg.text for seg in segments])
            else:
                result = model.transcribe(
                    audio,
                    language=language,
                    fp16=(self.device == "cuda"),
                    verbose=False,
                    temperature=0
                )
                text = result['text']
            
            processing_time = time.time() - start_time
            audio_duration = len(audio) / 16000
            rtf = processing_time / audio_duration if audio_duration > 0 else 0
            
            return {
                'text': text.strip(),
                'language': language,
                'processing_time': processing_time,
                'audio_duration': audio_duration,
                'rtf': rtf,
                'backend': self.backend,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {'text': '', 'error': str(e)}

# =====================================
# WebRTC Handler with Enhanced Debugging
# =====================================

class AudioProcessor:
    """Process audio from WebRTC track"""
    def __init__(self, stt_model: UnifiedSTT, websocket: WebSocket, language: str = "hi"):
        self.stt_model = stt_model
        self.websocket = websocket
        self.language = language
        self.audio_buffer = AudioBuffer()
        self.frame_count = 0
        self.total_frames = 0
        self.resampler = None
        
    async def process_audio_track(self, track):
        """Process incoming audio track"""
        logger.info(f"Starting audio processing for track: {track.kind}")
        
        try:
            while True:
                frame = await track.recv()
                self.frame_count += 1
                
                # Log every 100 frames to confirm audio is being received
                if self.frame_count % 100 == 0:
                    logger.debug(f"Received {self.frame_count} audio frames")
                
                # Convert to numpy
                audio_data = frame.to_ndarray()
                
                # Resample if needed
                if frame.sample_rate != 16000:
                    if self.resampler is None:
                        self.resampler = torchaudio.transforms.Resample(
                            orig_freq=frame.sample_rate,
                            new_freq=16000
                        )
                        logger.info(f"Resampling from {frame.sample_rate}Hz to 16000Hz")
                    
                    audio_tensor = torch.from_numpy(audio_data).float()
                    audio_data = self.resampler(audio_tensor).numpy()
                
                # Add to buffer
                processable_audio = self.audio_buffer.add_chunk(audio_data)
                
                if processable_audio is not None:
                    # Process audio
                    asyncio.create_task(self._transcribe(processable_audio))
                    self.audio_buffer.reset_processing()
                    
        except Exception as e:
            logger.error(f"Error processing audio track: {e}")
    
    async def _transcribe(self, audio: np.ndarray):
        """Transcribe audio chunk"""
        try:
            logger.debug(f"Processing audio chunk of {len(audio)/16000:.2f} seconds")
            result = self.stt_model.transcribe(audio, language=self.language)
            
            if result.get('text'):
                logger.info(f"Transcription: {result['text']}")
                await self.websocket.send_json({
                    'type': 'transcription',
                    'data': result
                })
            else:
                logger.debug("No speech detected in chunk")
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")

# =====================================
# FastAPI Application
# =====================================

app = FastAPI(title="Smart STT Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

peer_connections = {}
stt_model = None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global stt_model
    logger.info("="*60)
    logger.info("Starting STT Service")
    logger.info(f"Backend: {STT_BACKEND}")
    logger.info(f"Device: {Config.DEVICE}")
    logger.info("="*60)
    
    stt_model = UnifiedSTT(device=Config.DEVICE)
    logger.info("✓ Service ready at http://localhost:8000")
    
    yield
    
    logger.info("Shutting down...")

app = FastAPI(title="Smart STT Service", lifespan=lifespan)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    logger.info(f"✓ Client {client_id} connected via WebSocket")
    
    pc = RTCPeerConnection()
    peer_connections[client_id] = {"pc": pc, "audio_processor": None}
    
    language_map = {
        'hindi': 'hi',
        'english': 'en',
        'tamil': 'ta',
        'kannada': 'kn'
    }
    current_language = 'hi'
    
    # Set up connection state logging
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
    
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state: {pc.iceConnectionState}")
    
    @pc.on("track")
    async def on_track(track):
        logger.info(f"✓ Received {track.kind} track")
        
        if track.kind == "audio":
            # Create audio processor
            processor = AudioProcessor(stt_model, websocket, current_language)
            peer_connections[client_id]["audio_processor"] = processor
            
            # Start processing
            asyncio.create_task(processor.process_audio_track(track))
            
            # Send confirmation
            await websocket.send_json({
                'type': 'status',
                'message': 'Audio track connected successfully'
            })
    
    try:
        while True:
            message = await websocket.receive_json()
            logger.debug(f"Received message type: {message.get('type')}")
            
            if message["type"] == "offer":
                # Set remote description
                offer = RTCSessionDescription(
                    sdp=message["sdp"],
                    type=message["type"]
                )
                await pc.setRemoteDescription(offer)
                logger.info("✓ Set remote description (offer)")
                
                # Create answer
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                logger.info("✓ Created local description (answer)")
                
                # Send answer
                await websocket.send_json({
                    "type": "answer",
                    "sdp": pc.localDescription.sdp
                })
                logger.info("✓ Sent answer to client")
                
            elif message["type"] == "ice_candidate":
                # Add ICE candidate if present
                if message.get("candidate"):
                    candidate = message["candidate"]
                    await pc.addIceCandidate(candidate)
                    logger.debug("Added ICE candidate")
                    
            elif message["type"] == "change_language":
                lang = message.get("language", "hindi")
                current_language = language_map.get(lang, 'hi')
                logger.info(f"Language changed to: {current_language}")
                
                # Update processor language if it exists
                if peer_connections[client_id]["audio_processor"]:
                    peer_connections[client_id]["audio_processor"].language = current_language
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup
        if client_id in peer_connections:
            await peer_connections[client_id]["pc"].close()
            del peer_connections[client_id]

@app.get("/")
async def index():
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>STT Service - Debug Mode</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 40px auto;
                padding: 20px;
                background: #f5f5f5;
            }}
            .container {{
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .controls {{
                margin: 20px 0;
                display: flex;
                gap: 10px;
                align-items: center;
            }}
            button {{
                padding: 10px 20px;
                font-size: 16px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }}
            #start {{ background: #4CAF50; color: white; }}
            #stop {{ background: #f44336; color: white; }}
            button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
            select {{
                padding: 10px;
                font-size: 16px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            #status {{
                padding: 10px;
                margin: 20px 0;
                background: #e3f2fd;
                border-radius: 5px;
                font-family: monospace;
            }}
            #transcription {{
                border: 1px solid #ddd;
                border-radius: 5px;
                min-height: 200px;
                max-height: 400px;
                overflow-y: auto;
                padding: 20px;
                background: #fafafa;
            }}
            .transcript-item {{
                margin: 10px 0;
                padding: 10px;
                background: white;
                border-radius: 5px;
                border-left: 4px solid #4CAF50;
            }}
            .debug-log {{
                margin-top: 20px;
                padding: 10px;
                background: #263238;
                color: #aed581;
                font-family: monospace;
                font-size: 12px;
                max-height: 200px;
                overflow-y: auto;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎙️ Speech-to-Text Service (Debug Mode)</h1>
            <p>Backend: {STT_BACKEND} | Device: {Config.DEVICE}</p>
            
            <div class="controls">
                <button id="start">▶️ Start Recording</button>
                <button id="stop" disabled>⏹️ Stop</button>
                <select id="language">
                    <option value="hindi">Hindi</option>
                    <option value="english">English</option>
                    <option value="tamil">Tamil</option>
                    <option value="kannada">Kannada</option>
                </select>
            </div>
            
            <div id="status">Status: Ready</div>
            
            <div id="transcription">
                <p style="color: #999;">Transcriptions will appear here...</p>
            </div>
            
            <div class="debug-log" id="debug">
                Debug log...
            </div>
        </div>
        
        <script>
            let pc = null;
            let ws = null;
            let stream = null;
            const clientId = Math.random().toString(36).substring(7);
            
            function log(message) {{
                const debug = document.getElementById('debug');
                const time = new Date().toLocaleTimeString();
                debug.innerHTML += `[${{time}}] ${{message}}<br>`;
                debug.scrollTop = debug.scrollHeight;
                console.log(message);
            }}
            
            function updateStatus(message) {{
                document.getElementById('status').textContent = `Status: ${{message}}`;
            }}
            
            document.getElementById('start').onclick = async () => {{
                try {{
                    log('Starting...');
                    updateStatus('Connecting to server...');
                    
                    // Connect WebSocket
                    ws = new WebSocket(`ws://localhost:8000/ws/${{clientId}}`);
                    log(`WebSocket connecting to /ws/${{clientId}}`);
                    
                    ws.onopen = async () => {{
                        log('✓ WebSocket connected');
                        updateStatus('WebSocket connected, setting up WebRTC...');
                        
                        // Create peer connection
                        pc = new RTCPeerConnection({{
                            iceServers: [{{urls: 'stun:stun.l.google.com:19302'}}]
                        }});
                        log('Created RTCPeerConnection');
                        
                        // Add connection state monitoring
                        pc.onconnectionstatechange = () => {{
                            log(`Connection state: ${{pc.connectionState}}`);
                            updateStatus(`Connection: ${{pc.connectionState}}`);
                        }};
                        
                        pc.oniceconnectionstatechange = () => {{
                            log(`ICE state: ${{pc.iceConnectionState}}`);
                        }};
                        
                        pc.onicegatheringstatechange = () => {{
                            log(`ICE gathering: ${{pc.iceGatheringState}}`);
                        }};
                        
                        // Get user media
                        log('Requesting microphone access...');
                        stream = await navigator.mediaDevices.getUserMedia({{
                            audio: {{
                                echoCancellation: true,
                                noiseSuppression: true,
                                autoGainControl: true,
                                sampleRate: 16000
                            }}
                        }});
                        log(`✓ Got audio stream (tracks: ${{stream.getTracks().length}})`);
                        
                        // Add tracks to peer connection
                        stream.getTracks().forEach(track => {{
                            const sender = pc.addTrack(track, stream);
                            log(`Added ${{track.kind}} track to peer connection`);
                        }});
                        
                        // Create offer
                        log('Creating offer...');
                        const offer = await pc.createOffer();
                        await pc.setLocalDescription(offer);
                        log('✓ Created and set local description');
                        
                        // Send offer
                        ws.send(JSON.stringify({{
                            type: 'offer',
                            sdp: offer.sdp
                        }}));
                        log('✓ Sent offer to server');
                        
                        // Send language preference
                        ws.send(JSON.stringify({{
                            type: 'change_language',
                            language: document.getElementById('language').value
                        }}));
                    }};
                    
                    ws.onmessage = async (event) => {{
                        const msg = JSON.parse(event.data);
                        log(`Received: ${{msg.type}}`);
                        
                        if (msg.type === 'answer') {{
                            const answer = new RTCSessionDescription({{
                                type: 'answer',
                                sdp: msg.sdp
                            }});
                            await pc.setRemoteDescription(answer);
                            log('✓ Set remote description (answer)');
                            updateStatus('Connected - Speak now!');
                            
                        }} else if (msg.type === 'transcription') {{
                            const div = document.getElementById('transcription');
                            if (div.querySelector('p')) {{
                                div.innerHTML = '';
                            }}
                            
                            const item = document.createElement('div');
                            item.className = 'transcript-item';
                            item.innerHTML = `
                                <strong>${{new Date().toLocaleTimeString()}}:</strong> 
                                ${{msg.data.text}}<br>
                                <small>RTF: ${{msg.data.rtf?.toFixed(2)}} | 
                                Duration: ${{msg.data.audio_duration?.toFixed(1)}}s |
                                Backend: ${{msg.data.backend}}</small>
                            `;
                            div.insertBefore(item, div.firstChild);
                            log(`Transcription received: "${{msg.data.text}}"`);
                            
                        }} else if (msg.type === 'status') {{
                            updateStatus(msg.message);
                            log(`Status: ${{msg.message}}`);
                        }}
                    }};
                    
                    ws.onerror = (error) => {{
                        log(`WebSocket error: ${{error}}`);
                        updateStatus('WebSocket error');
                    }};
                    
                    ws.onclose = () => {{
                        log('WebSocket closed');
                        updateStatus('Disconnected');
                    }};
                    
                    document.getElementById('start').disabled = true;
                    document.getElementById('stop').disabled = false;
                    
                }} catch (error) {{
                    log(`ERROR: ${{error.message}}`);
                    updateStatus(`Error: ${{error.message}}`);
                    alert('Error: ' + error.message);
                }}
            }};
            
            document.getElementById('stop').onclick = () => {{
                log('Stopping...');
                
                if (stream) {{
                    stream.getTracks().forEach(track => {{
                        track.stop();
                        log(`Stopped ${{track.kind}} track`);
                    }});
                    stream = null;
                }}
                
                if (pc) {{
                    pc.close();
                    log('Closed peer connection');
                    pc = null;
                }}
                
                if (ws) {{
                    ws.close();
                    log('Closed WebSocket');
                    ws = null;
                }}
                
                document.getElementById('start').disabled = false;
                document.getElementById('stop').disabled = true;
                updateStatus('Stopped');
            }};
            
            document.getElementById('language').onchange = (e) => {{
                if (ws && ws.readyState === WebSocket.OPEN) {{
                    ws.send(JSON.stringify({{
                        type: 'change_language',
                        language: e.target.value
                    }}));
                    log(`Language changed to: ${{e.target.value}}`);
                }}
            }};
            
            // Initial log
            log('Page loaded, ready to start');
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)