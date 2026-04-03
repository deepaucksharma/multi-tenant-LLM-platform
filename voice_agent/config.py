"""Voice agent configuration."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class VoiceConfig:
    # STT (Speech-to-Text)
    whisper_model: str = "base"  # tiny, base, small — fits 8GB with LLM
    whisper_device: str = "cpu"  # Keep on CPU to save GPU for LLM
    whisper_language: str = "en"

    # TTS (Text-to-Speech)
    tts_engine: str = "edge-tts"  # edge-tts is free and fast
    tts_voice_sis: str = "en-US-JennyNeural"
    tts_voice_mfg: str = "en-US-GuyNeural"
    tts_rate: str = "+10%"  # Slightly faster speech

    # Server
    voice_port: int = int(os.getenv("VOICE_PORT", "8001"))
    inference_url: str = os.getenv("INFERENCE_URL", "http://localhost:8000")

    # Audio
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 30  # WebRTC standard

    # Pipeline
    silence_threshold: float = 0.01
    silence_duration_s: float = 1.5  # Silence before processing
    max_recording_s: float = 30.0  # Maximum recording length
    min_recording_s: float = 0.5   # Minimum recording length


VOICE_CONFIG = VoiceConfig()
