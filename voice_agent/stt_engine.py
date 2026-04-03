"""
Speech-to-Text engine using faster-whisper.
Runs on CPU to keep GPU free for LLM inference.
"""
import time
import io
import numpy as np
from typing import Dict

from loguru import logger

from voice_agent.config import VOICE_CONFIG

# Lazy-loaded model
_whisper_model = None
_model_lock = None


def get_whisper_model():
    """Load Whisper model (singleton, lazy-loaded)."""
    global _whisper_model, _model_lock

    if _model_lock is None:
        import threading
        _model_lock = threading.Lock()

    if _whisper_model is not None:
        return _whisper_model

    with _model_lock:
        if _whisper_model is not None:
            return _whisper_model

        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading faster-whisper model: {VOICE_CONFIG.whisper_model}")
            _whisper_model = WhisperModel(
                VOICE_CONFIG.whisper_model,
                device=VOICE_CONFIG.whisper_device,
                compute_type="int8",  # CPU-optimized
            )
            logger.info("Whisper model loaded (faster-whisper)")
            return _whisper_model
        except ImportError:
            logger.info("faster-whisper not available, trying openai-whisper")

        try:
            import whisper
            logger.info(f"Loading openai-whisper model: {VOICE_CONFIG.whisper_model}")
            _whisper_model = whisper.load_model(
                VOICE_CONFIG.whisper_model,
                device=VOICE_CONFIG.whisper_device,
            )
            logger.info("Whisper model loaded (openai-whisper)")
            return _whisper_model
        except ImportError:
            logger.error("No whisper library available. Install faster-whisper or openai-whisper.")
            raise


def transcribe_audio(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    language: str = "en",
) -> Dict:
    """
    Transcribe audio data to text.

    Args:
        audio_data: numpy array of float32 audio samples
        sample_rate: Audio sample rate
        language: Language code

    Returns:
        Dict with 'text', 'language', 'duration_s', 'transcription_time_s'
    """
    model = get_whisper_model()
    t_start = time.time()

    # Normalize audio
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    if np.max(np.abs(audio_data)) > 1.0:
        audio_data = audio_data / 32768.0  # int16 to float32

    duration_s = len(audio_data) / sample_rate

    try:
        # Try faster-whisper API
        if hasattr(model, 'transcribe') and not hasattr(model, 'decode'):
            segments, info = model.transcribe(
                audio_data,
                language=language,
                beam_size=1,
                best_of=1,
                vad_filter=True,
            )
            text = " ".join(segment.text for segment in segments).strip()
            detected_language = info.language if hasattr(info, 'language') else language
        else:
            # openai-whisper API
            result = model.transcribe(
                audio_data,
                language=language,
                fp16=False,
            )
            text = result.get("text", "").strip()
            detected_language = result.get("language", language)

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        text = ""
        detected_language = language

    transcription_time = round(time.time() - t_start, 3)

    logger.info(
        f"STT: '{text[:80]}{'...' if len(text) > 80 else ''}' "
        f"({duration_s:.1f}s audio → {transcription_time}s transcription)"
    )

    return {
        "text": text,
        "language": detected_language,
        "duration_s": round(duration_s, 2),
        "transcription_time_s": transcription_time,
        "realtime_factor": round(transcription_time / max(duration_s, 0.01), 2),
    }


def transcribe_wav_bytes(wav_bytes: bytes) -> Dict:
    """Transcribe from WAV file bytes."""
    import wave

    with io.BytesIO(wav_bytes) as wav_io:
        with wave.open(wav_io, 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

    # Convert to numpy
    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

    # Convert to mono if stereo
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return transcribe_audio(audio, sample_rate)


if __name__ == "__main__":
    print("Testing STT engine...")
    test_audio = np.zeros(16000 * 2, dtype=np.float32)  # 2 seconds silence
    result = transcribe_audio(test_audio)
    print(f"Result: {result}")
