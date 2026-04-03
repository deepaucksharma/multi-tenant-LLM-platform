"""
Text-to-Speech engine using edge-tts.
Free, fast, and doesn't require API keys.
Supports tenant-specific voices.
"""
import io
import asyncio
import time
from typing import Optional, AsyncGenerator
from pathlib import Path

from loguru import logger

from voice_agent.config import VOICE_CONFIG


async def synthesize_speech(
    text: str,
    tenant_id: str = "sis",
    output_path: Optional[str] = None,
) -> bytes:
    """
    Synthesize speech from text using edge-tts.

    Args:
        text: Text to speak
        tenant_id: Tenant ID for voice selection
        output_path: Optional file path to save audio

    Returns:
        MP3 audio bytes
    """
    import edge_tts

    # Select voice based on tenant
    voice = (
        VOICE_CONFIG.tts_voice_sis
        if tenant_id == "sis"
        else VOICE_CONFIG.tts_voice_mfg
    )

    t_start = time.time()

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=VOICE_CONFIG.tts_rate,
    )

    # Collect audio bytes
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])

    audio_bytes = b"".join(audio_chunks)
    tts_time = round(time.time() - t_start, 3)

    logger.info(
        f"TTS: {len(text)} chars → {len(audio_bytes)} bytes "
        f"({tts_time}s, voice={voice})"
    )

    # Save to file if requested
    if output_path:
        Path(output_path).write_bytes(audio_bytes)

    return audio_bytes


async def synthesize_speech_streaming(
    text: str,
    tenant_id: str = "sis",
) -> AsyncGenerator[bytes, None]:
    """
    Stream TTS audio chunks as they're generated.
    Yields MP3 audio chunks for real-time playback.
    """
    import edge_tts

    voice = (
        VOICE_CONFIG.tts_voice_sis
        if tenant_id == "sis"
        else VOICE_CONFIG.tts_voice_mfg
    )

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=VOICE_CONFIG.tts_rate,
    )

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]


def synthesize_speech_sync(text: str, tenant_id: str = "sis") -> bytes:
    """Synchronous wrapper for TTS synthesis."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    synthesize_speech(text, tenant_id),
                )
                return future.result()
        else:
            return loop.run_until_complete(synthesize_speech(text, tenant_id))
    except RuntimeError:
        return asyncio.run(synthesize_speech(text, tenant_id))


async def list_voices(language: str = "en") -> list:
    """List available TTS voices for a language."""
    import edge_tts
    voices = await edge_tts.list_voices()
    return [
        {"name": v["ShortName"], "gender": v["Gender"], "locale": v["Locale"]}
        for v in voices
        if v["Locale"].startswith(language)
    ]


if __name__ == "__main__":
    async def test():
        print("Testing TTS engine...")

        audio = await synthesize_speech(
            "The enrollment process requires proof of residency and immunization records.",
            tenant_id="sis",
            output_path="test_sis_voice.mp3",
        )
        print(f"SIS voice: {len(audio)} bytes → test_sis_voice.mp3")

        audio = await synthesize_speech(
            "The lockout tagout procedure must be followed for all maintenance activities.",
            tenant_id="mfg",
            output_path="test_mfg_voice.mp3",
        )
        print(f"MFG voice: {len(audio)} bytes → test_mfg_voice.mp3")

        voices = await list_voices("en")
        print(f"\nAvailable English voices: {len(voices)}")
        for v in voices[:10]:
            print(f"  {v['name']} ({v['gender']})")

    asyncio.run(test())
