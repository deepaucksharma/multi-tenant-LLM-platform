"""
Complete voice pipeline: STT → LLM (via inference API) → TTS
Orchestrates the full voice conversation flow.
"""
import time
import asyncio
import json
import numpy as np
from typing import Dict, Optional, AsyncGenerator
from dataclasses import dataclass, field

import httpx
from loguru import logger

from voice_agent.config import VOICE_CONFIG
from voice_agent.stt_engine import transcribe_audio
from voice_agent.tts_engine import synthesize_speech, synthesize_speech_streaming


@dataclass
class VoiceTurn:
    """A single turn in the voice conversation."""
    turn_id: int
    user_audio_duration_s: float = 0.0
    user_text: str = ""
    assistant_text: str = ""
    assistant_audio_bytes: int = 0
    stt_time_s: float = 0.0
    llm_time_s: float = 0.0
    tts_time_s: float = 0.0
    total_time_s: float = 0.0
    tenant_id: str = ""
    model_type: str = ""


@dataclass
class ConversationState:
    """Tracks the state of a voice conversation."""
    session_id: str
    tenant_id: str
    turns: list = field(default_factory=list)
    history: list = field(default_factory=list)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)

    def add_turn(self, user_text: str, assistant_text: str):
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})
        # Keep last 6 messages (3 turns)
        if len(self.history) > 6:
            self.history = self.history[-6:]


class VoicePipeline:
    """
    End-to-end voice pipeline.
    STT (Whisper) → LLM (inference API) → TTS (edge-tts)
    """

    def __init__(self, inference_url: str = None):
        self.inference_url = inference_url or VOICE_CONFIG.inference_url
        self._sessions: Dict[str, ConversationState] = {}

    def create_session(self, tenant_id: str) -> ConversationState:
        """Create a new conversation session."""
        session_id = f"voice_{int(time.time()*1000)}"
        session = ConversationState(
            session_id=session_id,
            tenant_id=tenant_id,
        )
        self._sessions[session_id] = session
        logger.info(f"Voice session created: {session_id} (tenant: {tenant_id})")
        return session

    def get_session(self, session_id: str) -> Optional[ConversationState]:
        return self._sessions.get(session_id)

    async def process_audio(
        self,
        audio_data: np.ndarray,
        session: ConversationState,
        sample_rate: int = 16000,
    ) -> Dict:
        """
        Process a complete audio utterance through the full pipeline.

        Returns:
            Dict with text, audio, and timing information
        """
        t_total_start = time.time()
        turn_id = len(session.turns) + 1
        tenant_id = session.tenant_id

        # ---- Step 1: STT ----
        t_stt_start = time.time()
        stt_result = transcribe_audio(audio_data, sample_rate)
        stt_time = time.time() - t_stt_start

        user_text = stt_result["text"]

        if not user_text.strip():
            return {
                "status": "empty",
                "message": "No speech detected",
                "stt_time_s": round(stt_time, 3),
            }

        logger.info(f"[Turn {turn_id}] User: '{user_text}'")

        # ---- Step 2: LLM Inference ----
        t_llm_start = time.time()
        try:
            llm_response = await self._call_inference(
                user_text, tenant_id, session.history
            )
            assistant_text = llm_response.get("message", "I couldn't generate a response.")
            model_type = llm_response.get("model_type", "")
        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            assistant_text = "I'm sorry, I'm having trouble responding right now. Please try again."
            model_type = "error"
        llm_time = time.time() - t_llm_start

        logger.info(f"[Turn {turn_id}] Assistant: '{assistant_text[:100]}...'")

        # ---- Step 3: TTS ----
        t_tts_start = time.time()
        try:
            audio_bytes = await synthesize_speech(assistant_text, tenant_id)
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            audio_bytes = b""
        tts_time = time.time() - t_tts_start

        total_time = time.time() - t_total_start

        # Update session
        session.add_turn(user_text, assistant_text)

        turn = VoiceTurn(
            turn_id=turn_id,
            user_audio_duration_s=stt_result["duration_s"],
            user_text=user_text,
            assistant_text=assistant_text,
            assistant_audio_bytes=len(audio_bytes),
            stt_time_s=round(stt_time, 3),
            llm_time_s=round(llm_time, 3),
            tts_time_s=round(tts_time, 3),
            total_time_s=round(total_time, 3),
            tenant_id=tenant_id,
            model_type=model_type,
        )
        session.turns.append(turn)

        logger.info(
            f"[Turn {turn_id}] Pipeline: STT={stt_time:.2f}s → "
            f"LLM={llm_time:.2f}s → TTS={tts_time:.2f}s = "
            f"Total={total_time:.2f}s"
        )

        return {
            "status": "success",
            "turn_id": turn_id,
            "user_text": user_text,
            "assistant_text": assistant_text,
            "audio_bytes": audio_bytes,
            "audio_format": "mp3",
            "timing": {
                "stt_s": turn.stt_time_s,
                "llm_s": turn.llm_time_s,
                "tts_s": turn.tts_time_s,
                "total_s": turn.total_time_s,
                "audio_duration_s": turn.user_audio_duration_s,
            },
            "model_type": model_type,
            "session_id": session.session_id,
        }

    async def process_audio_streaming(
        self,
        audio_data: np.ndarray,
        session: ConversationState,
        sample_rate: int = 16000,
    ) -> AsyncGenerator[Dict, None]:
        """
        Process audio with streaming results.
        Yields intermediate results as they become available.
        """
        t_start = time.time()
        tenant_id = session.tenant_id

        # Step 1: STT
        yield {"event": "stt_start", "timestamp": time.time()}
        stt_result = transcribe_audio(audio_data, sample_rate)
        user_text = stt_result["text"]
        yield {
            "event": "stt_complete",
            "text": user_text,
            "duration_s": stt_result["duration_s"],
            "stt_time_s": stt_result["transcription_time_s"],
        }

        if not user_text.strip():
            yield {"event": "empty", "message": "No speech detected"}
            return

        # Step 2: LLM
        yield {"event": "llm_start", "timestamp": time.time()}
        try:
            llm_response = await self._call_inference(
                user_text, tenant_id, session.history
            )
            assistant_text = llm_response.get("message", "")
        except Exception as e:
            assistant_text = "I'm having trouble responding right now."
            yield {"event": "llm_error", "error": str(e)}

        yield {
            "event": "llm_complete",
            "text": assistant_text,
            "llm_time_s": round(time.time() - t_start, 3),
        }

        # Step 3: TTS (streaming)
        yield {"event": "tts_start", "timestamp": time.time()}
        try:
            async for audio_chunk in synthesize_speech_streaming(
                assistant_text, tenant_id
            ):
                yield {"event": "audio_chunk", "data": audio_chunk}
        except Exception as e:
            yield {"event": "tts_error", "error": str(e)}

        session.add_turn(user_text, assistant_text)

        yield {
            "event": "complete",
            "total_time_s": round(time.time() - t_start, 3),
            "user_text": user_text,
            "assistant_text": assistant_text,
        }

    async def _call_inference(
        self,
        message: str,
        tenant_id: str,
        history: list,
    ) -> Dict:
        """Call the inference API."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.inference_url}/chat",
                json={
                    "tenant_id": tenant_id,
                    "message": message,
                    "conversation_history": history[-6:],
                    "use_rag": True,
                    "use_streaming": False,
                    "max_new_tokens": 256,  # Shorter for voice
                    "temperature": 0.7,
                },
            )
            response.raise_for_status()
            return response.json()

    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """Get summary of a voice session."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        turns = session.turns
        if not turns:
            return {"session_id": session_id, "turns": 0}

        return {
            "session_id": session_id,
            "tenant_id": session.tenant_id,
            "total_turns": len(turns),
            "avg_total_time_s": round(
                sum(t.total_time_s for t in turns) / len(turns), 2
            ),
            "avg_stt_time_s": round(
                sum(t.stt_time_s for t in turns) / len(turns), 2
            ),
            "avg_llm_time_s": round(
                sum(t.llm_time_s for t in turns) / len(turns), 2
            ),
            "avg_tts_time_s": round(
                sum(t.tts_time_s for t in turns) / len(turns), 2
            ),
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "user": t.user_text[:100],
                    "assistant": t.assistant_text[:100],
                    "total_s": t.total_time_s,
                }
                for t in turns
            ],
        }


# Module-level singleton
_pipeline: Optional[VoicePipeline] = None


def get_voice_pipeline() -> VoicePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VoicePipeline()
    return _pipeline
