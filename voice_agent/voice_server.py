"""
Voice agent WebSocket server and HTTP endpoints.

Provides:
  1. WebSocket endpoint for real-time voice streaming
  2. HTTP endpoint for upload-based voice processing
  3. Browser-based voice interface

Usage:
    python voice_agent/voice_server.py
"""
import os
import io
import json
import time
import wave
import asyncio
import base64
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from voice_agent.config import VOICE_CONFIG
from voice_agent.voice_pipeline import get_voice_pipeline
from voice_agent.stt_engine import transcribe_wav_bytes
from voice_agent.tts_engine import synthesize_speech

app = FastAPI(
    title="Voice Agent API",
    description="Multi-tenant voice agent: STT → LLM → TTS",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Browser Voice Interface
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def voice_ui():
    """Serve the browser-based voice interface."""
    return VOICE_HTML


# ============================================================
# HTTP Voice Endpoint
# ============================================================

@app.post("/voice/process")
async def process_voice(
    audio: UploadFile = File(..., description="WAV audio file"),
    tenant_id: str = Form(default="sis"),
    session_id: str = Form(default=""),
):
    """
    Process an uploaded audio file through the full voice pipeline.
    Returns JSON with transcription, response text, and base64 audio.
    """
    pipeline = get_voice_pipeline()

    # Get or create session
    session = None
    if session_id:
        session = pipeline.get_session(session_id)
    if not session:
        session = pipeline.create_session(tenant_id)

    # Read and convert audio
    audio_bytes = await audio.read()

    try:
        # Parse WAV
        with io.BytesIO(audio_bytes) as wav_io:
            with wave.open(wav_io, 'rb') as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())

        # Convert to float32 mono
        if sample_width == 2:
            audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_np = np.frombuffer(frames, dtype=np.float32)

        if n_channels > 1:
            audio_np = audio_np.reshape(-1, n_channels).mean(axis=1)

    except Exception as e:
        return {"status": "error", "message": f"Invalid audio format: {e}"}

    # Process through pipeline
    result = await pipeline.process_audio(audio_np, session, sample_rate)

    # Encode response audio as base64 for JSON transport
    response_data = {
        "status": result["status"],
        "session_id": session.session_id,
        "user_text": result.get("user_text", ""),
        "assistant_text": result.get("assistant_text", ""),
        "timing": result.get("timing", {}),
        "model_type": result.get("model_type", ""),
    }

    if result.get("audio_bytes"):
        response_data["audio_base64"] = base64.b64encode(
            result["audio_bytes"]
        ).decode("ascii")
        response_data["audio_format"] = "mp3"

    return response_data


# ============================================================
# WebSocket Voice Endpoint
# ============================================================

@app.websocket("/voice/ws")
async def voice_websocket(
    websocket: WebSocket,
    tenant_id: str = Query(default="sis"),
):
    """
    WebSocket endpoint for real-time voice interaction.

    Protocol:
    Client → Server:
      - {"type": "audio", "data": "<base64 WAV>"} — Send audio chunk
      - {"type": "end_turn"} — Signal end of speech
      - {"type": "set_tenant", "tenant_id": "sis"|"mfg"} — Change tenant

    Server → Client:
      - {"type": "stt_result", "text": "..."} — Transcription
      - {"type": "llm_text", "text": "..."} — LLM response text
      - {"type": "audio_chunk", "data": "<base64 MP3>"} — TTS audio chunk
      - {"type": "turn_complete", "timing": {...}} — Turn complete
      - {"type": "error", "message": "..."} — Error
    """
    await websocket.accept()
    logger.info(f"Voice WebSocket connected (tenant: {tenant_id})")

    pipeline = get_voice_pipeline()
    session = pipeline.create_session(tenant_id)
    audio_buffer = bytearray()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type", "")

            if msg_type == "set_tenant":
                new_tenant = message.get("tenant_id", tenant_id)
                session = pipeline.create_session(new_tenant)
                tenant_id = new_tenant
                await websocket.send_json({
                    "type": "tenant_changed",
                    "tenant_id": tenant_id,
                    "session_id": session.session_id,
                })

            elif msg_type == "audio":
                audio_b64 = message.get("data", "")
                if audio_b64:
                    audio_buffer.extend(base64.b64decode(audio_b64))

            elif msg_type == "end_turn":
                if len(audio_buffer) < 1000:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Audio too short",
                    })
                    audio_buffer.clear()
                    continue

                try:
                    # Parse WAV from buffer
                    with io.BytesIO(bytes(audio_buffer)) as wav_io:
                        with wave.open(wav_io, 'rb') as wf:
                            sr = wf.getframerate()
                            nc = wf.getnchannels()
                            sw = wf.getsampwidth()
                            frames = wf.readframes(wf.getnframes())

                    if sw == 2:
                        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    else:
                        audio_np = np.frombuffer(frames, dtype=np.float32)
                    if nc > 1:
                        audio_np = audio_np.reshape(-1, nc).mean(axis=1)

                    # Stream results
                    async for event in pipeline.process_audio_streaming(
                        audio_np, session, sr
                    ):
                        evt_type = event.get("event", "")

                        if evt_type == "stt_complete":
                            await websocket.send_json({
                                "type": "stt_result",
                                "text": event.get("text", ""),
                                "stt_time_s": event.get("stt_time_s", 0),
                            })

                        elif evt_type == "llm_complete":
                            await websocket.send_json({
                                "type": "llm_text",
                                "text": event.get("text", ""),
                                "llm_time_s": event.get("llm_time_s", 0),
                            })

                        elif evt_type == "audio_chunk":
                            chunk_b64 = base64.b64encode(
                                event["data"]
                            ).decode("ascii")
                            await websocket.send_json({
                                "type": "audio_chunk",
                                "data": chunk_b64,
                            })

                        elif evt_type == "complete":
                            await websocket.send_json({
                                "type": "turn_complete",
                                "user_text": event.get("user_text", ""),
                                "assistant_text": event.get("assistant_text", ""),
                                "total_time_s": event.get("total_time_s", 0),
                            })

                        elif evt_type == "empty":
                            await websocket.send_json({
                                "type": "error",
                                "message": "No speech detected",
                            })

                except Exception as e:
                    logger.error(f"Voice processing error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                    })

                audio_buffer.clear()

    except WebSocketDisconnect:
        logger.info(f"Voice WebSocket disconnected (session: {session.session_id})")
    except Exception as e:
        logger.error(f"Voice WebSocket error: {e}")


# ============================================================
# Session Info
# ============================================================

@app.get("/voice/session/{session_id}")
async def get_session(session_id: str):
    """Get voice session summary."""
    pipeline = get_voice_pipeline()
    summary = pipeline.get_session_summary(session_id)
    if not summary:
        return {"error": "Session not found"}
    return summary


# ============================================================
# TTS Test
# ============================================================

@app.get("/voice/tts")
async def test_tts(
    text: str = Query(default="Hello, this is a test of the text to speech system."),
    tenant_id: str = Query(default="sis"),
):
    """Test TTS endpoint — returns MP3 audio."""
    audio_bytes = await synthesize_speech(text, tenant_id)
    return Response(content=audio_bytes, media_type="audio/mpeg")


# ============================================================
# Browser Voice Interface HTML
# ============================================================

VOICE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Voice Agent — Multi-Tenant LLM</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; background: #f8f9fa; color: #333; }
.container { max-width: 600px; margin: 0 auto; padding: 20px; }
h1 { text-align: center; margin-bottom: 4px; font-size: 1.5em; }
.subtitle { text-align: center; color: #888; font-size: 0.85em; margin-bottom: 20px; }

.tenant-select { display: flex; gap: 8px; justify-content: center; margin-bottom: 20px; }
.tenant-btn { padding: 10px 24px; border: 2px solid #ddd; border-radius: 12px; background: white;
  cursor: pointer; font-size: 14px; transition: all 0.2s; }
.tenant-btn.active-sis { border-color: #3b82f6; background: #eff6ff; color: #1d4ed8; }
.tenant-btn.active-mfg { border-color: #22c55e; background: #f0fdf4; color: #15803d; }

.mic-area { text-align: center; margin: 30px 0; }
.mic-btn { width: 80px; height: 80px; border-radius: 50%; border: none; cursor: pointer;
  font-size: 28px; transition: all 0.2s; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
.mic-btn.idle { background: #3b82f6; color: white; }
.mic-btn.recording { background: #ef4444; color: white; animation: pulse 1s infinite; }
.mic-btn.processing { background: #f59e0b; color: white; cursor: wait; }
.mic-btn:disabled { opacity: 0.5; cursor: not-allowed; }
@keyframes pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.08); } }

.status { text-align: center; font-size: 13px; color: #888; min-height: 24px; margin: 10px 0; }

.transcript-area { margin: 20px 0; }
.turn { margin-bottom: 16px; padding: 12px; border-radius: 12px; }
.turn-user { background: #e8f0fe; border-left: 3px solid #3b82f6; }
.turn-assistant { background: white; border: 1px solid #e5e7eb; border-left: 3px solid #22c55e; }
.turn-label { font-size: 11px; font-weight: 600; color: #888; margin-bottom: 4px; text-transform: uppercase; }
.turn-text { font-size: 14px; line-height: 1.5; }
.turn-meta { font-size: 11px; color: #aaa; margin-top: 6px; }

.connection-bar { text-align: center; padding: 6px; font-size: 12px; border-radius: 8px; margin-bottom: 16px; }
.connected { background: #dcfce7; color: #166534; }
.disconnected { background: #fef2f2; color: #991b1b; }
</style>
</head>
<body>
<div class="container">
  <h1>Voice Agent</h1>
  <p class="subtitle">Multi-Tenant LLM &bull; Speak to ask a question</p>

  <div id="connectionBar" class="connection-bar disconnected">Connecting...</div>

  <div class="tenant-select">
    <button class="tenant-btn active-sis" id="btnSis" onclick="setTenant('sis')">SIS</button>
    <button class="tenant-btn" id="btnMfg" onclick="setTenant('mfg')">MFG</button>
  </div>

  <div class="mic-area">
    <button class="mic-btn idle" id="micBtn" onclick="toggleRecording()">&#127908;</button>
  </div>
  <div class="status" id="status">Tap the microphone to start speaking</div>

  <div class="transcript-area" id="transcripts"></div>
</div>

<script>
let tenantId = 'sis';
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;

const micBtn = document.getElementById('micBtn');
const statusEl = document.getElementById('status');
const transcriptsEl = document.getElementById('transcripts');
const connBar = document.getElementById('connectionBar');

async function checkConnection() {
  try {
    await fetch('/voice/tts?text=test&tenant_id=sis', { method: 'HEAD' });
    connBar.className = 'connection-bar connected';
    connBar.textContent = 'Connected to voice server';
  } catch {
    connBar.className = 'connection-bar disconnected';
    connBar.textContent = 'Voice server not reachable';
  }
}
checkConnection();
setInterval(checkConnection, 10000);

function setTenant(t) {
  tenantId = t;
  document.getElementById('btnSis').className = 'tenant-btn' + (t === 'sis' ? ' active-sis' : '');
  document.getElementById('btnMfg').className = 'tenant-btn' + (t === 'mfg' ? ' active-mfg' : '');
  transcriptsEl.innerHTML = '';
  statusEl.textContent = 'Switched to ' + t.toUpperCase() + ' tenant';
}

async function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true }
    });

    audioContext = new AudioContext({ sampleRate: 16000 });
    audioChunks = [];

    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      await processAudio(blob);
    };

    mediaRecorder.start();
    isRecording = true;
    micBtn.className = 'mic-btn recording';
    micBtn.textContent = '\\u23F9';
    statusEl.textContent = 'Listening... Tap to stop';
  } catch (err) {
    statusEl.textContent = 'Microphone access denied: ' + err.message;
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  isRecording = false;
  micBtn.className = 'mic-btn processing';
  micBtn.textContent = '\\u23F3';
  micBtn.disabled = true;
  statusEl.textContent = 'Processing...';
}

async function processAudio(webmBlob) {
  try {
    const arrayBuffer = await webmBlob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const wavBlob = audioBufferToWav(audioBuffer);

    const formData = new FormData();
    formData.append('audio', wavBlob, 'recording.wav');
    formData.append('tenant_id', tenantId);

    statusEl.textContent = 'Transcribing speech...';

    const response = await fetch('/voice/process', {
      method: 'POST',
      body: formData,
    });

    const result = await response.json();

    if (result.status === 'success') {
      addTurn('user', result.user_text, null);

      statusEl.textContent = 'Generating response...';

      addTurn('assistant', result.assistant_text, result.timing);

      if (result.audio_base64) {
        statusEl.textContent = 'Playing response...';
        await playAudio(result.audio_base64);
      }

      statusEl.textContent = 'Done (' + result.timing.total_s + 's total)';
    } else {
      statusEl.textContent = result.message || 'No speech detected';
    }
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
  }

  micBtn.className = 'mic-btn idle';
  micBtn.textContent = '\\u{1F3A4}';
  micBtn.disabled = false;
}

function addTurn(role, text, timing) {
  const div = document.createElement('div');
  div.className = 'turn turn-' + role;

  let meta = '';
  if (timing) {
    meta = '<div class="turn-meta">STT: ' + timing.stt_s + 's | LLM: ' + timing.llm_s + 's | TTS: ' + timing.tts_s + 's | Total: ' + timing.total_s + 's</div>';
  }

  div.innerHTML =
    '<div class="turn-label">' + (role === 'user' ? 'You' : tenantId.toUpperCase() + ' Assistant') + '</div>' +
    '<div class="turn-text">' + text + '</div>' +
    meta;
  transcriptsEl.appendChild(div);
  div.scrollIntoView({ behavior: 'smooth' });
}

async function playAudio(base64Data) {
  return new Promise((resolve) => {
    const audio = new Audio('data:audio/mpeg;base64,' + base64Data);
    audio.onended = resolve;
    audio.onerror = resolve;
    audio.play().catch(resolve);
  });
}

function audioBufferToWav(buffer) {
  const numChannels = 1;
  const sampleRate = buffer.sampleRate;
  const bitsPerSample = 16;
  const data = buffer.getChannelData(0);
  const dataLength = data.length * (bitsPerSample / 8);
  const headerLength = 44;
  const totalLength = headerLength + dataLength;

  const arrayBuffer = new ArrayBuffer(totalLength);
  const view = new DataView(arrayBuffer);

  function writeString(v, offset, string) {
    for (let i = 0; i < string.length; i++) {
      v.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(view, 0, 'RIFF');
  view.setUint32(4, totalLength - 8, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * (bitsPerSample / 8), true);
  view.setUint16(32, numChannels * (bitsPerSample / 8), true);
  view.setUint16(34, bitsPerSample, true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataLength, true);

  let offset = 44;
  for (let i = 0; i < data.length; i++) {
    const sample = Math.max(-1, Math.min(1, data[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
    offset += 2;
  }

  return new Blob([arrayBuffer], { type: 'audio/wav' });
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn

    port = VOICE_CONFIG.voice_port
    logger.info(f"Starting voice server on port {port}")
    uvicorn.run(
        "voice_agent.voice_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
