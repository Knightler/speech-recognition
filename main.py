from typing import AsyncIterator # -> Annotates objects you can loop over with async for
import asyncio
from assemblyai_stt import AssemblyAISTT # -> STT (AssemblyAI)
from events import VoiceAgentEvent # -> Represents an event coming from a voice agent

async def stt_stream(
        audio_stream: AsyncIterator[bytes]) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Audio (Bytes) -> Voice Events (VoiceAgentEvent)

    Uses a producer-consumer pattern where:
    - Producer: Reads audio chunk and sends them to AssemblyAI
    - Consumer: Reveives transcription events from AssemblyAI
    """
    stt = AssemblyAISTT(sample_rate=16000)

    async def send_audio():
        """Background task that pumps audio chunks to AssemblyAI."""
        try:
            async for audio_chunk in audio_stream:
                await stt.send_audio(audio_chunk)
            finally:
                # Signal completion when audio stream ends
                await stt.close()

    # Launch audio sending in background
    send_task = asyncio.create_task(send_audio())

    try:
        # Reveive and yield transcription events as they arrive
        async for event in stt.receive_events():
            yield event
    finally:
        # Cleanup
        with contextlib.suppress(asyncio.CancelledError):
            send_task.cancel()
            await send_task
        await stt.close()
