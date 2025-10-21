import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from elevenlabs.client import AsyncElevenLabs
from pydantic import BaseModel
import io

app = FastAPI()
logger = logging.getLogger("uvicorn")

class VoiceRequest(BaseModel):
    text: str
    voice_id: str
    api_key: str

@app.get("/")
async def root():
    """A simple root endpoint to check if the API is online."""
    return {"status": "Zero Voice API is online."}

@app.post("/generate_voice")
async def generate_voice(request: VoiceRequest):
    """
    Receives text, voice_id, and api_key, then returns the generated
    audio file from ElevenLabs.
    """
    logger.info(f"Received voice generation request for voice_id: {request.voice_id}")
    
    try:
        eleven_client = AsyncElevenLabs(api_key=request.api_key)

        audio_stream = eleven_client.text_to_speech.convert(
            voice_id=request.voice_id,
            model_id="eleven_v3",
            text=request.text,
            # --- THIS IS THE FIX ---
            # Using opus_48000_32, a more standard format.
            output_format="opus_48000_32", 
        )

        audio_bytes_io = io.BytesIO()
        async for chunk in audio_stream:
            audio_bytes_io.write(chunk)
        
        audio_bytes_io.seek(0)
        
        if not audio_bytes_io.getbuffer().nbytes > 0:
            raise ValueError("Received empty audio stream from ElevenLabs.")

        logger.info("Successfully generated voice stream.")
        
        # media_type="audio/ogg" is correct for Opus
        return StreamingResponse(audio_bytes_io, media_type="audio/ogg")

    except Exception as e:
        logger.error(f"ElevenLabs API call failed: {e}")
        raise HTTPException(status_code=401, detail=f"ElevenLabs API Error: {str(e)}")
