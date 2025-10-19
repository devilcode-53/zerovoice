import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from elevenlabs.client import AsyncElevenLabs
from pydantic import BaseModel
import io

app = FastAPI()
logger = logging.getLogger("uvicorn")

# This is the request model. The main bot will send JSON in this format.
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
        # Initialize the client with the API key provided in the request
        eleven_client = AsyncElevenLabs(api_key=request.api_key)

        # Generate the audio stream
        audio_stream = await eleven_client.text_to_speech.convert(
            voice_id=request.voice_id,
            model_id="eleven_v3",
            text=request.text,
            output_format="mp3_44100_128",
        )

        # Assemble the audio chunks from the async stream
        audio_bytes_io = io.BytesIO()
        async for chunk in audio_stream:
            audio_bytes_io.write(chunk)
        
        audio_bytes_io.seek(0) # Rewind the in-memory file to the beginning
        
        if not audio_bytes_io.getbuffer().nbytes > 0:
            raise ValueError("Received empty audio stream from ElevenLabs.")

        logger.info("Successfully generated voice stream.")
        
        # Return the audio as a streaming response
        return StreamingResponse(audio_bytes_io, media_type="audio/mpeg")

    except Exception as e:
        logger.error(f"ElevenLabs API call failed: {e}")
        # If ElevenLabs fails (e.g., bad key, quota), send a specific error
        # The main bot will catch this and rotate the key.
        raise HTTPException(status_code=401, detail=f"ElevenLabs API Error: {str(e)}")
