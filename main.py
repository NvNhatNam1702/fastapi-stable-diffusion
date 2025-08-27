# main.py

import io
import base64
import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional # Import Optional for optional fields

from generator import StableDiffusionGenerator

# --- 1. Initialize FastAPI App ---
app = FastAPI()

# --- 2. Define Request and Response Models ---
class ImageRequest(BaseModel):
    prompt: str
    lora_name: Optional[str] = None  # Make lora_name optional
    lora_scale: float = 0.8         # Provide a default scale

class ImageResponse(BaseModel):
    image_base64: str
    prompt: str
    filename: str
    lora_used: Optional[str] = None # Add lora_used to the response

# --- 3. Create a single, global instance of our generator ---
generator = StableDiffusionGenerator()

# --- 4. Define the output folder ---
OUTPUT_FOLDER = "output_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --- 5. Define the API Endpoint ---
@app.post("/generate-image", response_model=ImageResponse)
async def generate_image(request: ImageRequest):
    try:
        # Call the generator with all parameters from the request
        image = generator(
            prompt=request.prompt,
            lora_name=request.lora_name,
            lora_scale=request.lora_scale
        )

        # --- Save the image to a folder ---
        unique_id = uuid.uuid4()
        filename = f"{unique_id}.png"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        image.save(filepath)
        print(f"✅ Image saved to: {filepath}")

        # --- Convert to Base64 for the response ---
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        base64_image = base64.b64encode(img_bytes).decode("utf-8")

        return ImageResponse(
            image_base64=base64_image,
            prompt=request.prompt,
            filename=filename,
            lora_used=request.lora_name
        )
    
    except Exception as e:
        # Provide more specific error details in the response
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/")
def read_root():
    return {"status": "Stable Diffusion API is running."}
