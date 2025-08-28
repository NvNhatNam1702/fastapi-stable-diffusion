import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from .base import BaseModelAdapter

class StableDiffusionAdapter(BaseModelAdapter):
    """Adapter for the Stable Diffusion model."""

    def _load_model(self, **kwargs):
        print(f"--- Loading Stable Diffusion model: {self.model_id} ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Using device: {self.device} ---")
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to(self.device)
        print("--- Stable Diffusion model loaded successfully ---")

    def predict(self, prompt: str, **kwargs) -> Image.Image:
        """
        Generates an image based on the prompt and optional LoRA.
        """
        # LoRA handling can be added here if needed, similar to previous versions
        with torch.inference_mode():
            image = self.pipe(prompt).images[0]
        return image
