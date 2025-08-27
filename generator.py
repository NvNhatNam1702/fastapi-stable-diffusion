# generator.py

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

LORA_FOLDER = "loras"

class StableDiffusionGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        print("--- Loading Stable Diffusion model ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Using device: {self.device} ---")

        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            self.pipe = self.pipe.to(self.device)
            print("--- Stable Diffusion model loaded successfully ---")
        except Exception as e:
            print(f"!!! Error loading the model: {e} !!!")
            self.pipe = None

        # --- NEW: Discover available LoRAs ---
        self.available_loras = self._discover_loras()
        if self.available_loras:
            print(f"--- Discovered LoRAs: {list(self.available_loras.keys())} ---")
        else:
            print("--- No LoRAs found in the 'loras' folder. ---")

    def _discover_loras(self):
        """Scans the LORA_FOLDER for .safetensors files."""
        loras = {}
        if os.path.exists(LORA_FOLDER):
            for filename in os.listdir(LORA_FOLDER):
                if filename.endswith(".safetensors"):
                    lora_name = os.path.splitext(filename)[0]
                    loras[lora_name] = os.path.join(LORA_FOLDER, filename)
        return loras

    def __call__(self, prompt: str, lora_name: str = None, lora_scale: float = 0.8) -> Image.Image:
        """
        Generates an image, optionally applying a LoRA.
        
        Args:
            prompt (str): The text prompt.
            lora_name (str, optional): The name of the LoRA to apply (filename without extension).
            lora_scale (float, optional): The weight/influence of the LoRA.
            
        Returns:
            Image.Image: The generated PIL Image.
        """
        if self.pipe is None:
            raise RuntimeError("Model is not loaded. Cannot generate image.")

        # --- NEW: LoRA Handling Logic ---
        lora_applied = False
        if lora_name:
            if lora_name in self.available_loras:
                lora_path = self.available_loras[lora_name]
                print(f"--- Applying LoRA: {lora_name} from {lora_path} ---")
                self.pipe.load_lora_weights(lora_path)
                lora_applied = True
            else:
                raise ValueError(f"LoRA '{lora_name}' not found. Available LoRAs: {list(self.available_loras.keys())}")
        
        # Define kwargs for the pipeline call. The LoRA scale is passed here.
        kwargs = {"cross_attention_kwargs": {"scale": lora_scale}} if lora_applied else {}

        with torch.inference_mode():
            generated_image = self.pipe(prompt, **kwargs).images[0]
        
        # --- Unload LoRA weights after generation ---
        # This ensures the base model is clean for the next request.
        if lora_applied:
            print(f"--- Unloading LoRA: {lora_name} ---")
            self.pipe.unload_lora_weights()

        return generated_image

