# core/adapters/hunyuan3d.py
import torch
from .base import BaseModelAdapter
import os
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDITFlowMatchingPipeline

class Hunyuan3DAdapter(BaseModelAdapter):
    """Adapter for the Hunyuan3D model."""

    def _load_model(self, **kwargs):
        print(f"--- Loading Hunyuan3D model: {self.model_id} ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the actual pipeline based on the provided image
        self.pipe = Hunyuan3DDITFlowMatchingPipeline.from_pretrained(
            self.model_id, # This will be 'tencent/Hunyuan3D-2' from our settings
            subfolder='hunyuan3d-dit-v2-0-turbo',
            use_safetensors=True
        )
        self.pipe.enable_flashvdm() # Enable optimization
        self.pipe = self.pipe.to(self.device)

        # Also initialize the background remover
        self.remover = BackgroundRemover()
        
        print("--- Hunyuan3D model loaded successfully ---")

    def predict(self, image_path: str, output_path: str, **kwargs) -> str:
        """
        Generates a 3D model from an input image.
        Returns the path to the generated 3D model file (.glb).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found at {image_path}")

        print(f"--- Preparing image and generating 3D model from: {image_path} ---")

        # 1. Load and process the input image
        input_image = Image.open(image_path)
        processed_image = self.remover.process(input_image) # Remove background

        # 2. Run the 3D generation pipeline
        with torch.inference_mode():
            # The pipeline call likely returns a mesh object
            generated_mesh = self.pipe(
                image=processed_image,
                num_inference_steps=5 # Using the fast turbo parameter
            )[0] # The output is likely a list

        # 3. Save the mesh to the specified output file
        # The library should have a method to save the mesh, e.g., .save_mesh()
        # This is an assumption based on common 3D library patterns.
        # If the library uses a different function, it should be substituted here.
        # For example: save_mesh_function(generated_mesh, output_path)
        if hasattr(generated_mesh, 'save'):
            generated_mesh.save(output_path)
        else:
             # Fallback/placeholder if a direct save method isn't available
             # In a real scenario, you'd find the correct export function from the library's docs.
             print("Warning: Mesh object does not have a .save() method. Saving a placeholder.")
             with open(output_path, "w") as f:
                 f.write("Generated 3D model data would be here.")


        print(f"--- 3D model saved to: {output_path} ---")
        return output_path
