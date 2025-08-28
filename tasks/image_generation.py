import importlib
import os
import uuid
from core.celery_app import celery_app
from config.settings import settings

# A simple in-memory cache to hold loaded model instances
MODEL_CACHE = {}

def get_model_adapter(model_name: str):
    """Dynamically loads and caches a model adapter based on settings."""
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    model_config = settings.MODELS.get(model_name)
    if not model_config:
        raise ValueError(f"Model '{model_name}' not found in settings.")

    # Dynamically import the adapter class
    adapter_path = model_config["adapter"] # core.adapters.stable_diffusion.StableDiffusionAdapter
    module_path, class_name = adapter_path.rsplit('.', 1) #strip the last dot
    module = importlib.import_module(module_path)
    AdapterClass = getattr(module, class_name)

    # Instantiate the adapter (which loads the model)
    adapter = AdapterClass(model_id=model_config["model_id"])
    MODEL_CACHE[model_name] = adapter
    return adapter

@celery_app.task(name="tasks.generate_image")
def generate_image(model_name: str, prompt: str, **kwargs):
    """
    Celery task to generate an image using a specified model.
    """
    try:
        adapter = get_model_adapter(model_name)
        image = adapter.predict(prompt=prompt, **kwargs)

        # Save the image
        OUTPUT_FOLDER = "output_images"
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        image.save(filepath)

        return {"status": "SUCCESS", "filename": filename}
    except Exception as e:
        # Log the error properly in a real application
        print(f"Task failed: {e}")
        return {"status": "FAILURE", "error": str(e)}
