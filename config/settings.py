from pydantic_settings import BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"

    # Celery Settings
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"
    
    #S3 settings 
    # AWS_ACCESS_KEY_ID: str
    # AWS_SECRET_ACCESS_KEY: str
    # AWS_REGION: str
    # S3_BUCKET_NAME: str
    S3_ENDPOINT_URL: str
    S3_ACCESS_KEY_ID: str
    S3_SECRET_ACCESS_KEY: str
    S3_REGION: str
    S3_BUCKET_NAME: str
    # Model Configurations
    # This dictionary will hold settings for all our models
    MODELS: Dict[str, Dict[str, Any]] = {
        "stable-diffusion-v1-5": {
            "model_id": "runwayml/stable-diffusion-v1-5",
            "adapter": "core.adapters.stable_diffusion.StableDiffusionAdapter",
            "default_lora_scale": 0.8
        },
        "hunyuan3d-v1": {
            "model_id": "Tencent-Hunyuan/Hunyuan3D", 
            "adapter": "core.adapters.hunyuan3d.Hunyuan3DAdapter",
        }
    }

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
