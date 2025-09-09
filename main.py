from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from typing import Optional
from core.celery_app import celery_app
from celery.result import AsyncResult
from config.settings import settings

app = FastAPI(title="Generative AI API")

# --- Pydantic Models for API Requests and Responses ---

class TextToImageRequest(BaseModel):
    model_name: str = "stable-diffusion-v1-5"
    prompt: str

class S3ImageRequest(BaseModel):
    s3_image_url: HttpUrl # Pydantic validates that this is a valid URL

class TaskResponse(BaseModel):
    task_id: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None


# --- API Endpoints ---

@app.post(f"{settings.API_V1_STR}/generate-image", response_model=TaskResponse)
async def start_text_to_image_generation(request: TextToImageRequest):
    """Create a text-to-image generation task."""
    task = celery_app.send_task(
        "tasks.generate_image",
        args=[request.model_name, request.prompt]
    )
    return TaskResponse(task_id=task.id)


@app.post(f"{settings.API_V1_STR}/generate-3d-from-s3", response_model=TaskResponse)
async def start_s3_image_to_3d_generation(request: S3ImageRequest):
    """Accepts an S3 URL and creates a task to generate a 3D model."""
    task = celery_app.send_task(
        "tasks.generate_3d_model_from_s3_image",
        args=[str(request.s3_image_url)] # Pass the URL string to the task
    )
    return TaskResponse(task_id=task.id)


@app.get(f"{settings.API_V1_STR}/tasks/{{task_id}}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of any task by its ID."""
    task_result = AsyncResult(task_id, app=celery_app)
    return TaskStatusResponse(
        task_id=task_id,
        status=task_result.status,
        result=task_result.result
    )

