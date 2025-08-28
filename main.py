# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from core.celery_app import celery_app
from celery.result import AsyncResult
from config.settings import settings

app = FastAPI(title="Stable Diffusion API")

class TaskRequest(BaseModel):
    model_name: str = "stable-diffusion-v1-5"
    prompt: str

class TaskResponse(BaseModel):
    task_id: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None

@app.post(f"{settings.API_V1_STR}/generate", response_model=TaskResponse)
async def start_generation(request: TaskRequest):
    """Create an image generation task."""
    task = celery_app.send_task(
        "tasks.generate_image",
        args=[request.model_name, request.prompt]
    )
    return TaskResponse(task_id=task.id)

@app.get(f"{settings.API_V1_STR}/tasks/{{task_id}}", response_model=TaskStatusResponse)
async def get_status(task_id: str):
    """Get the status of a task."""
    task_result = AsyncResult(task_id, app=celery_app)
    return TaskStatusResponse(
        task_id=task_id,
        status=task_result.status,
        result=task_result.result
    )
