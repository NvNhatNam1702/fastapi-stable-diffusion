from celery import Celery
from config.settings import settings

celery_app = Celery(
    "tasks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["tasks.image_generation"] # Tells Celery where to find tasks
)
