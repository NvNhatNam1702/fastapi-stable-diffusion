# core/adapters/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseModelAdapter(ABC):
    """Abstract base class for all AI model adapters."""

    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        self.model = None
        self._load_model(**kwargs)

    @abstractmethod
    def _load_model(self, **kwargs):
        """Loads the model into memory (VRAM)."""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Runs the model inference."""
        pass
