"""Inference API for X-Ray to 3D Bone Reconstruction."""

# Lazy imports to avoid pulling in heavy model dependencies at package level.
# Use:  from src.inference.predictor import Predictor
#       from src.inference.model_registry import MODEL_REGISTRY

__all__ = ["Predictor", "MODEL_REGISTRY", "get_model_info"]


def __getattr__(name):
    if name == "Predictor":
        from .predictor import Predictor
        return Predictor
    if name == "MODEL_REGISTRY":
        from .model_registry import MODEL_REGISTRY
        return MODEL_REGISTRY
    if name == "get_model_info":
        from .model_registry import get_model_info
        return get_model_info
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
