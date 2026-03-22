from __future__ import annotations
"""
Pydantic schemas for API request / response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional


class ModelInfo(BaseModel):
    model_key: str
    description: str
    config_name: str
    weights_exist: bool


class PredictionResponse(BaseModel):
    model_key: str
    volume_shape: list[int]
    clinical_metrics: Optional[dict] = None
    download_url: str = Field(description="URL to download the .nii.gz output volume")


class HealthResponse(BaseModel):
    status: str
    loaded_models: list[str]
    available_models: list[str]
