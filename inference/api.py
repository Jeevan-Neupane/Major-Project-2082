from __future__ import annotations
"""
FastAPI inference server for X-Ray to 3D Bone Reconstruction.

Endpoints:
    GET  /health              – server health + loaded models
    GET  /models              – list all available models
    POST /predict/{model_key} – run inference (returns downloadable .nii.gz)
    POST /predict/{model_key}/clinical – inference + clinical metrics
"""
import io
import os
import sys
import uuid
import shutil
import tempfile
from contextlib import asynccontextmanager

import numpy as np
import nibabel as nib
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from inference.predictor import Predictor
from inference.model_registry import list_available_models, MODEL_REGISTRY
from inference.schemas import ModelInfo, PredictionResponse, HealthResponse

# ------------------------------------------------------------------ #
# Globals
# ------------------------------------------------------------------ #

_loaded_predictors: dict[str, Predictor] = {}
_output_dir: str = ""


# ------------------------------------------------------------------ #
# Lifespan: create/clean temp dir
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _output_dir
    _output_dir = tempfile.mkdtemp(prefix="xray3d_outputs_")
    yield
    shutil.rmtree(_output_dir, ignore_errors=True)


app = FastAPI(
    title="X-Ray to 3D Bone Reconstruction API",
    version="1.0.0",
    description=(
        "Upload AP and (optionally) lateral X-ray images to reconstruct "
        "a 3D bone volume using SwinX2S or TransVert models."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:3001", "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _get_predictor(model_key: str) -> Predictor:
    """Lazily load the predictor for *model_key*."""
    if model_key not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_key}")
    if model_key not in _loaded_predictors:
        try:
            _loaded_predictors[model_key] = Predictor(model_key)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    return _loaded_predictors[model_key]


def _save_nifti(volume: np.ndarray, spacing: tuple, filename: str) -> str:
    """Save a 3D numpy volume as a NIfTI file, return the path."""
    affine = np.diag([*spacing, 1.0]).astype(np.float64)
    nii = nib.Nifti1Image(volume.astype(np.float32), affine)
    path = os.path.join(_output_dir, filename)
    nib.save(nii, path)
    return path


async def _read_upload(upload: UploadFile) -> bytes:
    """Read an UploadFile into bytes."""
    return await upload.read()


# ------------------------------------------------------------------ #
# Routes
# ------------------------------------------------------------------ #

@app.get("/health", response_model=HealthResponse)
async def health():
    """Server health check."""
    return HealthResponse(
        status="ok",
        loaded_models=list(_loaded_predictors.keys()),
        available_models=[m["model_key"] for m in list_available_models() if m["weights_exist"]],
    )


@app.get("/models", response_model=list[ModelInfo])
async def models():
    """List all registered models and whether weights exist on disk."""
    return [ModelInfo(**m) for m in list_available_models()]


@app.post("/predict/{model_key}")
async def predict(
    model_key: str,
    ap_image: UploadFile = File(..., description="AP (frontal) X-ray image"),
    lat_image: UploadFile | None = File(None, description="Lateral X-ray image (optional)"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Binarisation threshold"),
):
    """
    Run inference and return the reconstructed 3D volume as a downloadable .nii.gz file.
    """
    predictor = _get_predictor(model_key)

    ap_bytes = await _read_upload(ap_image)
    lat_bytes = await _read_upload(lat_image) if lat_image is not None else None

    result = predictor.predict(ap_bytes, lat_bytes, threshold=threshold)

    # Save as NIfTI
    uid = uuid.uuid4().hex[:12]
    fname = f"{model_key}_{uid}.nii.gz"
    path = _save_nifti(result["volume_binary"], (1.0, 1.0, 1.0), fname)

    return FileResponse(
        path,
        media_type="application/gzip",
        filename=fname,
    )


@app.post("/predict/{model_key}/prob")
async def predict_probability(
    model_key: str,
    ap_image: UploadFile = File(..., description="AP (frontal) X-ray image"),
    lat_image: UploadFile | None = File(None, description="Lateral X-ray image (optional)"),
):
    """
    Run inference and return the probability volume (float32) as .nii.gz.
    """
    predictor = _get_predictor(model_key)

    ap_bytes = await _read_upload(ap_image)
    lat_bytes = await _read_upload(lat_image) if lat_image is not None else None

    result = predictor.predict(ap_bytes, lat_bytes, threshold=0.5)

    uid = uuid.uuid4().hex[:12]
    fname = f"{model_key}_prob_{uid}.nii.gz"
    path = _save_nifti(result["volume_prob"], (1.0, 1.0, 1.0), fname)

    return FileResponse(
        path,
        media_type="application/gzip",
        filename=fname,
    )


@app.post("/predict/{model_key}/clinical", response_model=PredictionResponse)
async def predict_clinical(
    model_key: str,
    ap_image: UploadFile = File(..., description="AP (frontal) X-ray image"),
    lat_image: UploadFile | None = File(None, description="Lateral X-ray image (optional)"),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    spacing_z: float = Query(1.0, description="Voxel spacing Z (mm)"),
    spacing_y: float = Query(1.0, description="Voxel spacing Y (mm)"),
    spacing_x: float = Query(1.0, description="Voxel spacing X (mm)"),
):
    """
    Run inference, compute anatomy-specific clinical metrics,
    and return the volume as a downloadable .nii.gz alongside the metrics.
    """
    predictor = _get_predictor(model_key)

    ap_bytes = await _read_upload(ap_image)
    lat_bytes = await _read_upload(lat_image) if lat_image is not None else None

    result = predictor.predict_with_clinical(
        ap_bytes, lat_bytes,
        threshold=threshold,
        voxel_spacing=(spacing_z, spacing_y, spacing_x),
    )

    uid = uuid.uuid4().hex[:12]
    fname = f"{model_key}_clinical_{uid}.nii.gz"
    spacing = (spacing_z, spacing_y, spacing_x)
    _save_nifti(result["volume_binary"], spacing, fname)

    return PredictionResponse(
        model_key=model_key,
        volume_shape=list(result["shape"]),
        clinical_metrics=result.get("clinical_metrics"),
        download_url=f"/download/{fname}",
    )


@app.post("/predict/{model_key}/mesh")
async def predict_mesh(
    model_key: str,
    ap_image: UploadFile = File(..., description="AP (frontal) X-ray image"),
    lat_image: UploadFile | None = File(None, description="Lateral X-ray image (optional)"),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    smooth: float = Query(1.0, ge=0.0, le=3.0, description="Gaussian smoothing sigma"),
):
    """
    Run inference and return a triangle mesh (vertices + faces + normals)
    extracted via marching cubes.  Ready for 3D rendering in the browser.
    """
    predictor = _get_predictor(model_key)

    ap_bytes = await _read_upload(ap_image)
    lat_bytes = await _read_upload(lat_image) if lat_image is not None else None

    mesh = predictor.predict_mesh(
        ap_bytes, lat_bytes,
        threshold=threshold,
        smooth_sigma=smooth,
    )

    return JSONResponse(content={
        "model_key": model_key,
        "volume_shape": list(mesh["shape"]),
        "vertex_count": len(mesh["vertices"]),
        "face_count": len(mesh["faces"]),
        "vertices": mesh["vertices"],
        "faces": mesh["faces"],
        "normals": mesh["normals"],
    })


@app.get("/download/{filename}")
async def download(filename: str):
    """Download a previously generated NIfTI volume."""
    # Sanitise filename to prevent directory traversal
    safe_name = os.path.basename(filename)
    path = os.path.join(_output_dir, safe_name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found or expired.")
    return FileResponse(
        path,
        media_type="application/gzip",
        filename=safe_name,
    )


@app.post("/load/{model_key}")
async def load_model(model_key: str):
    """Pre-load a model into memory so subsequent predictions are fast."""
    _get_predictor(model_key)
    return {"status": "loaded", "model_key": model_key}


@app.post("/unload/{model_key}")
async def unload_model(model_key: str):
    """Unload a model from memory to free GPU/RAM."""
    if model_key in _loaded_predictors:
        del _loaded_predictors[model_key]
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "unloaded", "model_key": model_key}
    raise HTTPException(status_code=404, detail=f"Model '{model_key}' is not loaded.")
