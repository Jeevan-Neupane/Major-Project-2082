# Inference API — X-Ray to 3D Bone Reconstruction

FastAPI-based REST API for running inference with trained SwinX2S and TransVert models.

## Setup

```bash
# Install additional dependency (from project root)
pip install fastapi uvicorn python-multipart
```

## Quick Start

```bash
# Start the server (from project root)
python -m inference.run_server

# With model preloading
python -m inference.run_server --preload swinx2s_femur transvert_hip

# Expose to network
python -m inference.run_server --host 0.0.0.0 --port 8000
```

The Swagger UI is at **http://localhost:8000/docs** once the server is running.

## Available Models

| Model Key            | Architecture | Anatomy   | Size  |
| -------------------- | ------------ | --------- | ----- |
| `swinx2s_femur`      | SwinX2S      | Femur     | ~130M |
| `swinx2s_hip`        | SwinX2S      | Hip       | ~130M |
| `swinx2s_rib`        | SwinX2S      | Rib       | ~314M |
| `swinx2s_verse19`    | SwinX2S      | Verse19   | ~32M  |
| `swinx2s_verse20`    | SwinX2S      | Verse20   | ~32M  |
| `swinx2s_ctpelvic`   | SwinX2S      | CT-Pelvic | ~130M |
| `transvert_femur`    | TransVert    | Femur     | —     |
| `transvert_hip`      | TransVert    | Hip       | —     |
| `transvert_rib`      | TransVert    | Rib       | —     |
| `transvert_verse19`  | TransVert    | Verse19   | —     |
| `transvert_verse20`  | TransVert    | Verse20   | —     |
| `transvert_ctpelvic` | TransVert    | CT-Pelvic | —     |

## API Endpoints

### `GET /health`

Server health check. Shows loaded and available models.

### `GET /models`

List all registered models and whether checkpoint files exist on disk.

### `POST /predict/{model_key}`

Upload AP (and optionally lateral) X-ray images → get back a binary `.nii.gz` 3D volume.

```bash
# Binary volume (AP only — lateral is duplicated)
curl -X POST "http://localhost:8000/predict/swinx2s_femur" \
  -F "ap_image=@ap_xray.png" \
  -o output.nii.gz

# With lateral view
curl -X POST "http://localhost:8000/predict/swinx2s_femur" \
  -F "ap_image=@ap_xray.png" \
  -F "lat_image=@lat_xray.png" \
  -o output.nii.gz
```

### `POST /predict/{model_key}/prob`

Same as above but returns the probability map (float32) instead of binary volume.

### `POST /predict/{model_key}/clinical`

Run inference + compute anatomy-specific clinical metrics (e.g. femoral head diameter, neck-shaft angle).

```bash
curl -X POST "http://localhost:8000/predict/swinx2s_femur/clinical" \
  -F "ap_image=@ap_xray.png" \
  -F "lat_image=@lat_xray.png"
```

Returns JSON:

```json
{
  "model_key": "swinx2s_femur",
  "volume_shape": [128, 128, 160],
  "clinical_metrics": {
    "femoral_head_diameter_mm": 48.2,
    "neck_shaft_angle_deg": 131.5,
    "femur_length_mm": 420.3
  },
  "download_url": "/download/swinx2s_femur_clinical_abc123.nii.gz"
}
```

### `POST /load/{model_key}`

Pre-load a model into GPU/RAM for faster subsequent predictions.

### `POST /unload/{model_key}`

Unload a model from memory to free resources.

### `GET /download/{filename}`

Download a previously generated NIfTI volume.

## Python Client Example

```python
import requests

# Predict with AP only
with open("ap_xray.png", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/predict/transvert_femur",
        files={"ap_image": f},
    )
with open("output.nii.gz", "wb") as f:
    f.write(resp.content)

# Clinical metrics
with open("ap_xray.png", "rb") as ap, open("lat_xray.png", "rb") as lat:
    resp = requests.post(
        "http://localhost:8000/predict/swinx2s_femur/clinical",
        files={"ap_image": ap, "lat_image": lat},
    )
print(resp.json())
```

## File Structure

```
inference/
├── __init__.py          # Package exports
├── api.py               # FastAPI application and routes
├── model_registry.py    # Maps model keys → configs + weights paths
├── predictor.py         # Core inference engine (load, preprocess, predict)
├── schemas.py           # Pydantic request/response schemas
├── run_server.py        # CLI entry point to start the server
└── README.md            # This file
```
