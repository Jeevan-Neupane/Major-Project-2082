"""
Model registry: maps each saved checkpoint to its config and weights path.
"""
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SAVED = os.path.join(_PROJECT_ROOT, "saved_models")


# Each entry: {model_key: {config_name, weights_path, description}}
MODEL_REGISTRY = {
    # ── SwinX2S models ──
    "swinx2s_femur": {
        "config_name": "swinx2s_femur_small",
        "weights": os.path.join(_SAVED, "femur", "swinx2s", "saved_models_swinx2s_small_femur_femur_swin_best.pth"),
        "description": "SwinX2S-Small for Femur (~130M params)",
    },
    "swinx2s_hip": {
        "config_name": "swinx2s_hip",
        "weights": os.path.join(_SAVED, "hip", "swinx2s", "saved_models_swinx2s_small_hip_hip_swin_best.pth"),
        "description": "SwinX2S-Small for Hip (~130M params)",
    },
    "swinx2s_rib": {
        "config_name": "swinx2s_rib",
        "weights": os.path.join(_SAVED, "rib", "swinx2s", "saved_models_swinx2s_base_rib_rib_swinbase_best.pth"),
        "description": "SwinX2S-Base for Rib (~314M params)",
    },
    "swinx2s_verse19": {
        "config_name": "swinx2s_verse19_small",
        "weights": os.path.join(_SAVED, "verse19", "swin", "saved_models_swinx2s_small_verse19_best_model_weights.pth"),
        "description": "SwinX2S-Small for Verse19 (~130M params)",
    },
    "swinx2s_verse20": {
        "config_name": "swinx2s_verse20",
        "weights": os.path.join(_SAVED, "verse20", "swin", "saved_models_swinx2s_tiny_verse20_verse20_swin_best.pth"),
        "description": "SwinX2S-Tiny for Verse20 (~32M params)",
    },
    "swinx2s_ctpelvic": {
        "config_name": "swinx2s_ctpelvic",
        "weights": os.path.join(_SAVED, "ctpelvik", "swinx2s", "saved_models_swinx2s_small_ctpelvic_ctpelvik_swinsmall_best.pth"),
        "description": "SwinX2S-Small for CT-Pelvic (~130M params)",
    },

    # ── TransVert models ──
    "transvert_femur": {
        "config_name": "transvert_femur",
        "weights": os.path.join(_SAVED, "femur", "transvert", "femur_transvert_best.pth"),
        "description": "TransVert for Femur",
    },
    "transvert_hip": {
        "config_name": "transvert_hip",
        "weights": os.path.join(_SAVED, "hip", "transvert", "hip_transvert_best.pth"),
        "description": "TransVert for Hip",
    },
    "transvert_rib": {
        "config_name": "transvert_rib",
        "weights": os.path.join(_SAVED, "rib", "transvert", "rib_transvert_best.pth"),
        "description": "TransVert for Rib",
    },
    "transvert_verse19": {
        "config_name": "transvert_verse19",
        "weights": os.path.join(_SAVED, "verse19", "transvert", "transvert_verse19.pth"),
        "description": "TransVert for Verse19",
    },
    "transvert_verse20": {
        "config_name": "transvert_verse20",
        "weights": os.path.join(_SAVED, "verse20", "transvert", "verse20_transvert_best.pth"),
        "description": "TransVert for Verse20",
    },
    "transvert_ctpelvic": {
        "config_name": "transvert_ctpelvic",
        "weights": os.path.join(_SAVED, "ctpelvik", "transvert", "ctpelvic_transvert_best.pth"),
        "description": "TransVert for CT-Pelvic",
    },
}


def get_model_info(model_key: str) -> dict:
    """Return registry entry for *model_key*, or raise ValueError."""
    if model_key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_key}'. Available: {available}")
    return MODEL_REGISTRY[model_key]


def list_available_models() -> list[dict]:
    """Return a list of dicts with key, description, and weights_exist."""
    out = []
    for key, info in MODEL_REGISTRY.items():
        out.append({
            "model_key": key,
            "description": info["description"],
            "config_name": info["config_name"],
            "weights_exist": os.path.isfile(info["weights"]),
        })
    return out
