from __future__ import annotations
"""
Core inference engine: load a model, preprocess inputs, run prediction, postprocess.
"""
import io
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.config import get_config
from models.swinx2s import SwinX2S
from models.transvert import TransVert
from inference.model_registry import get_model_info


class Predictor:
    """
    Loads a trained model and provides a `predict` method that accepts
    raw image bytes (AP + lateral X-rays) and returns a 3D binary volume.
    """

    def __init__(self, model_key: str, device: str | None = None):
        info = get_model_info(model_key)
        self.model_key = model_key
        self.config = get_config(info["config_name"])
        self.weights_path = info["weights"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Resolve 2D / 3D sizes (same logic as train.py)
        if self.config.MODEL_TYPE == "swinx2s":
            self.img_size_2d = tuple(self.config.IMG_SIZE[1:])  # (H, W)
            self.img_size_3d = tuple(self.config.IMG_SIZE)      # (D, H, W)
        else:
            self.img_size_2d = tuple(self.config.IMG_SIZE)      # (H, W)
            self.img_size_3d = (
                self.config.INTERMEDIATE_DEPTH * 4,
                self.config.IMG_SIZE[0],
                self.config.IMG_SIZE[1],
            )

        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Resize(self.img_size_2d, antialias=True),
        ])

        self.model = self._build_and_load()

    # ------------------------------------------------------------------ #
    # Model construction
    # ------------------------------------------------------------------ #

    def _build_and_load(self) -> torch.nn.Module:
        cfg = self.config
        if cfg.MODEL_TYPE == "swinx2s":
            model = SwinX2S(
                img_size=cfg.IMG_SIZE,
                in_channels=cfg.IN_CHANNELS,
                out_channels=cfg.OUT_CHANNELS,
                feature_size=cfg.FEATURE_SIZE,
                depths=cfg.DEPTHS,
                num_heads=cfg.NUM_HEADS,
                window_size=cfg.WINDOW_SIZE,
                norm_name=cfg.NORM_NAME,
                fusion_depths=cfg.FUSION_DEPTHS,
                spatial_dims=cfg.SPATIAL_DIMS,
            )
        elif cfg.MODEL_TYPE == "transvert":
            model = TransVert(
                in_channels=cfg.IN_CHANNELS,
                mid_channels=cfg.MID_CHANNELS,
                out_channels=cfg.OUT_CHANNELS,
                intermediate_depth=cfg.INTERMEDIATE_DEPTH,
            )
        else:
            raise ValueError(f"Unknown model type: {cfg.MODEL_TYPE}")

        # Load checkpoint
        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        # Support both raw state_dict and wrapped checkpoint
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Load image bytes → [1, 1, H, W] tensor on device."""
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        tensor = self.transform(img).unsqueeze(0)  # [1, 1, H, W]
        return tensor.to(self.device)

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict(
        self,
        ap_bytes: bytes,
        lat_bytes: bytes | None = None,
        threshold: float = 0.5,
    ) -> dict:
        """
        Run inference on a pair of X-ray images.

        Args:
            ap_bytes: Raw bytes of the AP X-ray image (PNG/JPEG/etc.).
            lat_bytes: Raw bytes of the lateral X-ray image. If None,
                       the AP image is duplicated (same behaviour as training).
            threshold: Binarisation threshold for the sigmoid output.

        Returns:
            dict with keys:
                volume_binary  – np.ndarray uint8 (D, H, W)
                volume_prob    – np.ndarray float32 (D, H, W) probability map
                shape          – tuple (D, H, W)
                model_key      – str
        """
        xa = self.preprocess_image(ap_bytes)
        xb = self.preprocess_image(lat_bytes) if lat_bytes is not None else xa.clone()

        logits = self.model(xa, xb)  # [1, 1, D, H, W]
        probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
        binary = (probs >= threshold).astype(np.uint8)

        return {
            "volume_binary": binary,
            "volume_prob": probs,
            "shape": binary.shape,
            "model_key": self.model_key,
        }

    @torch.no_grad()
    def predict_with_clinical(
        self,
        ap_bytes: bytes,
        lat_bytes: bytes | None = None,
        threshold: float = 0.5,
        voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> dict:
        """
        Run inference and compute clinical metrics for the predicted volume.

        Returns the same dict as `predict` plus a `clinical_metrics` key.
        """
        result = self.predict(ap_bytes, lat_bytes, threshold)
        mask = result["volume_binary"]
        spacing = np.array(voxel_spacing, dtype=np.float64)

        clinical = {}
        dataset = self.config.DATASET_NAME.lower()
        try:
            if "femur" in dataset:
                from utils.clinical.femur_metrics import compute_femur_metrics
                diameter, nsa, length = compute_femur_metrics(mask, spacing)
                clinical = {"femoral_head_diameter_mm": diameter, "neck_shaft_angle_deg": nsa, "femur_length_mm": length}
            elif "hip" in dataset or "pelv" in dataset:
                from utils.clinical.hip_metrics import compute_hip_metrics
                pw, ad, adepth = compute_hip_metrics(mask, spacing)
                clinical = {"pelvic_width_mm": pw, "acetabular_diameter_mm": ad, "acetabular_depth_mm": adepth}
            elif "verse" in dataset:
                from utils.clinical.verse_metrics import compute_verse_metrics
                h, w, d = compute_verse_metrics(mask, spacing)
                clinical = {"vertebra_height_mm": h, "vertebra_width_mm": w, "vertebra_depth_mm": d}
        except Exception as e:
            clinical = {"error": str(e)}

        result["clinical_metrics"] = clinical
        return result

    @torch.no_grad()
    def predict_mesh(
        self,
        ap_bytes: bytes,
        lat_bytes: bytes | None = None,
        threshold: float = 0.5,
        smooth_sigma: float = 1.0,
        step_size: int = 1,
    ) -> dict:
        """
        Run inference and convert the predicted volume to a triangle mesh
        using marching cubes (scikit-image).

        Returns dict with:
            vertices  – list of [x,y,z] floats
            faces     – list of [i,j,k] int indices
            normals   – list of [nx,ny,nz] floats
            shape     – volume shape (D,H,W)
            model_key – str
        """
        from scipy.ndimage import gaussian_filter
        from skimage.measure import marching_cubes

        result = self.predict(ap_bytes, lat_bytes, threshold)
        vol = result["volume_prob"]

        # Smooth the probability volume for a cleaner surface
        if smooth_sigma > 0:
            vol = gaussian_filter(vol.astype(np.float32), sigma=smooth_sigma)

        verts, faces, normals, _ = marching_cubes(
            vol, level=threshold, step_size=step_size,
        )

        # Centre the mesh around origin
        centre = np.array(result["shape"], dtype=np.float32) / 2.0
        verts = verts - centre

        return {
            "vertices": verts.tolist(),
            "faces": faces.tolist(),
            "normals": normals.tolist(),
            "shape": result["shape"],
            "model_key": self.model_key,
        }
