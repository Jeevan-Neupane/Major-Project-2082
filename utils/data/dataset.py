"""
Bone dataset for X-Ray to 3D reconstruction.

Loads paired X-ray images (AP + lateral) and 3D CT volumes from
a folder structure of the form:
    root_dir/
        p_0001/
            x-ray/  *.png (AP and lateral views)
            ct-scan/ *.nii.gz
        ...
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from torchvision import transforms as T


class PairedBiplanarAugmentation:
    """Geometry-consistent augmentation for paired X-rays and a 3D mask."""
    def __init__(
        self, flip_probability=0.5, intensity_probability=0.8,
        gamma_range=(0.85, 1.15), gain_range=(0.90, 1.10),
        bias_range=(-0.05, 0.05), noise_std=0.02,
    ):
        self.flip_probability = flip_probability
        self.intensity_probability = intensity_probability
        self.gamma_range = gamma_range
        self.gain_range = gain_range
        self.bias_range = bias_range
        self.noise_std = noise_std

    @staticmethod
    def _sample_uniform(bounds):
        return torch.empty(1).uniform_(*bounds).item()

    def _intensity(self, image):
        if torch.rand(1).item() >= self.intensity_probability:
            return image
        image = image.clamp(0, 1).pow(self._sample_uniform(self.gamma_range))
        image = image * self._sample_uniform(self.gain_range) + self._sample_uniform(self.bias_range)
        if self.noise_std > 0:
            image = image + torch.randn_like(image) * self.noise_std
        return image.clamp(0, 1)

    def __call__(self, ap, lat, volume):
        # Volume is [C,D,H,W]. AP maps (H,W); LAT maps (H,D).
        if torch.rand(1).item() < self.flip_probability:
            ap, lat, volume = ap.flip(-2), lat.flip(-2), volume.flip(-2)
        if torch.rand(1).item() < self.flip_probability:
            ap, volume = ap.flip(-1), volume.flip(-1)
        if torch.rand(1).item() < self.flip_probability:
            lat, volume = lat.flip(-1), volume.flip(-3)
        return self._intensity(ap), self._intensity(lat), volume


class BoneDataset(Dataset):
    """Dataset for paired X-ray → 3D bone volume reconstruction."""

    def __init__(
        self,
        root_dir,
        img_size_2d=(128, 128),
        img_size_3d=(64, 128, 128),
        transform_2d=None,
        require_lat=False,
        return_volume=True,
        duplicate_ap_if_missing_lat=True,
        exclude_pids=None,
        include_pids=None,
        augment=False,
        flip_ap_lr=False,
    ):
        """flip_ap_lr: mirror the AP image left-right after loading. The AP PNGs follow the
        radiographic display convention (patient's right on the image left), the opposite of the
        volume's W axis; True makes AP image column w correspond to volume column w. Default False
        reproduces the November 2026 models."""
        self.flip_ap_lr = flip_ap_lr
        self.root_dir = root_dir
        self.img_size_2d = img_size_2d
        self.img_size_3d = img_size_3d
        self.return_volume = return_volume
        self.require_lat = require_lat
        self.duplicate_ap_if_missing_lat = duplicate_ap_if_missing_lat
        self.exclude_pids = set(exclude_pids or [])
        self.include_pids = None if include_pids is None else set(include_pids)
        self.augment = PairedBiplanarAugmentation() if augment is True else augment

        self.transform_2d = transform_2d or T.Compose(
            [
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
                T.Resize(img_size_2d, antialias=True),
            ]
        )

        self.samples = []
        found_pids = set()

        # Check for train/val/test splits or direct p_* folders
        splits = ["train", "val", "test"]
        search_dirs = []

        for split in splits:
            split_path = os.path.join(root_dir, split)
            if os.path.exists(split_path):
                search_dirs.append(split_path)

        if not search_dirs:
            search_dirs = [root_dir]

        for search_dir in search_dirs:
            for pdir in sorted(glob.glob(os.path.join(search_dir, "p_*"))):
                pid = os.path.basename(pdir)
                if pid in self.exclude_pids:
                    continue
                if self.include_pids is not None and pid not in self.include_pids:
                    continue
                ap = glob.glob(os.path.join(pdir, "x-ray", "*ap*.png"))
                lat = glob.glob(os.path.join(pdir, "x-ray", "*lat*.png"))
                vol = glob.glob(os.path.join(pdir, "ct-scan", "*.nii.gz"))

                if not ap or not vol:
                    continue
                if self.require_lat and not lat:
                    continue

                self.samples.append(
                    {
                        "pid": pid,
                        "ap": ap[0],
                        "lat": lat[0] if lat else None,
                        "vol": vol[0],
                    }
                )
                found_pids.add(pid)

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found under {root_dir}")
        if self.include_pids is not None:
            missing = self.include_pids - found_pids
            if missing:
                preview = ", ".join(sorted(missing)[:5])
                raise RuntimeError(
                    f"{len(missing)} manifest PID(s) are missing or invalid under {root_dir}: {preview}"
                )

    def __len__(self):
        return len(self.samples)

    @property
    def pids(self):
        """Patient identifiers represented by this dataset."""
        return {sample["pid"] for sample in self.samples}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe_minmax(self, a, eps=1e-8):
        a = a.astype(np.float32)
        amin, amax = float(np.min(a)), float(np.max(a))
        if not np.isfinite(amin) or not np.isfinite(amax) or (amax - amin) < eps:
            return np.zeros_like(a, dtype=np.float32)
        return (a - amin) / (amax - amin + eps)

    def _load_volume_resized(self, path):
        nii = nib.load(path)
        vol_np = nii.get_fdata(dtype=np.float32)
        # Extract voxel spacing from NIfTI header (mm)
        voxel_spacing = np.array(nii.header.get_zooms()[:3], dtype=np.float32)
        # Scale spacing to account for resampling
        original_shape = np.array(vol_np.shape[:3], dtype=np.float32)
        target_shape = np.array(self.img_size_3d, dtype=np.float32)
        resampled_spacing = voxel_spacing * (original_shape / target_shape)

        vol_np = self._safe_minmax(vol_np)
        vol = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0)
        D, H, W = self.img_size_3d
        vol = F.interpolate(vol, size=(D, H, W), mode="trilinear", align_corners=False)
        return vol.squeeze(0), resampled_spacing

    def _load_xray(self, path):
        img = Image.open(path).convert("L")
        return self.transform_2d(img)

    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        item = self.samples[idx]
        pid = item["pid"]

        xa = self._load_xray(item["ap"])
        if self.flip_ap_lr:
            xa = xa.flip(-1)
        if item["lat"] is not None:
            xb = self._load_xray(item["lat"])
        else:
            xb = (
                xa.clone() if self.duplicate_ap_if_missing_lat else torch.zeros_like(xa)
            )

        voxel_spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        if self.return_volume:
            vol, voxel_spacing = self._load_volume_resized(item["vol"])
        else:
            vol = None

        if self.augment and vol is not None:
            xa, xb, vol = self.augment(xa, xb, vol)

        return {
            "pid": pid,
            "xa": xa,
            "xb": xb,
            "vol": vol,
            "voxel_spacing": voxel_spacing,
            "paths": item,
        }


def collate_bone(batch):
    """Collates a batch of bone data, keeping voxel spacing as a list."""
    xa = torch.stack([b["xa"] for b in batch], dim=0)
    xb = torch.stack([b["xb"] for b in batch], dim=0)
    vol = None
    if batch[0]["vol"] is not None:
        vol = torch.stack([b["vol"] for b in batch], dim=0)
    pids = [b["pid"] for b in batch]
    paths = [b["paths"] for b in batch]
    voxel_spacings = [b["voxel_spacing"] for b in batch]
    return {
        "pid": pids,
        "xa": xa,
        "xb": xb,
        "vol": vol,
        "voxel_spacing": voxel_spacings,
        "paths": paths,
    }
