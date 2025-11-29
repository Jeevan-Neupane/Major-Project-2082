import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from torchvision import transforms as T

class SmallBoneDataset(Dataset):
    def __init__(
        self,
        root_dir,
        img_size_2d=(128, 128),
        img_size_3d=(64, 128, 128),
        transform_2d=None,
        require_lat=False,
        return_volume=True,
        duplicate_ap_if_missing_lat=True
    ):
        self.root_dir = root_dir
        self.img_size_2d = img_size_2d
        self.img_size_3d = img_size_3d
        self.return_volume = return_volume
        self.require_lat = require_lat
        self.duplicate_ap_if_missing_lat = duplicate_ap_if_missing_lat

        self.transform_2d = transform_2d or T.Compose([
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Resize(img_size_2d, antialias=True),
        ])

        self.samples = []
        for pdir in sorted(glob.glob(os.path.join(root_dir, "p_*"))):
            ap = glob.glob(os.path.join(pdir, "x-ray", "*ap*.png"))
            lat = glob.glob(os.path.join(pdir, "x-ray", "*lat*.png"))
            vol = glob.glob(os.path.join(pdir, "ct-scan", "*.nii.gz"))

            if not ap or not vol:
                continue
            if self.require_lat and not lat:
                continue

            self.samples.append({
                "pid": os.path.basename(pdir),
                "ap": ap[0],
                "lat": lat[0] if lat else None,
                "vol": vol[0],
            })

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found under {root_dir}")

    def __len__(self):
        return len(self.samples)

    def _safe_minmax(self, a, eps=1e-8):
        a = a.astype(np.float32)
        amin, amax = float(np.min(a)), float(np.max(a))
        if not np.isfinite(amin) or not np.isfinite(amax) or (amax - amin) < eps:
            return np.zeros_like(a, dtype=np.float32)
        return (a - amin) / (amax - amin + eps)

    def _load_volume_resized(self, path):
        vol_np = nib.load(path).get_fdata(dtype=np.float32)
        vol_np = self._safe_minmax(vol_np)
        vol = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0)
        D, H, W = self.img_size_3d
        vol = F.interpolate(vol, size=(D, H, W), mode="trilinear", align_corners=False)
        return vol.squeeze(0)

    def _load_xray(self, path):
        img = Image.open(path).convert("L")
        return self.transform_2d(img)

    def __getitem__(self, idx):
        item = self.samples[idx]
        pid = item["pid"]

        xa = self._load_xray(item["ap"])
        if item["lat"] is not None:
            xb = self._load_xray(item["lat"])
        else:
            if self.duplicate_ap_if_missing_lat:
                xb = xa.clone()
            else:
                xb = torch.zeros_like(xa)

        vol = self._load_volume_resized(item["vol"]) if self.return_volume else None

        return {
            "pid": pid,
            "xa": xa,
            "xb": xb,
            "vol": vol,
            "paths": item,
        }

def collate_small(batch):
    xa = torch.stack([b["xa"] for b in batch], dim=0)
    xb = torch.stack([b["xb"] for b in batch], dim=0)
    vol = None
    if batch[0]["vol"] is not None:
        vol = torch.stack([b["vol"] for b in batch], dim=0)
    pids = [b["pid"] for b in batch]
    paths = [b["paths"] for b in batch]
    return {"pid": pids, "xa": xa, "xb": xb, "vol": vol, "paths": paths}