"""Do the Modal-trained fair SwinX2S models reproduce their logged best validation Dice when evaluated
locally with scripts/evaluate_checkpoints.py::score (CPU)? Validation data only; the test split is not touched."""
import glob, os, sys, json
ROOT = "/home/ubuntu/Major-Project-2082"
sys.path.insert(0, ROOT)
import torch
from configs.config import CONFIG_REGISTRY as R
from train import create_model, load_split_manifest
from scripts.evaluate_checkpoints import sizes, score
from utils.data.dataset import BoneDataset

RUNS = {"verse20": "fair_swinx2s_verse20_20261116_044047", "femur": "fair_swinx2s_femur_20261116_043849"}

if __name__ == "__main__":
    os.chdir(ROOT)
    torch.set_num_threads(8)
    for ds in sys.argv[1:] or RUNS:
        run = RUNS[ds]
        cfg = R[f"fair_swinx2s_{ds}"]
        d = glob.glob(f"modal_results/results/saved_models/swinx2s/tiny/{ds}/{run}")[0]
        best = glob.glob(f"{d}/best_epoch*.pth")[0]
        meta = torch.load(best, map_location="cpu", weights_only=False)
        model = create_model(cfg, torch.device("cpu"))
        model.load_state_dict(torch.load(f"{d}/best_model_weights.pth", map_location="cpu"), strict=True)
        model.eval()
        _, sp = load_split_manifest(cfg)
        s2, s3 = sizes(cfg)
        data = BoneDataset(root_dir=os.path.join(cfg.DATA_ROOT, "train"), img_size_2d=s2, img_size_3d=s3,
                           return_volume=True, include_pids=sp["val"], flip_ap_lr=cfg.FLIP_AP_LR)
        cases = score(model, data, cfg, torch.device("cpu"))
        mean = sum(c["dice"] for c in cases) / len(cases)
        print(f"{ds:8s} n={len(cases):3d}  local per-case mean val Dice {mean:.4f}  |  logged best val Dice "
              f"{float(meta['val_dice']):.4f} (epoch {meta['epoch']})  |  sizes 2D {s2} 3D {s3}  flip={cfg.FLIP_AP_LR}  "
              f"geometry={model.geometry} output={model.output_mode}", flush=True)
