"""One CPU training step of fair_swinx2s (official Swin windows + drop path) through the real loss path."""
import os, sys
sys.path.insert(0, "/home/ubuntu/Major-Project-2082")
import torch
from configs.config import CONFIG_REGISTRY as R
from train import create_model, load_split_manifest
from utils.data.dataset import BoneDataset, collate_bone
from utils.training.losses import DiceBCELoss
from utils.training.trainer import _objective
from models.swinX2S.swin_transformer import SwinTransformerBlock
if __name__ == "__main__":
    os.chdir("/home/ubuntu/Major-Project-2082"); torch.manual_seed(0)
    cfg = R["fair_swinx2s_verse20"]
    _, sp = load_split_manifest(cfg)
    ds = BoneDataset(root_dir=os.path.join(cfg.DATA_ROOT, "train"), img_size_2d=tuple(cfg.IMG_SIZE[1:]),
                     img_size_3d=tuple(cfg.IMG_SIZE), include_pids=sp["train"][:2], augment=True, flip_ap_lr=cfg.FLIP_AP_LR)
    b = collate_bone([ds[i] for i in range(len(ds))])
    m = create_model(cfg, torch.device("cpu")).train()
    blocks = [x for x in m.modules() if isinstance(x, SwinTransformerBlock)]
    dp = [type(x.drop_path).__name__ for x in blocks]
    print("swin_impl:", m.swin_impl, "| blocks using official windows:", sum(x.official_windows for x in blocks), "/", len(blocks),
          "| drop path modules:", dp.count("DropPath"), "DropPath +", dp.count("Identity"), "Identity (first block has rate 0)")
    logits = m(b["xa"], b["xb"])
    loss = _objective(m, DiceBCELoss(bce_weight=cfg.BCE_WEIGHT, dice_weight=cfg.DICE_WEIGHT), logits, b["vol"])
    loss.backward()
    g = [p.grad for p in m.parameters()]
    print("loss", round(loss.item(), 4), "| finite grads", sum(x is not None and torch.isfinite(x).all().item() for x in g), "/", len(g),
          "| two-view loss used:", getattr(m, "view_logits", None) is not None, "| params %.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
