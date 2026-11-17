"""Pick one representative test case per subset for the paper's qualitative figure.

The case is the one whose MultiScale test Dice is closest to that model's median
test Dice (no cherry-picking of best cases). Saves inputs, ground truth and the
three models' binary predictions to paper/evidence/qualitative/<subset>.npz.

Usage: python scripts/select_qualitative.py --pred-dir <dir written by evaluate_checkpoints.py>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from configs import get_config  # noqa: E402
from scripts.evaluate_checkpoints import DATASETS, MODELS, run_id  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--evidence", default=os.path.join(ROOT, "paper", "evidence"))
    args = ap.parse_args()
    out_dir = os.path.join(args.evidence, "qualitative")
    os.makedirs(out_dir, exist_ok=True)
    for ds in DATASETS:
        res = {m: json.load(open(os.path.join(args.evidence, "test", f"{run_id(ds, m)}.json"))) for m in MODELS}
        cases = res["multiscale_dualviewcnn"]["test_cases"]
        med = np.median([c["dice"] for c in cases])
        pid = min(cases, key=lambda c: (abs(c["dice"] - med), c["pid"]))["pid"]
        out = {"pid": pid}
        for m in MODELS:
            z = np.load(os.path.join(args.pred_dir, f"{run_id(ds, m)}.npz"))
            i = list(z["pids"]).index(pid)
            shape = tuple(z["shape"])
            n = int(np.prod(shape))
            out[m] = np.unpackbits(z["pred"][i])[:n].reshape(shape).astype(bool)
            gt = np.unpackbits(z["gt"][i])[:n].reshape(shape).astype(bool)
            if "gt" in out:
                assert (out["gt"] == gt).all()
            out["gt"] = gt
            out[m + "_dice"] = next(c["dice"] for c in res[m]["test_cases"] if c["pid"] == pid)
        H, W = out["gt"].shape[1:]
        D = out["gt"].shape[0]
        xdir = os.path.join(get_config(f"multiscale_dualviewcnn_{ds}").DATA_ROOT, "train", pid, "x-ray")
        apf = glob.glob(os.path.join(xdir, "*ap*.png"))[0]
        latf = glob.glob(os.path.join(xdir, "*lat*.png"))[0]
        # Inputs shown at the model's input resolution; AP is (H,W), lateral is (H,D).
        out["ap"] = np.asarray(Image.open(apf).convert("L").resize((W, H), Image.LANCZOS))
        out["lat"] = np.asarray(Image.open(latf).convert("L").resize((D, H), Image.LANCZOS))
        np.savez_compressed(os.path.join(out_dir, f"{ds}.npz"), **out)
        print(ds, pid, {m: round(100 * out[m + "_dice"], 1) for m in MODELS})


if __name__ == "__main__":
    main()
