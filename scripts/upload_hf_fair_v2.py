#!/usr/bin/env python3
"""Upload the staged fair_v2 folder (see scripts/stage_hf_fair_v2.py) to the Hugging Face model repo in one commit.

The write token of the account that owns the repo (jeevan1234) is typed in by the user (hidden input). It is
never printed, logged or saved. Old files in the repo are not removed.

Usage: python scripts/upload_hf_fair_v2.py /tmp/hf_stage
"""
import getpass
import sys

from huggingface_hub import HfApi

REPO = "jeevan1234/biplanar-bone-reconstruction-models"


def main():
    stage = sys.argv[1]
    token = getpass.getpass("Paste the jeevan1234 Hugging Face WRITE token (input is hidden), then press Enter: ").strip()
    api = HfApi(token=token)
    try:
        who = api.whoami()["name"]
    except Exception as e:
        print("Token check failed:", type(e).__name__, "- please check the token and run again.")
        return 1
    print("Logged in as:", who, flush=True)
    print("Uploading about 0.9 GB to", REPO, "... (a few minutes)", flush=True)
    info = api.upload_folder(folder_path=stage, repo_id=REPO, repo_type="model",
                             commit_message="Add fair_v2: 18 retrained models of paper version 2 (corrected view "
                                            "placement, shared training settings) and update the model card")
    print("UPLOAD DONE:", getattr(info, "commit_url", info), flush=True)
    files = HfApi(token=False).list_repo_files(REPO)
    weights = [f for f in files if f.startswith("fair_v2/") and f.endswith("best_model_weights.pth")]
    print(f"Check: {len(weights)} model weight files under fair_v2/ in the public repo (expected 18).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
