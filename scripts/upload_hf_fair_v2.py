#!/usr/bin/env python3
"""Replace the contents of the Hugging Face model repo with the staged models (see scripts/stage_hf_fair_v2.py).

One commit adds every staged file and removes every other file in the repo (the earlier models), except
.gitattributes. The earlier files stay in the repo's commit history.

The write token of the account that owns the repo (jeevan1234) is typed in by the user (hidden input). It is
never printed, logged or saved.

Usage: python scripts/upload_hf_fair_v2.py /tmp/hf_stage
"""
import getpass
import os
import sys

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

REPO = "jeevan1234/biplanar-bone-reconstruction-models"


def main():
    stage = sys.argv[1]
    local = sorted(os.path.relpath(os.path.join(d, f), stage).replace(os.sep, "/")
                   for d, _, fs in os.walk(stage) for f in fs)
    token = getpass.getpass("Paste the jeevan1234 Hugging Face WRITE token (input is hidden), then press Enter: ").strip()
    api = HfApi(token=token)
    try:
        who = api.whoami()["name"]
    except Exception as e:
        print("Token check failed:", type(e).__name__, "- please check the token and run again.")
        return 1
    print("Logged in as:", who, flush=True)
    old = sorted(set(api.list_repo_files(REPO)) - set(local) - {".gitattributes"})
    print(f"Adding {len(local)} files (about 0.9 GB) and removing {len(old)} old files ... (a few minutes)", flush=True)
    ops = ([CommitOperationAdd(path_in_repo=f, path_or_fileobj=os.path.join(stage, f)) for f in local]
           + [CommitOperationDelete(path_in_repo=f) for f in old])
    info = api.create_commit(repo_id=REPO, repo_type="model", operations=ops,
                             commit_message="Replace the earlier models with the 18 models of the paper "
                                            "(corrected view placement, shared training settings)")
    print("UPLOAD DONE:", getattr(info, "commit_url", info), flush=True)
    files = HfApi(token=False).list_repo_files(REPO)
    weights = [f for f in files if f.startswith("fair_v2/") and f.endswith("best_model_weights.pth")]
    left = [f for f in files if f in old]
    print(f"Check: {len(weights)} model weight files under fair_v2/ (expected 18); "
          f"{len(left)} old files left (expected 0); {len(files)} files in the repo.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
