#!/usr/bin/env python3
"""Write a cleaned split manifest that keeps only one copy of each duplicated sample.

Two samples count as copies when their AP image, lateral image and 3D mask are byte-identical
(same MD5 hashes), the same test used for the paper (paper/arxiv_submission/scripts/find_duplicates.py).

For every group of copies we keep one sample and drop the others:
  * keep the copy in train if there is one, otherwise the one in val, otherwise the one in test;
  * if several copies are in the same split, keep the lowest PID.
So test cases that had a copy in train or val are removed from test (the same 10 CTPelvic cases the
paper already excludes), val cases with a copy in train are removed from val, and repeated training
samples are kept once.

Nothing is deleted or moved. The original manifest (splits/<name>_seed42.json), which the trained
models used, stays as it is. A new file splits/<name>_seed42_dedup.json is written only for datasets
that have copies. Use it for new training runs with:
    python train.py --config <config> --split_manifest splits/ctpelvic_seed42_dedup.json

Usage (from the repository root):
    python scripts/make_clean_splits.py            # all datasets
    python scripts/make_clean_splits.py --dataset ctpelvic
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "femur": PROJECT_ROOT / "datasets" / "Total Segmentator" / "Femur",
    "hip": PROJECT_ROOT / "datasets" / "Total Segmentator" / "Hip",
    "rib": PROJECT_ROOT / "datasets" / "Total Segmentator" / "Rib",
    "verse19": PROJECT_ROOT / "datasets" / "Verse" / "verse19",
    "verse20": PROJECT_ROOT / "datasets" / "Verse" / "verse20",
    "ctpelvic": PROJECT_ROOT / "datasets" / "ctpelvik",
}
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def sample_key(sample_dir: Path) -> tuple[str, str, str]:
    """MD5 of the AP image, the lateral image and the mask of one sample."""
    ap = sorted((sample_dir / "x-ray").glob("*ap*.png"))[0]
    lat = sorted((sample_dir / "x-ray").glob("*lat*.png"))[0]
    mask = sorted((sample_dir / "ct-scan").glob("*.nii.gz"))[0]
    return md5(ap), md5(lat), md5(mask)


def clean_dataset(name: str, seed: int) -> dict | None:
    source = PROJECT_ROOT / "splits" / f"{name}_seed{seed}.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    splits = manifest["splits"]
    where = {pid: split for split, pids in splits.items() for pid in pids}
    store = DATASETS[name] / manifest.get("canonical_source", "train")

    groups = collections.defaultdict(list)
    for pid in sorted(where):
        groups[sample_key(store / pid)].append(pid)
    copies = [pids for pids in groups.values() if len(pids) > 1]

    removed = []
    for pids in copies:
        keep = min(pids, key=lambda p: (SPLIT_PRIORITY[where[p]], p))
        for pid in sorted(pids):
            if pid != keep:
                removed.append({"pid": pid, "split": where[pid],
                                "copy_of": keep, "copy_of_split": where[keep]})
    dropped = {r["pid"] for r in removed}
    clean = {split: [p for p in pids if p not in dropped] for split, pids in splits.items()}

    counts = {split: f"{len(splits[split])} -> {len(clean[split])}" for split in ("train", "val", "test")}
    print(f"{name:9s} samples={len(where):4d} unique={len(groups):4d} copy groups={len(copies):3d} "
          f"removed={len(removed):3d}  " + "  ".join(f"{s} {c}" for s, c in counts.items()))
    if not removed:
        return None

    return {
        "schema_version": manifest.get("schema_version", 1),
        "dataset": name,
        "seed": seed,
        "canonical_source": manifest.get("canonical_source", "train"),
        "derived_from": str(source.relative_to(PROJECT_ROOT)),
        "rule": ("Samples whose AP image, lateral image and mask are byte-identical (MD5) are copies. "
                 "One copy per group is kept: the one in train, else val, else test; lowest PID within a split."),
        "counts": {split: len(pids) for split, pids in clean.items()},
        "removed": removed,
        "splits": clean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write split manifests without duplicated samples")
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing *_dedup.json")
    args = parser.parse_args()

    names = DATASETS if args.dataset == "all" else [args.dataset]
    for name in names:
        result = clean_dataset(name, args.seed)
        if result is None:
            continue
        output = PROJECT_ROOT / "splits" / f"{name}_seed{args.seed}_dedup.json"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"          wrote {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
