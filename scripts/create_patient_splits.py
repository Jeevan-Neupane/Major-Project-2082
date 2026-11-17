#!/usr/bin/env python3
"""Create deterministic, disjoint PID split manifests without moving data files.

The existing val/ and test/ directories are overlapping subsets of train/.
This utility keeps the data untouched and makes 80/10/10 PID-level manifests
from the complete canonical train/ copy.  Training consumes these manifests.
"""
from __future__ import annotations

import argparse
import json
import random
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


def valid_pids(directory: Path) -> set[str]:
    """Return PIDs with both X-ray views and a target volume."""
    pids = set()
    for sample_dir in directory.glob("p_*"):
        if not sample_dir.is_dir():
            continue
        has_ap = any((sample_dir / "x-ray").glob("*ap*.png"))
        has_lat = any((sample_dir / "x-ray").glob("*lat*.png"))
        has_volume = any((sample_dir / "ct-scan").glob("*.nii.gz"))
        if has_ap and has_lat and has_volume:
            pids.add(sample_dir.name)
    return pids


def create_split(pids: set[str], seed: int, train_ratio: float, val_ratio: float) -> dict[str, list[str]]:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than one")
    ordered = sorted(pids)
    random.Random(seed).shuffle(ordered)
    train_end = round(len(ordered) * train_ratio)
    val_end = train_end + round(len(ordered) * val_ratio)
    return {
        "train": sorted(ordered[:train_end]),
        "val": sorted(ordered[train_end:val_end]),
        "test": sorted(ordered[val_end:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create disjoint PID split manifests")
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "splits"
    output_dir.mkdir(exist_ok=True)
    selected = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for name, root in selected.items():
        canonical = valid_pids(root / "train")
        available = set().union(*(valid_pids(root / split) for split in ("train", "val", "test")))
        if not canonical:
            raise RuntimeError(f"No complete paired samples under {root / 'train'}")
        if canonical != available:
            raise RuntimeError(
                f"Canonical train/ does not contain exactly the available sample IDs for {name}; refusing to split."
            )
        splits = create_split(canonical, args.seed, args.train_ratio, args.val_ratio)
        split_sets = [set(splits[name]) for name in ("train", "val", "test")]
        assert not (split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2])
        manifest = {
            "schema_version": 1,
            "dataset": name,
            "seed": args.seed,
            "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": 1 - args.train_ratio - args.val_ratio},
            "canonical_source": "train",
            "splits": splits,
        }
        output = output_dir / f"{name}_seed{args.seed}.json"
        output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} -> {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
