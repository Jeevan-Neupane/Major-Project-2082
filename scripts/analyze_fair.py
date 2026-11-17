#!/usr/bin/env python3
"""Statistics for the fair-benchmark test results (paper/evidence_fair/test/*.json).

Same method as the paper (paper/arxiv_submission/scripts/make_revision_tables.py):
  * per-case paired Dice differences MultiScale - other model, per dataset;
  * 95% bootstrap CI of the mean difference (10,000 resamples, numpy default_rng(0));
  * two-sided Wilcoxon signed-rank test; Holm step-down correction over the 12 MultiScale comparisons.
Also reported (exploratory, separate Holm family of 6): SwinX2S - DualViewCNN.
Also compares each model's new test Dice with the earlier November runs (results_revised.json).
Writes paper/evidence_fair/results_fair.json.
"""
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "paper", "evidence_fair")
DATASETS = ["femur", "hip", "rib", "verse19", "verse20", "ctpelvic"]
MODELS = ["dualviewcnn", "multiscale", "swinx2s"]
OLD_NAME = {"dualviewcnn": "dualviewcnn", "multiscale": "multiscale_dualviewcnn", "swinx2s": "swinx2s"}


def load(model, ds):
    return json.load(open(os.path.join(EV, "test", f"fair_{model}_{ds}.json")))


def paired(a, b, rng):
    assert set(a) == set(b)
    d = np.array([a[p] - b[p] for p in sorted(a)])
    boot = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    return dict(n=len(d), mean_diff=float(d.mean()), median_diff=float(np.median(d)),
                ci_low=float(np.percentile(boot, 2.5)), ci_high=float(np.percentile(boot, 97.5)),
                wins=int((d > 0).sum()), p=float(stats.wilcoxon(d).pvalue))


def holm(comps):
    order = np.argsort([c["p"] for c in comps])
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(comps) - rank) * comps[idx]["p"]))
        comps[idx]["p_holm"] = running


def main():
    R = {(m, ds): load(m, ds) for m in MODELS for ds in DATASETS}
    dice = {k: {c["pid"]: c["dice"] for c in v["cases"]} for k, v in R.items()}

    rng = np.random.default_rng(0)
    primary = []
    for ds in DATASETS:
        for other in ["dualviewcnn", "swinx2s"]:
            primary.append(dict(dataset=ds, comparison=f"multiscale - {other}",
                                **paired(dice["multiscale", ds], dice[other, ds], rng)))
    holm(primary)
    exploratory = [dict(dataset=ds, comparison="swinx2s - dualviewcnn",
                        **paired(dice["swinx2s", ds], dice["dualviewcnn", ds], rng)) for ds in DATASETS]
    holm(exploratory)

    old = json.load(open(os.path.join(ROOT, "paper", "arxiv_submission", "results_revised.json")))["test_summary_clean"]
    before_after = {f"{m}/{ds}": {"old_test_dice": old[f"{OLD_NAME[m]}/{ds}"]["dice"],
                                  "new_test_dice": R[m, ds]["summary"]["dice"]} for m in MODELS for ds in DATASETS}
    means = {m: float(np.mean([R[m, ds]["summary"]["dice"] for ds in DATASETS])) for m in MODELS}
    old_means = {m: float(np.mean([old[f"{OLD_NAME[m]}/{ds}"]["dice"] for ds in DATASETS])) for m in MODELS}
    n_cases = sum(R["multiscale", ds]["n_cases"] for ds in DATASETS)

    out = {"n_test_cases": n_cases,
           "summary": {f"{m}/{ds}": R[m, ds]["summary"] for m in MODELS for ds in DATASETS},
           "params": {f"{m}/{ds}": R[m, ds]["params"] for m in MODELS for ds in DATASETS},
           "best_epoch": {f"{m}/{ds}": R[m, ds]["best_epoch"] for m in MODELS for ds in DATASETS},
           "mean_of_datasets_dice": means, "old_mean_of_datasets_dice": old_means,
           "paired_primary": primary, "paired_exploratory": exploratory, "before_after_test": before_after}
    json.dump(out, open(os.path.join(EV, "results_fair.json"), "w"), indent=1)

    print(f"test cases: {n_cases}")
    print("mean test Dice over datasets (old -> new):",
          ", ".join(f"{m} {100*old_means[m]:.1f} -> {100*means[m]:.1f}" for m in MODELS))
    for c in primary + exploratory:
        print(f"{c['dataset']:9s} {c['comparison']:24s} n={c['n']:3d} diff {100*c['mean_diff']:+5.1f} "
              f"[{100*c['ci_low']:+5.1f}, {100*c['ci_high']:+5.1f}] wins {c['wins']:3d}/{c['n']:<3d} "
              f"p={c['p']:.2g} p_holm={c['p_holm']:.2g}")


if __name__ == "__main__":
    main()
