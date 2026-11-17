"""Build the data-driven tables and figures of the paper from the benchmark results.

Writes the box plot, the qualitative figure, the validation and loss curves and the LaTeX tables that
main.tex includes.

Nothing is trained, evaluated or re-scored here. Inputs:
  paper/evidence_fair/test/fair_<model>_<dataset>.json   per-case test metrics (scored once, 17 Nov 2026)
  paper/evidence_fair/val/fair_<model>_<dataset>.json    validation metrics of the same checkpoints
  paper/evidence_fair/results_fair.json                  paired statistics (scripts/analyze_fair.py)
  paper/evidence_fair/test_preds/fair_<model>_<dataset>.npz  thresholded test predictions (packbits)
  <run_id>.csv / <run_id>.log                            per-epoch training logs of the 18 runs
  splits/<dataset>_seed42.json, splits/ctpelvic_seed42_dedup.json   split manifests
  <DATA_ROOT>/train/<pid>/x-ray/*ap*.png, *lat*.png      inputs shown in the qualitative figure
Only the run whose run_dir is named in the evidence JSON is read for each model and dataset, so
timing tests and aborted runs in the same log folders are never used.

Notes:
  * CTPelvic uses the split with one copy of each scan (412/48/44).
  * The epoch limit is read from each log (200 for every model).
  * SwinX2S hip and CTPelvic stopped at epochs 195 and 190 because of the cloud GPU time limit
    (learning rate already ~0); stability.tex shows the epochs actually run.
  * The qualitative figure shows the AP input mirrored, as the models saw it (see render()).

Usage (repository layout):
  /opt/Major-Project-2082/rootvenv/bin/python paper/arxiv_v3/scripts/make_fair_tables.py
Needs numpy, scipy, matplotlib and Pillow.
"""
import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent                     # paper/arxiv_v3/scripts
REPO = HERE.parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--evidence", default=str(REPO / "paper" / "evidence_fair"))
ap.add_argument("--repo", default=str(REPO), help="repository root (splits/, logs, datasets)")
ap.add_argument("--out", default=str(HERE.parent))
args = ap.parse_args()
E, ROOT, OUT = Path(args.evidence), Path(args.repo), Path(args.out)
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)
LOG_DIRS = [ROOT / "logs", ROOT / "modal_results_second" / "results" / "logs"]

MODELS = ["dualviewcnn", "multiscale", "swinx2s"]
NAMES = ["DualViewCNN", "MultiScale", "SwinX2S"]
DATASETS = ["femur", "hip", "rib", "verse19", "verse20", "ctpelvic"]
LABELS = ["Femur", "Hip", "Rib", "VerSe19", "VerSe20", "CTPelvic"]
# Runs that ended on the cloud GPU's hard time limit, not on the epoch limit or early stopping.
# Checked against the logs below; if another run ends without either, the script stops.
TIME_LIMITED = {("hip", "swinx2s"), ("ctpelvic", "swinx2s")}
# The three runs of the loss figure.
LOSS_PANELS = [("rib", "multiscale", "MultiScale, rib"), ("hip", "swinx2s", "SwinX2S, hip"),
               ("ctpelvic", "swinx2s", "SwinX2S, CTPelvic")]
# Validated categorical slots 1-3 (dataviz reference palette, light mode, all-pairs).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

# ------------------------------------------------------------------ evidence and logs
R = {(ds, m): json.loads((E / "test" / f"fair_{m}_{ds}.json").read_text()) for ds in DATASETS for m in MODELS}
V = {(ds, m): json.loads((E / "val" / f"fair_{m}_{ds}.json").read_text()) for ds in DATASETS for m in MODELS}
FAIR = json.loads((E / "results_fair.json").read_text())


def run_id(ds, m):
    return Path(R[ds, m]["run_dir"]).name


def find_log(rid):
    hits = [d / f"{rid}.csv" for d in LOG_DIRS if (d / f"{rid}.csv").exists()]
    assert len(hits) == 1, (rid, hits)
    return hits[0]


LOGS, TRAIN = {}, {}
for ds in DATASETS:
    for m in MODELS:
        rid = run_id(ds, m)
        assert V[ds, m]["run_dir"] == R[ds, m]["run_dir"] and V[ds, m]["best_epoch"] == R[ds, m]["best_epoch"]
        path = find_log(rid)
        rows = list(csv.DictReader(path.open()))
        log = path.with_suffix(".log").read_text()
        assert [int(r["epoch"]) for r in rows] == list(range(1, len(rows) + 1)), rid
        assert len(rows) > 2, f"{rid} is a timing test"
        for r in rows:
            for k in ["val_dice", "val_iou", "val_precision", "val_recall", "train_loss", "val_loss"]:
                assert math.isfinite(float(r[k])), (rid, r["epoch"], k)
        best = max(rows, key=lambda r: float(r["val_dice"]))          # first maximum
        assert int(best["epoch"]) == R[ds, m]["best_epoch"], rid
        assert abs(float(best["val_dice"]) - R[ds, m]["logged_val_dice"]) < 1e-9, rid
        # The validation re-check of the saved checkpoint reproduces the logged Dice.
        assert abs(V[ds, m]["summary"]["dice"] - float(best["val_dice"])) < 2e-5, rid
        cap = int(re.search(r"NUM_EPOCHS\s*: (\d+)", log)[1])
        if "Early stopping at epoch" in log:
            stop = "early stopping"
        elif len(rows) == cap:
            stop = "epoch limit"
        else:
            stop = "time limit"
        assert (stop == "time limit") == ((ds, m) in TIME_LIMITED), (rid, stop, len(rows))
        if stop != "time limit":
            assert int(re.findall(r"Best Epoch\s*: (\d+)", log)[-1]) == int(best["epoch"]), rid
        LOGS[ds, m] = rows
        TRAIN[ds, m] = dict(run_id=rid, log_csv=str(path.relative_to(ROOT)), epochs_run=len(rows), epoch_cap=cap,
                            stop=stop, best_epoch=int(best["epoch"]),
                            final_lr=float(rows[-1]["lr"]), final_val_dice=float(rows[-1]["val_dice"]),
                            final_train_loss=float(rows[-1]["train_loss"]), final_val_loss=float(rows[-1]["val_loss"]),
                            nonfinite_gradient_epochs=sum(not math.isfinite(float(r["grad_norm"])) for r in rows),
                            **{f"best_{k}": float(best[k]) for k in
                               ["val_dice", "val_iou", "val_precision", "val_recall", "val_loss"]})


def cases(ds, m):
    return R[ds, m]["cases"]


def summary(cs):
    a = lambda k: np.array([c[k] for c in cs], dtype=float)
    out = {"n": len(cs)}
    for k in ["dice", "iou", "precision", "recall", "nsd_1.5mm", "hd95_mm", "assd_mm"]:
        out[k] = float(a(k).mean())
        out[k + "_sd"] = float(a(k).std(ddof=1)) if len(cs) > 1 else 0.0
        out[k + "_median"] = float(np.median(a(k)))
    return out


S = {(ds, m): summary(cases(ds, m)) for ds in DATASETS for m in MODELS}
for (ds, m), s in S.items():
    # Surface metrics are defined for every case (no empty prediction), and the per-case means
    # agree with the summary stored by the evaluation script.
    assert all(c["hd95_mm"] is not None for c in cases(ds, m)), (ds, m)
    assert s["n"] == R[ds, m]["n_cases"] == R[ds, m]["summary"]["n"]
    for k in ["dice", "dice_sd", "iou", "precision", "recall", "nsd_1.5mm", "hd95_mm", "hd95_mm_median", "assd_mm"]:
        assert abs(s[k] - R[ds, m]["summary"][k]) < 1e-9, (ds, m, k)
for ds in DATASETS:
    pids = [sorted(c["pid"] for c in cases(ds, m)) for m in MODELS]
    assert pids[0] == pids[1] == pids[2], ds
    grid = R[ds, "multiscale"]["grid"]
    assert all(R[ds, m]["grid"] == grid and R[ds, m]["flip_ap_lr"] is True for m in MODELS), ds


def table(name, head, body, cols):
    (OUT / "tables" / name).write_text(
        "\\begin{tabular}{" + cols + "}\n\\toprule\n" + head + " \\\\\n\\midrule\n"
        + "\n".join(" & ".join(r) + " \\\\" for r in body) + "\n\\bottomrule\n\\end{tabular}\n")


pct = lambda x: f"{100 * x:.1f}"

# ------------------------------------------------------------------ splits and grids
body, SPLITS = [], {}
for ds, lab in zip(DATASETS, LABELS):
    manifest = R[ds, "multiscale"]["split_manifest"]
    assert all(R[ds, m]["split_manifest"] == manifest for m in MODELS)
    s = json.loads((ROOT / manifest).read_text())["splits"]
    sets = [set(v) for v in s.values()]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i)), ds
    assert sorted(s["test"]) == sorted(c["pid"] for c in cases(ds, "multiscale")), ds
    assert len(s["val"]) == V[ds, "multiscale"]["n_cases"], ds
    grid = R[ds, "multiscale"]["grid"]
    sp = np.median(np.array([c["voxel_spacing_mm"] for c in cases(ds, MODELS[0])]), axis=0)
    n = {k: len(s[k]) for k in ["train", "val", "test"]}
    SPLITS[ds] = dict(manifest=manifest, **n, total=sum(n.values()), grid=grid, voxel_mm=[float(x) for x in sp])
    body.append([lab, str(n["train"]), str(n["val"]), str(n["test"]), str(sum(n.values())),
                 r"$%d\times%d\times%d$" % tuple(grid), r"$%.2f\times%.2f\times%.2f$" % tuple(sp)])
table("splits.tex", r"Dataset & Train & Val. & Test & Total & Grid $(D,H,W)$ & Voxel (mm)", body, "lrrrrcc")

# ------------------------------------------------------------------ main test Dice table
body = []
for ds, lab in zip(DATASETS, LABELS):
    t = [S[ds, m] for m in MODELS]
    top = max(x["dice"] for x in t)
    cells = [(r"\textbf{%s $\pm$ %s}" if x["dice"] == top else r"%s $\pm$ %s") % (pct(x["dice"]), pct(x["dice_sd"]))
             for x in t]
    body.append([lab, str(t[0]["n"])] + cells)
# A mean over datasets is kept in results_fair_summary.json only: Dice is not comparable across anatomies.
means = {m: float(np.mean([S[ds, m]["dice"] for ds in DATASETS])) for m in MODELS}
for m in MODELS:
    assert abs(means[m] - FAIR["mean_of_datasets_dice"][m]) < 1e-9, m
table("test_dice.tex", r"Dataset & $n$ & DualViewCNN & MultiScale & SwinX2S", body, "lrccc")
test_dice_rows = body

# ------------------------------------------------------------------ full metrics table
body = []
for ds, lab in zip(DATASETS, LABELS):
    for i, (m, n) in enumerate(zip(MODELS, NAMES)):
        t = S[ds, m]
        body.append([lab if i == 0 else "", n] + [pct(t[k]) for k in ["dice", "iou", "precision", "recall", "nsd_1.5mm"]]
                    + [f"{t['hd95_mm']:.1f}", f"{t['hd95_mm_median']:.1f}", f"{t['assd_mm']:.2f}"])
table("test_metrics.tex",
      r"&&&&&&& \multicolumn{2}{c}{HD95 (mm)} & \\ \cmidrule(lr){8-9} "
      r"Dataset & Model & Dice & IoU & Prec. & Rec. & NSD & Mean & Median & ASSD (mm)", body, "llrrrrrrrr")

# ------------------------------------------------------------------ paired comparisons (from results_fair.json)
# The statistics are taken as stored (scripts/analyze_fair.py: 10,000
# bootstrap resamples with default_rng(0), two-sided Wilcoxon signed-rank test, Holm over the 12
# MultiScale comparisons). Only their deterministic parts are checked against the per-case Dice here.
comparisons = FAIR["paired_primary"]
assert len(comparisons) == 12
for c in comparisons:
    a, b = c["comparison"].split(" - ")
    assert a == "multiscale" and b in ("dualviewcnn", "swinx2s"), c["comparison"]
    c["vs"] = b
    ms = {x["pid"]: x["dice"] for x in cases(c["dataset"], "multiscale")}
    other = {x["pid"]: x["dice"] for x in cases(c["dataset"], b)}
    d = np.array([ms[p] - other[p] for p in sorted(ms)])
    assert c["n"] == len(d) and c["wins"] == int((d > 0).sum()), c
    assert abs(c["mean_diff"] - d.mean()) < 1e-9 and abs(c["median_diff"] - np.median(d)) < 1e-9, c
    assert math.isclose(c["p"], stats.wilcoxon(d).pvalue, rel_tol=1e-6), c
    assert c["ci_low"] <= c["mean_diff"] <= c["ci_high"], c
order = np.argsort([c["p"] for c in comparisons])
running = 0.0
for rank, idx in enumerate(order):  # Holm step-down over all 12 tests
    running = max(running, min(1.0, (len(comparisons) - rank) * comparisons[idx]["p"]))
    assert math.isclose(comparisons[idx]["p_holm"], running, rel_tol=1e-9), comparisons[idx]

fmt_p = lambda p: r"$<$0.001" if p < 0.001 else f"{p:.3f}"
body = []
for ds, lab in zip(DATASETS, LABELS):
    row = [lab]
    for m in ["dualviewcnn", "swinx2s"]:
        c = next(c for c in comparisons if c["dataset"] == ds and c["vs"] == m)
        row += [f"{100*c['mean_diff']:+.1f} [{100*c['ci_low']:+.1f}, {100*c['ci_high']:+.1f}]",
                f"{c['wins']}/{c['n']}", fmt_p(c["p_holm"])]
    body.append(row)
table("test_comparisons.tex",
      r"& \multicolumn{3}{c}{MultiScale $-$ DualViewCNN} & \multicolumn{3}{c}{MultiScale $-$ SwinX2S} \\"
      r" \cmidrule(lr){2-4}\cmidrule(lr){5-7} Dataset & $\Delta$Dice [95\% CI] & Wins & $p_{\rm Holm}$"
      r" & $\Delta$Dice [95\% CI] & Wins & $p_{\rm Holm}$", body, "lrrrrrr")

# ------------------------------------------------------------------ CTPelvic duplicate table: no longer applies
# The CTPelvic duplicates were removed from the split before training (splits/ctpelvic_seed42_dedup.json).
(OUT / "tables" / "ctpelvic_duplicates.tex").unlink(missing_ok=True)

# ------------------------------------------------------------------ appendix tables
body, st = [], []
for ds, lab in zip(DATASETS, LABELS):
    for i, (m, n) in enumerate(zip(MODELS, NAMES)):
        rows, tr = LOGS[ds, m], TRAIN[ds, m]
        best = rows[tr["best_epoch"] - 1]
        body.append([lab if i == 0 else "", n, str(best["epoch"])] + [pct(float(best[k])) for k in
                    ["val_dice", "val_iou", "val_precision", "val_recall"]] + [f"{float(best['val_loss']):.3f}"])
        # Epochs actually run out of the limit (SwinX2S hip 195 and CTPelvic 190: GPU time limit).
        st.append([lab if i == 0 else "", n, f"{tr['epochs_run']}/{tr['epoch_cap']}", pct(tr["final_val_dice"]),
                   f"{tr['final_train_loss']:.3f}", f"{tr['final_val_loss']:.3f}", str(tr["nonfinite_gradient_epochs"])])
table("val_metrics.tex", "Dataset & Model & Epoch & Dice & IoU & Prec. & Rec. & Loss", body, "llrrrrrr")
table("stability.tex", "Dataset & Model & Epochs & Final Dice & Train loss & Val.\\ loss & Non-finite", st, "llrrrrr")

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})

# Per-case test Dice (style of make_revision_tables.py)
fig, ax = plt.subplots(figsize=(6.5, 2.5))
width = 0.26
for j, (m, n, c) in enumerate(zip(MODELS, NAMES, COLORS)):
    data = [[100 * x["dice"] for x in cases(ds, m)] for ds in DATASETS]
    pos = np.arange(len(DATASETS)) + (j - 1) * (width + 0.02)
    ax.boxplot(data, positions=pos, widths=width, showfliers=False, patch_artist=True,
               medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=c, lw=1.0),
               capprops=dict(color=c, lw=1.0), boxprops=dict(facecolor=c + "33", edgecolor=c, lw=1.2))
    for p, d in zip(pos, data):
        jitter = np.random.default_rng(1).uniform(-width / 3, width / 3, len(d))
        ax.scatter(p + jitter, d, s=5, color=c, alpha=0.55, linewidths=0, zorder=3)
    ax.plot([], [], "s", color=c, label=n, ms=7)
ax.set_xticks(range(len(DATASETS)))
ax.set_xticklabels([f"{lab}\n($n$={S[ds, MODELS[0]]['n']})" for ds, lab in zip(DATASETS, LABELS)])
ax.set_ylabel("Test Dice (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", color=GRID, lw=0.6)
fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT / "figures" / "test_dice_distribution.pdf")
plt.close(fig)

# Validation curves (style of paper/generate_results.py)
fig, axs = plt.subplots(2, 3, figsize=(7.16, 3.9), sharex=False, sharey=True)
for ax, ds, lab in zip(axs.flat, DATASETS, LABELS):
    for m, n, c in zip(MODELS, NAMES, COLORS):
        rows, be = LOGS[ds, m], TRAIN[ds, m]["best_epoch"]
        ax.plot([int(r["epoch"]) for r in rows], [100 * float(r["val_dice"]) for r in rows], color=c, lw=1.4, label=n)
        ax.plot(be, 100 * float(rows[be - 1]["val_dice"]), "o", color=c, ms=5,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    ax.set_title(lab, loc="left", fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_xlabel("Epoch")
for ax in axs[:, 0]:
    ax.set_ylabel("Validation Dice (%)")
h, l = axs.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT / "figures" / "validation_curves.pdf")
plt.close(fig)

# Loss curves of three runs (style of paper/generate_results.py)
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.1))
for ax, (ds, m, title) in zip(axs, LOSS_PANELS):
    rows = LOGS[ds, m]
    x = [int(r["epoch"]) for r in rows]
    ax.plot(x, [float(r["train_loss"]) for r in rows], label="Training", color=COLORS[0], lw=1.4)
    ax.plot(x, [float(r["val_loss"]) for r in rows], label="Validation", color=COLORS[1], lw=1.4)
    ax.axvline(TRAIN[ds, m]["best_epoch"], color=INK2, ls=":", lw=1, label="Selected epoch")
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xlabel("Epoch")
    ax.grid(axis="y", color=GRID, lw=0.6)
axs[0].set_ylabel("BCE + soft Dice loss")
axs[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "figures" / "loss_curves.pdf")
plt.close(fig)


# ------------------------------------------------------------------ qualitative figure
def render(vol):
    """Depth-shaded surface of a binary (D,H,W) volume viewed along D.

    Image rows are H and columns are the volume's W axis, not flipped. The AP image is shown
    mirrored, as the models saw it (see below), so its columns also run along W.
    """
    hit = vol.any(axis=0)
    depth = np.where(hit, vol.argmax(axis=0), vol.shape[0]).astype(float)
    gy, gx = np.gradient(np.where(hit, depth, np.nan))
    nz = 1.0 / np.sqrt(1 + np.nan_to_num(gx) ** 2 + np.nan_to_num(gy) ** 2)
    shade = 0.25 + 0.75 * nz * (1 - 0.5 * depth / vol.shape[0])
    img = np.ones(hit.shape + (3,))
    img[hit] = np.clip(shade[hit, None] * np.array([0.93, 0.89, 0.80]), 0, 1)
    return img


def unpack(z, i):
    shape = tuple(int(s) for s in z["shape"])
    return np.unpackbits(z["pred"][i])[:int(np.prod(shape))].reshape(shape).astype(bool), \
        np.unpackbits(z["gt"][i])[:int(np.prod(shape))].reshape(shape).astype(bool)


def data_root(ds):
    log = find_log(run_id(ds, "multiscale")).with_suffix(".log").read_text()
    return Path(re.search(r"DATA_ROOT\s*: (.+)", log)[1].strip())


QUAL = {}
for ds in DATASETS:
    # The case whose MultiScale test Dice is closest to the median MultiScale test Dice; ties go to
    # the lower PID.
    cs = cases(ds, "multiscale")
    med = np.median([c["dice"] for c in cs])
    pid = min(cs, key=lambda c: (abs(c["dice"] - med), c["pid"]))["pid"]
    q = {"pid": pid}
    for m in MODELS:
        z = np.load(E / "test_preds" / f"fair_{m}_{ds}.npz")
        pred, gt = unpack(z, list(z["pids"]).index(pid))
        if "gt" in q:
            assert (q["gt"] == gt).all(), (ds, m)
        q["gt"], q[m] = gt, pred
        q[m + "_dice"] = next(c["dice"] for c in cases(ds, m) if c["pid"] == pid)
        # The stored prediction reproduces the Dice in the evidence JSON.
        assert abs(2 * (pred & gt).sum() / (pred.sum() + gt.sum()) - q[m + "_dice"]) < 1e-4, (ds, m)
    D, H, W = q["gt"].shape
    # The evaluation reads every sample from DATA_ROOT/train (manifest canonical_source "train").
    xdir = data_root(ds) / "train" / pid / "x-ray"
    apf, latf = sorted(glob.glob(str(xdir / "*ap*.png"))), sorted(glob.glob(str(xdir / "*lat*.png")))
    assert len(apf) == 1 and len(latf) == 1, xdir
    # Inputs shown at the output-grid resolution: AP spans (H, W), lateral spans (H, D).
    ap_img = np.asarray(Image.open(apf[0]).convert("L").resize((W, H), Image.LANCZOS))
    # The models were trained with FLIP_AP_LR=True: the AP PNG (radiographic display convention) is
    # mirrored left-right before it enters the network, so that its columns run along the volume's
    # W axis. We show the AP image as the model saw it (mirrored), which gives it the same left-right
    # orientation as the unflipped 3D renderings. The lateral image is shown as stored (not flipped),
    # which is also how the models saw it.
    q["ap"] = ap_img[:, ::-1]
    q["lat"] = np.asarray(Image.open(latf[0]).convert("L").resize((D, H), Image.LANCZOS))
    QUAL[ds] = q

cols = ["AP input", "Lateral input", "Ground truth"] + NAMES
fig, axs = plt.subplots(len(DATASETS), len(cols), figsize=(6.5, 1.15 * len(DATASETS)))
for r, (ds, lab) in enumerate(zip(DATASETS, LABELS)):
    q = QUAL[ds]
    panels = [q["ap"], q["lat"], render(q["gt"])] + [render(q[m]) for m in MODELS]
    for c, img in enumerate(panels):
        ax = axs[r, c]
        # Every panel covers the same cubic region around the bone, so all are drawn as equal squares.
        ax.imshow(img, cmap="gray" if c < 2 else None, interpolation="antialiased", aspect="auto")
        ax.set_box_aspect(1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        if r == 0:
            ax.set_title(cols[c], fontsize=8)
        if c >= 3:
            ax.text(0.5, -0.04, f"{100 * float(q[MODELS[c - 3] + '_dice']):.1f}%", transform=ax.transAxes,
                    ha="center", va="top", fontsize=7, color=INK2)
    axs[r, 0].set_ylabel(lab, fontsize=8)
fig.tight_layout(h_pad=0.9, w_pad=0.2)
fig.savefig(OUT / "figures" / "qualitative.pdf", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ machine-readable summary
r1 = lambda x: round(100 * x, 1)
out = {
    "source": {"evidence": str(E.relative_to(ROOT)) if E.is_relative_to(ROOT) else str(E),
               "test_scored_at": json.loads((E / "test" / "test_run_info.json").read_text()).get("test_scored_once_at"),
               "script": "paper/arxiv_v3/scripts/make_fair_tables.py"},
    "n_test_cases_total": sum(S[ds, "multiscale"]["n"] for ds in DATASETS),
    "n_test_cases": {ds: S[ds, "multiscale"]["n"] for ds in DATASETS},
    "test_dice_pct": {ds: {m: {"mean": r1(S[ds, m]["dice"]), "sd": r1(S[ds, m]["dice_sd"]),
                               "median": r1(S[ds, m]["dice_median"])} for m in MODELS} for ds in DATASETS},
    "best_model_per_dataset": {ds: max(MODELS, key=lambda m: S[ds, m]["dice"]) for ds in DATASETS},
    "mean_of_datasets_dice_pct": {m: r1(means[m]) for m in MODELS},
    "mean_of_datasets_dice": means,
    "test_summary": {f"{m}/{ds}": S[ds, m] for ds in DATASETS for m in MODELS},
    "paired_primary": [{k: c[k] for k in ["dataset", "comparison", "n", "mean_diff", "median_diff", "ci_low",
                                         "ci_high", "wins", "p", "p_holm"]} for c in comparisons],
    "paired_primary_wins_total": {m: {"wins": sum(c["wins"] for c in comparisons if c["vs"] == m),
                                      "n": sum(c["n"] for c in comparisons if c["vs"] == m)}
                                  for m in ["dualviewcnn", "swinx2s"]},
    "paired_exploratory": FAIR.get("paired_exploratory"),
    "training": {f"{m}/{ds}": TRAIN[ds, m] for ds in DATASETS for m in MODELS},
    "time_limited_runs": sorted(f"{m}/{ds}" for ds, m in TIME_LIMITED),
    "splits": SPLITS,
    "qualitative_cases": {ds: {"pid": QUAL[ds]["pid"], **{f"{m}_dice": QUAL[ds][m + "_dice"] for m in MODELS}}
                          for ds in DATASETS},
    "params": {f"{m}/{ds}": R[ds, m]["params"] for ds in DATASETS for m in MODELS},
    "notes": ["Per-dataset Dice is mean +- SD over the test cases (SD with ddof=1).",
              "The mean over datasets is not reported in the tables: Dice is not comparable across anatomies.",
              "CTPelvic uses splits/ctpelvic_seed42_dedup.json (412/48/44); no test case is excluded.",
              "SwinX2S hip and CTPelvic stopped at epochs 195 and 190 (cloud GPU time limit; learning rate ~0).",
              "Qualitative figure: AP input shown mirrored, as given to the models (FLIP_AP_LR=True); "
              "renderings unflipped, in the same orientation."],
}
(OUT / "results_fair_summary.json").write_text(json.dumps(out, indent=1) + "\n")

# ------------------------------------------------------------------ report
wins = out["paired_primary_wins_total"]
print(f"test cases: {out['n_test_cases_total']}; MultiScale wins vs DualViewCNN {wins['dualviewcnn']['wins']}/"
      f"{wins['dualviewcnn']['n']}, vs SwinX2S {wins['swinx2s']['wins']}/{wins['swinx2s']['n']}")
print("mean of datasets:", out["mean_of_datasets_dice_pct"])
for c in comparisons:
    print(f"{c['dataset']:9s} vs {c['vs']:12s} n={c['n']:3d} d={100*c['mean_diff']:+.2f} "
          f"[{100*c['ci_low']:+.2f},{100*c['ci_high']:+.2f}] wins={c['wins']} p={c['p']:.3g} pHolm={c['p_holm']:.3g}")
for ds, m in sorted(TIME_LIMITED):
    t = TRAIN[ds, m]
    print(f"time-limited: {m}/{ds} {t['epochs_run']}/{t['epoch_cap']} epochs, final lr {t['final_lr']:.1e}")
print("qualitative cases:", {ds: QUAL[ds]["pid"] for ds in DATASETS})
print("\nTest Dice (%, mean +- SD):")
print(f"{'Dataset':9s} {'n':>3s}  " + "  ".join(f"{n:>16s}" for n in NAMES))
for ds, lab in zip(DATASETS, LABELS):
    cells = [f"{r1(S[ds, m]['dice']):5.1f} +- {r1(S[ds, m]['dice_sd']):4.1f}"
             + ("*" if out["best_model_per_dataset"][ds] == m else " ") for m in MODELS]
    print(f"{lab:9s} {S[ds, 'multiscale']['n']:3d}  " + "  ".join(f"{c:>16s}" for c in cells))
print("(* = best; bold in tables/test_dice.tex)")
print("written:", ", ".join(f"tables/{t}" for t in ["splits.tex", "test_dice.tex", "test_metrics.tex",
                                                     "test_comparisons.tex", "val_metrics.tex", "stability.tex"]),
      "| figures/test_dice_distribution.pdf, validation_curves.pdf, loss_curves.pdf, qualitative.pdf"
      "| results_fair_summary.json | removed tables/ctpelvic_duplicates.tex")
