"""Schematic figures of the revised manuscript (no experimental data is drawn here).

overview.pdf               study pipeline, from CT collections to evaluation
lifting.pdf                how each model places the AP and lateral features in the 3D grid
swinx2s_implementation.pdf our SwinX2S implementation (replaces the reproduced published figure)
cnn_dualview_architecture.pdf, cnn_multiscale_architecture.pdf
                           the CNN schematics of paper/generate_cnn_architectures.py, redrawn
                           without an in-figure title

Each drawing follows the code in models/ (see the comments next to each figure).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 10})
C = {"ap": "#DDECF7", "lat": "#FCE8D5", "fusion": "#DFEEDF", "out": "#E9E3F2", "neutral": "#F0F3F6",
     "warn": "#FBE3E1"}
EDGE = "#536372"
AP_C, LAT_C = "#2a78d6", "#eb6834"


def canvas(w, h):
    f, a = plt.subplots(figsize=(w, h))
    a.set(xlim=(0, w), ylim=(0, h))
    a.axis("off")
    return f, a


def box(a, x, y, w, h, title, detail="", color="neutral", fs=10.5, dfs=8.8):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.02,rounding_size=.07", fc=C[color], ec=EDGE, lw=1))
    a.text(x + w / 2, y + h * (.68 if detail else .5), title, ha="center", va="center", weight="bold", fontsize=fs)
    if detail:
        a.text(x + w / 2, y + h * .3, detail, ha="center", va="center", fontsize=dfs, linespacing=1.15)


def edge(a, *pts, dashed=False):
    for p, q in zip(pts[:-2], pts[1:-1]):
        a.plot([p[0], q[0]], [p[1], q[1]], color=EDGE, lw=1.2, ls="--" if dashed else "-")
    a.annotate("", xy=pts[-1], xytext=pts[-2],
               arrowprops=dict(arrowstyle="->", lw=1.2, color=EDGE, linestyle="--" if dashed else "-"))


def save(f, name):
    f.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    f.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(f)


# ---------------------------------------------------------------- overview of the study
f, a = canvas(9.4, 5.2)
box(a, .1, 3.95, 2.7, 1.05, "CT collections", "TotalSegmentator:\nfemur, hip, rib\nVerSe19, VerSe20, CTPelvic1K", "neutral", dfs=8.4)
box(a, 3.3, 3.95, 2.9, 1.05, "Preprocessing (NAAMII)", "crop around the bone, reorient (PIR),\nAP + lateral DRRs (ITK ray casting),\nmask of the target bone", "neutral", dfs=8.4)
box(a, 6.7, 3.95, 2.6, 1.05, "Paired samples", "AP PNG, lateral PNG,\nNIfTI mask", "neutral", dfs=8.4)
edge(a, (2.8, 4.47), (3.3, 4.47)); edge(a, (6.2, 4.47), (6.7, 4.47))
box(a, 6.7, 2.35, 2.6, 1.05, "Split manifests", "seed 42, by sample folder,\n80% / 10% / 10%;\nCTPelvic copies removed", "neutral", dfs=8.4)
edge(a, (8.0, 3.95), (8.0, 3.4))
box(a, 3.3, 2.35, 2.9, 1.05, "18 training runs", "3 models x 6 datasets, same\nbudget; views placed by the\nX-ray geometry; best val. Dice", "fusion", dfs=8.4)
edge(a, (6.7, 2.87), (6.2, 2.87))
box(a, .1, 2.35, 2.7, 1.05, "Test evaluation", "run once (17 Nov 2026);\nDice, IoU, Prec., Rec.,\nHD95, ASSD, NSD", "out", dfs=8.4)
edge(a, (3.3, 2.87), (2.8, 2.87))
box(a, .1, .55, 2.7, 1.2, "Paired statistics", "per-case Dice differences,\nbootstrap CI, Wilcoxon test,\nHolm correction (12 tests)", "out", dfs=8.4)
edge(a, (1.45, 2.35), (1.45, 1.75))
save(f, "overview")

# ---------------------------------------------------------------- lifting geometry
# Volume axes follow utils/data/dataset.py: tensor (D, H, W) = (anterior-posterior,
# superior-inferior, left-right).  The AP image spans (H, W); the lateral image spans (H, D).
# MultiScale: models/multiscale_dualview_cnn.py::_lift_ap / _lift_lateral.
# DualViewCNN: models/transvert.py::DualViewCNN.forward (lateral slab permuted back so that the
# image's horizontal axis lands on W and the learned expansion fills D).
# SwinX2S (ours): models/swinx2s.py::_inflate_list_to_3d (same mapping for both views).
k = 0.45  # oblique projection: a step of 1 along D moves (k cos30, k sin30) on the page


def P(d, h, w):
    return (w + k * 0.866 * d, h + k * 0.5 * d)


def quad(a, pts, fc, alpha, ec=None, lw=0.8, z=1):
    a.add_patch(Polygon([P(*p) for p in pts], closed=True, fc=fc, ec=ec or fc, alpha=alpha, lw=lw, zorder=z))


def cube(a):
    L = 1.0
    corners = [(d, h, w) for d in (0, L) for h in (0, L) for w in (0, L)]
    for p in corners:
        for q in corners:
            if sum(x != y for x, y in zip(p, q)) == 1 and p < q:
                hidden = (p[0] == L and q[0] == L and (p[1] == 0 or p[2] == 0) and (q[1] == 0 or q[2] == 0)) \
                    or (p == (0, 0, 0) and q[0] == L) or (p[0] == 0 and p[1] == 0 and p[2] == 0)
                a.plot(*zip(P(*p), P(*q)), color="#8a949e", lw=0.8, ls=":" if hidden else "-", zorder=5)


def axes_labels(a):
    a.annotate("", xy=P(0, -.18, 1.05), xytext=P(0, -.18, 0), arrowprops=dict(arrowstyle="<->", color=EDGE, lw=.9))
    a.text(*P(0, -.32, .5), "W (left-right)", ha="center", va="center", fontsize=7.5, color=EDGE)
    a.annotate("", xy=P(0, 1.02, -.12), xytext=P(0, 0, -.12), arrowprops=dict(arrowstyle="<->", color=EDGE, lw=.9))
    a.text(*P(0, .5, -.2), "H (S-I)", ha="right", va="center", fontsize=7.5, color=EDGE, rotation=90)
    a.annotate("", xy=P(1.05, -.12, 1.12), xytext=P(0, -.12, 1.12), arrowprops=dict(arrowstyle="<->", color=EDGE, lw=.9))
    a.text(*P(.62, -.24, 1.3), "D (A-P)", ha="left", va="center", fontsize=7.5, color=EDGE)


def panel(a, x0, title, kind):
    a.set_xlim(-.45, 1.75); a.set_ylim(-.55, 2.05); a.axis("off"); a.set_aspect("equal")
    a.set_title(title, fontsize=8.6, loc="left", pad=2)
    if kind == "geometry":
        quad(a, [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)], AP_C, .25, z=2)          # AP image plane (H,W)
        quad(a, [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)], LAT_C, .25, z=2)         # lateral plane (H,D)
        a.annotate("", xy=P(0, .55, .45), xytext=P(1.25, .55, .45),
                   arrowprops=dict(arrowstyle="->", color=AP_C, lw=1.6))
        a.text(*P(0, 1.42, -.1), "AP rays along D", fontsize=7.2, color=AP_C, ha="left", va="bottom")
        a.annotate("", xy=P(.5, .3, 1.0), xytext=P(.5, .3, -.35),
                   arrowprops=dict(arrowstyle="->", color=LAT_C, lw=1.6))
        a.text(*P(.5, .2, -.42), "lateral rays\nalong W", fontsize=7.2, color=LAT_C, ha="right", va="top")
    elif kind == "consistent":
        for t in (0, .5, 1):
            quad(a, [(t, 0, 0), (t, 1, 0), (t, 1, 1), (t, 0, 1)], AP_C, .18, ec=AP_C, z=2)
            quad(a, [(0, 0, t), (0, 1, t), (1, 1, t), (1, 0, t)], LAT_C, .18, ec=LAT_C, z=3)
        a.text(*P(0, 1.42, -.1), "AP copied along D (not reversed L-R),\nlateral copied along W", fontsize=7.2, ha="left", va="bottom")
    else:
        for t in (0, .5, 1):
            quad(a, [(t, 0, 0), (t, 1, 0), (t, 1, 1), (t, 0, 1)], AP_C, .14, ec=AP_C, z=2)
            quad(a, [(t, .08, .08), (t, .92, .08), (t, .92, .92), (t, .08, .92)], LAT_C, .14, ec=LAT_C, z=3)
        a.text(*P(0, 1.42, -.1), "AP as in (b); lateral A-P axis\nplaced on W, expanded along D", fontsize=7.2, ha="left", va="bottom")
    cube(a)
    axes_labels(a)


f, axs = plt.subplots(1, 3, figsize=(7.0, 2.6))
panel(axs[0], 0, "(a) Projection geometry", "geometry")
panel(axs[1], 0, "(b) MultiScale", "consistent")
panel(axs[2], 0, "(c) DualViewCNN, our SwinX2S", "implemented")
f.tight_layout(w_pad=0.4)
save(f, "lifting")

# ---------------------------------------------------------------- our SwinX2S implementation
# models/swinx2s.py: one SwinTransformer (shared by both views); _inflate_list_to_3d expands
# each 2D feature map along D with a 1x1x1 Conv3d (initialised to ones), identically for both
# views; encoder1..encoder10 and decoder1..decoder5 are shared between the views; one TransFusion
# cross-attention block at the 1/32 bottleneck (models/swinX2S/transfusion.py: LayerNorm +
# cross-attention, no feed-forward layer, no residual); the two decoded volumes are concatenated
# and fused by a 1x1x1 Conv3d.  (v2: lateral inflation along W, averaged outputs, instance norm, Tiny only)
f, a = canvas(7.4, 8.3)
box(a, .3, 7.35, 2.6, .65, "AP image", "", "ap")
box(a, 4.5, 7.35, 2.6, .65, "Lateral image", "", "lat")
box(a, 1.6, 6.05, 4.2, .95, "2D Swin encoder (shared)", "Tiny: C=32, patch 2, window 7x7,\ndepths (2,2,6,2), drop path up to 0.2", "neutral")
edge(a, (1.6, 7.35), (2.6, 7.0)); edge(a, (5.8, 7.35), (4.8, 7.0))
box(a, 1.6, 4.75, 4.2, .95, "2D-to-3D expansion (6 levels)", "AP: copied along D; lateral: columns\nresized to D, copied along W", "neutral")
edge(a, (3.7, 6.05), (3.7, 5.7))
box(a, 1.6, 3.45, 4.2, .95, "3D residual encoder blocks", "shared weights, applied to each view", "neutral")
edge(a, (3.7, 4.75), (3.7, 4.4))
box(a, 1.6, 2.15, 4.2, .95, "Cross-view attention", "one block at 1/32 scale,\nno feed-forward layer, no residual", "fusion")
edge(a, (3.7, 3.45), (3.7, 3.1))
box(a, 1.6, .85, 4.2, .95, "3D decoder (shared)", "five upsampling blocks, instance norm;\none output volume per view", "neutral")
edge(a, (3.7, 2.15), (3.7, 1.8))
edge(a, (5.8, 3.92), (6.5, 3.92), (6.5, 1.32), (5.8, 1.32))
a.text(6.6, 2.6, "skip\nconnections", fontsize=8.5, color=EDGE, va="center", ha="left")
box(a, 1.6, -.35, 4.2, .8, "Average the two outputs", "as in the official code", "out")
edge(a, (3.7, .85), (3.7, .45))
a.set_ylim(-.45, 8.3)
save(f, "swinx2s_implementation")

# ---------------------------------------------------------------- CNN schematics (no in-figure title)
f, a = canvas(9, 6.2)
for x, c, name, detail in [(.35, "ap", "AP view", "Learned depth expansion"),
                           (5.15, "lat", "Lateral view", "Resize to D, expand along W")]:
    box(a, x, 5.25, 3.5, .7, name, "Input projection", c)
    box(a, x, 3.9, 3.5, .8, "Extract image features", "Two strided convolutions", c)
    box(a, x, 2.55, 3.5, .8, "Form 3D features", detail, c)
    edge(a, (x + 1.75, 5.25), (x + 1.75, 4.7)); edge(a, (x + 1.75, 3.9), (x + 1.75, 3.35))
    edge(a, (x + 1.75, 2.55), (x + 1.75, 2.2), (4.5, 2.2), (4.5, 1.85))
box(a, 2.9, 1.1, 3.2, .75, "Fuse + decode", "Concatenate + 3D upsampling", "fusion")
edge(a, (4.5, 1.1), (4.5, .75)); box(a, 2.9, .15, 3.2, .6, "3D prediction", "Single-channel logits", "out")
save(f, "cnn_dualview_architecture")

f, a = canvas(9, 8.7)
box(a, .2, 7.65, 2.65, .85, "AP encoder", "Four feature resolutions", "ap")
box(a, 6.15, 7.65, 2.65, .85, "Lateral encoder", "Four feature resolutions", "lat")
a.text(4.5, 8.02, "Independent\nview branches", ha="center", va="center", fontsize=10.5)
for i, (y, ch, scale) in enumerate([(6.05, 128, "1/8"), (4.55, 64, "1/4"), (3.05, 32, "1/2"), (1.55, 16, "Full")]):
    for x, c, lift in [(.2, "ap", "copy along D"), (6.15, "lat", "copy along W")]:
        box(a, x, y, 2.65, .85, "Decode + lift", f"{ch} channels, {lift}", c)
        edge(a, (x + 1.325, 7.65 if i == 0 else y + 1.5), (x + 1.325, y + .85))
    box(a, 3.35, y, 2.3, .85, "Fuse in 3D", f"{scale} size, {ch} channels", "fusion")
    edge(a, (2.85, y + .425), (3.35, y + .425)); edge(a, (6.15, y + .425), (5.65, y + .425))
    if i:
        edge(a, (4.5, y + 1.5), (4.5, y + .85))
    if i < 3:
        a.text(4.68, y - .32, "upsample", fontsize=9, va="center", color=EDGE)
edge(a, (4.5, 1.55), (4.5, 1.05)); box(a, 3.1, .3, 2.8, .75, "3D prediction", "1x1x1 head, logits", "out")
save(f, "cnn_multiscale_architecture")
print("figures written to", OUT)
