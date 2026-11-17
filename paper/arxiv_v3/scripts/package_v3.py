"""Assemble the arXiv source package.

arxiv_v3.zip contains main.tex, main.bbl, references.bib, the tables and figures that main.tex uses, and anc/
(ancillary evidence; see anc_README.md). Build the paper first:
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
The code, results, splits and checks in anc/ are taken from the clean code repository
(github.com/Jeevan-Neupane/biplanar-bone-reconstruction, checked out next to this repository).
"""
from pathlib import Path
import re
import shutil
import zipfile

A = Path(__file__).resolve().parent.parent          # paper/arxiv_v3
P = A.parent                                         # paper/
R = P.parent                                         # repository root
C = R.parent / "biplanar-bone-reconstruction"         # clean code repository
EV = C / "results"

shutil.copy2(A / "build" / "main.bbl", A / "main.bbl")
shutil.copy2(A / "build" / "main.pdf", A / "final_paper_v3.pdf")

tex = (A / "main.tex").read_text()
figures = sorted({A / f for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)})
tables = sorted({A / (t if t.endswith(".tex") else t + ".tex") for t in re.findall(r"\\input\{([^}]+)\}", tex)})
source = [A / "main.tex", A / "main.bbl", A / "references.bib"] + tables + figures
for f in source:
    assert f.exists(), f

anc = {"anc/README.md": A / "anc_README.md",
       "anc/results_fair.json": EV / "results_fair.json"}
for split in ["val", "test"]:
    for f in sorted((EV / split).glob("*.json")):
        anc[f"anc/{split}/{f.name}"] = f
for f in sorted((C / "splits").glob("*.json")):
    anc[f"anc/splits/{f.name}"] = f
for f in ["data_flip_ap.json"] + sorted(p.name for p in (EV / "geometry_check").glob("models_*.json")
                                         if "final" not in p.name):
    anc[f"anc/geometry_check/{f}"] = EV / "geometry_check" / f
# training logs of the 18 runs (the run folders are listed in the evidence JSONs)
import json
for f in sorted((EV / "test").glob("fair_*.json")):
    run = Path(json.loads(f.read_text())["run_dir"]).name
    for ext in [".csv", ".log"]:
        src = [p for p in [R / "logs" / f"{run}{ext}", R / "modal_results_second" / "results" / "logs" / f"{run}{ext}"]
               if p.exists()]
        assert src, f"missing training log {run}{ext}"
        anc[f"anc/logs/{run}{ext}"] = src[0]
for name in ["evaluate_fair.py", "analyze_fair.py", "evaluate_checkpoints.py", "check_geometry.py",
             "make_clean_splits.py", "modal_fair.py", "run_fair_queue.sh"]:
    anc[f"anc/scripts/{name}"] = C / "scripts" / name
for name in ["make_fair_tables.py", "make_figures.py"]:
    anc[f"anc/scripts/{name}"] = A / "scripts" / name
for rel in ["train.py", "configs/config.py", "configs/args_parser.py", "utils/data/dataset.py",
            "utils/training/losses.py", "utils/training/metrics.py", "utils/training/trainer.py",
            "utils/training/checkpoint.py", "utils/training/logger.py", "models/transvert.py",
            "models/multiscale_dualview_cnn.py", "models/swinx2s.py", "models/swinX2S/swin_transformer.py",
            "models/swinX2S/transfusion.py", "models/swinX2S/unetr_blocks.py"]:
    anc[f"anc/code/{rel}"] = C / rel

allowed = re.compile(r"^[A-Za-z0-9_+\-.,=/]+$")
for name in list(anc) + [str(f.relative_to(A)) for f in source]:
    assert allowed.match(name), f"arXiv does not allow this file name: {name}"

out = A / "arxiv_v3.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in source:
        z.write(f, f.relative_to(A))
    for name, f in anc.items():
        z.write(f, name)
print(f"{out.name}: {len(source)} source files + {len(anc)} ancillary files, {out.stat().st_size / 1e6:.1f} MB")
