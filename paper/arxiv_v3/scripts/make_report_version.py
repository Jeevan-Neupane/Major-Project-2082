"""Derive the major-project-report version of the paper (paper/major_report_v3/) from paper/arxiv_v3/main.tex.

As requested for the report: no mention of the CTPelvic duplicates.
"""
from pathlib import Path
import shutil

A = Path(__file__).resolve().parent.parent              # paper/arxiv_v3
OUT = A.parent / "major_report_v3"
OUT.mkdir(exist_ok=True)
for d in ["figures", "tables"]:
    shutil.copytree(A / d, OUT / d, dirs_exist_ok=True)
shutil.copy2(A / "references.bib", OUT / "references.bib")
s = (A / "main.tex").read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


def cut(start, end):
    """Remove from `start` up to (not including) `end`."""
    global s
    i = s.index(start)
    j = s.index(end, i)
    s = s[:i] + s[j:]


# splits: no duplicates
rep("""In CTPelvic, 37 scans were stored twice under different folder names, with identical images and masks. We kept one copy of each: the copy in the training split if there was one, otherwise the one in the validation split. This leaves 412 training, 48 validation and 44 test samples, with no scan in more than one split.""",
    """For CTPelvic the splits have 412 training, 48 validation and 44 test samples, with no scan in more than one split.""")
# overview figure without the duplicates note
rep("{figures/overview.pdf}", "{figures/overview_report.pdf}")
assert "earlier version" not in s, "an 'earlier version' mention is left"
assert "duplicat" not in s.lower(), "a duplicates mention is left"
(OUT / "main.tex").write_text(s)
print("report version written to", OUT)
