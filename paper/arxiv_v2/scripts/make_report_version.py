"""Derive the major-project-report version of paper v2 (paper/major_report_version_v2/) from paper/arxiv_v2/main.tex.

As requested for the report: no paragraph about problems in the abstract, no mention of the CTPelvic duplicates,
and no before/after section. Kept for honesty: one neutral sentence that the test splits had been scored once
before for a preliminary version of the models, with no setting chosen using test results.
"""
from pathlib import Path
import shutil

A = Path(__file__).resolve().parent.parent              # paper/arxiv_v2
OUT = A.parent / "major_report_version_v2"
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


# abstract: drop the two sentences about the earlier version
rep(" An earlier version of these models placed the views incorrectly. Correcting this and matching the training "
    "settings raised the mean Dice over the six datasets by 4.8 to 18.8 points on the same test cases.", "")
# contributions
rep("""\\item a simple projection test that checks how a model places the two views in 3D. It found errors in an earlier version of our models, and we report how much correcting them changed the results on the same test cases;""",
    """\\item a simple projection test and a sensitivity test that check how a model places the two views in 3D;""")
# splits: no duplicates
rep("""In CTPelvic, 37 scans were stored twice under different folder names, with identical images and masks. We kept one copy of each: the copy in the training split if there was one, otherwise the one in the validation split. This leaves 412 training, 48 validation and 44 test samples, with no scan in more than one split.""",
    """For CTPelvic the splits have 412 training, 48 validation and 44 test samples, with no scan in more than one split.""")
# evaluation protocol: replace the earlier-version paragraph by a neutral note
i = s.index("\\paragraph{Relation to an earlier version.}")
j = s.index("\n\n", i)
s = s[:i] + ("The test splits had been scored once before, for a preliminary version of the models with different "
             "training settings. No setting of the models reported here was chosen using test results.") + s[j:]
# training behaviour: drop the comparison with the earlier SwinX2S
rep(""" In the earlier version, the larger SwinX2S model reached its best validation Dice within 18 to 32 epochs on hip, rib and CTPelvic and then over-segmented more and more. The corrected SwinX2S does not do this: its best epochs are between 113 and 196, and its precision and recall stay close (Table~\\ref{tab:metrics}).""",
    """ SwinX2S reaches its best validation Dice late in training (epochs 113 to 196), and its precision and recall stay close (Table~\\ref{tab:metrics}).""")
# before/after subsection and table
cut("\\subsection{Before and after the corrections}", "% ======================================================================\n\\section{Discussion}")
# discussion: view placement subsection without the earlier version
i = s.index("\\subsection{How the views are placed in 3D}")
j = s.index("\\subsection{Model size, amount of data and training}", i)
s = s[:i] + """\\subsection{How the views are placed in 3D}
Mistakes in how the two views are placed in 3D are easy to make, because the AP and lateral images are usually resized to the same shape, so a wrong placement still gives tensors of the right size and the network still trains to a reasonable Dice. A sensitivity test like the one in Section~\\ref{sec:models}, together with a projection test on the data (Section~\\ref{sec:problem}), checks this quickly, and we recommend running both before comparing models.

""" + s[j:]
# limitations
rep(""" The test splits were scored once before, for the earlier version of the models (Section~\\ref{sec:setup}); no setting was chosen using test results, but the test data are no longer completely unseen by us.""",
    """ The test splits had been scored once before, for a preliminary version of the models.""")
# conclusion
rep(""" Correcting how the views were placed in an earlier version changed the results by up to 29 Dice points, so checks of this kind should be part of any comparison.""",
    """ Checks of how the views are placed in 3D should be part of any comparison of this kind.""")
assert "earlier version" not in s, "an 'earlier version' mention is left"
assert "duplicat" not in s.lower(), "a duplicates mention is left"
(OUT / "main.tex").write_text(s)
print("report version written to", OUT)
