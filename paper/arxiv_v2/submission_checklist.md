# Version 2: checklist before posting (17 Nov 2026)

Nothing has been uploaded or submitted.

## Files
| Item | Status |
|---|---|
| `arxiv_v2.zip` (source + `anc/`) | **Ready.** 18 source + 111 ancillary files, 2.0 MB. Compiles standalone with 0 errors (24 pages). |
| `final_paper_v2.pdf` | The compiled paper, for reading. |
| `arxiv_metadata.txt` | Title, categories, comments and plain-text abstract (1,845 characters; the limit is 1,920). |
| `CHANGES_V2.md` | What changed from version 1 and why. |
| Major report version | `paper/major_report_version_v2/major_report_paper_v2.pdf` (23 pages). It does not mention duplicates or the earlier version, and has one neutral sentence about the test split being scored before. |

## Things only you can do
1. **[YOU] Upload the 18 new models to Hugging Face** (`jeevan1234/biplanar-bone-reconstruction-models`). This needs your HF token.
   - Until then, the paper correctly says the models "will be available".
   - After the upload, change "will be available" to "are available" in the Code and Data Availability section.
   - The run folders are listed in `scripts/evaluate_fair.py` (`RUNS`).
2. **[YOU] Authors:** the paper still has no author names. The saved block is in `author_block_saved.tex`.
3. **[YOU] If version 1 is already on arXiv,** use **Replace** (not a new submission) and keep the "Version 2: …" comment. Otherwise, submit this as the first version; the paper still explains the earlier version, because its test split was used once before.
4. **[YOU] Endorsement, licence, funding and ethics:** unchanged from version 1. See `paper/arxiv_submission/submission_checklist.md`, items 3–5.
5. **[YOU] Push the exact code to GitHub** and tag it, for example `v2-arxiv`. The fixes are in `models/`, `utils/`, `configs/config.py` (fair configs), `train.py` and `scripts/`.
6. **[YOU] Add the Swin-X2S MIT notice** to the GitHub and Hugging Face repositories. The text is at the end of `anc_README.md`.

## Scientific status
- Validation was rescored first for all 18 runs (max difference 1.1e-5). The test split was scored once (17 Nov 2026).
- No setting was chosen using test results. The split had been scored once before (10 Nov) for the earlier version, and this is disclosed in the paper.
- Remaining limitations, all stated in the paper:
  - simulated X-ray images only;
  - splits are by sample;
  - one seed;
  - learning rates not tuned;
  - two SwinX2S runs were stopped by the time limit at epochs 195 and 190, with the learning rate already about 0.
