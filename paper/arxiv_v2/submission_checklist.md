# Checklist before posting (17 Nov 2026)

Nothing has been uploaded to arXiv.

## Files
| Item | Status |
|---|---|
| `arxiv_v2.zip` (source + `anc/`) | **Ready.** Compiles standalone with 0 errors (23 pages). The extra files in `anc/` come from the clean code repository. |
| `final_paper_v2.pdf` | The compiled paper, for reading. |
| `arxiv_metadata.txt` | Title, categories, comments and plain-text abstract (1,640 characters; the limit is 1,920). |
| Major report version | `paper/major_report_version_v2/major_report_paper_v2.pdf` (23 pages). Same as the paper, without the CTPelvic duplicates sentence. |

## Things only you can do
1. **[YOU] Upload the 18 models to Hugging Face** (`jeevan1234/biplanar-bone-reconstruction-models`). This needs your HF token (tmux session `hf-upload`).
   - Until then, the paper correctly says the models "will be available".
   - After the upload, change "will be available" to "are available" in the Code and Data Availability section.
2. **[YOU] Authors:** the paper still has no author names. The saved block is in `author_block_saved.tex`.
3. **[YOU] Submit as a new arXiv submission.** Our records show nothing was posted on arXiv before.
4. **[YOU] Endorsement, licence, funding and ethics:** see `paper/arxiv_submission/submission_checklist.md`, items 3–5.

## Done
- Code repository: https://github.com/Jeevan-Neupane/biplanar-bone-reconstruction (release v1.0). It includes the
  Swin-X2S MIT notice (`THIRD_PARTY_NOTICES.md`).
- The cleaned code gives bit-identical outputs to the training code for all 18 trained models.

## Scientific status
- Validation was rescored first for all 18 runs (max difference 1.1e-5). The test split was scored once for these models (17 Nov 2026).
- No model or training setting was chosen using test results.
- Remaining limitations, all stated in the paper:
  - simulated X-ray images only;
  - splits are by sample;
  - one seed;
  - learning rates not tuned;
  - two SwinX2S runs were stopped by the time limit at epochs 195 and 190, with the learning rate already about 0.
