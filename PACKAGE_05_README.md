# APSB submission package — carbamazepine EQA (NCCL)

Self-contained tree for manuscript + SI + analysis code + (coded) analysis CSV.  
**GitHub mirror:** https://github.com/00Qi/eqa-carbamazepine-apsb  
**Dropbox:** `D:\Dropbox\EQA\APSB\APSB_submission_1008` (machine 00Qi)

## Folder map

| Path | Contents |
|---|---|
| `00_Title_page.docx` | Title page |
| `01_Manuscript_anonymized.docx` | Blinded manuscript |
| `01_Manuscript_anonymized_*_TEXT.txt` | Round/phase plain-text extracts |
| `01b_Manuscript_full_identified_NOT_for_review.docx` | Identified MS (not for peer review) |
| `EQAManuscript_APSB_revised_clean_1008.docx` | Clean revised MS |
| `02_Graphical_abstract.png` + caption | Graphical abstract |
| `02b_Highlights.txt` | Highlights |
| `03_Cover_letter_draft.txt` | Cover letter draft |
| `04_Submission_checklist.md` | Checklist |
| `Abstract_trim_le200_*.txt` | Abstract ≤200 words |
| `Figure_legends_all.txt` | Figure legends |
| `Response_to_reviewers_*.md` | Response drafts |
| `改稿说明_*.md` | Revision notes (major / round2–5 / phase2 / round3sim) |
| `figures/` | Figure 1–5 (PNG) |
| `supplementary/` | Tables S0–S2, S0b, S1b, Figures S1–S5, Love plot |
| `phase2_analysis/` | Phase-2 LMM/rlmer **scripts + output CSVs/logs** |
| `round5_analysis/` | Round-5 scripts + PSM/Love SMD CSV |
| `round3sim_analysis/` | Round-3sim common-mode & risk scripts + tables |
| `code/` | Clean script mirror (phase2 / round5 / round3sim / historical / utils) — see `code/README.md` |
| `data/` | `EQA_with_LabInfo_integrated.csv` + sharing note — see `data/README.md` |

## Standing analysis locks

- Unit **ng/mL**; assigned value = robust median; `New_Bias%` signed relative bias.  
- Balance code **2** = High.  
- Primary: NCCL 2014–2020; NIFDC descriptive only.  
- Double-blind peer review materials as labeled.  
- **No** 0526 ML/SMOTE as primary claims (`code/historical/` archival only).  
- Round-3sim narrative center: **relative** common-mode share change + simple historical failure-rate risk enrichment; QA/SIL-IS exploratory; do **not** claim industry-wide absolute Var(α) decline.

## Recompute (examples)

```bash
# from package root
export APSB_DATA="$PWD/data/EQA_with_LabInfo_integrated.csv"

python round3sim_analysis/run_common_mode_round3sim.py
python round3sim_analysis/run_risk_round3sim.py

# R (micromamba env with lme4 / robustlmm as used in phase2)
cd phase2_analysis
Rscript run_phase2_robust.R
Rscript run_rlmer_phase2.R
```

Outputs land in the respective `*_analysis/` folders and/or `supplementary/` (see script headers).

## Release zip

GitHub release tag `round3sim-20261030` also carries a full-package zip for offline sync.
