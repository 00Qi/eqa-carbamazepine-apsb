# Code index (self-contained mirror)

Primary working copies with outputs also live in `../phase2_analysis/`, `../round5_analysis/`, `../round3sim_analysis/`.  
This `code/` tree is a clean script mirror so nothing exists only on GitHub.

**GitHub mirror:** https://github.com/00Qi/eqa-carbamazepine-apsb

**Input data:** `../data/EQA_with_LabInfo_integrated.csv` (see `../data/README.md`). Set `APSB_DATA` to override.

## `phase2/` — primary LMM / robust / rlmer

| Script | Needs | Writes (typical) |
|---|---|---|
| `run_phase2_robust.R` | data CSV; R packages `lme4`, `lmerTest`, … | S1/S1b style coef tables, LOO, bootstrap |
| `run_rlmer_phase2.R` | data CSV; `robustlmm` | `S1b_rlmer_*.csv` |

Run from `phase2_analysis/` or set paths via package `data/`.

## `round3sim/` — common-mode + risk (third-round mock review)

| Script | Needs | Writes |
|---|---|---|
| `run_common_mode_round3sim.py` | data CSV; Python `pandas`/`numpy` | `Table_S_common_mode*`, decision JSON |
| `run_risk_round3sim.py` | data CSV; `sklearn` optional for logistic | `Table_S_risk*` |
| `build_table_s2.py` | data CSV | Table S2 ranking helpers |

## `round5/` — figure regen / LMM R2 / PSM check

| Script | Needs | Notes |
|---|---|---|
| `S&F_0527_fixed_round5.R` (= b/c snapshot) | `setwd` to folder containing CSV (historically `D:/Dropbox/EQA`) | Full pipeline archetype; **not** ML/SMOTE primary |
| `run_lmm_s1b_r2.R` | CSV in cwd | ICC/R2 add-on |
| `regen_fig3.R` | CSV; `ggplot2` | Regenerates Figure 3 |
| `regen_love_s1.R` | CSV | Love plot / Figure S1 |
| `psm_check.py` | path rewrite helper | Diagnostic |

## `historical/` — archival only (not manuscript primary)

- `S&F_0527.R` — earlier full script  
- `EQA_ML_Analysis_Final.py`, `EQA_PSM_Analysis_Final.py` — **not** used as APSB primary (standing: no 0526 ML/SMOTE claims)

## `utils/` — one-off feasibility / earlier S1b helpers

- `feasibility_alpha_delta_nested_R2_20261021.py`  
- `run_lmm_s1b.R`  
- `build_table_s2_src.py` — earlier Table S2 builder (prefer `round3sim/build_table_s2.py`)
