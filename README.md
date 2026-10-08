# EQA carbamazepine PT analysis (APSB manuscript)

Analysis code accompanying the manuscript on decade-long Chinese external quality assessment (EQA) / proficiency testing for carbamazepine bioanalysis (2014–2024), focused on process control (independent QA), error-structure evolution (PCA), and laboratory-level risk stratification.

## Scripts

| File | Role |
|------|------|
| `S&F_0527.R` | Linear mixed-effects models, PCA, and main figures (R ≥ 4.4) |
| `EQA_PSM_Analysis_Final.py` | Propensity score matching (Python ≥ 3.11) |
| `EQA_ML_Analysis_Final.py` | Laboratory-level risk stratification / triage metrics |

## Data

Anonymized proficiency-testing records are **not** included in this repository. They are available from the corresponding author upon reasonable request, subject to institutional and program data-sharing policies.

Place `EQA_with_LabInfo_integrated.csv` in the working directory before running the R script (path expected by `S&F_0527.R`).

## Citation

Please cite the published APSB article (DOI to be added upon acceptance) and this repository.

## License

MIT (code only). Underlying PT data remain under program/institutional restrictions.
