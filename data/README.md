# Data (analysis input)

## File

- `EQA_with_LabInfo_integrated.csv` — NCCL carbamazepine EQA record-level table used for LMM, common-mode, and risk analyses (primary years 2014–2020).

## Identifiers / sharing note

- `Lab_ID` is a **coded** EQA laboratory identifier (not a hospital legal name). The author retains any crosswalk to real-world lab names.
- Columns include Region, Lab_type, and questionnaire practice fields (QA, SIL-IS, CNAS, Balance, etc.).
- For public redistribution beyond co-authors/reviewers, confirm NCCL data-use terms. This package copy is for **reproducibility of the APSB analyses**.

## Required columns (minimum)

`Year`, `Lab_ID`, `Sample`, `Conc`, `Sample_Level`, `Median_Conc`, `New_Bias%`,  
plus practice/method fields used in models: `QA`, `SIL-IS`, `CNAS`, `Method Validation`, `Balance` / `Balance_Precision`, `Lab_type`, `Region`, `Method`, `Quantitation_Method`, `Process_Method`.

## Unit / coding locks

- Concentration unit: **ng/mL**.
- Assigned value = robust median (`Median_Conc`).
- Relative bias = `New_Bias%`.
- Balance code **2** = High precision class (as used in primary coding).

## Override path

Scripts look under this folder first. Or set environment variable:

```bash
export APSB_DATA=/absolute/path/to/EQA_with_LabInfo_integrated.csv
```
