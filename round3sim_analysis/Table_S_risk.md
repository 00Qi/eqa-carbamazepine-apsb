# Table S_risk. 2020 holdout risk enrichment (Round-3sim reproducibility)

**Cohort.** N=131 laboratories; events=23 (17.6%) with ≥1 PT-defined deviation (abs(New_Bias%)>15).
**Design.** Train 2014–2019; test 2020. Historical PT features use only pre-2020 records. Cold-start labs (no pre-2020 history) scored 0 on heuristics (ranked last).
**Bootstrap.** Lab resample on the 2020 holdout, B=2000, seed=20261030.

## A. Performance with 95% CIs (Capture at 10/20/30/40%)

| Scorer | AUC (95% CI) | PR-AUC (95% CI) | Cap@10% | Cap@20% | Cap@30% | Cap@40% |
|---|---|---|---:|---:|---:|---:|
| Historical failure-rate | 0.640 (0.529–0.749) | 0.235 (0.148–0.397) | 17.4% | 34.8% | 56.5% | 60.9% |
| Historical max abs bias | 0.641 (0.506–0.769) | 0.339 (0.191–0.521) | 21.7% | 30.4% | 52.2% | 60.9% |
| Regularized logistic | 0.623 (0.504–0.737) | 0.264 (0.150–0.411) | 8.7% | 21.7% | 43.5% | 52.2% |
| Blended | 0.669 (0.542–0.785) | 0.341 (0.185–0.529) | 13.0% | 34.8% | 56.5% | 65.2% |

## B. Incremental value vs simple historical failure-rate

| Scorer | ΔAUC | ΔCap@10% | ΔCap@20% | ΔCap@30% | ΔCap@40% |
|---|---:|---:|---:|---:|---:|
| Historical failure-rate | +0.000 | +0.0 pp | +0.0 pp | +0.0 pp | +0.0 pp |
| Historical max abs bias | +0.001 | +4.3 pp | -4.3 pp | -4.3 pp | +0.0 pp |
| Regularized logistic | -0.018 | -8.7 pp | -13.0 pp | -13.0 pp | -8.7 pp |
| Blended | +0.029 | -4.3 pp | +0.0 pp | +0.0 pp | +4.3 pp |

## C. Cold-start

- Cold-start labs (no pre-2020 history): **17** (events=3).
- Warm labs: **114** (events=20).
- Cold labs placed in top-20% by failure-rate ranking: 0 (cold events captured there: 0) — expected near zero under score=0 rule.

Warm-only AUC/Capture@20% (labs with history):

| Scorer | AUC | Cap@20% (within warm) |
|---|---:|---:|
| Historical failure-rate | 0.660 | 35.0% |
| Historical max abs bias | 0.687 | 35.0% |
| Regularized logistic | 0.633 | 25.0% |
| Blended | 0.705 | 30.0% |

## D. Leakage checklist

- Historical PT summaries: **pre-2020 only** (computed only from Year < prediction year (cum_fail_rate, hist_max_abs, slopes, etc.)).
- Practice flags in logistic FEATURE_COLS: **QA/SIL/CNAS/MV/bal_num/is_CRO taken from CURRENT lab-year row in build_feature_rows — for 2020 holdout these are 2020 qu...**
- `n_rec_year` in logistic features: disclosed (included in FEATURE_COLS (same-year record count) — potential weak leakage for logistic; heuristics (fail/max) do NOT use it).
- Heuristic scorers (failure-rate / max abs): **no same-year outcome leakage**.

## E. Decision

- Enrichment vs random at 20% audit budget: **YES** (Capture@20%=34.8% vs 20% random; AUC=0.640, 95% CI 0.529–0.749).
- Complex score beats simple failure-rate on Capture@20%: **NO**.
- Complex score beats simple failure-rate on AUC (>+0.02): **YES**.
- Primary triage signal for manuscript: **Historical failure-rate (simple rule); blended offers little Capture gain**.

Not a regulatory deployment claim.