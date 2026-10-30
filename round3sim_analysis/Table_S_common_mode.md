# Table S_common_mode. Signed common-mode vs level-specific bias variance (Round-3sim)

**Definition.** For each complete laboratory-year (five concentration levels), signed relative bias
`New_Bias%` is decomposed as \(b_{il} = \alpha_i + \delta_{il}\), where \(\alpha_i\) is the
across-level mean (common mode) and \(\delta_{il}\) the level-specific residual.
Share = Var(\(\alpha\)) / [Var(\(\alpha\)) + mean_i Var(\(\delta\))].
Assigned value = robust median of participating laboratories (consensus). Unit = ng/mL for concentrations; bias in %.

**Bootstrap.** Within each year, laboratories were resampled with replacement (B=2000); percentile 95% CIs for share and Var(\(\alpha\)).

**Subsets.** (1) all complete laboratory-years; (2) laboratories participating in ≥3 calendar years; (3) laboratories present in both 2014 and 2020 (n=31).

## A. All complete laboratories (primary)

| Year | N labs | Var(α) | Var(α) 95% CI | mean Var(δ) | share | share 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| 2014 | 47 | 124.9 | 34.3–283.1 | 27.3 | 0.820 | 0.627–0.878 |
| 2015 | 68 | 92.3 | 58.7–129.5 | 58.6 | 0.612 | 0.483–0.713 |
| 2016 | 67 | 188.1 | 30.9–482.2 | 974.2 | 0.162 | 0.135–0.761 |
| 2017 | 82 | 8068.1 | 30.3–23595.9 | 33656.7 | 0.193 | 0.189–0.721 |
| 2018 | 111 | 82.5 | 41.8–149.2 | 151.4 | 0.353 | 0.228–0.845 |
| 2019 | 133 | 57.5 | 30.9–92.7 | 137.4 | 0.295 | 0.176–0.804 |
| 2020 | 131 | 146.6 | 38.0–351.3 | 618.3 | 0.192 | 0.160–0.782 |

## B. Continuous participants (≥3 years; n unique labs in filter = 106)

| Year | N labs | Var(α) | Var(α) 95% CI | mean Var(δ) | share | share 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| 2014 | 45 | 130.0 | 34.3–296.0 | 28.0 | 0.823 | 0.623–0.880 |
| 2015 | 62 | 82.7 | 53.6–114.2 | 53.8 | 0.606 | 0.469–0.726 |
| 2016 | 62 | 199.3 | 27.2–504.0 | 1050.9 | 0.159 | 0.135–0.761 |
| 2017 | 78 | 8474.2 | 28.8–24746.7 | 35381.2 | 0.193 | 0.189–0.721 |
| 2018 | 100 | 84.2 | 40.8–158.0 | 166.2 | 0.336 | 0.218–0.843 |
| 2019 | 91 | 65.6 | 26.6–112.6 | 194.9 | 0.252 | 0.152–0.816 |
| 2020 | 85 | 202.9 | 40.2–507.1 | 945.4 | 0.177 | 0.149–0.799 |

## C. Present in both 2014 and 2020 (n=31)

| Year | N labs | Var(α) | Var(α) 95% CI | mean Var(δ) | share | share 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| 2014 | 31 | 159.1 | 33.5–377.3 | 29.1 | 0.845 | 0.607–0.894 |
| 2015 | 29 | 66.7 | 32.6–104.6 | 66.3 | 0.501 | 0.322–0.688 |
| 2016 | 29 | 360.3 | 19.7–975.1 | 2215.6 | 0.140 | 0.122–0.811 |
| 2017 | 29 | 51.1 | 30.8–72.2 | 14.7 | 0.777 | 0.619–0.878 |
| 2018 | 30 | 37.7 | 18.2–57.4 | 18.3 | 0.673 | 0.446–0.824 |
| 2019 | 29 | 36.2 | 17.4–55.9 | 159.0 | 0.185 | 0.053–0.809 |
| 2020 | 31 | 59.8 | 35.0–82.7 | 12.0 | 0.833 | 0.717–0.900 |

## Trend decision (Round-3sim)

- Relative common-mode **share** 2014→2020: **YES — declines across subsets**
  (all-complete share 0.820 → 0.192).
- Absolute Var(\(\alpha\)) decline 2014→2020: **NO — not supported**
  (all-complete Var(α) 124.9 → 146.6;
  early median 2014–16 = 124.9,
  late median 2018–20 = 82.5).
- 2017 absolute components remain outlier-inflated; interpret absolute Var(α) cautiously.
- PCA PC1 variance share remains exploratory support only for the relative pattern change.

**Manuscript implication.** Center the error-structure claim on the **relative** common-mode / PC1 share decline;
do **not** claim that absolute systematic common-mode variance declined industry-wide.
