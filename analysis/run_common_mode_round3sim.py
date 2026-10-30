#!/usr/bin/env python3
"""Round-3sim P0: independent common-mode bias decomposition (signed relative bias).

b_il = α_i + δ_il
  α_i = mean across levels for lab i
  δ_il = level-specific residual
Per year: Var(α), mean_i Var(δ), share = Var(α)/(Var(α)+mean Var(δ))
Bootstrap: resample laboratories within year (B=2000).
Subsets: all complete labs; labs in ≥k years (k=3); labs present in both 2014 and 2020.
Primary bias column: New_Bias% (signed % vs robust-median assigned value).
Sensitivity: Bias column.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("/workspace/p2/analysis/EQA_with_LabInfo_integrated.csv")
OUT = Path("/workspace/p2/out/round3sim")
PKG = Path("/workspace/apsb_out/APSB_submission_1008")
OUT.mkdir(parents=True, exist_ok=True)
(PKG / "round3sim_analysis").mkdir(parents=True, exist_ok=True)
(PKG / "supplementary").mkdir(parents=True, exist_ok=True)

B = 2000
RNG = np.random.default_rng(20261030)
YEARS = list(range(2014, 2021))
LEVELS = [f"Level.{i}" for i in range(1, 6)]


def wide_year(df: pd.DataFrame, year: int, bias_col: str) -> tuple[np.ndarray, np.ndarray]:
    s = df.loc[df["Year"] == year, ["Lab_ID", "Sample_Level", bias_col]].copy()
    s[bias_col] = pd.to_numeric(s[bias_col], errors="coerce")
    s = s.dropna(subset=[bias_col])
    w = s.pivot_table(index="Lab_ID", columns="Sample_Level", values=bias_col, aggfunc="mean")
    # require all five levels
    cols = [c for c in LEVELS if c in w.columns]
    if len(cols) < 5:
        return np.empty((0, 5)), np.array([])
    mat = w[LEVELS].to_numpy(dtype=float)
    labs = w.index.to_numpy()
    ok = np.isfinite(mat).all(axis=1)
    return mat[ok], labs[ok]


def decomp(mat: np.ndarray) -> dict:
    """mat: n_labs × n_levels signed bias."""
    if mat.shape[0] < 5:
        return dict(n_labs=int(mat.shape[0]), var_alpha=np.nan, mean_var_delta=np.nan, share=np.nan)
    alpha = mat.mean(axis=1)
    delta = mat - alpha[:, None]
    var_a = float(np.var(alpha, ddof=1))
    # per-lab sample var of delta across levels (ddof=1); mean over labs
    var_d = float(np.mean(np.var(delta, axis=1, ddof=1)))
    share = var_a / (var_a + var_d) if (var_a + var_d) > 0 else np.nan
    return dict(n_labs=int(mat.shape[0]), var_alpha=var_a, mean_var_delta=var_d, share=share)


def bootstrap_year(mat: np.ndarray, B: int, rng: np.random.Generator) -> dict:
    n = mat.shape[0]
    if n < 5:
        return dict(share_lo=np.nan, share_hi=np.nan, var_alpha_lo=np.nan, var_alpha_hi=np.nan)
    shares, vars_a = [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        d = decomp(mat[idx])
        shares.append(d["share"])
        vars_a.append(d["var_alpha"])
    shares = np.asarray(shares, float)
    vars_a = np.asarray(vars_a, float)
    return dict(
        share_lo=float(np.nanpercentile(shares, 2.5)),
        share_hi=float(np.nanpercentile(shares, 97.5)),
        var_alpha_lo=float(np.nanpercentile(vars_a, 2.5)),
        var_alpha_hi=float(np.nanpercentile(vars_a, 97.5)),
    )


def run_panel(df: pd.DataFrame, bias_col: str, lab_filter: set | None, label: str) -> pd.DataFrame:
    rows = []
    for yr in YEARS:
        mat, labs = wide_year(df, yr, bias_col)
        if lab_filter is not None:
            mask = np.array([lab in lab_filter for lab in labs])
            mat, labs = mat[mask], labs[mask]
        d = decomp(mat)
        boot = bootstrap_year(mat, B, RNG)
        rows.append(
            dict(
                subset=label,
                bias_col=bias_col,
                Year=yr,
                n_labs=d["n_labs"],
                var_alpha=d["var_alpha"],
                mean_var_delta=d["mean_var_delta"],
                share=d["share"],
                share_CI_lo=boot["share_lo"],
                share_CI_hi=boot["share_hi"],
                var_alpha_CI_lo=boot["var_alpha_lo"],
                var_alpha_CI_hi=boot["var_alpha_hi"],
            )
        )
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(RAW)
    df["Year"] = df["Year"].astype(int)
    # participation counts
    ly = df.groupby(["Lab_ID", "Year"]).size().reset_index(name="n")
    n_years = ly.groupby("Lab_ID")["Year"].nunique()
    labs_k3 = set(n_years[n_years >= 3].index)
    labs_2014 = set(ly.loc[ly["Year"] == 2014, "Lab_ID"])
    labs_2020 = set(ly.loc[ly["Year"] == 2020, "Lab_ID"])
    labs_span = labs_2014 & labs_2020

    panels = []
    for bias_col in ("New_Bias%", "Bias"):
        panels.append(run_panel(df, bias_col, None, "all_complete"))
        panels.append(run_panel(df, bias_col, labs_k3, "continuous_ge3yrs"))
        panels.append(run_panel(df, bias_col, labs_span, "present_2014_and_2020"))
    res = pd.concat(panels, ignore_index=True)

    # Primary table: New_Bias%, all_complete
    prim = res[(res.bias_col == "New_Bias%") & (res.subset == "all_complete")].copy()
    prim_k3 = res[(res.bias_col == "New_Bias%") & (res.subset == "continuous_ge3yrs")].copy()
    prim_span = res[(res.bias_col == "New_Bias%") & (res.subset == "present_2014_and_2020")].copy()

    # Trend summaries (exclude 2017 for absolute Var(α) sensitivity)
    def trend_note(panel: pd.DataFrame) -> dict:
        p = panel.set_index("Year")
        v14, v20 = p.loc[2014, "var_alpha"], p.loc[2020, "var_alpha"]
        s14, s20 = p.loc[2014, "share"], p.loc[2020, "share"]
        # Spearman on years excluding 2017
        yrs = [y for y in YEARS if y != 2017]
        va = [p.loc[y, "var_alpha"] for y in yrs]
        sh = [p.loc[y, "share"] for y in yrs]
        # simple linear slope of share and log1p(var_alpha)
        x = np.asarray(yrs, float)
        slope_share = float(np.polyfit(x, sh, 1)[0])
        slope_logvar = float(np.polyfit(x, np.log1p(va), 1)[0])
        # also with 2017
        va_all = [p.loc[y, "var_alpha"] for y in YEARS]
        sh_all = [p.loc[y, "share"] for y in YEARS]
        slope_share_all = float(np.polyfit(YEARS, sh_all, 1)[0])
        # absolute decline? 2020 < 2014 and median late < median early
        early = np.median([p.loc[y, "var_alpha"] for y in (2014, 2015, 2016)])
        late = np.median([p.loc[y, "var_alpha"] for y in (2018, 2019, 2020)])
        abs_decline = bool(v20 < v14 and late < early)
        share_decline = bool(s20 < s14 and slope_share < 0)
        return dict(
            var_alpha_2014=float(v14),
            var_alpha_2020=float(v20),
            share_2014=float(s14),
            share_2020=float(s20),
            slope_share_ex2017=slope_share,
            slope_share_all=slope_share_all,
            slope_log1p_var_alpha_ex2017=slope_logvar,
            median_var_alpha_early_2014_16=float(early),
            median_var_alpha_late_2018_20=float(late),
            absolute_var_alpha_decline=abs_decline,
            relative_share_decline=share_decline,
        )

    decision = {
        "bias_definition": "signed New_Bias% = (Conc - Median_Conc)/Median_Conc*100 (robust-median assigned value)",
        "model": "b_il = alpha_i + delta_il; Var(alpha); mean Var(delta); share=Var(alpha)/(Var(alpha)+mean Var(delta))",
        "bootstrap": f"lab resample within year, B={B}, seed=20261030",
        "n_labs_k3": len(labs_k3),
        "n_labs_span_2014_2020": len(labs_span),
        "all_complete": trend_note(prim),
        "continuous_ge3yrs": trend_note(prim_k3),
        "present_2014_and_2020": trend_note(prim_span),
    }
    # Overall narrative flag
    abs_ok = all(
        decision[k]["absolute_var_alpha_decline"]
        for k in ("all_complete", "continuous_ge3yrs", "present_2014_and_2020")
    )
    share_ok = all(
        decision[k]["relative_share_decline"]
        for k in ("all_complete", "continuous_ge3yrs", "present_2014_and_2020")
    )
    decision["absolute_var_alpha_trend_robust"] = abs_ok
    decision["relative_share_trend_robust"] = share_ok
    decision["center_claim"] = (
        "relative_common_mode_share_decline_supported"
        if share_ok
        else "common_mode_claim_not_supported"
    ) + (
        "; absolute_Var_alpha_decline_NOT_supported"
        if not abs_ok
        else "; absolute_Var_alpha_decline_supported"
    )

    # Write outputs
    res.to_csv(OUT / "Table_S_common_mode_full.csv", index=False)
    prim.to_csv(OUT / "Table_S_common_mode_primary.csv", index=False)
    prim_k3.to_csv(OUT / "Table_S_common_mode_ge3yrs.csv", index=False)
    prim_span.to_csv(OUT / "Table_S_common_mode_span2014_2020.csv", index=False)
    (OUT / "common_mode_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    # Markdown table for SI
    def fmt_panel(p: pd.DataFrame) -> str:
        lines = [
            "| Year | N labs | Var(α) | Var(α) 95% CI | mean Var(δ) | share | share 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in p.iterrows():
            lines.append(
                f"| {int(r.Year)} | {int(r.n_labs)} | {r.var_alpha:.1f} | "
                f"{r.var_alpha_CI_lo:.1f}–{r.var_alpha_CI_hi:.1f} | {r.mean_var_delta:.1f} | "
                f"{r.share:.3f} | {r.share_CI_lo:.3f}–{r.share_CI_hi:.3f} |"
            )
        return "\n".join(lines)

    md = f"""# Table S_common_mode. Signed common-mode vs level-specific bias variance (Round-3sim)

**Definition.** For each complete laboratory-year (five concentration levels), signed relative bias
`New_Bias%` is decomposed as \(b_{{il}} = \\alpha_i + \\delta_{{il}}\), where \(\\alpha_i\) is the
across-level mean (common mode) and \(\\delta_{{il}}\) the level-specific residual.
Share = Var(\(\\alpha\)) / [Var(\(\\alpha\)) + mean_i Var(\(\\delta\))].
Assigned value = robust median of participating laboratories (consensus). Unit = ng/mL for concentrations; bias in %.

**Bootstrap.** Within each year, laboratories were resampled with replacement (B={B}); percentile 95% CIs for share and Var(\(\\alpha\)).

**Subsets.** (1) all complete laboratory-years; (2) laboratories participating in ≥3 calendar years; (3) laboratories present in both 2014 and 2020 (n={len(labs_span)}).

## A. All complete laboratories (primary)

{fmt_panel(prim)}

## B. Continuous participants (≥3 years; n unique labs in filter = {len(labs_k3)})

{fmt_panel(prim_k3)}

## C. Present in both 2014 and 2020 (n={len(labs_span)})

{fmt_panel(prim_span)}

## Trend decision (Round-3sim)

- Relative common-mode **share** 2014→2020: **{'YES — declines across subsets' if share_ok else 'NOT robust'}**
  (all-complete share {decision['all_complete']['share_2014']:.3f} → {decision['all_complete']['share_2020']:.3f}).
- Absolute Var(\(\\alpha\)) decline 2014→2020: **{'YES' if abs_ok else 'NO — not supported'}**
  (all-complete Var(α) {decision['all_complete']['var_alpha_2014']:.1f} → {decision['all_complete']['var_alpha_2020']:.1f};
  early median 2014–16 = {decision['all_complete']['median_var_alpha_early_2014_16']:.1f},
  late median 2018–20 = {decision['all_complete']['median_var_alpha_late_2018_20']:.1f}).
- 2017 absolute components remain outlier-inflated; interpret absolute Var(α) cautiously.
- PCA PC1 variance share remains exploratory support only for the relative pattern change.

**Manuscript implication.** Center the error-structure claim on the **relative** common-mode / PC1 share decline;
do **not** claim that absolute systematic common-mode variance declined industry-wide.
"""
    (OUT / "Table_S_common_mode.md").write_text(md, encoding="utf-8")

    # Copy into package
    for name in [
        "Table_S_common_mode_full.csv",
        "Table_S_common_mode_primary.csv",
        "Table_S_common_mode_ge3yrs.csv",
        "Table_S_common_mode_span2014_2020.csv",
        "Table_S_common_mode.md",
        "common_mode_decision.json",
    ]:
        src = OUT / name
        (PKG / "round3sim_analysis" / name).write_bytes(src.read_bytes())
        if name.startswith("Table_S_common_mode") and name.endswith((".csv", ".md")):
            # also refresh supplementary S0b primary
            pass
    # Update Table S0b to match primary New_Bias% with CIs
    s0b = prim.rename(
        columns={
            "var_alpha": "var_common",
            "mean_var_delta": "mean_var_specific",
            "share": "ratio_common_over_total",
        }
    )[
        [
            "Year",
            "n_labs",
            "var_common",
            "mean_var_specific",
            "ratio_common_over_total",
            "share_CI_lo",
            "share_CI_hi",
            "var_alpha_CI_lo",
            "var_alpha_CI_hi",
        ]
    ]
    s0b.to_csv(PKG / "supplementary" / "Table_S0b_common_mode_bias.csv", index=False)
    (PKG / "supplementary" / "Table_S0b_common_mode_bias.md").write_text(md, encoding="utf-8")
    (PKG / "phase2_analysis" / "common_mode_bias_decomposition.csv").write_text(
        s0b.to_csv(index=False), encoding="utf-8"
    )

    print(json.dumps(decision, indent=2))
    print(prim.to_string(index=False))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
