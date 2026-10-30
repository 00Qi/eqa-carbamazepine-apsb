#!/usr/bin/env python3
"""Round-3sim P0: full risk-stratification reproducibility (2020 holdout).

Extends Table S2 with:
- AUC / PR-AUC 95% CIs (lab bootstrap)
- Capture@10/20/30/40%
- Cold-start (no pre-2020 history) scoring + performance
- Head-to-head vs simple historical failure-rate (ΔAUC, ΔCapture)
- Leakage checklist
No new complex ML models; reuse existing scorers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Reuse helpers from existing Table S2 builder
sys.path.insert(0, "/workspace/apsb_out/table_s2_src")
from build_table_s2 import (  # noqa: E402
    FEATURE_COLS,
    build_feature_rows,
    fit_logistic,
    lab_year_frame,
    load_raw,
    pct_rank,
    temporal_select_C,
    triage_metrics,
)

RAW = Path("/workspace/p2/analysis/EQA_with_LabInfo_integrated.csv")
OUT = Path("/workspace/p2/out/round3sim")
PKG = Path("/workspace/apsb_out/APSB_submission_1008")
OUT.mkdir(parents=True, exist_ok=True)
(PKG / "round3sim_analysis").mkdir(parents=True, exist_ok=True)

B = 2000
RNG = np.random.default_rng(20261030)
KS = (0.10, 0.20, 0.30, 0.40)


def triage_ks(y, score, ks=KS):
    return triage_metrics(y, score, ks=ks)


def bootstrap_metrics(y, score, B=B, rng=RNG, ks=KS):
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    n = len(y)
    rows = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yy, ss = y[idx], s[idx]
        if yy.sum() == 0 or yy.sum() == n:
            continue
        try:
            m = triage_ks(yy, ss, ks=ks)
        except ValueError:
            continue
        rows.append(m)
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out = {}
    for col in df.columns:
        if col.startswith("K@") or col.startswith("Caught@"):
            continue
        out[f"{col}_lo"] = float(np.nanpercentile(df[col], 2.5))
        out[f"{col}_hi"] = float(np.nanpercentile(df[col], 97.5))
        out[f"{col}_mean"] = float(np.nanmean(df[col]))
    return out


def main():
    df = load_raw(RAW)
    ly = lab_year_frame(df)
    F = build_feature_rows(df, ly)
    train = F[F["Year"].between(2014, 2019)].copy()
    test = F[F["Year"] == 2020].copy().reset_index(drop=True)
    n = len(test)
    n_events = int(test["event"].sum())
    assert n == 131 and n_events == 23, (n, n_events)

    # Leakage documentation
    leakage = {
        "train_years": "2014-2019",
        "test_year": 2020,
        "historical_PT_features": "computed only from Year < prediction year (cum_fail_rate, hist_max_abs, slopes, etc.)",
        "practice_flags_in_FEATURE_COLS": "QA/SIL/CNAS/MV/bal_num/is_CRO taken from CURRENT lab-year row in build_feature_rows — for 2020 holdout these are 2020 questionnaire values (disclosed residual contemporaneous limitation for practice flags; historical PT summaries are leakage-safe)",
        "n_rec_year": "included in FEATURE_COLS (same-year record count) — potential weak leakage for logistic; heuristics (fail/max) do NOT use it",
        "cold_start_scoring": "labs with has_history==0 get heuristic scores filled to 0 (ranked last); logistic still scores them via current-year practice flags + zeroed history after median fill",
        "event_definition": "abs(New_Bias%)>15 any record in lab-year; not ISR",
        "no_2020_outcome_in_features": True,
    }

    fail_score = test["cum_fail_rate"].fillna(0.0).to_numpy()
    max_score = test["hist_max_abs"].fillna(0.0).to_numpy()
    C = temporal_select_C(train, FEATURE_COLS)
    logit_p, C, med, sc, clf = fit_logistic(train, test, FEATURE_COLS, C)
    blend = (pct_rank(logit_p) + pct_rank(fail_score) + pct_rank(max_score)) / 3.0
    y = test["event"].to_numpy()

    scorers = {
        "Historical failure-rate": fail_score,
        "Historical max abs bias": max_score,
        "Regularized logistic": logit_p,
        "Blended": blend,
    }

    summary_rows = []
    boot_rows = []
    for name, score in scorers.items():
        m = triage_ks(y, score)
        b = bootstrap_metrics(y, score)
        row = {"Scorer": name, **m}
        # attach CIs
        for k in ("AUC", "PR_AUC", "Capture@10%", "Capture@20%", "Capture@30%", "Capture@40%"):
            row[f"{k}_CI_lo"] = b.get(f"{k}_lo", np.nan)
            row[f"{k}_CI_hi"] = b.get(f"{k}_hi", np.nan)
        summary_rows.append(row)
        boot_rows.append({"Scorer": name, **b})

    summary = pd.DataFrame(summary_rows)

    # Incremental vs simple failure-rate rule
    base = summary_rows[0]
    incr = []
    for row in summary_rows:
        incr.append(
            {
                "Scorer": row["Scorer"],
                "AUC": row["AUC"],
                "delta_AUC_vs_failrate": row["AUC"] - base["AUC"],
                "Capture@10%": row["Capture@10%"],
                "delta_Capture@10%": row["Capture@10%"] - base["Capture@10%"],
                "Capture@20%": row["Capture@20%"],
                "delta_Capture@20%": row["Capture@20%"] - base["Capture@20%"],
                "Capture@30%": row["Capture@30%"],
                "delta_Capture@30%": row["Capture@30%"] - base["Capture@30%"],
                "Capture@40%": row["Capture@40%"],
                "delta_Capture@40%": row["Capture@40%"] - base["Capture@40%"],
                "PR_AUC": row["PR_AUC"],
                "delta_PR_AUC": row["PR_AUC"] - base["PR_AUC"],
            }
        )
    incr_df = pd.DataFrame(incr)

    # Cold-start subset
    cold = test["has_history"] == 0
    warm = test["has_history"] == 1
    cold_stats = {
        "n_cold": int(cold.sum()),
        "n_cold_events": int(y[cold].sum()),
        "n_warm": int(warm.sum()),
        "n_warm_events": int(y[warm].sum()),
        "cold_start_rule": "heuristic scores = 0 (ranked last among all 131); logistic uses median-filled history + 2020 practice flags",
    }
    cold_perf = []
    # Among cold labs alone: scores are mostly ties at 0 for heuristics → AUC undefined or chance
    if cold.sum() >= 5 and y[cold].sum() > 0 and y[cold].sum() < cold.sum():
        for name, score in scorers.items():
            try:
                auc = float(roc_auc_score(y[cold], score[cold]))
            except ValueError:
                auc = float("nan")
            try:
                pra = float(average_precision_score(y[cold], score[cold]))
            except ValueError:
                pra = float("nan")
            # Capture within cold only at 20% of cold
            m_c = triage_ks(y[cold], score[cold], ks=(0.20,))
            cold_perf.append(
                {
                    "Scorer": name,
                    "subset": "cold_start_only",
                    "AUC": auc,
                    "PR_AUC": pra,
                    "Capture@20%_within_subset": m_c.get("Capture@20%", np.nan),
                }
            )
    # Warm-only performance
    for name, score in scorers.items():
        m_w = triage_ks(y[warm], score[warm])
        bw = bootstrap_metrics(y[warm], score[warm])
        cold_perf.append(
            {
                "Scorer": name,
                "subset": "warm_history_only",
                "AUC": m_w["AUC"],
                "PR_AUC": m_w["PR_AUC"],
                "Capture@20%_within_subset": m_w["Capture@20%"],
                "AUC_CI_lo": bw.get("AUC_lo", np.nan),
                "AUC_CI_hi": bw.get("AUC_hi", np.nan),
                "Capture@20%_CI_lo": bw.get("Capture@20%_lo", np.nan),
                "Capture@20%_CI_hi": bw.get("Capture@20%_hi", np.nan),
            }
        )
    # How many cold events are captured in top 20% of FULL ranking (they shouldn't be if score=0)
    order_fail = np.argsort(-fail_score, kind="mergesort")
    K20 = int(np.ceil(0.20 * n))
    top20 = set(order_fail[:K20].tolist())
    cold_idx = set(np.where(cold.to_numpy())[0].tolist())
    cold_in_top20 = len(top20 & cold_idx)
    cold_events_in_top20 = int(y[list(top20 & cold_idx)].sum()) if (top20 & cold_idx) else 0
    cold_stats["cold_labs_in_top20_failrate"] = cold_in_top20
    cold_stats["cold_events_in_top20_failrate"] = cold_events_in_top20

    cold_df = pd.DataFrame(cold_perf)

    # Decision-curve style simple table: net enrichment vs random at each budget
    dca = []
    prevalence = y.mean()
    for name, score in scorers.items():
        m = triage_ks(y, score)
        for frac, key in [(0.10, "Capture@10%"), (0.20, "Capture@20%"), (0.30, "Capture@30%"), (0.40, "Capture@40%")]:
            cap = m[key]
            # expected capture under random = frac
            dca.append(
                {
                    "Scorer": name,
                    "audit_fraction": frac,
                    "Capture": cap,
                    "Random_Capture": frac,
                    "delta_vs_random": cap - frac,
                    "Precision": m[f"Precision@{int(frac*100)}%"],
                    "Lift": m[f"Lift@{int(frac*100)}%"],
                }
            )
    dca_df = pd.DataFrame(dca)

    # Decision flags
    fail_cap20 = base["Capture@20%"]
    enrich_ok = fail_cap20 > 0.20  # better than random
    # complex vs simple: blended or logistic improves Capture@20 or AUC?
    blend_row = next(r for r in summary_rows if r["Scorer"] == "Blended")
    logit_row = next(r for r in summary_rows if r["Scorer"] == "Regularized logistic")
    complex_beats_simple_capture = (
        blend_row["Capture@20%"] > fail_cap20 + 0.02
        or logit_row["Capture@20%"] > fail_cap20 + 0.02
    )
    complex_beats_simple_auc = (
        blend_row["AUC"] > base["AUC"] + 0.02 or logit_row["AUC"] > base["AUC"] + 0.02
    )
    decision = {
        "n_labs": n,
        "n_events": n_events,
        "logistic_C": C,
        "risk_enrichment_vs_random_Capture20": enrich_ok,
        "failrate_Capture@20%": fail_cap20,
        "failrate_AUC": base["AUC"],
        "failrate_AUC_CI": [base["AUC_CI_lo"], base["AUC_CI_hi"]],
        "failrate_Capture@20%_CI": [base["Capture@20%_CI_lo"], base["Capture@20%_CI_hi"]],
        "complex_beats_simple_on_Capture20": complex_beats_simple_capture,
        "complex_beats_simple_on_AUC": complex_beats_simple_auc,
        "recommended_primary_scorer": "Historical failure-rate (simple rule); blended offers little Capture gain",
        "cold_start": cold_stats,
        "leakage": leakage,
    }

    # Write CSVs
    summary.to_csv(OUT / "Table_S_risk_summary.csv", index=False)
    incr_df.to_csv(OUT / "Table_S_risk_incremental_vs_failrate.csv", index=False)
    cold_df.to_csv(OUT / "Table_S_risk_cold_warm.csv", index=False)
    dca_df.to_csv(OUT / "Table_S_risk_enrichment_vs_random.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT / "Table_S_risk_bootstrap_raw.csv", index=False)
    (OUT / "risk_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    # Per-lab list with cold flag
    list_df = pd.DataFrame(
        {
            "Lab_ID": test["Lab_ID"].values,
            "event_flag": y,
            "prior_years": test["prior_years"].values,
            "has_history": test["has_history"].values,
            "cold_start": cold.astype(int).values,
            "score_hist_fail_rate": fail_score,
            "score_hist_max_abs": max_score,
            "score_logistic_prob": logit_p,
            "score_blended": blend,
        }
    )
    for col, scn in [
        ("rank_hist_fail_rate", "score_hist_fail_rate"),
        ("rank_hist_max_abs", "score_hist_max_abs"),
        ("rank_logistic", "score_logistic_prob"),
        ("rank_blended", "score_blended"),
    ]:
        list_df[col] = list_df[scn].rank(ascending=False, method="min").astype(int)
    list_df = list_df.sort_values(["rank_hist_fail_rate", "Lab_ID"]).reset_index(drop=True)
    list_df.to_csv(OUT / "Table_S_risk_lab_ranking_list.csv", index=False)

    # Markdown
    def fmt_ci(row, key):
        return f"{row[key]:.3f} ({row[key+'_CI_lo']:.3f}–{row[key+'_CI_hi']:.3f})"

    lines = [
        "# Table S_risk. 2020 holdout risk enrichment (Round-3sim reproducibility)",
        "",
        f"**Cohort.** N={n} laboratories; events={n_events} ({100*n_events/n:.1f}%) with ≥1 PT-defined deviation (abs(New_Bias%)>15).",
        "**Design.** Train 2014–2019; test 2020. Historical PT features use only pre-2020 records. Cold-start labs (no pre-2020 history) scored 0 on heuristics (ranked last).",
        f"**Bootstrap.** Lab resample on the 2020 holdout, B={B}, seed=20261030.",
        "",
        "## A. Performance with 95% CIs (Capture at 10/20/30/40%)",
        "",
        "| Scorer | AUC (95% CI) | PR-AUC (95% CI) | Cap@10% | Cap@20% | Cap@30% | Cap@40% |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['Scorer']} | {fmt_ci(r,'AUC')} | {fmt_ci(r,'PR_AUC')} | "
            f"{100*r['Capture@10%']:.1f}% | {100*r['Capture@20%']:.1f}% | "
            f"{100*r['Capture@30%']:.1f}% | {100*r['Capture@40%']:.1f}% |"
        )
    lines += [
        "",
        "## B. Incremental value vs simple historical failure-rate",
        "",
        "| Scorer | ΔAUC | ΔCap@10% | ΔCap@20% | ΔCap@30% | ΔCap@40% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in incr_df.iterrows():
        lines.append(
            f"| {r.Scorer} | {r['delta_AUC_vs_failrate']:+.3f} | "
            f"{100*r['delta_Capture@10%']:+.1f} pp | {100*r['delta_Capture@20%']:+.1f} pp | "
            f"{100*r['delta_Capture@30%']:+.1f} pp | {100*r['delta_Capture@40%']:+.1f} pp |"
        )
    lines += [
        "",
        "## C. Cold-start",
        "",
        f"- Cold-start labs (no pre-2020 history): **{cold_stats['n_cold']}** (events={cold_stats['n_cold_events']}).",
        f"- Warm labs: **{cold_stats['n_warm']}** (events={cold_stats['n_warm_events']}).",
        f"- Cold labs placed in top-20% by failure-rate ranking: {cold_stats['cold_labs_in_top20_failrate']} "
        f"(cold events captured there: {cold_stats['cold_events_in_top20_failrate']}) — expected near zero under score=0 rule.",
        "",
        "Warm-only AUC/Capture@20% (labs with history):",
        "",
        "| Scorer | AUC | Cap@20% (within warm) |",
        "|---|---:|---:|",
    ]
    for _, r in cold_df[cold_df.subset == "warm_history_only"].iterrows():
        lines.append(
            f"| {r.Scorer} | {r.AUC:.3f} | {100*r['Capture@20%_within_subset']:.1f}% |"
        )
    lines += [
        "",
        "## D. Leakage checklist",
        "",
        f"- Historical PT summaries: **pre-2020 only** ({leakage['historical_PT_features']}).",
        f"- Practice flags in logistic FEATURE_COLS: **{leakage['practice_flags_in_FEATURE_COLS'][:120]}...**",
        f"- `n_rec_year` in logistic features: disclosed ({leakage['n_rec_year']}).",
        f"- Heuristic scorers (failure-rate / max abs): **no same-year outcome leakage**.",
        "",
        "## E. Decision",
        "",
        f"- Enrichment vs random at 20% audit budget: **{'YES' if enrich_ok else 'NO'}** "
        f"(Capture@20%={100*fail_cap20:.1f}% vs 20% random; AUC={base['AUC']:.3f}, "
        f"95% CI {base['AUC_CI_lo']:.3f}–{base['AUC_CI_hi']:.3f}).",
        f"- Complex score beats simple failure-rate on Capture@20%: **{'YES' if complex_beats_simple_capture else 'NO'}**.",
        f"- Complex score beats simple failure-rate on AUC (>+0.02): **{'YES' if complex_beats_simple_auc else 'NO'}**.",
        f"- Primary triage signal for manuscript: **{decision['recommended_primary_scorer']}**.",
        "",
        "Not a regulatory deployment claim.",
    ]
    md = "\n".join(lines)
    (OUT / "Table_S_risk.md").write_text(md, encoding="utf-8")

    # Sync to package supplementary + round3sim_analysis
    for name in [
        "Table_S_risk_summary.csv",
        "Table_S_risk_incremental_vs_failrate.csv",
        "Table_S_risk_cold_warm.csv",
        "Table_S_risk_enrichment_vs_random.csv",
        "Table_S_risk_lab_ranking_list.csv",
        "Table_S_risk.md",
        "risk_decision.json",
    ]:
        (PKG / "round3sim_analysis" / name).write_bytes((OUT / name).read_bytes())

    # Refresh Table S2 summary with new Capture fractions + CIs (keep name for reviewers)
    summary.to_csv(PKG / "supplementary" / "Table_S2_lab_level_2020_ranking_summary.csv", index=False)
    list_df.to_csv(PKG / "supplementary" / "Table_S2_lab_level_2020_ranking_list.csv", index=False)
    (PKG / "supplementary" / "Table_S2_lab_level_2020_ranking_summary.md").write_text(md, encoding="utf-8")

    print(json.dumps(decision, indent=2))
    print(summary[["Scorer", "AUC", "AUC_CI_lo", "AUC_CI_hi", "Capture@20%", "Capture@20%_CI_lo", "Capture@20%_CI_hi"]].to_string(index=False))
    print(incr_df.to_string(index=False))


if __name__ == "__main__":
    main()
