#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
EQA Propensity Score Matching (PSM) Analysis — Final Version
================================================================================
Purpose:
    Estimate the causal effect of independent Quality Assurance (QA) on
    analytical bias in bioanalytical proficiency testing, controlling for
    confounders (lab type, region, experience, instrumentation).

Analysis Steps:
    1. Lab-level data aggregation and covariate preparation
    2. Propensity score estimation (logistic regression, McFadden R²)
    3. 1:1 nearest-neighbour matching with caliper = 0.2 × SD(logit PS)
    4. Covariate balance assessment (SMD, Love Plot)
    5. Outcome comparison: |Bias| and risk rate (Mann-Whitney U)
    6. Level-specific QA effect analysis in matched sample

Outputs:
    PSM_Analysis_Final.tiff        — 4-panel publication figure (300 DPI)
    psm_results_final.csv          — Summary statistics
    psm_level_results_final.csv    — Level-specific results

Style:
    theme_jpa — Journal of Pharmaceutical Analysis / APSB publication standard
    (Arial font, full four-sided border, light-grey dashed grid, academic palette)

Author: EQA Research Group
================================================================================
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import log_loss
from pathlib import Path

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# THEME_JPA — APSB Journal Publication Style
# ══════════════════════════════════════════════════════════════════════════════

# Global rcParams
rcParams['font.family']        = 'Arial'
rcParams['font.size']          = 11
rcParams['axes.titlesize']     = 12
rcParams['axes.titleweight']   = 'bold'
rcParams['axes.labelsize']     = 12
rcParams['axes.labelweight']   = 'bold'
rcParams['axes.facecolor']     = 'white'
rcParams['axes.edgecolor']     = 'black'
rcParams['axes.linewidth']     = 1.0
rcParams['axes.grid']          = True
rcParams['axes.axisbelow']     = True
rcParams['grid.linestyle']     = '--'
rcParams['grid.color']         = '#e0e0e0'
rcParams['grid.alpha']         = 0.6
rcParams['grid.linewidth']     = 0.6
rcParams['xtick.color']        = 'black'
rcParams['ytick.color']        = 'black'
rcParams['xtick.labelsize']    = 11
rcParams['ytick.labelsize']    = 11
rcParams['xtick.direction']    = 'out'
rcParams['ytick.direction']    = 'out'
rcParams['xtick.major.width']  = 1.0
rcParams['ytick.major.width']  = 1.0
rcParams['xtick.major.size']   = 4.0
rcParams['ytick.major.size']   = 4.0
rcParams['xtick.minor.visible']= False
rcParams['ytick.minor.visible']= False
rcParams['legend.frameon']     = True
rcParams['legend.framealpha']  = 0.95
rcParams['legend.edgecolor']   = '#cccccc'
rcParams['legend.fontsize']    = 10
rcParams['figure.facecolor']   = 'white'
rcParams['savefig.facecolor']  = 'white'
rcParams['savefig.dpi']        = 300

# APSB Academic Colour Palette
BLUE   = '#00468B'   # Academic blue  — QA=Yes / compliance / control
RED    = '#ED0000'   # Academic red   — QA=No  / risk / treatment
GREEN  = '#117A65'   # Auxiliary green
ORANGE = '#E67E22'
GREY   = '#7F8C8D'
LIGHT_BLUE = '#AED6F1'
LIGHT_RED  = '#F1948A'


def set_theme_jpa(ax, grid_axis='both', full_border=True):
    """
    Apply theme_jpa (APSB publication style) to a matplotlib Axes object.

    Parameters
    ----------
    ax         : matplotlib.axes.Axes
    grid_axis  : 'both' | 'x' | 'y' | None — which axes to show grid
    full_border: bool — if True, show all four spines (journal standard)
    """
    ax.set_facecolor('white')
    # Four-sided border (APSB requirement)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(full_border if spine in ['top', 'right'] else True)
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color('black')
    # Ticks
    ax.tick_params(axis='both', which='major', direction='out',
                   length=4, width=1.0, colors='black', labelsize=11)
    ax.tick_params(axis='both', which='minor', bottom=False, left=False)
    # Grid
    ax.grid(False)
    if grid_axis == 'both':
        ax.grid(True, axis='both', linestyle='--', color='#e0e0e0',
                alpha=0.6, linewidth=0.6, zorder=0)
    elif grid_axis in ('x', 'y'):
        ax.grid(True, axis=grid_axis, linestyle='--', color='#e0e0e0',
                alpha=0.6, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def add_panel_label(ax, letter, x=-0.13, y=1.07):
    """Add bold panel label (A, B, C, D) to upper-left of axes."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left',
            color='black')


def save_figure(fig, name, output_dir):
    """Save figure as both TIFF (300 DPI) and PNG preview (150 DPI)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tiff_path = str(Path(output_dir) / f'{name}.tiff')
    png_path  = str(Path(output_dir) / f'{name}_preview.png')
    fig.savefig(tiff_path, dpi=300, format='tiff',
                bbox_inches='tight', facecolor='white')
    fig.savefig(png_path,  dpi=150, format='png',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [saved] {name}.tiff  |  {name}_preview.png")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH  = 'D:/Dropbox/EQA/EQA_with_LabInfo_integrated.csv'
OUTPUT_DIR = 'D:/Dropbox/EQA/eqa_ml/EQA_PSM_outputs_final'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & LAB-LEVEL AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 1 — DATA LOADING & LAB-LEVEL AGGREGATION")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df['New_Bias_abs'] = df['New_Bias%'].abs()
df['Risk_Flag']    = (df['New_Bias_abs'] > 15).astype(int)
df['Lab_type_bin'] = df['Lab_type'].str.strip().str.upper().map(
    lambda x: 1 if 'CRO' in str(x) else 0)
df['QA_bin']       = (df['QA'] == 'Yes').astype(int)
df['Level_num']    = df['Sample_Level'].str.extract(r'(\d+)')[0].astype(float)

# Balance precision encoding
bal_map = {
    '1': 1, '2': 2, '1,2': 1,
    '10^-6 Precision': 3,
    'Model not found; possibly 10^-5 precision': 2,
    '1 (Explicitly Stated)': 1,
}
df['Bal_enc'] = df['Balance_Precision'].astype(str).str.strip().map(bal_map).fillna(1)

# Lab-level aggregation
lab_df = df.groupby('Lab_ID').agg(
    QA_bin       = ('QA_bin',       lambda x: int(x.mean() > 0.5)),
    Lab_type_bin = ('Lab_type_bin', 'first'),
    Years_exp    = ('Year',         'nunique'),
    First_year   = ('Year',         'min'),
    CNAS_bin     = ('CNAS',         lambda x: int((x == 'Yes').mean() > 0.5)),
    SIL_IS_bin   = ('SIL-IS',       lambda x: int((x == 'Yes').mean() > 0.5)),
    Bal_enc      = ('Bal_enc',      'median'),
    mean_abs_bias= ('New_Bias_abs', 'mean'),
    risk_rate    = ('Risk_Flag',    'mean'),
    n_records    = ('Lab_ID',       'count'),
).reset_index()

# Region encoding (top 5 regions + Other)
region_map  = df.groupby('Lab_ID')['Region'].first()
lab_df['Region'] = lab_df['Lab_ID'].map(region_map)
top_regions = df['Region'].value_counts().head(5).index.tolist()
lab_df['Region_enc'] = lab_df['Region'].apply(
    lambda x: top_regions.index(x) if x in top_regions else len(top_regions))

n_qa_yes = int(lab_df['QA_bin'].sum())
n_qa_no  = int((lab_df['QA_bin'] == 0).sum())
print(f"  Total labs: {len(lab_df)}  |  QA=Yes: {n_qa_yes}  |  QA=No: {n_qa_no}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PROPENSITY SCORE ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2 — PROPENSITY SCORE ESTIMATION")
print("=" * 70)

COVARIATES = ['Lab_type_bin', 'Years_exp', 'First_year',
              'CNAS_bin', 'SIL_IS_bin', 'Bal_enc', 'Region_enc']
COV_LABELS = ['Lab Type (CRO)', 'Years Experience', 'First Year Joined',
              'CNAS Accreditation', 'SIL-IS Usage', 'Balance Precision', 'Region']

X_ps = lab_df[COVARIATES].fillna(lab_df[COVARIATES].median())
y_ps = lab_df['QA_bin'].values

scaler  = StandardScaler()
X_ps_sc = scaler.fit_transform(X_ps)

lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr.fit(X_ps_sc, y_ps)
lab_df['ps'] = lr.predict_proba(X_ps_sc)[:, 1]

# McFadden pseudo-R²
null_ll     = log_loss(y_ps, np.full(len(y_ps), y_ps.mean()), normalize=False)
model_ll    = log_loss(y_ps, lr.predict_proba(X_ps_sc)[:, 1], normalize=False)
mcfadden_r2 = 1 - model_ll / null_ll
print(f"  PS model McFadden R² = {mcfadden_r2:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — 1:1 NEAREST-NEIGHBOUR MATCHING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3 — 1:1 NEAREST-NEIGHBOUR MATCHING")
print("=" * 70)

lab_df['logit_ps'] = np.log(
    lab_df['ps'] / (1 - lab_df['ps'].clip(1e-6, 1 - 1e-6)))
caliper = 0.2 * lab_df['logit_ps'].std()
print(f"  Caliper (0.2 × SD logit PS) = {caliper:.4f}")

treated = lab_df[lab_df['QA_bin'] == 1].copy()
control = lab_df[lab_df['QA_bin'] == 0].copy()

nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
nn.fit(control[['logit_ps']].values)
dists, idxs = nn.kneighbors(treated[['logit_ps']].values)

matched_pairs = []
used_ctrl = set()
for i, (dist, idx) in enumerate(zip(dists.flatten(), idxs.flatten())):
    ctrl_lab = control.iloc[idx]['Lab_ID']
    if dist <= caliper and ctrl_lab not in used_ctrl:
        matched_pairs.append({
            'treated_lab': treated.iloc[i]['Lab_ID'],
            'control_lab': ctrl_lab,
            'ps_diff':     dist,
        })
        used_ctrl.add(ctrl_lab)

matched_df   = pd.DataFrame(matched_pairs)
matched_labs = set(matched_df['treated_lab']) | set(matched_df['control_lab'])
lab_matched  = lab_df[lab_df['Lab_ID'].isin(matched_labs)].copy()
print(f"  Matched pairs: {len(matched_df)}  (out of {len(control)} control labs)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — COVARIATE BALANCE ASSESSMENT (SMD)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4 — COVARIATE BALANCE (SMD)")
print("=" * 70)

def smd(x1, x2):
    """Standardised Mean Difference."""
    return ((x1.mean() - x2.mean()) /
            np.sqrt((x1.std() ** 2 + x2.std() ** 2) / 2 + 1e-10))

smd_before, smd_after = {}, {}
for cov in COVARIATES:
    t_b = lab_df.loc[lab_df['QA_bin'] == 1, cov].fillna(0)
    c_b = lab_df.loc[lab_df['QA_bin'] == 0, cov].fillna(0)
    t_a = lab_matched.loc[lab_matched['QA_bin'] == 1, cov].fillna(0)
    c_a = lab_matched.loc[lab_matched['QA_bin'] == 0, cov].fillna(0)
    smd_before[cov] = abs(smd(t_b, c_b))
    smd_after[cov]  = abs(smd(t_a, c_a))
    print(f"  {cov:20s}  SMD before={smd_before[cov]:.3f}  after={smd_after[cov]:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — OUTCOME ANALYSIS IN MATCHED SAMPLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5 — OUTCOME ANALYSIS")
print("=" * 70)

df['QA_bin_rec'] = (df['QA'] == 'Yes').astype(int)
df_matched = df[df['Lab_ID'].isin(matched_labs)].copy()
lab_qa     = lab_df[['Lab_ID', 'QA_bin']].rename(columns={'QA_bin': 'QA_bin_lab'})
df_matched = df_matched.merge(lab_qa, on='Lab_ID', how='left')

# Unmatched comparison
qa_yes_bias_u = df.loc[df['QA_bin_rec'] == 1, 'New_Bias_abs'].dropna()
qa_no_bias_u  = df.loc[df['QA_bin_rec'] == 0, 'New_Bias_abs'].dropna()
_, p_unmatched = stats.mannwhitneyu(qa_yes_bias_u, qa_no_bias_u, alternative='two-sided')

# Matched comparison
qa_yes_bias = df_matched.loc[df_matched['QA_bin_lab'] == 1, 'New_Bias_abs'].dropna()
qa_no_bias  = df_matched.loc[df_matched['QA_bin_lab'] == 0, 'New_Bias_abs'].dropna()
qa_yes_risk = df_matched.loc[df_matched['QA_bin_lab'] == 1, 'Risk_Flag'].mean()
qa_no_risk  = df_matched.loc[df_matched['QA_bin_lab'] == 0, 'Risk_Flag'].mean()
_, p_val    = stats.mannwhitneyu(qa_yes_bias, qa_no_bias, alternative='two-sided')

print(f"  Unmatched — QA=Yes: {qa_yes_bias_u.mean():.2f}%  QA=No: {qa_no_bias_u.mean():.2f}%  p={p_unmatched:.4f}")
print(f"  Matched   — QA=Yes: {qa_yes_bias.mean():.2f}%  QA=No: {qa_no_bias.mean():.2f}%  p={p_val:.4f}")
print(f"  Risk rate — QA=Yes: {qa_yes_risk:.3f}  QA=No: {qa_no_risk:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — LEVEL-SPECIFIC QA EFFECT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6 — LEVEL-SPECIFIC QA EFFECT")
print("=" * 70)

level_results = []
for lv in ['Level.1', 'Level.2', 'Level.3', 'Level.4', 'Level.5']:
    sub = df_matched[df_matched['Sample_Level'] == lv]
    yes = sub.loc[sub['QA_bin_lab'] == 1, 'New_Bias_abs'].dropna()
    no  = sub.loc[sub['QA_bin_lab'] == 0, 'New_Bias_abs'].dropna()
    if len(yes) > 5 and len(no) > 5:
        _, p = stats.mannwhitneyu(yes, no, alternative='two-sided')
        level_results.append({
            'Level':        lv,
            'QA_Yes_mean':  yes.mean(),
            'QA_No_mean':   no.mean(),
            'p_value':      p,
            'Effect_size':  yes.mean() - no.mean(),
        })

level_res_df = pd.DataFrame(level_results)
print(level_res_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
results_summary = {
    'n_labs_total':              len(lab_df),
    'n_qa_yes':                  n_qa_yes,
    'n_qa_no':                   n_qa_no,
    'n_matched_pairs':           len(matched_df),
    'caliper':                   round(caliper, 4),
    'ps_mcfadden_r2':            round(mcfadden_r2, 4),
    'unmatched_p':               round(p_unmatched, 4),
    'matched_p':                 round(p_val, 4),
    'qa_yes_mean_bias_unmatched':round(qa_yes_bias_u.mean(), 3),
    'qa_no_mean_bias_unmatched': round(qa_no_bias_u.mean(), 3),
    'qa_yes_mean_bias_matched':  round(qa_yes_bias.mean(), 3),
    'qa_no_mean_bias_matched':   round(qa_no_bias.mean(), 3),
    'qa_yes_risk_matched':       round(qa_yes_risk, 4),
    'qa_no_risk_matched':        round(qa_no_risk, 4),
}
pd.DataFrame([results_summary]).to_csv(
    str(Path(OUTPUT_DIR) / 'psm_results_final.csv'), index=False)
level_res_df.to_csv(
    str(Path(OUTPUT_DIR) / 'psm_level_results_final.csv'), index=False)
print("\n  Results saved.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FIGURE: PSM ANALYSIS (4-PANEL, theme_jpa)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8 — GENERATING FIGURE (theme_jpa)")
print("=" * 70)

fig = plt.figure(figsize=(14, 11), facecolor='white')
gs  = GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.40,
               left=0.09, right=0.96, top=0.93, bottom=0.08)

# ── Panel A: Propensity Score Distribution (Before Matching) ─────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.hist(lab_df.loc[lab_df['QA_bin'] == 1, 'ps'], bins=20, alpha=0.65,
          color=BLUE, label=f'QA = Yes  (n={n_qa_yes})', density=True, zorder=3)
ax_a.hist(lab_df.loc[lab_df['QA_bin'] == 0, 'ps'], bins=20, alpha=0.65,
          color=RED,  label=f'QA = No   (n={n_qa_no})',  density=True, zorder=3)
ax_a.set_xlabel('Propensity Score', fontweight='bold')
ax_a.set_ylabel('Density', fontweight='bold')
ax_a.set_title('Propensity Score Distribution\n(Before Matching)')
ax_a.legend(loc='upper right', title=None)
set_theme_jpa(ax_a, grid_axis='y')
add_panel_label(ax_a, 'A')

# ── Panel B: Love Plot (SMD Before vs After Matching) ────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
y_pos      = np.arange(len(COVARIATES))
smd_b_vals = [smd_before[c] for c in COVARIATES]
smd_a_vals = [smd_after[c]  for c in COVARIATES]

for i in range(len(COVARIATES)):
    ax_b.plot([smd_b_vals[i], smd_a_vals[i]], [i, i],
              color='#AAAAAA', lw=0.9, alpha=0.6, zorder=2)
ax_b.scatter(smd_b_vals, y_pos, color=RED,  s=65, zorder=5,
             label='Before matching', marker='D', edgecolors='white', linewidths=0.5)
ax_b.scatter(smd_a_vals, y_pos, color=BLUE, s=65, zorder=5,
             label='After matching',  marker='o', edgecolors='white', linewidths=0.5)
ax_b.axvline(0.10, color='black', lw=1.2, linestyle='--', alpha=0.7,
             label='SMD = 0.10 threshold', zorder=4)
ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(COV_LABELS, fontsize=10)
ax_b.set_xlabel('Standardised Mean Difference (SMD)', fontweight='bold')
ax_b.set_title('Covariate Balance: Love Plot\n(SMD Before vs. After Matching)')
ax_b.legend(loc='lower right', title=None, fontsize=9)
ax_b.set_xlim(-0.05, max(max(smd_b_vals), 0.5) + 0.12)
set_theme_jpa(ax_b, grid_axis='x')
add_panel_label(ax_b, 'B')

# ── Panel C: |Bias| Boxplot (Unmatched vs Matched) ───────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
data_plot = [
    qa_yes_bias_u.clip(0, 60),
    qa_no_bias_u.clip(0, 60),
    qa_yes_bias.clip(0, 60),
    qa_no_bias.clip(0, 60),
]
labels_c = ['QA=Yes\n(Unmatched)', 'QA=No\n(Unmatched)',
            'QA=Yes\n(Matched)',   'QA=No\n(Matched)']
colors_c = [LIGHT_BLUE, LIGHT_RED, BLUE, RED]

bp = ax_c.boxplot(data_plot, patch_artist=True, widths=0.52,
                  medianprops=dict(color='black', lw=2.0),
                  whiskerprops=dict(lw=1.2, color='black'),
                  capprops=dict(lw=1.2, color='black'),
                  flierprops=dict(marker='o', ms=3, alpha=0.35,
                                  markerfacecolor='grey', markeredgecolor='none'),
                  zorder=3)
for patch, color in zip(bp['boxes'], colors_c):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.0)

ax_c.set_xticklabels(labels_c, fontsize=10)
ax_c.set_ylabel('|Bias| % (clipped at 60%)', fontweight='bold')
ax_c.set_title('QA Effect on |Bias|:\nUnmatched vs. Matched Samples')

# Significance brackets
y_bracket = 54
for x1, x2, p_br in [(1, 2, p_unmatched), (3, 4, p_val)]:
    ax_c.plot([x1, x2], [y_bracket, y_bracket], 'k-', lw=1.3, zorder=5)
    ax_c.plot([x1, x1], [y_bracket - 1.5, y_bracket], 'k-', lw=1.0, zorder=5)
    ax_c.plot([x2, x2], [y_bracket - 1.5, y_bracket], 'k-', lw=1.0, zorder=5)
    p_label = (f'p < 0.001' if p_br < 0.001 else
               f'p = {p_br:.3f}' if p_br < 0.05 else f'p = {p_br:.3f} (n.s.)')
    ax_c.text((x1 + x2) / 2, y_bracket + 0.8, p_label,
              ha='center', va='bottom', fontsize=9.5, color='black')

set_theme_jpa(ax_c, grid_axis='y')
add_panel_label(ax_c, 'C')

# ── Panel D: Level-Specific QA Effect (Matched Sample) ───────────────────────
ax_d = fig.add_subplot(gs[1, 1])
x_lv  = np.arange(len(level_res_df))
width = 0.36
b1 = ax_d.bar(x_lv - width / 2, level_res_df['QA_Yes_mean'], width,
              color=BLUE, alpha=0.85, label='QA = Yes',
              edgecolor='black', linewidth=0.6, zorder=3)
b2 = ax_d.bar(x_lv + width / 2, level_res_df['QA_No_mean'],  width,
              color=RED,  alpha=0.85, label='QA = No',
              edgecolor='black', linewidth=0.6, zorder=3)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax_d.text(bar.get_x() + bar.get_width() / 2, h + 0.25,
              f'{h:.1f}', ha='center', va='bottom', fontsize=8.5, color='black')

for i, row in level_res_df.iterrows():
    sig = ('***' if row['p_value'] < 0.001 else
           '**'  if row['p_value'] < 0.01  else
           '*'   if row['p_value'] < 0.05  else 'n.s.')
    y_top = max(row['QA_Yes_mean'], row['QA_No_mean']) + 1.8
    ax_d.text(i, y_top, sig, ha='center', va='bottom', fontsize=10, color='black')

ax_d.set_xticks(x_lv)
ax_d.set_xticklabels([f'Level {i + 1}' for i in range(len(level_res_df))], fontsize=10)
ax_d.set_ylabel('Mean |Bias| % (Matched Sample)', fontweight='bold')
ax_d.set_title('Level-Specific QA Effect\n(PSM-Matched Laboratories)')
ax_d.legend(loc='upper right', title=None)
set_theme_jpa(ax_d, grid_axis='y')
add_panel_label(ax_d, 'D')

# Figure footer
fig.text(
    0.5, 0.015,
    (f'PSM: 1:1 nearest-neighbour matching, caliper = {caliper:.3f} (0.2 × SD logit PS)  |  '
     f'Matched pairs: n = {len(matched_df)}  |  '
     f'PS model McFadden R\u00b2 = {mcfadden_r2:.3f}'),
    ha='center', fontsize=9, color='#555555', style='italic'
)

save_figure(fig, 'PSM_Analysis_Final', OUTPUT_DIR)

print("\n" + "=" * 70)
print("PSM ANALYSIS COMPLETE")
print(f"  Outputs: {OUTPUT_DIR}/")
print("=" * 70)
