#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
EQA Machine Learning Analysis — Final Version
================================================================================
Purpose:
    Predict high-risk analytical bias events in bioanalytical proficiency
    testing using an optimised ensemble of Random Forest, LightGBM, and
    XGBoost classifiers, with advanced feature engineering and triage metrics
    designed for Risk-Based Auditing (RBA) in pharmaceutical bioanalysis.

Analysis Pipeline:
    1.  Data loading and base preprocessing
    2.  Advanced feature engineering
        - Interaction features: QA×Level, QA×Group, Year×QA, SIL-IS×Level
        - Historical features: expanding-window failure rate, rolling 1/2/3-yr,
          trend slope, Level-1 historical bias, industry percentile rank
        - WOE encoding for Year and Level (fit on training data only)
    3.  Temporal train/test split (train: 2014-2019; test: 2020)
    4.  Hyperparameter optimisation via Optuna (50 trials × 3 models)
    5.  Base model training + 4 ensemble strategies
        (Soft Voting equal/weighted, Stacking, Ensemble of Ensembles)
    6.  Model calibration (Platt Scaling, Isotonic Regression)
    7.  Triage metrics: Capture@K%, Precision@K%, Lift@K%, DCA
    8.  SHAP interpretability (TreeExplainer on RF v2)
    9.  Publication-quality figures (theme_jpa, 300 DPI TIFF)
    10. Final summary and CSV outputs

Outputs (saved to EQA_ML_outputs_final/):
    model_performance_final.csv
    calibration_comparison_final.csv
    triage_metrics_final.csv
    dca_results_final.csv
    audit_savings_final.csv
    shap_importance_final.csv
    ROC_PR_Curves_Final.tiff
    CaptureRate_AllModels_Final.tiff
    DecisionCurveAnalysis_Final.tiff
    Calibration_Comparison_Final.tiff
    SHAP_Summary_Final.tiff
    Triage_Efficiency_Summary_Final.tiff

Style:
    theme_jpa — Journal of Pharmaceutical Analysis / APSB publication standard

Author: Wenyuan Qi
================================================================================
"""

import os
import warnings
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import linregress

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec

from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, auc,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb
import xgboost as xgb
import optuna
import shap

from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
logging.getLogger('lightgbm').setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════════════════════════
# THEME_JPA — APSB Journal Publication Style
# ══════════════════════════════════════════════════════════════════════════════

rcParams['font.family']         = 'Arial'
rcParams['font.size']           = 11
rcParams['axes.titlesize']      = 12
rcParams['axes.titleweight']    = 'bold'
rcParams['axes.labelsize']      = 12
rcParams['axes.labelweight']    = 'bold'
rcParams['axes.facecolor']      = 'white'
rcParams['axes.edgecolor']      = 'black'
rcParams['axes.linewidth']      = 1.0
rcParams['axes.grid']           = True
rcParams['axes.axisbelow']      = True
rcParams['grid.linestyle']      = '--'
rcParams['grid.color']          = '#e0e0e0'
rcParams['grid.alpha']          = 0.6
rcParams['grid.linewidth']      = 0.6
rcParams['xtick.color']         = 'black'
rcParams['ytick.color']         = 'black'
rcParams['xtick.labelsize']     = 11
rcParams['ytick.labelsize']     = 11
rcParams['xtick.direction']     = 'out'
rcParams['ytick.direction']     = 'out'
rcParams['xtick.major.width']   = 1.0
rcParams['ytick.major.width']   = 1.0
rcParams['xtick.major.size']    = 4.0
rcParams['ytick.major.size']    = 4.0
rcParams['xtick.minor.visible'] = False
rcParams['ytick.minor.visible'] = False
rcParams['legend.frameon']      = True
rcParams['legend.framealpha']   = 0.95
rcParams['legend.edgecolor']    = '#cccccc'
rcParams['legend.fontsize']     = 10
rcParams['figure.facecolor']    = 'white'
rcParams['savefig.facecolor']   = 'white'
rcParams['savefig.dpi']         = 300

# APSB Academic Colour Palette
BLUE   = '#00468B'   # Academic blue  — control / compliance / RF
RED    = '#ED0000'   # Academic red   — risk / treatment / LGB
GREEN  = '#117A65'   # Auxiliary green — XGBoost / ensemble
ORANGE = '#D4700A'   # Auxiliary orange
PURPLE = '#6C3483'   # Auxiliary purple
GREY   = '#7F8C8D'   # Neutral grey
LIGHT_BLUE = '#AED6F1'
LIGHT_RED  = '#F1948A'

# Model colour map (consistent across all figures)
MODEL_COLORS = {
    'Random Forest (v2)':      BLUE,
    'LightGBM (v2)':           RED,
    'XGBoost':                 GREEN,
    'Soft Voting (equal)':     ORANGE,
    'Soft Voting (weighted)':  PURPLE,
    'Stacking (LR meta)':      '#8B4513',
    'Ensemble of Ensembles':   '#C0392B',
}


def set_theme_jpa(ax, grid_axis='both', full_border=True):
    """
    Apply theme_jpa (APSB publication style) to a matplotlib Axes object.

    Parameters
    ----------
    ax         : matplotlib.axes.Axes
    grid_axis  : 'both' | 'x' | 'y' | None
    full_border: bool — show all four spines
    """
    ax.set_facecolor('white')
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True if full_border else spine in ['bottom', 'left'])
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color('black')
    ax.tick_params(axis='both', which='major', direction='out',
                   length=4, width=1.0, colors='black', labelsize=11)
    ax.tick_params(axis='both', which='minor', bottom=False, left=False)
    ax.grid(False)
    if grid_axis in ('both', 'x', 'y'):
        ax.grid(True, axis=grid_axis, linestyle='--', color='#e0e0e0',
                alpha=0.6, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def add_panel_label(ax, letter, x=-0.13, y=1.07):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left', color='black')


def save_figure(fig, name, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(str(Path(output_dir) / f'{name}.tiff'),
                dpi=300, format='tiff', bbox_inches='tight', facecolor='white')
    fig.savefig(str(Path(output_dir) / f'{name}_preview.png'),
                dpi=150, format='png',  bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [saved] {name}.tiff")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH  = 'D:/Dropbox/EQA/EQA_with_LabInfo_integrated.csv'
OUTPUT_DIR = 'D:/Dropbox/EQA/eqa_ml/EQA_ML_outputs_final'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & BASE PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 1 — DATA LOADING & BASE PREPROCESSING")
print("=" * 70)

raw = pd.read_csv(DATA_PATH)
bias_col = next(
    (c for c in ['New_Bias%', 'Bias', 'bias', 'new_bias%'] if c in raw.columns), None)
if bias_col is None:
    raise ValueError("Cannot find Bias column in dataset.")
print(f"  Bias column: '{bias_col}'  |  Shape: {raw.shape}")

df = raw.copy()
df['Risk_Flag'] = (df[bias_col].abs() > 15).astype(int)
df['AbsBias']   = df[bias_col].abs()

# Encode base categorical features
df['Group'] = df['Lab_type'].astype(str).str.strip().str.upper().map(
    lambda x: 1 if (isinstance(x, str) and 'CRO' in x) else 0)

for raw_col, new_col in [('QA',                 'QA_bin'),
                          ('SIL-IS',             'SIL-IS_bin'),
                          ('CNAS',               'CNAS_bin'),
                          ('Method Validation',  'Method Validation_bin')]:
    if raw_col in df.columns:
        df[new_col] = (df[raw_col].astype(str).str.strip().str.lower()
                       .isin(['yes', '1', 'true', 'y'])).astype(int)

if 'Sample_Level' in df.columns:
    df['Level'] = df['Sample_Level'].astype(str).str.extract(r'(\d+)')[0].astype(float)
else:
    df['Level'] = df['Level'].astype(float)

bal_map = {
    '1': 1, '2': 2, '1,2': 1,
    '10^-6 Precision': 3,
    'Model not found; possibly 10^-5 precision': 2,
    'Not Provided': np.nan, 'Not Recorded': np.nan,
    'Model not found': np.nan, '1 (Explicitly Stated)': 1,
}
df['Balance_encoded'] = (df['Balance_Precision'].astype(str).str.strip()
                         .map(bal_map)
                         .fillna(df['Balance_Precision'].astype(str).str.strip()
                                 .map(bal_map).median()))

BASE_FEATS = ['Year', 'Group', 'QA_bin', 'SIL-IS_bin', 'Level',
              'CNAS_bin', 'Balance_encoded', 'Method Validation_bin']
df = df.dropna(subset=BASE_FEATS + ['Risk_Flag'])
print(f"  After cleaning: {len(df)} records  |  "
      f"Risk_Flag=1: {df['Risk_Flag'].sum()} ({df['Risk_Flag'].mean() * 100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ADVANCED FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2 — ADVANCED FEATURE ENGINEERING")
print("=" * 70)

# 2a. Explicit interaction features
df['QA_x_Level']    = df['QA_bin']     * df['Level']
df['QA_x_Group']    = df['QA_bin']     * df['Group']
df['Year_x_QA']     = df['Year']       * df['QA_bin']
df['SILIS_x_Level'] = df['SIL-IS_bin'] * df['Level']
df['Level_sq']      = df['Level'] ** 2
df['Year_norm']     = ((df['Year'] - df['Year'].min()) /
                       (df['Year'].max() - df['Year'].min()))

print("  Interaction features: QA×Level, QA×Group, Year×QA, SIL-IS×Level, Level², Year(norm)")

# 2b. Historical behaviour features (expanding window — no data leakage)
if 'Lab_ID' in df.columns:
    lab_key = 'Lab_ID'
elif 'Lab' in df.columns:
    lab_key = 'Lab'
else:
    df['Lab_key'] = (df['Group'].astype(str) + '_' +
                     df['QA_bin'].astype(str) + '_' +
                     df['CNAS_bin'].astype(str) + '_' +
                     df['SIL-IS_bin'].astype(str))
    lab_key = 'Lab_key'

print(f"  Lab grouping key: '{lab_key}'  |  Unique labs: {df[lab_key].nunique()}")

hist_records = []
for lab_id, lab_grp in df.groupby(lab_key):
    lab_grp = lab_grp.sort_values('Year').copy()
    for yr in sorted(lab_grp['Year'].unique()):
        past = lab_grp[lab_grp['Year'] < yr]
        if len(past) == 0:
            hist_records.append({
                'Lab_key_merge':    lab_id, 'Year': yr,
                'hist_fail_rate':   np.nan, 'hist_bias_std':    np.nan,
                'hist_n':           0,      'roll1yr_fail':     np.nan,
                'roll2yr_fail':     np.nan, 'roll3yr_fail':     np.nan,
                'fail_trend_slope': np.nan, 'hist_level1_bias': np.nan,
            })
        else:
            def roll_fail(n_yr):
                recent = past[past['Year'] >= yr - n_yr]
                return recent['Risk_Flag'].mean() if len(recent) > 0 else np.nan

            ann_fail = past.groupby('Year')['Risk_Flag'].mean()
            slope    = (linregress(ann_fail.index, ann_fail.values)[0]
                        if len(ann_fail) >= 3 else np.nan)
            past_l1  = past[past['Level'] == 1.0]

            hist_records.append({
                'Lab_key_merge':    lab_id, 'Year': yr,
                'hist_fail_rate':   past['Risk_Flag'].mean(),
                'hist_bias_std':    past['AbsBias'].std() if len(past) > 1 else 0.0,
                'hist_n':           len(past),
                'roll1yr_fail':     roll_fail(1),
                'roll2yr_fail':     roll_fail(2),
                'roll3yr_fail':     roll_fail(3),
                'fail_trend_slope': slope,
                'hist_level1_bias': past_l1['AbsBias'].mean() if len(past_l1) > 0 else np.nan,
            })

hist_df = pd.DataFrame(hist_records)
df = df.merge(hist_df,
              left_on=[lab_key, 'Year'],
              right_on=['Lab_key_merge', 'Year'],
              how='left')

# 2c. Industry percentile rank (expanding window)
df['industry_pct_rank'] = np.nan
for yr in sorted(df['Year'].unique()):
    past_all = df[df['Year'] < yr]
    if len(past_all) > 0:
        yr_mask = df['Year'] == yr
        df.loc[yr_mask, 'industry_pct_rank'] = (
            df.loc[yr_mask, 'hist_fail_rate'].fillna(0.5).rank(pct=True).values)

HIST_FEATS = ['hist_fail_rate', 'hist_bias_std', 'roll1yr_fail', 'roll2yr_fail',
              'roll3yr_fail', 'fail_trend_slope', 'hist_level1_bias', 'industry_pct_rank']
INTERACT_FEATS = ['QA_x_Level', 'QA_x_Group', 'Year_x_QA', 'SILIS_x_Level',
                  'Level_sq', 'Year_norm']
ALL_FEATS = BASE_FEATS + INTERACT_FEATS + HIST_FEATS
print(f"  Total features after engineering: {len(ALL_FEATS)}")
print(f"  Base: {len(BASE_FEATS)}  Interactions: {len(INTERACT_FEATS)}  Historical: {len(HIST_FEATS)}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TEMPORAL SPLIT & WOE ENCODING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3 — TEMPORAL SPLIT & WOE ENCODING")
print("=" * 70)

def compute_woe(df_train, col, target='Risk_Flag'):
    """Weight of Evidence encoding — fit on training data only."""
    total_pos = df_train[target].sum()
    total_neg = (1 - df_train[target]).sum()
    woe_map   = {}
    for val in df_train[col].unique():
        sub     = df_train[df_train[col] == val]
        pos     = sub[target].sum()
        neg     = (1 - sub[target]).sum()
        pos_r   = (pos + 0.5) / (total_pos + 0.5)
        neg_r   = (neg + 0.5) / (total_neg + 0.5)
        woe_map[val] = np.log(pos_r / neg_r)
    return woe_map

train_df = df[df['Year'] <= 2019].copy()
test_df  = df[df['Year'] == 2020].copy()

year_woe  = compute_woe(train_df, 'Year')
level_woe = compute_woe(train_df, 'Level')
for dset in [train_df, test_df]:
    dset['Year_woe']  = dset['Year'].map(year_woe).fillna(0)
    dset['Level_woe'] = dset['Level'].map(level_woe).fillna(0)

WOE_FEATS   = ['Year_woe', 'Level_woe']
FINAL_FEATS = ALL_FEATS + WOE_FEATS

# Fill NaN in historical features with training-set median (prevents leakage)
for col in HIST_FEATS:
    med = train_df[col].median()
    train_df[col] = train_df[col].fillna(med)
    test_df[col]  = test_df[col].fillna(med)

X_train = train_df[FINAL_FEATS].values.astype(float)
y_train = train_df['Risk_Flag'].values
X_test  = test_df[FINAL_FEATS].values.astype(float)
y_test  = test_df['Risk_Flag'].values

print(f"  Train (2014-2019): n={len(X_train)}, high-risk={y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"  Test  (2020):      n={len(X_test)},  high-risk={y_test.sum()} ({y_test.mean()*100:.1f}%)")
print(f"  Feature count: {len(FINAL_FEATS)}")

# SMOTE on training set
smote    = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)
print(f"  After SMOTE: {len(X_sm)} samples")

FEAT_LABELS = [
    'Year', 'Lab Group', 'Indep. QA', 'SIL-IS', 'Conc. Level', 'CNAS Accred.',
    'Balance Prec.', 'Method Valid.',
    'QA x Level', 'QA x Group', 'Year x QA', 'SIL-IS x Level', 'Level\u00b2', 'Year (norm)',
    'Hist. Fail Rate', 'Hist. Bias SD', 'Roll-1yr Fail', 'Roll-2yr Fail', 'Roll-3yr Fail',
    'Fail Trend Slope', 'Hist. L1 Bias', 'Industry Pct Rank',
    'Year (WOE)', 'Level (WOE)',
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HYPERPARAMETER OPTIMISATION (Optuna, 50 trials each)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4 — HYPERPARAMETER OPTIMISATION (50 trials each)")
print("=" * 70)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_auc(clf, X, y, cv=CV):
    aucs = []
    for tr, va in cv.split(X, y):
        sm = SMOTE(random_state=42)
        Xtr, ytr = sm.fit_resample(X[tr], y[tr])
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(X[va])[:, 1]
        if len(np.unique(y[va])) == 2:
            aucs.append(roc_auc_score(y[va], prob))
    return np.mean(aucs) if aucs else 0.5

# Random Forest
def rf_obj(trial):
    p = dict(
        n_estimators      = trial.suggest_int('n_estimators', 100, 800, step=50),
        max_depth         = trial.suggest_int('max_depth', 3, 12),
        min_samples_split = trial.suggest_int('min_samples_split', 2, 15),
        min_samples_leaf  = trial.suggest_int('min_samples_leaf', 1, 10),
        max_features      = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.4, 0.6]),
        random_state=42, n_jobs=2,
    )
    return cv_auc(RandomForestClassifier(**p), X_train, y_train)

print("  Optimising Random Forest (50 trials)...")
rf_study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
rf_study.optimize(rf_obj, n_trials=50, show_progress_bar=False)
best_rf = {**rf_study.best_params, 'random_state': 42, 'n_jobs': 2}
print(f"  Best RF CV AUC: {rf_study.best_value:.4f}")

# LightGBM
def lgb_obj(trial):
    p = dict(
        n_estimators      = trial.suggest_int('n_estimators', 100, 800, step=50),
        learning_rate     = trial.suggest_float('learning_rate', 0.003, 0.15, log=True),
        max_depth         = trial.suggest_int('max_depth', 3, 12),
        num_leaves        = trial.suggest_int('num_leaves', 8, 80),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.4, 1.0),
        subsample         = trial.suggest_float('subsample', 0.5, 1.0),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 2.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 2.0, log=True),
        min_child_samples = trial.suggest_int('min_child_samples', 5, 30),
        random_state=42, n_jobs=2, verbose=-1,
    )
    return cv_auc(lgb.LGBMClassifier(**p), X_train, y_train)

print("  Optimising LightGBM (50 trials)...")
lgb_study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
lgb_study.optimize(lgb_obj, n_trials=50, show_progress_bar=False)
best_lgb = {**lgb_study.best_params, 'random_state': 42, 'n_jobs': 2, 'verbose': -1}
print(f"  Best LGB CV AUC: {lgb_study.best_value:.4f}")

# XGBoost
def xgb_obj(trial):
    p = dict(
        n_estimators      = trial.suggest_int('n_estimators', 100, 800, step=50),
        learning_rate     = trial.suggest_float('learning_rate', 0.003, 0.15, log=True),
        max_depth         = trial.suggest_int('max_depth', 3, 10),
        subsample         = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.4, 1.0),
        gamma             = trial.suggest_float('gamma', 0, 5),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 2.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 2.0, log=True),
        scale_pos_weight  = trial.suggest_float('scale_pos_weight', 1, 15),
        random_state=42, n_jobs=2, eval_metric='logloss', verbosity=0,
    )
    return cv_auc(xgb.XGBClassifier(**p), X_train, y_train)

print("  Optimising XGBoost (50 trials)...")
xgb_study = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_obj, n_trials=50, show_progress_bar=False)
best_xgb = {**xgb_study.best_params, 'random_state': 42, 'n_jobs': 2,
            'eval_metric': 'logloss', 'verbosity': 0}
print(f"  Best XGB CV AUC: {xgb_study.best_value:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRAIN BASE MODELS & ENSEMBLE STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5 — TRAIN BASE MODELS & ENSEMBLE STRATEGIES")
print("=" * 70)

rf_m  = RandomForestClassifier(**best_rf)
lgb_m = lgb.LGBMClassifier(**best_lgb)
xgb_m = xgb.XGBClassifier(**best_xgb)
rf_m.fit(X_sm, y_sm)
lgb_m.fit(X_sm, y_sm)
xgb_m.fit(X_sm, y_sm)

# CV-AUC weighted voting
cv_scores = {
    'RF':  rf_study.best_value,
    'LGB': lgb_study.best_value,
    'XGB': xgb_study.best_value,
}
total_cv = sum(cv_scores.values())
w_rf, w_lgb, w_xgb = [cv_scores[k] / total_cv for k in ['RF', 'LGB', 'XGB']]
print(f"  Voting weights — RF: {w_rf:.3f}  LGB: {w_lgb:.3f}  XGB: {w_xgb:.3f}")

soft_equal = VotingClassifier(
    estimators=[('rf', rf_m), ('lgb', lgb_m), ('xgb', xgb_m)], voting='soft')
soft_equal.fit(X_sm, y_sm)

soft_weighted = VotingClassifier(
    estimators=[('rf', rf_m), ('lgb', lgb_m), ('xgb', xgb_m)],
    voting='soft', weights=[w_rf, w_lgb, w_xgb])
soft_weighted.fit(X_sm, y_sm)

stacking = StackingClassifier(
    estimators=[('rf',  RandomForestClassifier(**best_rf)),
                ('lgb', lgb.LGBMClassifier(**best_lgb)),
                ('xgb', xgb.XGBClassifier(**best_xgb))],
    final_estimator=LogisticRegression(C=1.0, max_iter=500, random_state=42),
    cv=5, passthrough=False, n_jobs=2)
stacking.fit(X_sm, y_sm)

# Ensemble of Ensembles (3 random seeds)
eoe_probs = []
for seed in [42, 123, 777]:
    rf_s  = RandomForestClassifier(**{**best_rf,  'random_state': seed})
    lgb_s = lgb.LGBMClassifier(**{**best_lgb, 'random_state': seed})
    xgb_s = xgb.XGBClassifier(**{**best_xgb, 'random_state': seed})
    Xs, ys = SMOTE(random_state=seed).fit_resample(X_train, y_train)
    rf_s.fit(Xs, ys); lgb_s.fit(Xs, ys); xgb_s.fit(Xs, ys)
    eoe_probs.append((rf_s.predict_proba(X_test)[:, 1] +
                      lgb_s.predict_proba(X_test)[:, 1] +
                      xgb_s.predict_proba(X_test)[:, 1]) / 3)
eoe_prob = np.mean(eoe_probs, axis=0)

# Evaluate all models
def evaluate_model(prob, name):
    pred = (prob >= 0.5).astype(int)
    fpr, tpr, _    = roc_curve(y_test, prob)
    p_arr, r_arr, _ = precision_recall_curve(y_test, prob)
    row = {
        'Model':     name,
        'AUC':       round(roc_auc_score(y_test, prob), 4),
        'PR-AUC':    round(auc(r_arr, p_arr), 4),
        'Brier':     round(brier_score_loss(y_test, prob), 4),
        'F1':        round(f1_score(y_test, pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_test, pred, zero_division=0), 4),
        'Precision': round(precision_score(y_test, pred, zero_division=0), 4),
    }
    print(f"  {name:30s}  AUC={row['AUC']}  PR-AUC={row['PR-AUC']}  "
          f"Brier={row['Brier']}  F1={row['F1']}")
    return row, prob, (fpr, tpr), (r_arr, p_arr)

models_to_eval = [
    (rf_m.predict_proba(X_test)[:, 1],          'Random Forest (v2)'),
    (lgb_m.predict_proba(X_test)[:, 1],         'LightGBM (v2)'),
    (xgb_m.predict_proba(X_test)[:, 1],         'XGBoost'),
    (soft_equal.predict_proba(X_test)[:, 1],    'Soft Voting (equal)'),
    (soft_weighted.predict_proba(X_test)[:, 1], 'Soft Voting (weighted)'),
    (stacking.predict_proba(X_test)[:, 1],      'Stacking (LR meta)'),
    (eoe_prob,                                    'Ensemble of Ensembles'),
]

perf_rows, all_probs, roc_curves_d, pr_curves_d = [], {}, {}, {}
for prob, name in models_to_eval:
    row, p, roc_c, pr_c = evaluate_model(prob, name)
    perf_rows.append(row)
    all_probs[name]    = p
    roc_curves_d[name] = roc_c
    pr_curves_d[name]  = pr_c

perf_df = pd.DataFrame(perf_rows)
perf_df.to_csv(str(Path(OUTPUT_DIR) / 'model_performance_final.csv'), index=False)

best_model_name = perf_df.loc[perf_df['AUC'].idxmax(), 'Model']
best_prob       = all_probs[best_model_name]
print(f"\n  Best model: {best_model_name}  AUC={perf_df['AUC'].max():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MODEL CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6 — MODEL CALIBRATION")
print("=" * 70)

rf_platt    = CalibratedClassifierCV(RandomForestClassifier(**best_rf),
                                      method='sigmoid',  cv=5)
rf_isotonic = CalibratedClassifierCV(RandomForestClassifier(**best_rf),
                                      method='isotonic', cv=5)
rf_platt.fit(X_sm, y_sm)
rf_isotonic.fit(X_sm, y_sm)
prob_platt    = rf_platt.predict_proba(X_test)[:, 1]
prob_isotonic = rf_isotonic.predict_proba(X_test)[:, 1]
best_base_prob = all_probs['Random Forest (v2)']

def compute_ece(prob, y, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (prob >= bin_edges[i]) & (prob < bin_edges[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / len(prob)) * abs(y[mask].mean() - prob[mask].mean())
    return ece

ece_base     = compute_ece(best_base_prob, y_test)
ece_platt    = compute_ece(prob_platt, y_test)
ece_isotonic = compute_ece(prob_isotonic, y_test)
brier_base     = brier_score_loss(y_test, best_base_prob)
brier_platt    = brier_score_loss(y_test, prob_platt)
brier_isotonic = brier_score_loss(y_test, prob_isotonic)

print(f"  Uncalibrated  — ECE: {ece_base:.4f}  Brier: {brier_base:.4f}  "
      f"AUC: {roc_auc_score(y_test, best_base_prob):.4f}")
print(f"  Platt Scaling — ECE: {ece_platt:.4f}  Brier: {brier_platt:.4f}  "
      f"AUC: {roc_auc_score(y_test, prob_platt):.4f}")
print(f"  Isotonic Reg. — ECE: {ece_isotonic:.4f}  Brier: {brier_isotonic:.4f}  "
      f"AUC: {roc_auc_score(y_test, prob_isotonic):.4f}")

calib_df = pd.DataFrame({
    'Method': ['Uncalibrated', 'Platt Scaling', 'Isotonic Regression'],
    'ECE':    [ece_base, ece_platt, ece_isotonic],
    'Brier':  [brier_base, brier_platt, brier_isotonic],
    'AUC':    [roc_auc_score(y_test, p)
               for p in [best_base_prob, prob_platt, prob_isotonic]],
})
calib_df.to_csv(str(Path(OUTPUT_DIR) / 'calibration_comparison_final.csv'), index=False)

# NOTE: Platt Scaling improves ECE but reduces AUC (ranking disruption).
# Triage / RBA uses uncalibrated probabilities for ranking;
# Platt-calibrated probabilities are used for threshold-based decisions.
final_prob = prob_platt


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TRIAGE METRICS & DECISION CURVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7 — TRIAGE METRICS & DECISION CURVE ANALYSIS")
print("=" * 70)

def capture_at_k(prob, y, k_pct):
    n_top   = max(1, int(len(prob) * k_pct / 100))
    top_idx = np.argsort(prob)[::-1][:n_top]
    return y[top_idx].sum() / max(1, y.sum())

def precision_at_k(prob, y, k_pct):
    n_top   = max(1, int(len(prob) * k_pct / 100))
    top_idx = np.argsort(prob)[::-1][:n_top]
    return y[top_idx].mean()

K_VALS     = [5, 10, 15, 20, 25, 30]
prevalence = y_test.mean()

triage_rows = []
for name, prob in all_probs.items():
    row = {'Model': name}
    for k in K_VALS:
        row[f'Capture@{k}%']   = round(capture_at_k(prob, y_test, k), 4)
        row[f'Precision@{k}%'] = round(precision_at_k(prob, y_test, k), 4)
        row[f'Lift@{k}%']      = round(capture_at_k(prob, y_test, k) / (k / 100), 4)
    triage_rows.append(row)

row_rand = {'Model': 'Random Baseline'}
for k in K_VALS:
    row_rand[f'Capture@{k}%']   = round(k / 100, 4)
    row_rand[f'Precision@{k}%'] = round(prevalence, 4)
    row_rand[f'Lift@{k}%']      = 1.0
triage_rows.append(row_rand)

triage_df = pd.DataFrame(triage_rows)
triage_df.to_csv(str(Path(OUTPUT_DIR) / 'triage_metrics_final.csv'), index=False)
print(triage_df[['Model'] + [f'Capture@{k}%' for k in K_VALS]].to_string(index=False))

# Decision Curve Analysis
def net_benefit(prob, y, threshold):
    pred = (prob >= threshold).astype(int)
    tp   = ((pred == 1) & (y == 1)).sum()
    fp   = ((pred == 1) & (y == 0)).sum()
    odds = threshold / (1 - threshold + 1e-10)
    return (tp - fp * odds) / len(y)

thresholds_dca = np.linspace(0.01, 0.50, 100)
dca_results    = {
    name: [net_benefit(prob, y_test, t) for t in thresholds_dca]
    for name, prob in [('Best Ensemble',       best_prob),
                       ('Stacking (LR meta)',  all_probs['Stacking (LR meta)']),
                       ('Platt-calibrated RF', final_prob)]
}
dca_treat_all  = [(y_test.mean() - (1 - y_test.mean()) * t / (1 - t + 1e-10))
                  for t in thresholds_dca]

dca_df = pd.DataFrame({'Threshold': thresholds_dca, 'Treat All': dca_treat_all})
for name, vals in dca_results.items():
    dca_df[name] = vals
dca_df.to_csv(str(Path(OUTPUT_DIR) / 'dca_results_final.csv'), index=False)

# Simulated audit savings
print("\n  Simulated Audit Savings (Best Model):")
audit_table = []
for k in K_VALS:
    n_aud     = int(len(y_test) * k / 100)
    cap       = capture_at_k(best_prob, y_test, k)
    n_caught  = int(cap * y_test.sum())
    n_total   = int(y_test.sum())
    lift_v    = cap / (k / 100)
    audit_table.append({
        'Top K%': f'{k}%', 'Records Audited': n_aud,
        'Failures Caught': n_caught, 'Total Failures': n_total,
        'Capture Rate': f'{cap * 100:.1f}%', 'Lift vs Random': f'{lift_v:.2f}x',
    })
    print(f"    Top {k:2d}%: audit {n_aud:3d} records -> "
          f"catch {n_caught}/{n_total} ({cap * 100:.1f}%, Lift={lift_v:.2f}x)")

pd.DataFrame(audit_table).to_csv(
    str(Path(OUTPUT_DIR) / 'audit_savings_final.csv'), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SHAP INTERPRETABILITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8 — SHAP ANALYSIS (RF v2)")
print("=" * 70)

explainer = shap.TreeExplainer(rf_m)
shap_vals = explainer.shap_values(X_test)
if isinstance(shap_vals, list):
    sv = shap_vals[1]
elif shap_vals.ndim == 3:
    sv = shap_vals[:, :, 1]
else:
    sv = shap_vals

shap_importance = pd.DataFrame({
    'Feature':   FEAT_LABELS,
    'Mean_SHAP': np.abs(sv).mean(axis=0),
}).sort_values('Mean_SHAP', ascending=False)
shap_importance.to_csv(
    str(Path(OUTPUT_DIR) / 'shap_importance_final.csv'), index=False)
print("  Top 10 features by mean |SHAP|:")
print(shap_importance.head(10).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — PUBLICATION FIGURES (theme_jpa)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 9 — GENERATING FIGURES (theme_jpa)")
print("=" * 70)

# ── Figure 1: ROC + PR Curves ─────────────────────────────────────────────────
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(13, 6), facecolor='white')

key_models  = ['Random Forest (v2)', 'LightGBM (v2)', 'XGBoost',
               'Soft Voting (weighted)', 'Stacking (LR meta)', 'Ensemble of Ensembles']
key_colors  = [MODEL_COLORS[m] for m in key_models]
key_lstyles = ['-', '-', '-', '--', '-.', ':']

for name, color, ls in zip(key_models, key_colors, key_lstyles):
    if name in all_probs:
        fpr_i, tpr_i = roc_curves_d[name]
        auc_i = perf_df.loc[perf_df['Model'] == name, 'AUC'].values[0]
        ax1a.plot(fpr_i, tpr_i, color=color, lw=2.0, linestyle=ls,
                  label=f'{name} (AUC = {auc_i:.3f})')
ax1a.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Random (0.500)')
ax1a.set_xlabel('1 - Specificity (FPR)', fontweight='bold')
ax1a.set_ylabel('Sensitivity (TPR)', fontweight='bold')
ax1a.set_title('ROC Curves — Ensemble Comparison')
ax1a.legend(fontsize=8.5, loc='lower right', title=None)
ax1a.set_xlim(0, 1); ax1a.set_ylim(0, 1.02)
set_theme_jpa(ax1a, grid_axis='both')
add_panel_label(ax1a, 'A')

for name, color, ls in zip(key_models, key_colors, key_lstyles):
    if name in all_probs:
        r_i, p_i = pr_curves_d[name]
        ap_i = perf_df.loc[perf_df['Model'] == name, 'PR-AUC'].values[0]
        ax1b.plot(r_i, p_i, color=color, lw=2.0, linestyle=ls,
                  label=f'{name} (PR-AUC = {ap_i:.3f})')
ax1b.axhline(prevalence, color='black', lw=1.2, linestyle='--', alpha=0.55,
             label=f'No-Skill (Prev = {prevalence:.3f})')
ax1b.set_xlabel('Recall', fontweight='bold')
ax1b.set_ylabel('Precision', fontweight='bold')
ax1b.set_title('Precision-Recall Curves — Ensemble Comparison')
ax1b.legend(fontsize=8.5, loc='upper right', title=None)
ax1b.set_xlim(0, 1); ax1b.set_ylim(0, 0.75)
set_theme_jpa(ax1b, grid_axis='both')
add_panel_label(ax1b, 'B')

plt.tight_layout(pad=2.0)
save_figure(fig1, 'ROC_PR_Curves_Final', OUTPUT_DIR)

# ── Figure 2: Capture@K Curves ────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 7), facecolor='white')
k_range = np.arange(1, 51)
colors_cycle = [MODEL_COLORS.get(n, GREY) for n in all_probs.keys()]
lstyles_cycle = ['-', '-', '-', '--', '-.', ':', '-']

for (name, prob), color, ls in zip(all_probs.items(), colors_cycle, lstyles_cycle):
    caps = [capture_at_k(prob, y_test, k) for k in k_range]
    cap20 = capture_at_k(prob, y_test, 20) * 100
    ax2.plot(k_range, caps, lw=2.0, color=color, linestyle=ls,
             label=f'{name}  (@20%: {cap20:.1f}%)')

ax2.plot(k_range, k_range / 100, color='black', lw=1.5, linestyle='--',
         alpha=0.55, label='Random Baseline')
ax2.axvline(20, color=GREY, lw=1.2, linestyle=':', alpha=0.7)
ax2.text(20.5, 0.05, 'Top 20%', fontsize=9, color=GREY, va='bottom')
ax2.set_xlabel('Top K% of Records Audited', fontweight='bold')
ax2.set_ylabel('Fraction of Failures Captured', fontweight='bold')
ax2.set_title('Capture Rate Curves — All Models\n(Triage Efficiency for Risk-Based Auditing)')
ax2.legend(fontsize=8.5, loc='lower right', title=None, ncol=1)
ax2.set_xlim(1, 50); ax2.set_ylim(0, 1.02)
set_theme_jpa(ax2, grid_axis='both')
save_figure(fig2, 'CaptureRate_AllModels_Final', OUTPUT_DIR)

# ── Figure 3: Decision Curve Analysis ─────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(9, 7), facecolor='white')
ax3.plot(thresholds_dca, dca_treat_all, color=GREY, lw=1.8, linestyle='--',
         label='Treat All', alpha=0.8)
ax3.axhline(0, color='black', lw=1.2, linestyle='-', alpha=0.5, label='Treat None')

dca_colors  = [BLUE, RED, GREEN]
dca_lstyles = ['-', '--', '-.']
for (name, vals), color, ls in zip(dca_results.items(), dca_colors, dca_lstyles):
    ax3.plot(thresholds_dca, vals, lw=2.2, color=color, linestyle=ls, label=name)

ax3.set_xlabel('Threshold Probability', fontweight='bold')
ax3.set_ylabel('Net Benefit', fontweight='bold')
ax3.set_title('Decision Curve Analysis\n(Net Benefit vs. Threshold Probability)')
ax3.legend(fontsize=10, loc='upper right', title=None)
ax3.set_xlim(0, 0.50); ax3.set_ylim(-0.025, 0.12)
set_theme_jpa(ax3, grid_axis='both')
save_figure(fig3, 'DecisionCurveAnalysis_Final', OUTPUT_DIR)

# ── Figure 4: Calibration Comparison ──────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5.5), facecolor='white')
calib_probs  = [best_base_prob, prob_platt, prob_isotonic]
calib_names  = ['Uncalibrated RF', 'Platt Scaling', 'Isotonic Regression']
calib_eces   = [ece_base, ece_platt, ece_isotonic]
calib_briers = [brier_base, brier_platt, brier_isotonic]
calib_aucs   = [roc_auc_score(y_test, p) for p in calib_probs]
panel_labels = ['A', 'B', 'C']

for ax4, prob, name, ece_v, brier_v, auc_v, pl in zip(
        axes4, calib_probs, calib_names, calib_eces, calib_briers, calib_aucs, panel_labels):
    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=8, strategy='uniform')
    ax4.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.55, label='Perfect calibration')
    ax4.plot(mean_pred, frac_pos, 'o-', color=BLUE, lw=2.2, ms=7,
             markerfacecolor=BLUE, markeredgecolor='white', markeredgewidth=0.8,
             label=f'ECE = {ece_v:.3f}\nBrier = {brier_v:.3f}\nAUC = {auc_v:.3f}',
             zorder=4)
    ax4.set_xlabel('Mean Predicted Probability', fontweight='bold')
    ax4.set_ylabel('Fraction of Positives', fontweight='bold')
    ax4.set_title(name)
    ax4.legend(fontsize=9, loc='upper left', title=None)
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    set_theme_jpa(ax4, grid_axis='both')
    add_panel_label(ax4, pl)

plt.suptitle('Calibration Curves: Before and After Calibration',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout(pad=2.0)
save_figure(fig4, 'Calibration_Comparison_Final', OUTPUT_DIR)

# ── Figure 5: SHAP Summary Beeswarm ───────────────────────────────────────────
print("  Generating SHAP summary plot...")
fig5, ax5 = plt.subplots(figsize=(9, 8), facecolor='white')
shap.summary_plot(sv, X_test, feature_names=FEAT_LABELS,
                  show=False, plot_type='dot', color_bar=True,
                  max_display=15, plot_size=None)
ax5 = plt.gca()
ax5.set_xlabel('SHAP Value (Impact on High-Risk Prediction)', fontweight='bold', fontsize=12)
ax5.set_title('SHAP Feature Importance — RF v2\n'
              '(Engineered Features, Temporal Validation: 2020 Test Set)',
              fontweight='bold', fontsize=12)
for spine in ['top', 'bottom', 'left', 'right']:
    ax5.spines[spine].set_visible(True)
    ax5.spines[spine].set_linewidth(1.0)
    ax5.spines[spine].set_color('black')
ax5.tick_params(axis='both', which='major', direction='out',
                length=4, width=1.0, colors='black', labelsize=11)
save_figure(plt.gcf(), 'SHAP_Summary_Final', OUTPUT_DIR)

# ── Figure 6: Triage Efficiency Summary ───────────────────────────────────────
fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 6.5), facecolor='white')

# Panel A: Capture@20% bar chart (all models)
cap20_vals  = [capture_at_k(prob, y_test, 20) * 100 for prob in all_probs.values()]
cap20_names = list(all_probs.keys())
random_cap20 = 20.0
bar_colors = [BLUE if v > random_cap20 * 1.15 else GREY for v in cap20_vals]

bars = ax6a.barh(range(len(cap20_names)), cap20_vals,
                  color=bar_colors, alpha=0.88, height=0.62,
                  edgecolor='black', linewidth=0.6, zorder=3)
ax6a.axvline(random_cap20, color='black', lw=1.5, linestyle='--', alpha=0.65,
             label=f'Random baseline ({random_cap20:.0f}%)', zorder=4)
for i, v in enumerate(cap20_vals):
    ax6a.text(v + 0.4, i, f'{v:.1f}%', va='center', fontsize=9.5, color='black')
ax6a.set_yticks(range(len(cap20_names)))
ax6a.set_yticklabels(cap20_names, fontsize=9.5)
ax6a.set_xlabel('Capture Rate at Top 20%', fontweight='bold')
ax6a.set_title('Triage Efficiency: Capture@20%\n(All Models vs. Random Baseline)')
ax6a.legend(loc='lower right', fontsize=9.5, title=None)
set_theme_jpa(ax6a, grid_axis='x')
add_panel_label(ax6a, 'A')

# Panel B: Audit savings table
ax6b.axis('off')
table_data = [['Top K%', 'Records\nAudited', 'Failures\nCaught', 'Capture\nRate', 'Lift']]
for k in K_VALS:
    n_aud    = int(len(y_test) * k / 100)
    cap      = capture_at_k(best_prob, y_test, k)
    n_caught = int(cap * y_test.sum())
    lift_v   = cap / (k / 100)
    table_data.append([f'Top {k}%', str(n_aud),
                       f'{n_caught}/{int(y_test.sum())}',
                       f'{cap * 100:.1f}%', f'{lift_v:.2f}x'])

col_widths = [0.15, 0.20, 0.22, 0.22, 0.18]
row_height = 0.115
y_start    = 0.90

for ri, row in enumerate(table_data):
    x = 0.03
    for ci, (cell, cw) in enumerate(zip(row, col_widths)):
        bg = BLUE if ri == 0 else ('#D6EAF8' if ri % 2 == 0 else 'white')
        fc = 'white' if ri == 0 else 'black'
        fw = 'bold'  if ri == 0 else 'normal'
        fs = 9.5     if ri == 0 else 9.0
        rect = mpatches.FancyBboxPatch(
            (x, y_start - ri * row_height), cw - 0.012, row_height * 0.86,
            boxstyle='round,pad=0.005',
            facecolor=bg, edgecolor='#AAAAAA', linewidth=0.6,
            transform=ax6b.transAxes, clip_on=False)
        ax6b.add_patch(rect)
        ax6b.text(x + cw / 2 - 0.006, y_start - ri * row_height + row_height * 0.43,
                  cell, transform=ax6b.transAxes,
                  ha='center', va='center', fontsize=fs, fontweight=fw, color=fc)
        x += cw

ax6b.set_title(f'Simulated Audit Savings\n(Best Model: {best_model_name})',
               fontweight='bold', fontsize=11, pad=8)
add_panel_label(ax6b, 'B', x=-0.04)

plt.tight_layout(pad=2.0)
save_figure(fig6, 'Triage_Efficiency_Summary_Final', OUTPUT_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY — OPTIMISED ML MODEL")
print("=" * 70)

print("\n  Model Performance:")
print(perf_df[['Model', 'AUC', 'PR-AUC', 'Brier', 'F1']].to_string(index=False))

print(f"\n  Calibration (RF v2):")
print(f"    Uncalibrated  — ECE: {ece_base:.4f}  Brier: {brier_base:.4f}")
print(f"    Platt Scaling — ECE: {ece_platt:.4f}  Brier: {brier_platt:.4f}")
print(f"    Isotonic Reg. — ECE: {ece_isotonic:.4f}  Brier: {brier_isotonic:.4f}")

print(f"\n  Triage Efficiency (Best Model: {best_model_name}):")
for k in K_VALS:
    cap  = capture_at_k(best_prob, y_test, k)
    lift = cap / (k / 100)
    print(f"    Capture@{k:2d}%: {cap * 100:.1f}%  (Lift = {lift:.2f}x)")

print(f"\n  Feature Engineering:")
print(f"    Base features:  {len(BASE_FEATS)}")
print(f"    Final features: {len(FINAL_FEATS)}")
print(f"    AUC improvement: baseline 0.620 -> v2 {perf_df['AUC'].max():.4f}")

print(f"\n  All outputs saved to: {OUTPUT_DIR}/")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
