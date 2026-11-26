# =============================================================================
# SCRIPT 03 — CROSS-DATASET ANALYSIS
# =============================================================================
# Purpose:
#   Take a trained model from ONE dataset (the "reference", e.g. CIC18) and
#   evaluate it on ALL other prepared datasets (e.g. CIC17). This answers:
#   "How well does a model trained on CIC18 generalize to CIC17?"
#
# Prerequisites:
#   1. Run 00_prepare_datasets.py   (prepare all datasets)
#   2. Run 01_train_save_models.py  (train models — at least the reference)
#
# Usage:
#   1. Set REFERENCE_MODEL_DIR to the model trained on your reference dataset
#   2. python 03_cross_dataset_analysis.py
#
# Output:
#   cross_dataset_results/from_CIC18/
#   ├── cross_dataset_metrics.csv        (full results table)
#   ├── cross_dataset_heatmap.png        (MCC/F1 heatmap)
#   ├── cross_dataset_bar.png            (bar chart comparison)
#   ├── generalization_gap.png           (within vs cross gap)
#   ├── confusion_matrices/
#   │   ├── cm_CIC18.png
#   │   ├── cm_CIC17.png
#   │   └── ...
#   ├── shap_stability/
#   │   ├── shap_ranking_<dataset>.png
#   │   ├── shap_ranking_heatmap.png     (feature rank stability)
#   │   ├── shap_pairwise_correlation.csv
#   │   └── shap_stability_report.csv
#   ├── lime_stability/
#   │   ├── lime_ranking_<dataset>.png
#   │   ├── lime_ranking_heatmap.png
#   │   ├── lime_pairwise_correlation.csv
#   │   └── lime_stability_report.csv
#   ├── xai_consistency/
#   │   ├── shap_vs_lime_<dataset>.png
#   │   └── shap_vs_lime_spearman.csv
#   └── cross_dataset_report.txt         (text summary)
# =============================================================================

# ┌─────────────────────────────────────────────────────┐
# │  AMD AOCL ENVIRONMENT TUNING                        │
# └─────────────────────────────────────────────────────┘
import os, sys, psutil
from log_utils import setup_run_logger

if __name__ == "__main__":
    setup_run_logger("log_03", __file__)

N_PHYSICAL = psutil.cpu_count(logical=False) or 4

os.environ["BLIS_NUM_THREADS"]       = str(N_PHYSICAL)
os.environ["OMP_NUM_THREADS"]        = str(N_PHYSICAL)
os.environ["OPENBLAS_NUM_THREADS"]   = str(N_PHYSICAL)
os.environ["NUMEXPR_NUM_THREADS"]    = str(N_PHYSICAL)
os.environ["OPENBLAS_CORETYPE"]      = "ZEN"

os.environ["OMP_PROC_BIND"] = "true"
os.environ["OMP_PLACES"]    = "cores"
os.environ["OMP_DYNAMIC"]   = "FALSE"

print(f"[AMD TUNING] ✔ Optimized for {N_PHYSICAL} physical threads (AOCL/OpenBLAS).")

# =============================================================================
import time, warnings, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
    classification_report
)

import shap
import lime
import lime.lime_tabular

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")


# =============================================================================
# CONFIGURATION  ← EDIT THESE
# =============================================================================

# ── Reference model (trained on ONE dataset via Script 01) ──
# Dataset utama = CIC-IDS-2018 ("CIC18"); target pengujian silang = CIC-IDS-2017 ("CIC17")
REFERENCE_MODEL_DIR = "model_outputs-v5/CIC18"

# ── Where prepared datasets live (from Script 00) ──
PREPARED_ROOT       = "prepared_data"

# ── Output folder ──
# Auto-named: cross_dataset_results/from_<reference>/
_ref_name = os.path.basename(REFERENCE_MODEL_DIR.rstrip("/\\"))
OUTPUT_DIR = os.path.join("cross_dataset_results", f"from_{_ref_name}")

# ── Which model to use ──
# "best" loads best_model.pkl, or specify e.g. "Random_Forest"
MODEL_TO_USE = "best"

# ── SHAP settings ──
RUN_SHAP        = True      # set False to skip SHAP (faster)
SHAP_MAX_ROWS   = 300       # samples per dataset for SHAP

# ── LIME settings ──
RUN_LIME        = True      # set False to skip LIME (faster)
LIME_AGG_N      = 100       # samples per dataset for aggregated LIME importance
LIME_SAMPLES    = 5         # individual instances to plot per dataset

RANDOM_STATE    = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "confusion_matrices"), exist_ok=True)
if RUN_SHAP:
    os.makedirs(os.path.join(OUTPUT_DIR, "shap_stability"), exist_ok=True)
if RUN_LIME:
    os.makedirs(os.path.join(OUTPUT_DIR, "lime_stability"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "lime_stability", "instances"), exist_ok=True)
if RUN_SHAP and RUN_LIME:
    os.makedirs(os.path.join(OUTPUT_DIR, "xai_consistency"), exist_ok=True)


# =============================================================================
# LOAD REFERENCE MODEL + ARTIFACTS
# =============================================================================

def load_reference_model():
    print("\n" + "="*60)
    print(f"  LOADING REFERENCE MODEL: {REFERENCE_MODEL_DIR}")
    print("="*60)

    if MODEL_TO_USE == "best":
        model_path = os.path.join(REFERENCE_MODEL_DIR, "best_model.pkl")
    else:
        model_path = os.path.join(REFERENCE_MODEL_DIR, f"{MODEL_TO_USE}.pkl")

    model    = joblib.load(model_path)
    scaler   = joblib.load(os.path.join(REFERENCE_MODEL_DIR, "scaler.pkl"))
    features = joblib.load(os.path.join(REFERENCE_MODEL_DIR, "feature_names.pkl"))

    try:
        config = joblib.load(os.path.join(REFERENCE_MODEL_DIR, "train_config.pkl"))
        ref_name    = config.get("DATASET_NAME", _ref_name)
        binary_mode = config.get("BINARY_MODE", True)
    except Exception:
        ref_name    = _ref_name
        binary_mode = True

    try:
        le = joblib.load(os.path.join(REFERENCE_MODEL_DIR, "label_encoder.pkl"))
        class_names = list(le.classes_)
    except Exception:
        class_names = ["Normal", "Attack"]

    print(f"  ✔ Model        : {type(model).__name__}  ← {model_path}")
    print(f"  ✔ Features ({len(features)}): {features[:5]}...")
    print(f"  ✔ Trained on   : {ref_name}")
    print(f"  ✔ Mode         : {'Binary' if binary_mode else 'Multi-class'}")
    print(f"  ✔ Classes      : {class_names}")

    return model, scaler, features, ref_name, binary_mode, class_names


# =============================================================================
# DISCOVER & LOAD TARGET DATASETS
# =============================================================================

def discover_datasets():
    """Find all prepared datasets in PREPARED_ROOT."""
    candidates = sorted([
        d for d in os.listdir(PREPARED_ROOT)
        if os.path.isdir(os.path.join(PREPARED_ROOT, d))
        and os.path.exists(os.path.join(PREPARED_ROOT, d, "X_test_raw.pkl"))
    ])
    print(f"\n  Discovered {len(candidates)} prepared datasets: {candidates}")
    return candidates


def load_and_prepare_target(ds_name, features, scaler, binary_mode):
    """
    Load a target dataset's TEST partition, select the reference model's
    features, fill missing, and apply the reference scaler.

    For within-dataset evaluation (target == reference), we use the saved
    X_test.pkl from model_outputs/ which is already scaled.
    """
    prep_dir = os.path.join(PREPARED_ROOT, ds_name)

    # Load raw test data
    X_test_raw = joblib.load(os.path.join(prep_dir, "X_test_raw.pkl"))
    y_test     = joblib.load(os.path.join(prep_dir, "y_test.pkl"))

    # Also load train for potential SHAP background
    X_train_raw = joblib.load(os.path.join(prep_dir, "X_train_raw.pkl"))

    # Select features from reference model — fill missing with 0
    missing = [f for f in features if f not in X_test_raw.columns]
    if missing:
        print(f"    ⚠ {len(missing)} features missing → filled with 0: {missing[:5]}...")
        for f in missing:
            X_test_raw[f]  = 0.0
            X_train_raw[f] = 0.0

    X_test_sel  = X_test_raw[features].fillna(0)
    X_train_sel = X_train_raw[features].fillna(0)

    # Apply reference scaler
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_sel),
        columns=features
    ).reset_index(drop=True)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train_sel),
        columns=features
    ).reset_index(drop=True)

    y_test = y_test.reset_index(drop=True)

    n_features_matched = len(features) - len(missing)
    print(f"    Loaded: {len(X_test_scaled):,} test rows | "
          f"Features matched: {n_features_matched}/{len(features)}")

    return X_test_scaled, y_test, X_train_scaled, n_features_matched


# =============================================================================
# EVALUATE MODEL ON ONE TARGET DATASET
# =============================================================================

def evaluate_on_target(model, X_test, y_test, ds_name, ref_name, binary_mode, class_names):
    """Evaluate the reference model on one target dataset."""
    is_within = (ds_name == ref_name) or (ds_name in ref_name) or (ref_name in ds_name)
    exp_type  = "Within" if is_within else "Cross"

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    mcc    = matthews_corrcoef(y_test, y_pred)

    try:
        y_prob = model.predict_proba(X_test)
        if binary_mode:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        auc = None

    metrics = {
        "Target": ds_name, "Type": exp_type,
        "Accuracy": round(acc, 4), "F1": round(f1, 4),
        "Precision": round(prec, 4), "Recall": round(rec, 4),
        "MCC": round(mcc * 100, 2),   # percentage
        "AUC-ROC": round(auc, 4) if auc else None,
        "N_test": len(y_test),
    }

    print(f"    {exp_type:6s} | F1={f1:.4f} | MCC={mcc*100:6.2f}% | "
          f"Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f}", end="")
    if auc: print(f" | AUC={auc:.4f}")
    else: print()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5) if binary_mode else (8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, linecolor="gray")
    cm_title = f"Train: {ref_name} → Test: {ds_name}"
    plt.title(cm_title, fontweight="bold", fontsize=11)
    plt.ylabel("Actual"); plt.xlabel("Predicted"); plt.tight_layout()
    safe = ds_name.replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices", f"cm_{safe}.png"), dpi=150)
    plt.close()

    # Classification report to text
    report_txt = classification_report(y_test, y_pred, zero_division=0)

    return metrics, report_txt


# =============================================================================
# SHAP PER TARGET DATASET
# =============================================================================

def compute_shap_for_target(model, X_test, features, ds_name):
    """Compute SHAP feature importance on a target dataset."""
    n = min(SHAP_MAX_ROWS, len(X_test))
    X_sample = X_test.iloc[:n].values.astype(np.float64)

    tree_models = {"RandomForestClassifier", "DecisionTreeClassifier",
                   "GradientBoostingClassifier", "XGBClassifier", "LGBMClassifier"}
    model_type = type(model).__name__

    if model_type not in tree_models:
        print(f"    SHAP skipped for {model_type} (not tree-based)")
        return None

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)

        if isinstance(sv, list):
            if len(sv) == 2:
                vals = np.abs(np.array(sv[1]))
            else:
                vals = np.abs(np.stack([np.array(s) for s in sv], axis=-1)).mean(axis=-1)
        else:
            vals = np.abs(np.array(sv))

        if vals.ndim == 3:
            vals = vals.mean(axis=-1)

        # Trim if needed
        n_feat = min(vals.shape[1], len(features))
        mean_abs = vals[:, :n_feat].mean(axis=0).flatten()
        importance = pd.Series(mean_abs, index=features[:n_feat])

        # Save bar plot
        plt.figure(figsize=(10, 6))
        importance.nlargest(15).sort_values().plot(
            kind="barh", color="steelblue", edgecolor="black", alpha=0.85)
        plt.title(f"SHAP Importance — Model: {_ref_name} → Data: {ds_name}",
                  fontsize=11, fontweight="bold")
        plt.xlabel("Mean |SHAP|"); plt.tight_layout()
        safe = ds_name.replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_stability",
                    f"shap_ranking_{safe}.png"), dpi=150)
        plt.close()

        return importance

    except Exception as e:
        print(f"    ⚠ SHAP failed for {ds_name}: {e}")
        return None


# =============================================================================
# SHAP STABILITY ANALYSIS
# =============================================================================

def analyze_shap_stability(shap_rankings):
    """
    Compare SHAP feature rankings across datasets.
    High stability = the model relies on the same features regardless of data.
    Low stability  = features that matter change per dataset (explains poor transfer).
    """
    if len(shap_rankings) < 2:
        print("  ⚠ Need ≥2 datasets for stability analysis")
        return

    print("\n" + "="*60)
    print("  SHAP FEATURE STABILITY ANALYSIS")
    print("="*60)

    # Build rank DataFrame
    rank_data = {}
    for ds_name, importance in shap_rankings.items():
        rank_data[ds_name] = importance.rank(ascending=False)

    df_ranks = pd.DataFrame(rank_data)

    # Average rank + stability (std)
    df_ranks["Avg Rank"] = df_ranks.mean(axis=1)
    df_ranks["Std"]      = df_ranks.drop(columns=["Avg Rank"]).std(axis=1)
    df_ranks = df_ranks.sort_values("Avg Rank")

    # Top 15 for heatmap
    top15 = df_ranks.head(15).index
    df_heatmap = df_ranks.loc[top15].drop(columns=["Avg Rank", "Std"])

    plt.figure(figsize=(max(10, len(shap_rankings)*2.5), 7))
    sns.heatmap(df_heatmap, annot=True, fmt=".0f", cmap="YlOrRd_r",
                linewidths=0.5, linecolor="white")
    plt.title(f"SHAP Feature Ranking Stability\n"
              f"(Model trained on {_ref_name} — lower rank = more important)",
              fontsize=12, fontweight="bold")
    plt.xlabel("Target Dataset"); plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_stability",
                "shap_ranking_heatmap.png"), dpi=150)
    plt.close()
    print("  ✔ shap_ranking_heatmap.png")

    # Spearman correlations between all pairs
    from itertools import combinations
    ds_names = [c for c in df_ranks.columns if c not in ["Avg Rank", "Std"]]

    if len(ds_names) >= 2:
        try:
            from scipy.stats import spearmanr
            corr_rows = []
            for a, b in combinations(ds_names, 2):
                rho, pval = spearmanr(df_ranks[a], df_ranks[b])
                corr_rows.append({"Dataset A": a, "Dataset B": b,
                                  "Spearman rho": round(rho, 4),
                                  "p-value": f"{pval:.4e}"})
                print(f"    {a} ↔ {b}: rho={rho:.4f} (p={pval:.2e})")
            df_corr = pd.DataFrame(corr_rows)
            df_corr.to_csv(os.path.join(OUTPUT_DIR, "shap_stability",
                           "shap_pairwise_correlation.csv"), index=False)
        except ImportError:
            print("  ⚠ scipy not installed — Spearman skipped")

    # Stable vs unstable features
    print(f"\n  ── Top 10 Most Stable Features (low std across datasets) ──")
    stable = df_ranks.sort_values("Std").head(10)
    for i, (feat, row) in enumerate(stable.iterrows(), 1):
        mark = "★" if row["Std"] < 3 else "△"
        print(f"    {i:2}. {feat:35s} avg_rank={row['Avg Rank']:5.1f}  "
              f"std={row['Std']:4.1f} {mark}")

    print(f"\n  ── Top 5 Most Unstable Features (high std) ──")
    unstable = df_ranks.sort_values("Std", ascending=False).head(5)
    for i, (feat, row) in enumerate(unstable.iterrows(), 1):
        print(f"    {i:2}. {feat:35s} avg_rank={row['Avg Rank']:5.1f}  "
              f"std={row['Std']:4.1f} ✗")

    # Save full report
    df_report = df_ranks.round(2)
    df_report.to_csv(os.path.join(OUTPUT_DIR, "shap_stability",
                     "shap_stability_report.csv"))
    print(f"\n  ✔ shap_stability_report.csv saved")


# =============================================================================
# LIME PER TARGET DATASET & STABILITY ANALYSIS (CROSS-DATASET)
# =============================================================================

def _stratified_sample_for_lime(X_test, y_test, n_target):
    """Select up to n_target rows stratified by y_test for LIME aggregation."""
    if len(X_test) <= n_target:
        return X_test.copy()
    try:
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=n_target, random_state=RANDOM_STATE
        )
        idx, _ = next(sss.split(np.zeros(len(y_test)), y_test))
        return X_test.iloc[np.sort(idx)]
    except Exception:
        return X_test.sample(n=n_target, random_state=RANDOM_STATE)


def compute_lime_for_target(model, X_train_bg, X_test, y_test, features, class_names, ds_name):
    """Compute aggregated LIME feature importance on a target dataset."""
    if not hasattr(model, "predict_proba"):
        print(f"    LIME skipped for {type(model).__name__} (no predict_proba)")
        return None

    try:
        bg_df = X_train_bg[features] if all(f in X_train_bg.columns for f in features) else X_train_bg
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=bg_df.values,
            feature_names=features,
            class_names=class_names,
            mode="classification",
            discretize_continuous=True,
            random_state=RANDOM_STATE,
        )

        safe = ds_name.replace(" ", "_").replace("/", "_")
        model_name = type(model).__name__

        # Individual LIME instance explanations
        for i in range(min(LIME_SAMPLES, len(X_test))):
            instance = X_test.iloc[i].values
            true_idx = int(y_test.iloc[i])
            pred_idx = int(model.predict(instance.reshape(1, -1))[0])
            pred_prob = model.predict_proba(instance.reshape(1, -1))[0]
            true_str = class_names[true_idx] if true_idx < len(class_names) else f"Class_{true_idx}"
            pred_str = class_names[pred_idx] if pred_idx < len(class_names) else f"Class_{pred_idx}"
            conf = float(np.max(pred_prob)) * 100

            exp = explainer.explain_instance(
                instance, model.predict_proba, num_features=10, top_labels=1
            )
            feats = exp.as_list(label=exp.top_labels[0])

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ["crimson" if w > 0 else "steelblue" for _, w in feats]
            labels_ = [f[0] for f in feats]
            weights_ = [f[1] for f in feats]
            ax.barh(labels_[::-1], weights_[::-1], color=colors[::-1],
                    edgecolor="black", alpha=0.85)
            ax.axvline(0, color="black", lw=0.8, ls="--")
            ax.set_xlabel(f"LIME Weight (red → {pred_str} | blue → other)")
            ax.set_title(
                f"LIME — {model_name} | {ds_name} #{i}\n"
                f"True: {true_str} | Pred: {pred_str} | Conf: {conf:.1f}%",
                fontsize=11, fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR, "lime_stability", "instances", f"lime_{safe}_inst{i}.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close()

        # Aggregated LIME importance over stratified subset
        sample = _stratified_sample_for_lime(X_test, y_test, LIME_AGG_N)
        imp_sum = {f: 0.0 for f in features}
        features_by_len = sorted(features, key=len, reverse=True)

        print(f"    LIME aggregating over {len(sample)} samples...")
        for idx, (_, row) in enumerate(sample.iterrows()):
            try:
                exp = explainer.explain_instance(
                    row.values, model.predict_proba,
                    num_features=len(features), top_labels=1,
                )
                top_lbl = exp.top_labels[0]
                mapped = exp.as_map().get(top_lbl)
                if mapped:
                    for feat_idx, wt in mapped:
                        if 0 <= feat_idx < len(features):
                            imp_sum[features[feat_idx]] += abs(wt)
                else:
                    for ft, wt in exp.as_list(label=top_lbl):
                        for fn in features_by_len:
                            if fn in ft:
                                imp_sum[fn] += abs(wt)
                                break
            except Exception:
                pass
            if (idx + 1) % 25 == 0:
                print(f"      LIME {idx + 1}/{len(sample)} ...")

        importance = pd.Series(imp_sum, index=features)

        # Save bar plot
        plt.figure(figsize=(10, 6))
        importance.nlargest(15).sort_values().plot(
            kind="barh", color="darkcyan", edgecolor="black", alpha=0.85
        )
        plt.title(f"LIME Importance — Model: {_ref_name} → Data: {ds_name}",
                  fontsize=11, fontweight="bold")
        plt.xlabel("Sum |LIME Weights|")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, "lime_stability", f"lime_ranking_{safe}.png"),
            dpi=150,
        )
        plt.close()

        return importance

    except Exception as e:
        print(f"    ⚠ LIME failed for {ds_name}: {e}")
        return None


def analyze_lime_stability(lime_rankings):
    """Compare LIME feature rankings across datasets."""
    if len(lime_rankings) < 2:
        print("  ⚠ Need ≥2 datasets for LIME stability analysis")
        return

    print("\n" + "="*60)
    print("  LIME FEATURE STABILITY ANALYSIS")
    print("="*60)

    rank_data = {}
    for ds_name, importance in lime_rankings.items():
        rank_data[ds_name] = importance.rank(ascending=False)

    df_ranks = pd.DataFrame(rank_data)
    df_ranks["Avg Rank"] = df_ranks.mean(axis=1)
    df_ranks["Std"]      = df_ranks.drop(columns=["Avg Rank"]).std(axis=1)
    df_ranks = df_ranks.sort_values("Avg Rank")

    top15 = df_ranks.head(15).index
    df_heatmap = df_ranks.loc[top15].drop(columns=["Avg Rank", "Std"])

    plt.figure(figsize=(max(10, len(lime_rankings)*2.5), 7))
    sns.heatmap(df_heatmap, annot=True, fmt=".0f", cmap="YlGnBu_r",
                linewidths=0.5, linecolor="white")
    plt.title(f"LIME Feature Ranking Stability\n"
              f"(Model trained on {_ref_name} — lower rank = more important)",
              fontsize=12, fontweight="bold")
    plt.xlabel("Target Dataset")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lime_stability",
                "lime_ranking_heatmap.png"), dpi=150)
    plt.close()
    print("  ✔ lime_ranking_heatmap.png")

    from itertools import combinations
    ds_names = [c for c in df_ranks.columns if c not in ["Avg Rank", "Std"]]
    if len(ds_names) >= 2:
        try:
            from scipy.stats import spearmanr
            corr_rows = []
            for a, b in combinations(ds_names, 2):
                rho, pval = spearmanr(df_ranks[a], df_ranks[b])
                corr_rows.append({"Dataset A": a, "Dataset B": b,
                                  "Spearman rho": round(rho, 4),
                                  "p-value": f"{pval:.4e}"})
                print(f"    LIME {a} ↔ {b}: rho={rho:.4f} (p={pval:.2e})")
            pd.DataFrame(corr_rows).to_csv(
                os.path.join(OUTPUT_DIR, "lime_stability", "lime_pairwise_correlation.csv"),
                index=False,
            )
        except ImportError:
            print("  ⚠ scipy not installed — LIME pairwise Spearman skipped")

    df_ranks.round(2).to_csv(
        os.path.join(OUTPUT_DIR, "lime_stability", "lime_stability_report.csv")
    )
    print("  ✔ lime_stability_report.csv saved")


def compute_shap_vs_lime_spearman(shap_rankings, lime_rankings, df_valid, model_name):
    """
    Compute Spearman Rank Correlation between SHAP and LIME feature rankings
    for every evaluated dataset (both Within-dataset and Cross-dataset).
    """
    common_ds = [ds for ds in shap_rankings if ds in lime_rankings]
    if not common_ds:
        return pd.DataFrame()

    print("\n" + "="*60)
    print("  SHAP vs LIME CONSISTENCY (SPEARMAN RANK CORRELATION)")
    print("="*60)

    type_map = {}
    if len(df_valid) > 0 and "Target" in df_valid.columns and "Type" in df_valid.columns:
        type_map = dict(zip(df_valid["Target"], df_valid["Type"]))

    rows = []
    for ds_name in common_ds:
        shap_imp = shap_rankings[ds_name]
        lime_imp = lime_rankings[ds_name]
        common_feats = sorted(set(shap_imp.index) & set(lime_imp.index))
        if len(common_feats) < 5:
            continue

        try:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(
                shap_imp[common_feats].rank(ascending=False),
                lime_imp.reindex(common_feats).fillna(0).rank(ascending=False),
            )
        except ImportError:
            print("  ⚠ scipy not installed — Spearman skipped")
            return pd.DataFrame()

        strength = "Strong" if abs(rho) > 0.7 else ("Moderate" if abs(rho) > 0.4 else "Weak")

        s_top5 = set(shap_imp.nlargest(5).index)
        l_top5 = set(lime_imp.nlargest(5).index)
        s_top10 = set(shap_imp.nlargest(10).index)
        l_top10 = set(lime_imp.nlargest(10).index)

        exp_type = type_map.get(ds_name, "Unknown")
        rows.append({
            "Target": ds_name,
            "Type": exp_type,
            "Spearman_rho": round(float(rho), 4),
            "p_value": float(pval),
            "Strength": strength,
            "Top5_Overlap": len(s_top5 & l_top5),
            "Top10_Overlap": len(s_top10 & l_top10),
            "N_Features": len(common_feats),
        })

        print(f"    {ds_name:20s} ({exp_type:6s}) | rho={rho:+.4f} (p={pval:.2e}) "
              f"[{strength}] | Top5={len(s_top5 & l_top5)}/5  Top10={len(s_top10 & l_top10)}/10")

        # Side-by-side normalized bar plot (SHAP vs LIME)
        shap_n = shap_imp / shap_imp.max() if shap_imp.max() > 0 else shap_imp
        lime_n = lime_imp / lime_imp.max() if lime_imp.max() > 0 else lime_imp
        top15 = shap_n.nlargest(min(15, len(shap_n))).index
        df_c = pd.DataFrame({
            "SHAP": shap_n[top15],
            "LIME": lime_n.reindex(top15).fillna(0),
        }).sort_values("SHAP", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 7))
        x = np.arange(len(df_c))
        w = 0.35
        ax.barh(x - w/2, df_c["SHAP"], w, label="SHAP",
                color="steelblue", edgecolor="black", alpha=0.85)
        ax.barh(x + w/2, df_c["LIME"], w, label="LIME",
                color="darkorange", edgecolor="black", alpha=0.85)
        ax.set_yticks(x)
        ax.set_yticklabels(df_c.index)
        ax.set_xlabel("Normalized Importance")
        ax.legend()
        ax.set_title(
            f"SHAP vs LIME — {model_name} | {ds_name} ({exp_type})\n"
            f"Spearman ρ = {rho:.4f} (p={pval:.2e}, {strength})",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        safe = ds_name.replace(" ", "_").replace("/", "_")
        plt.savefig(
            os.path.join(OUTPUT_DIR, "xai_consistency", f"shap_vs_lime_{safe}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()

    df_spearman = pd.DataFrame(rows)
    if not df_spearman.empty:
        csv_path = os.path.join(OUTPUT_DIR, "xai_consistency", "shap_vs_lime_spearman.csv")
        df_spearman.to_csv(csv_path, index=False)
        print(f"\n  ✔ {csv_path} saved")

    return df_spearman


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_visualizations(df_results, ref_name):
    print("\n" + "="*60)
    print("  GENERATING VISUALIZATIONS")
    print("="*60)

    n_datasets = len(df_results)

    # ── 1. Bar chart: F1 + MCC per target ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["forestgreen" if t == "Within" else "steelblue"
              for t in df_results["Type"]]

    axes[0].bar(df_results["Target"], df_results["F1"], color=colors,
                edgecolor="black", alpha=0.85)
    for i, (_, row) in enumerate(df_results.iterrows()):
        axes[0].text(i, row["F1"] + 0.01, f"{row['F1']:.3f}",
                     ha="center", fontsize=8)
    axes[0].set_title(f"F1-Score (Model: {ref_name})", fontweight="bold")
    axes[0].set_ylabel("F1"); axes[0].set_ylim(0, 1.15)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].axhline(y=df_results[df_results["Type"]=="Within"]["F1"].values[0]
                    if "Within" in df_results["Type"].values else 0,
                    color="red", ls="--", lw=0.8, alpha=0.5)

    axes[1].bar(df_results["Target"], df_results["MCC"], color=colors,
                edgecolor="black", alpha=0.85)
    for i, (_, row) in enumerate(df_results.iterrows()):
        axes[1].text(i, max(row["MCC"] + 1, 1), f"{row['MCC']:.1f}%",
                     ha="center", fontsize=8)
    axes[1].set_title(f"MCC % (Model: {ref_name})", fontweight="bold")
    axes[1].set_ylabel("MCC (%)"); axes[1].set_ylim(-10, 110)
    axes[1].tick_params(axis="x", rotation=30)

    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(color="forestgreen", label="Within-dataset"),
              Patch(color="steelblue", label="Cross-dataset")]
    axes[0].legend(handles=legend, fontsize=8)

    plt.suptitle(f"Cross-Dataset Generalization — Trained on {ref_name}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cross_dataset_bar.png"), dpi=150)
    plt.close()
    print("  ✔ cross_dataset_bar.png")

    # ── 2. Generalization gap ──
    within = df_results[df_results["Type"] == "Within"]
    cross  = df_results[df_results["Type"] == "Cross"]

    if len(within) > 0 and len(cross) > 0:
        w_f1  = within["F1"].values[0]
        w_mcc = within["MCC"].values[0]
        c_f1  = cross["F1"].mean()
        c_mcc = cross["MCC"].mean()

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(2); w = 0.3
        ax.bar(x - w/2, [w_f1, w_mcc], w, label="Within", color="forestgreen",
               edgecolor="black", alpha=0.85)
        ax.bar(x + w/2, [c_f1, c_mcc], w, label="Cross (avg)", color="steelblue",
               edgecolor="black", alpha=0.85)

        # Gap annotations
        f1_gap  = w_f1 - c_f1
        mcc_gap = w_mcc - c_mcc
        ax.annotate(f"↓{f1_gap:.3f}", xy=(0, max(w_f1, c_f1) + 0.02),
                    ha="center", fontsize=10, color="red", fontweight="bold")
        ax.annotate(f"↓{mcc_gap:.1f}%", xy=(1, max(w_mcc, c_mcc) + 2),
                    ha="center", fontsize=10, color="red", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(["F1-Score", "MCC (%)"])
        ax.set_title(f"Generalization Gap — {ref_name}", fontweight="bold")
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "generalization_gap.png"), dpi=150)
        plt.close()
        print(f"  ✔ generalization_gap.png  (F1 gap={f1_gap:.4f}, MCC gap={mcc_gap:.1f}%)")


# =============================================================================
# TEXT REPORT
# =============================================================================

def generate_report(df_results, ref_name, model_name, shap_rankings, class_names,
                    lime_rankings=None, df_spearman=None):
    lines = []
    lines.append("="*70)
    lines.append(f"  CROSS-DATASET GENERALIZATION REPORT")
    lines.append(f"  Reference model : {model_name}")
    lines.append(f"  Trained on      : {ref_name}")
    lines.append(f"  Classes         : {class_names}")
    lines.append(f"  Tested on       : {len(df_results)} datasets")
    lines.append("="*70)

    within = df_results[df_results["Type"] == "Within"]
    cross  = df_results[df_results["Type"] == "Cross"]

    if len(within) > 0:
        w = within.iloc[0]
        lines.append(f"\n  Within-dataset ({w['Target']}):")
        lines.append(f"    F1={w['F1']:.4f}  MCC={w['MCC']:.2f}%  "
                     f"Acc={w['Accuracy']:.4f}  N={w['N_test']:,}")

    if len(cross) > 0:
        lines.append(f"\n  Cross-dataset (avg over {len(cross)} targets):")
        lines.append(f"    F1={cross['F1'].mean():.4f}  "
                     f"MCC={cross['MCC'].mean():.2f}%  "
                     f"Acc={cross['Accuracy'].mean():.4f}")

        if len(within) > 0:
            f1_gap  = within["F1"].values[0] - cross["F1"].mean()
            mcc_gap = within["MCC"].values[0] - cross["MCC"].mean()
            lines.append(f"\n  Generalization gap:")
            lines.append(f"    F1  drop : {f1_gap:+.4f}")
            lines.append(f"    MCC drop : {mcc_gap:+.2f}%")

        lines.append(f"\n  Best cross-dataset:")
        best = cross.loc[cross["MCC"].idxmax()]
        lines.append(f"    {best['Target']:20s} MCC={best['MCC']:.2f}%  F1={best['F1']:.4f}")

        lines.append(f"\n  Worst cross-dataset:")
        worst = cross.loc[cross["MCC"].idxmin()]
        lines.append(f"    {worst['Target']:20s} MCC={worst['MCC']:.2f}%  F1={worst['F1']:.4f}")

    lines.append(f"\n  ── Full Results ──")
    lines.append(df_results.to_string(index=False))

    if shap_rankings and len(shap_rankings) >= 2:
        lines.append(f"\n  ── SHAP Stability ──")
        # Top-5 overlap between within and each cross
        within_name = within["Target"].values[0] if len(within) > 0 else None
        if within_name and within_name in shap_rankings:
            w_top5 = set(shap_rankings[within_name].nlargest(5).index)
            for ds_name, imp in shap_rankings.items():
                if ds_name == within_name:
                    continue
                c_top5 = set(imp.nlargest(5).index)
                overlap = w_top5 & c_top5
                lines.append(f"    {within_name} ↔ {ds_name}: "
                             f"top-5 overlap = {len(overlap)}/5 "
                             f"({', '.join(sorted(overlap)) if overlap else 'none'})")

    if lime_rankings and len(lime_rankings) >= 2:
        lines.append(f"\n  ── LIME Stability ──")
        within_name = within["Target"].values[0] if len(within) > 0 else None
        if within_name and within_name in lime_rankings:
            w_top5 = set(lime_rankings[within_name].nlargest(5).index)
            for ds_name, imp in lime_rankings.items():
                if ds_name == within_name:
                    continue
                c_top5 = set(imp.nlargest(5).index)
                overlap = w_top5 & c_top5
                lines.append(f"    {within_name} ↔ {ds_name}: "
                             f"top-5 overlap = {len(overlap)}/5 "
                             f"({', '.join(sorted(overlap)) if overlap else 'none'})")

    if df_spearman is not None and not df_spearman.empty:
        lines.append(f"\n  ── SHAP vs LIME Consistency (Spearman Rank Correlation) ──")
        lines.append(df_spearman.to_string(index=False))
        w_sp = df_spearman[df_spearman["Type"] == "Within"]
        c_sp = df_spearman[df_spearman["Type"] == "Cross"]
        if len(w_sp) > 0 and len(c_sp) > 0:
            w_rho = w_sp["Spearman_rho"].mean()
            c_rho = c_sp["Spearman_rho"].mean()
            lines.append(f"\n    Within Spearman avg : {w_rho:+.4f}")
            lines.append(f"    Cross  Spearman avg : {c_rho:+.4f}")
            lines.append(f"    Spearman diff (W-C) : {w_rho - c_rho:+.4f}")

    lines.append("\n" + "="*70)

    report_text = "\n".join(lines)
    print(report_text)

    with open(os.path.join(OUTPUT_DIR, "cross_dataset_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  ✔ cross_dataset_report.txt saved")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print(f"  SCRIPT 03 — CROSS-DATASET ANALYSIS")
    print(f"  Reference : {REFERENCE_MODEL_DIR}")
    print(f"  Output    : {OUTPUT_DIR}")
    print("="*70)

    t_start = time.time()

    # ── Load reference model ──
    model, scaler, features, ref_name, binary_mode, class_names = load_reference_model()
    model_name = type(model).__name__

    # Reference training background for LIME (mirrors Script 02 Mode B)
    ref_bg_path = os.path.join(REFERENCE_MODEL_DIR, "X_train_sample.pkl")
    try:
        ref_train_bg = joblib.load(ref_bg_path)
        print(f"  ✔ LIME background : {ref_train_bg.shape} from {ref_bg_path}")
    except Exception:
        ref_train_bg = None
        print(f"  ⚠ X_train_sample.pkl not found — LIME will use target train sample")

    # ── Discover target datasets ──
    datasets = discover_datasets()
    if not datasets:
        print(f"  ✗ No prepared datasets in '{PREPARED_ROOT}/'")
        sys.exit(1)

    # ── Evaluate on each target ──
    all_results = []
    all_reports = {}
    shap_rankings = {}
    lime_rankings = {}

    for i, ds_name in enumerate(datasets, 1):
        print(f"\n  [{i}/{len(datasets)}] Target: {ds_name}")

        try:
            X_test, y_test, X_train_bg, n_matched = \
                load_and_prepare_target(ds_name, features, scaler, binary_mode)

            metrics, report_txt = evaluate_on_target(
                model, X_test, y_test, ds_name, ref_name, binary_mode, class_names)
            metrics["Features_matched"] = n_matched
            all_results.append(metrics)
            all_reports[ds_name] = report_txt

            # SHAP
            if RUN_SHAP:
                imp = compute_shap_for_target(model, X_test, features, ds_name)
                if imp is not None:
                    shap_rankings[ds_name] = imp

            # LIME
            if RUN_LIME:
                bg_for_lime = ref_train_bg if ref_train_bg is not None else X_train_bg
                lime_imp = compute_lime_for_target(
                    model, bg_for_lime, X_test, y_test, features, class_names, ds_name
                )
                if lime_imp is not None:
                    lime_rankings[ds_name] = lime_imp

        except Exception as e:
            print(f"    ✗ FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results.append({
                "Target": ds_name, "Type": "Error",
                "Accuracy": None, "F1": None, "Precision": None,
                "Recall": None, "MCC": None, "AUC-ROC": None,
                "N_test": None, "Features_matched": None,
            })

        gc.collect()

    # ── Save results ──
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, "cross_dataset_metrics.csv"), index=False)

    # Save all classification reports
    with open(os.path.join(OUTPUT_DIR, "classification_reports.txt"), "w", encoding="utf-8") as f:
        for ds_name, report in all_reports.items():
            f.write(f"{'='*60}\nTrain: {ref_name} → Test: {ds_name}\n{'='*60}\n")
            f.write(report)
            f.write("\n\n")

    # ── Filter valid results for visualization ──
    df_valid = df_results[df_results["F1"].notna()].copy()

    if len(df_valid) > 0:
        generate_visualizations(df_valid, ref_name)

    if RUN_SHAP and len(shap_rankings) >= 2:
        analyze_shap_stability(shap_rankings)

    if RUN_LIME and len(lime_rankings) >= 2:
        analyze_lime_stability(lime_rankings)

    df_spearman = pd.DataFrame()
    if RUN_SHAP and RUN_LIME and shap_rankings and lime_rankings:
        df_spearman = compute_shap_vs_lime_spearman(
            shap_rankings, lime_rankings, df_valid, model_name
        )

    generate_report(
        df_valid, ref_name, model_name, shap_rankings, class_names,
        lime_rankings=lime_rankings, df_spearman=df_spearman,
    )

    elapsed = time.time() - t_start

    # Summary counts
    n_ok   = len(df_valid)
    n_fail = len(df_results) - n_ok
    within = df_valid[df_valid["Type"] == "Within"]
    cross  = df_valid[df_valid["Type"] == "Cross"]

    print(f"""
  ╔═══════════════════════════════════════════════════════════╗
  ║  SCRIPT 03 COMPLETE ✔  ({elapsed:.1f}s)
  ╠═══════════════════════════════════════════════════════════╣
  ║  Reference   : {ref_name} ({model_name})
  ║  Tested on   : {n_ok} datasets ({n_fail} failed)
  ║  Within MCC  : {within['MCC'].values[0] if len(within) else 'N/A'}%
  ║  Cross MCC   : {cross['MCC'].mean() if len(cross) else 'N/A'}% (avg)
  ║  SHAP ranked : {len(shap_rankings)} datasets
  ║  LIME ranked : {len(lime_rankings)} datasets
  ║  Spearman    : {len(df_spearman)} dataset(s) compared
  ║
  ║  Output → ./{OUTPUT_DIR}/
  ║   ├── cross_dataset_metrics.csv
  ║   ├── cross_dataset_bar.png
  ║   ├── generalization_gap.png
  ║   ├── confusion_matrices/cm_*.png
  ║   ├── classification_reports.txt
  ║   ├── shap_stability/
  ║   │   ├── shap_ranking_*.png
  ║   │   ├── shap_ranking_heatmap.png
  ║   │   ├── shap_pairwise_correlation.csv
  ║   │   └── shap_stability_report.csv
  ║   ├── lime_stability/
  ║   │   ├── lime_ranking_*.png
  ║   │   ├── lime_ranking_heatmap.png
  ║   │   ├── lime_pairwise_correlation.csv
  ║   │   └── lime_stability_report.csv
  ║   ├── xai_consistency/
  ║   │   ├── shap_vs_lime_*.png
  ║   │   └── shap_vs_lime_spearman.csv
  ║   └── cross_dataset_report.txt
  ╚═══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
