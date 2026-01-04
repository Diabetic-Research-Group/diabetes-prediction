"""
Feature Importance & Ablation Study
====================================
This script:
1. Loads the best trained model
2. Extracts feature importances
3. Iteratively removes least important features
4. Evaluates on test set after each removal
5. Logs metrics and visualizations to MLflow
6. Identifies elbow points where performance degrades significantly
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.data.prepare_diabetes_data import AGG_COL, LABEL_COL, load_from_huggingface_or_local

# Feature groups for reference (optional)
blood_serum_labs = [
    "LBDHDD__response", "LBDLDL__response", "VNTOTHDRATIO__response", "LBDINSI__response",
    "LBXGH__response", "LBXGLU__response", "LBXSGL__response", "LBXGLT__response",
    "LBXCRP__response", "LBDSALSI__response", "LBDSBUSI__response", "LBXSCR__response",
    "LBXB12__chemicals", "MHPSI__response",
]
urine_kidney_labs = ["URXUCR__response", "URXUMA__response", "URXUMS__response", "VNEGFR__response"]
lifestyle_factors = ["ALQ130__questionnaire", "PAD680__questionnaire", "PAQ655__questionnaire", "PAQ670__questionnaire"]
body_measurements = ["BMXBMI__response", "BMXHT__response", "BMXWT__response", "BMXWAIST__response", "RIDAGEYR__demographics"]
pressure_cardio = ["BPXSAR__response", "BPXDAR__response", "BPXPLS__response", "BPXPULS__response"]
med_history_comorbidities = [
    "BPD035__questionnaire", "BPQ040A__questionnaire", "BPQ050A__questionnaire", "HAE5A__questionnaire",
    "BPQ080__questionnaire", "BPQ090D__questionnaire", "BPQ100D__questionnaire", "MCQ160C__questionnaire",
    "MCQ160D__questionnaire", "MCQ160L__questionnaire", "MCQ140__questionnaire", "MCQ300C__questionnaire",
]
all_features = blood_serum_labs + urine_kidney_labs + lifestyle_factors + body_measurements + pressure_cardio + med_history_comorbidities

CLASS_TO_ID = {"Not diabetic": 0, "T2D": 1, "Other": 2}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}


def _select_existing_features(df: pd.DataFrame, features: List[str]) -> List[str]:
    return [c for c in features if c in df.columns]


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute all multiclass metrics."""
    labels = [0, 1, 2]
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    pr_per_class, rc_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    for cls_idx, pr_val in zip(labels, pr_per_class):
        metrics[f"precision_class_{cls_idx}"] = float(pr_val)
    for cls_idx, rc_val in zip(labels, rc_per_class):
        metrics[f"recall_class_{cls_idx}"] = float(rc_val)
    for cls_idx, f1_val in zip(labels, f1_per_class):
        metrics[f"f1_class_{cls_idx}"] = float(f1_val)

    pr_macro, rc_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    pr_weighted, rc_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics.update({
        "precision_macro": float(pr_macro),
        "recall_macro": float(rc_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(pr_weighted),
        "recall_weighted": float(rc_weighted),
        "f1_weighted": float(f1_weighted),
    })

    try:
        metrics["roc_auc_macro"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        metrics["roc_auc_weighted"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
    except ValueError:
        metrics["roc_auc_macro"] = 0.0
        metrics["roc_auc_weighted"] = 0.0

    pr_aucs = []
    weighted_sum = 0.0
    for cls_idx in labels:
        y_true_bin = (y_true == cls_idx).astype(int)
        try:
            pr_val = average_precision_score(y_true_bin, y_proba[:, cls_idx])
        except ValueError:
            pr_val = 0.0
        pr_aucs.append(pr_val)
        weighted_sum += pr_val * (y_true_bin.mean())
        metrics[f"pr_auc_class_{cls_idx}"] = float(pr_val)
    metrics["pr_auc_macro"] = float(np.mean(pr_aucs)) if pr_aucs else 0.0
    metrics["pr_auc_weighted"] = float(weighted_sum) if len(y_true) > 0 else 0.0

    return metrics


def get_feature_importance_dataframe(model, feature_names: List[str]) -> pd.DataFrame:
    """Extract feature importances and return sorted DataFrame."""
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, title: str = "Feature Importance", artifact_name: str = "feature_importance.png"):
    """Plot and log feature importance bar chart."""
    plt.figure(figsize=(12, max(6, len(importance_df) // 3)))
    plt.barh(range(len(importance_df)), importance_df["importance"])
    plt.yticks(range(len(importance_df)), importance_df["feature"], fontsize=8)
    plt.xlabel("Importance Score")
    plt.title(title)
    plt.tight_layout()
    tmp_path = os.path.join("mlflow_tmp_arts", artifact_name)
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    plt.savefig(tmp_path, dpi=100, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(tmp_path, artifact_path="plots")


def plot_metric_curves(ablation_results: List[Dict], metric_keys: List[str] = None):
    """Plot metric degradation vs number of features removed."""
    if metric_keys is None:
        metric_keys = ["roc_auc_weighted", "pr_auc_weighted", "recall_macro", "accuracy"]

    # Extract data
    features_used = [r["features_used"] for r in ablation_results]
    
    for metric_key in metric_keys:
        if metric_key in ablation_results[0]["metrics"]:
            metric_values = [r["metrics"][metric_key] for r in ablation_results]
            
            plt.figure(figsize=(10, 6))
            plt.plot(features_used, metric_values, marker="o", linewidth=2, markersize=6)
            plt.xlabel("Number of Features Used")
            plt.ylabel(metric_key.upper())
            plt.title(f"{metric_key.upper()} vs Number of Features")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            tmp_path = os.path.join("mlflow_tmp_arts", f"metric_{metric_key}.png")
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            plt.savefig(tmp_path, dpi=100)
            plt.close()
            mlflow.log_artifact(tmp_path, artifact_path="metric_curves")


def run_ablation_study(model, X_test: pd.DataFrame, y_test: np.ndarray, test_df: pd.DataFrame):
    """
    Iteratively remove least important features and evaluate performance.
    """
    feature_names = list(X_test.columns)
    importance_df = get_feature_importance_dataframe(model, feature_names)
    
    # Log initial feature importance
    mlflow.log_table(importance_df, "feature_importance.json")
    plot_feature_importance(importance_df, title="Initial Feature Importance", artifact_name="feature_importance_full.png")
    
    ablation_results: List[Dict] = []
    
    # Start with all features
    features_to_remove = importance_df["feature"].tolist()  # Least important first (already sorted desc)
    features_to_remove.reverse()  # Now in ascending importance order
    
    print(f"\nStarting ablation study with {len(feature_names)} features...")
    print(f"Features will be zeroed out in order of increasing importance:\n")
    
    # Evaluation step 1: All features
    X_current = X_test.copy()
    y_pred = model.predict(X_current)
    y_proba = model.predict_proba(X_current)
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    active_features = set(feature_names)
    result = {
        "features_used": len(active_features),
        "feature_removed": "NONE (all features)",
        "metrics": metrics,
    }
    ablation_results.append(result)
    print(f"Step 0: {len(active_features):3d} features | ROC-AUC-W: {metrics['roc_auc_weighted']:.4f} | PR-AUC-W: {metrics['pr_auc_weighted']:.4f}")
    
    # Iteratively zero out least important features
    for step, feature_to_remove in enumerate(features_to_remove[:-1], 1):  # Keep at least 1 feature
        active_features.discard(feature_to_remove)
        
        if len(active_features) == 0:
            break
        
        # Create copy and set removed features to NaN (keep all 43 columns for LightGBM)
        X_current = X_test.copy()
        for col in feature_names:
            if col not in active_features:
                X_current[col] = np.nan
        
        y_pred = model.predict(X_current)
        y_proba = model.predict_proba(X_current)
        metrics = compute_metrics(y_test, y_pred, y_proba)
        
        result = {
            "features_used": len(active_features),
            "feature_removed": feature_to_remove,
            "metrics": metrics,
        }
        ablation_results.append(result)
        print(f"Step {step}: {len(active_features):3d} features | ROC-AUC-W: {metrics['roc_auc_weighted']:.4f} | PR-AUC-W: {metrics['pr_auc_weighted']:.4f}")
    
    # Log all ablation results
    ablation_df = pd.DataFrame([
        {
            "step": i,
            "features_used": r["features_used"],
            "feature_removed": r["feature_removed"],
            **{f"metric_{k}": v for k, v in r["metrics"].items()},
        }
        for i, r in enumerate(ablation_results)
    ])
    mlflow.log_table(ablation_df, "ablation_study_results.json")
    mlflow.log_dict({"ablation_results_summary": str(ablation_df.to_dict())}, "ablation_summary.json")
    
    # Plot metric curves
    plot_metric_curves(ablation_results)
    
    return importance_df, ablation_results, ablation_df


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("diabetes-feature-importance-ablation")
    
    # Load data
    train_df, test_df = load_from_huggingface_or_local()
    
    # Load best model from optuna tuning run
    experiment = mlflow.get_experiment_by_name("diabetes-optuna-multiclass")
    if experiment is None:
        raise ValueError("Could not find 'diabetes-optuna-multiclass' experiment. Run optuna tuning first.")
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string="tags.mlflow.runName = 'optuna_tuning'")
    if len(runs) == 0:
        raise ValueError("Could not find 'optuna_tuning' run in experiment.")
    
    best_run_id = runs.iloc[0].run_id
    print(f"Loading model from run: {best_run_id}")
    model = mlflow.lightgbm.load_model(f"runs:/{best_run_id}/model")
    
    # Prepare features
    all_cols = _select_existing_features(test_df, all_features)
    X_test = _to_numeric(test_df, all_cols)
    y_test = test_df[AGG_COL].map(CLASS_TO_ID).astype(int).to_numpy()
    
    print(f"Test set shape: {X_test.shape}")
    print(f"Test set class distribution: {np.bincount(y_test)}")
    
    with mlflow.start_run(run_name="feature_ablation_study"):
        mlflow.log_param("test_rows", int(len(test_df)))
        mlflow.log_param("initial_feature_count", X_test.shape[1])
        mlflow.log_dict(test_df[LABEL_COL].value_counts().to_dict(), "test_label_counts.json")
        mlflow.log_dict(test_df[AGG_COL].value_counts().to_dict(), "test_label_counts_agg.json")
        
        # Run ablation
        importance_df, ablation_results, ablation_df = run_ablation_study(model, X_test, y_test, test_df)
        
        print(f"\nAblation study complete!")
        print(f"Results logged to MLflow experiment 'diabetes-feature-importance-ablation'")
        print(f"\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
