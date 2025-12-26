from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from lightgbm import LGBMClassifier

import mlflow
import mlflow.lightgbm


# 1. Blood / Serum laboratory measurements
blood_serum_labs = [
    "LBDHDD__response",
    "LBDLDL__response",
    "VNTOTHDRATIO__response",
    "LBDINSI__response",
    "LBXGH__response",
    "LBXGLU__response",
    "LBXSGL__response",
    "LBXGLT__response",
    "LBXCRP__response",
    "LBDSALSI__response",
    "LBDSBUSI__response",
    "LBXSCR__response",
    "LBXB12__chemicals",
    "MHPSI__response",
]

# 2. Urine / Kidney-related laboratory measurements
urine_kidney_labs = [
    "URXUCR__response",
    "URXUMA__response",
    "URXUMS__response",
    "VNEGFR__response",
]

# 3. Lifestyle & behavioral factors
lifestyle_factors = [
    "ALQ130__questionnaire",
    "PAD680__questionnaire",
    "PAQ655__questionnaire",
    "PAQ670__questionnaire",
]

# 4. Body measurements & demographics
body_measurements = [
    "BMXBMI__response",
    "BMXHT__response",
    "BMXWT__response",
    "BMXWAIST__response",
    "RIDAGEYR__demographics",
]

# 5. Blood pressure & cardiovascular measurements
pressure_cardio = [
    "BPXSAR__response",
    "BPXDAR__response",
    "BPXPLS__response",
    "BPXPULS__response",
]

# 6. Medication usage, medical history & comorbidities
med_history_comorbidities = [
    "BPD035__questionnaire",
    "BPQ040A__questionnaire",
    "BPQ050A__questionnaire",
    "HAE5A__questionnaire",
    "BPQ080__questionnaire",
    "BPQ090D__questionnaire",
    "BPQ100D__questionnaire",
    "MCQ160C__questionnaire",
    "MCQ160D__questionnaire",
    "MCQ160L__questionnaire",
    "MCQ140__questionnaire",
    "MCQ300C__questionnaire",
]

all_features = (
    blood_serum_labs
    + urine_kidney_labs
    + lifestyle_factors
    + body_measurements
    + pressure_cardio
    + med_history_comorbidities
)

label_col = "Diabetes_Type"  # ["T2D", "Possible-T2D", "T1D", "Not Diabetic", "Skipped", "Excluded"]


def _select_existing_features(df: pd.DataFrame, features: List[str]) -> List[str]:
    return [c for c in features if c in df.columns]


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def prepare_train_val(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df[df[label_col].notna()].copy()
    # First split the full dataset into train/val 50/50 with stratification
    train_full, val_df = train_test_split(
        df, test_size=0.5, random_state=0, stratify=df[label_col]
    )
    # From train_full, filter to only T2D and Not diabetic (lowercase)
    allowed = {"T2D", "Not diabetic"}
    train_df = train_full[train_full[label_col].isin(allowed)].copy()
    # Undersample Not diabetic to match T2D count
    t2d_df = train_df[train_df[label_col] == "T2D"]
    not_df = train_df[train_df[label_col] == "Not diabetic"]
    n_t2d = len(t2d_df)
    if len(not_df) > n_t2d:
        not_df = not_df.sample(n=n_t2d, random_state=0)
    train_bal = pd.concat([t2d_df, not_df], axis=0).sample(frac=1.0, random_state=0)
    return train_bal, val_df


def evaluate_and_log(
    y_true_bin_val: np.ndarray,
    y_pred_bin_val: np.ndarray,
    y_proba_val: np.ndarray,
    feature_group: str = "",
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    acc = float(accuracy_score(y_true_bin_val, y_pred_bin_val))
    metrics["val_accuracy_bin"] = acc
    print(f"  [{feature_group}] Binary Accuracy: {acc:.4f}, Unique preds: {np.unique(y_pred_bin_val)}, Unique true: {np.unique(y_true_bin_val)}")
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true_bin_val, y_pred_bin_val, average="binary", zero_division=0
    )
    metrics["val_precision_bin"] = float(pr)
    metrics["val_recall_bin"] = float(rc)
    metrics["val_f1_bin"] = float(f1)
    # Log AUCs only if both classes are present in validation subset
    if len(np.unique(y_true_bin_val)) == 2:
        try:
            metrics["val_roc_auc_bin"] = float(roc_auc_score(y_true_bin_val, y_proba_val))
            metrics["val_pr_auc_bin"] = float(average_precision_score(y_true_bin_val, y_proba_val))
        except Exception:
            pass
    mlflow.log_metrics(metrics)
    return metrics


def plot_and_log_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int], title: str, artifact_name: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    tmp_path = os.path.join("mlflow_tmp_arts", f"{artifact_name}.png")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    plt.savefig(tmp_path)
    plt.close()
    mlflow.log_artifact(tmp_path, artifact_path="plots")


def run_for_feature_group(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    group_name: str,
    features: List[str],
):
    print(f"\n=== {group_name} ===")
    cols = _select_existing_features(df, features)
    missing = sorted(set(features) - set(cols))
    mlflow.log_param("feature_group", group_name)
    mlflow.log_param("selected_features_count", len(cols))
    mlflow.log_param("missing_features_count", len(missing))
    if missing:
        mlflow.log_text("\n".join(missing), "missing_features.txt")

    X_train = _to_numeric(train_df, cols)
    y_train = train_df[label_col].map({"Not diabetic": 0, "T2D": 1}).astype(int)

    X_val_full = _to_numeric(val_df, cols)
    y_val_full = val_df[label_col]

    print(f"  Train y_train: {y_train.value_counts().to_dict()}, NaN in X_train: {X_train.isna().sum().sum()}")
    print(f"  Val full shape: {X_val_full.shape}, y_val_full classes: {y_val_full.value_counts().to_dict()}")

    # Subset of validation with only the two classes for binary metrics
    val_mask_bin = y_val_full.isin(["T2D", "Not diabetic"])  # boolean mask
    X_val_bin = X_val_full[val_mask_bin]
    y_val_bin = y_val_full[val_mask_bin].map({"Not diabetic": 0, "T2D": 1}).astype(int)

    params = {
        "objective": "binary",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_leaves": 31,
        "min_data_in_leaf": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "n_jobs": -1,
        "random_state": 0,
    }
    mlflow.log_params(params)

    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_pred_bin = model.predict(X_val_bin)
    y_proba_bin = model.predict_proba(X_val_bin)[:, 1]
    print(f"  Train shape: {X_train.shape}, Val bin shape: {X_val_bin.shape}, NaN count in X_val_bin: {X_val_bin.isna().sum().sum()}")
    evaluate_and_log(y_val_bin.to_numpy(), y_pred_bin, y_proba_bin, feature_group=group_name)

    # Evaluate on full validation (includes other classes mapped to 2)
    y_val_full_mapped = y_val_full.map({"Not diabetic": 0, "T2D": 1}).fillna(2).astype(int)
    y_pred_full = model.predict(X_val_full)
    acc_full = float((y_pred_full == y_val_full_mapped).mean())
    mlflow.log_metric("val_accuracy_full", acc_full)

    # Confusion matrices
    plot_and_log_confusion(
        y_true=y_val_bin.to_numpy(),
        y_pred=y_pred_bin,
        labels=[0, 1],
        title=f"Confusion (Binary) - {group_name}",
        artifact_name=f"cm_binary_{group_name}",
    )
    plot_and_log_confusion(
        y_true=y_val_full_mapped.to_numpy(),
        y_pred=y_pred_full,
        labels=[0, 1, 2],
        title=f"Confusion (Full) - {group_name}",
        artifact_name=f"cm_full_{group_name}",
    )

    # Classification report on full validation
    report = classification_report(
        y_val_full_mapped, y_pred_full, labels=[0, 1, 2], digits=4, zero_division=0
    )
    mlflow.log_text(report, "classification_report_full.txt")

    # Log model
    mlflow.lightgbm.log_model(model, name="model")


def main():
    # Configure MLflow local tracking
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = "diabetes-feature-groups"
    mlflow.set_experiment(experiment_name)

    # Load dataset
    ds = load_dataset("rtweera/nhanes-data-converted", split="train")
    df = ds.to_pandas()

    # Prepare train/validation per requirements
    train_df, val_df = prepare_train_val(df, label_col)

    # Log dataset sizes and class counts
    with mlflow.start_run(run_name="feature_group_study"):
        mlflow.log_param("train_rows", int(len(train_df)))
        mlflow.log_param("val_rows", int(len(val_df)))
        class_counts_train = train_df[label_col].value_counts().to_dict()
        class_counts_val = val_df[label_col].value_counts().to_dict()
        mlflow.log_dict(class_counts_train, "train_class_counts.json")
        mlflow.log_dict(class_counts_val, "val_class_counts.json")

        feature_groups: Dict[str, List[str]] = {
            "blood_serum_labs": blood_serum_labs,
            "urine_kidney_labs": urine_kidney_labs,
            "lifestyle_factors": lifestyle_factors,
            "body_measurements": body_measurements,
            "pressure_cardio": pressure_cardio,
            "med_history_comorbidities": med_history_comorbidities,
            "all_features": all_features,
        }

        for group_name, feats in feature_groups.items():
            with mlflow.start_run(nested=True, run_name=group_name):
                run_for_feature_group(df, train_df, val_df, group_name, feats)


if __name__ == "__main__":
    main()