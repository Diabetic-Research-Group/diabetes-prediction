from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from lightgbm import LGBMClassifier

import mlflow
import mlflow.lightgbm

from src.data.prepare_diabetes_data import AGG_COL, LABEL_COL, load_from_huggingface_or_local


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

CLASS_TO_ID = {"Not diabetic": 0, "T2D": 1, "Other": 2}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}


def _select_existing_features(df: pd.DataFrame, features: List[str]) -> List[str]:
    return [c for c in features if c in df.columns]


def _to_numeric(df: pd.DataFrame, cols: List[str], dataset_tag: str | None = None) -> pd.DataFrame:
    out = df[cols].copy()
    invalid: Dict[str, Dict[str, object]] = {}
    for c in cols:
        raw = out[c]
        converted = pd.to_numeric(raw, errors="coerce")
        bad_mask = converted.isna() & raw.notna()
        bad_count = int(bad_mask.sum())
        if bad_count > 0:
            top_bad = raw[bad_mask].astype(str).value_counts().head(10).to_dict()
            invalid[c] = {"invalid_count": bad_count, "top_values": top_bad}
        out[c] = converted

    if invalid:
        target_name = dataset_tag or "dataset"
        artifact_name = f"non_numeric_values_{target_name}.json"
        try:
            mlflow.log_dict(invalid, artifact_name)
        except Exception:
            print(f"[WARN] Non-numeric values encountered for {target_name}: {invalid}")
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
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
    metrics.update(
        {
            "precision_macro": float(pr_macro),
            "recall_macro": float(rc_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(pr_weighted),
            "recall_weighted": float(rc_weighted),
            "f1_weighted": float(f1_weighted),
        }
    )

    try:
        metrics["roc_auc_macro"] = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        )
        metrics["roc_auc_weighted"] = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        )
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

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return metrics, cm


def plot_and_log_confusion(cm: np.ndarray, labels: List[int], title: str, artifact_name: str):
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
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_name: str,
    features: List[str],
    params: Dict[str, float],
):
    print(f"\n=== {group_name} ===")
    cols = _select_existing_features(train_df, features)
    missing = sorted(set(features) - set(cols))
    mlflow.log_param("feature_group", group_name)
    mlflow.log_param("selected_features_count", len(cols))
    mlflow.log_param("missing_features_count", len(missing))
    if missing:
        mlflow.log_text("\n".join(missing), f"missing_features_{group_name}.txt")

    X_train = _to_numeric(train_df, cols, dataset_tag="train")
    y_train = train_df[AGG_COL].map(CLASS_TO_ID).astype(int)

    X_test = _to_numeric(test_df, cols, dataset_tag="test")
    y_test = test_df[AGG_COL].map(CLASS_TO_ID).astype(int)

    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)
    test_metrics, test_cm = compute_metrics(y_test, y_pred_test, y_proba_test)
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
    mlflow.log_dict({"confusion_matrix": test_cm.tolist()}, f"test_confusion_matrix_{group_name}.json")

    plot_and_log_confusion(test_cm, labels=[0, 1, 2], title=f"Test Confusion - {group_name}", artifact_name=f"cm_test_{group_name}")
    report = classification_report(y_test, y_pred_test, labels=[0, 1, 2], target_names=[ID_TO_CLASS[i] for i in [0, 1, 2]], digits=4, zero_division=0)
    mlflow.log_text(report, f"classification_report_test_{group_name}.txt")

    mlflow.lightgbm.log_model(model, name=f"model_{group_name}")


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("diabetes-feature-groups-multiclass")

    # Load prepared splits (train balanced, test imbalanced)
    train_df, test_df = load_from_huggingface_or_local()

    # Load best hyperparameters from config
    with open(os.path.join("config", "best_hyperparameters.json"), "r") as f:
        best_params = json.load(f)
    best_params.update({"objective": "multiclass", "num_class": 3, "n_jobs": -1, "random_state": 42})

    feature_groups: Dict[str, List[str]] = {
        "blood_serum_labs": blood_serum_labs,
        "urine_kidney_labs": urine_kidney_labs,
        "lifestyle_factors": lifestyle_factors,
        "body_measurements": body_measurements,
        "pressure_cardio": pressure_cardio,
        "med_history_comorbidities": med_history_comorbidities,
        "all_features": all_features,
    }

    with mlflow.start_run(run_name="feature_group_study_multiclass"):
        mlflow.log_param("train_rows", int(len(train_df)))
        mlflow.log_param("test_rows", int(len(test_df)))
        mlflow.log_dict(train_df[LABEL_COL].value_counts().to_dict(), "train_label_counts.json")
        mlflow.log_dict(test_df[LABEL_COL].value_counts().to_dict(), "test_label_counts.json")
        mlflow.log_dict(train_df[AGG_COL].value_counts().to_dict(), "train_label_counts_agg.json")
        mlflow.log_dict(test_df[AGG_COL].value_counts().to_dict(), "test_label_counts_agg.json")
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        for group_name, feats in feature_groups.items():
            with mlflow.start_run(nested=True, run_name=group_name):
                run_for_feature_group(train_df, test_df, group_name, feats, best_params)


if __name__ == "__main__":
    main()