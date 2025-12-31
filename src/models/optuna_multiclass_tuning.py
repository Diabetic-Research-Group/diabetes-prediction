from __future__ import annotations

import os
from typing import Dict, List, Tuple

import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.data.prepare_diabetes_data import (
    AGG_COL,
    LABEL_COL,
    load_from_huggingface_or_local,
)


RANDOM_STATE = 42
N_SPLITS = 10

CLASS_TO_ID = {"Not diabetic": 0, "T2D": 1, "Other": 2}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

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

urine_kidney_labs = [
    "URXUCR__response",
    "URXUMA__response",
    "URXUMS__response",
    "VNEGFR__response",
]

lifestyle_factors = [
    "ALQ130__questionnaire",
    "PAD680__questionnaire",
    "PAQ655__questionnaire",
    "PAQ670__questionnaire",
]

body_measurements = [
    "BMXBMI__response",
    "BMXHT__response",
    "BMXWT__response",
    "BMXWAIST__response",
    "RIDAGEYR__demographics",
]

pressure_cardio = [
    "BPXSAR__response",
    "BPXDAR__response",
    "BPXPLS__response",
    "BPXPULS__response",
]

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

FEATURE_COLUMNS = (
    blood_serum_labs
    + urine_kidney_labs
    + lifestyle_factors
    + body_measurements
    + pressure_cardio
    + med_history_comorbidities
)


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
            # If no active run, fall back to stdout to avoid silent failures
            print(f"[WARN] Non-numeric values encountered for {target_name}: {invalid}")
    return out


def _calc_specificity(cm: np.ndarray, labels: List[int]) -> Tuple[float, Dict[int, float]]:
    per_class: Dict[int, float] = {}
    total = cm.sum()
    for idx, lbl in enumerate(labels):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        per_class[lbl] = tn / denom if denom > 0 else 0.0
    macro = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return macro, per_class


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    labels = [0, 1, 2]
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    pr_per_class, rc_per_class, f1_per_class, support = precision_recall_fscore_support(
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
    spec_macro, spec_per_class = _calc_specificity(cm, labels)
    metrics["specificity_macro"] = spec_macro
    for cls_idx, spec in spec_per_class.items():
        metrics[f"specificity_class_{cls_idx}"] = spec

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


def cross_validate_model(
    X: pd.DataFrame,
    y: np.ndarray,
    params: Dict[str, float],
    n_splits: int = N_SPLITS,
) -> Tuple[Dict[str, float], np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    metrics_list: List[Dict[str, float]] = []
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_proba: List[np.ndarray] = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        fold_metrics, _ = compute_metrics(y_val, y_pred, y_proba)
        metrics_list.append(fold_metrics)

        all_true.append(y_val)
        all_pred.append(y_pred)
        all_proba.append(y_proba)

    agg_metrics: Dict[str, float] = {}
    for key in metrics_list[0]:
        agg_metrics[key] = float(np.mean([m[key] for m in metrics_list]))

    y_true_concat = np.concatenate(all_true)
    y_pred_concat = np.concatenate(all_pred)
    cm_total = confusion_matrix(y_true_concat, y_pred_concat, labels=[0, 1, 2])
    return agg_metrics, cm_total


def suggest_params(trial: optuna.Trial) -> Dict[str, float]:
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    return params


def prepare_features(df: pd.DataFrame, feature_cols: List[str], dataset_tag: str) -> Tuple[pd.DataFrame, np.ndarray]:
    cols = _select_existing_features(df, feature_cols)
    X = _to_numeric(df, cols, dataset_tag=dataset_tag)
    y = df[AGG_COL].map(CLASS_TO_ID).astype(int).to_numpy()
    return X, y


def log_classification_report(y_true: np.ndarray, y_pred: np.ndarray, artifact_name: str):
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=[ID_TO_CLASS[i] for i in [0, 1, 2]], digits=4, zero_division=0)
    mlflow.log_text(report, artifact_name)


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("diabetes-optuna-multiclass")

    train_df, test_df = load_from_huggingface_or_local()
    feature_cols = _select_existing_features(train_df, FEATURE_COLUMNS)

    with mlflow.start_run(run_name="optuna_tuning"):
        X_train, y_train = prepare_features(train_df, feature_cols, dataset_tag="train")
        X_test, y_test = prepare_features(test_df, feature_cols, dataset_tag="test")

        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial)
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(params)
                cv_metrics, cv_cm = cross_validate_model(X_train, y_train, params)
                mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})
                mlflow.log_dict({"confusion_matrix": cv_cm.tolist()}, f"cv_cm_trial_{trial.number}.json")
                plot_and_log_confusion(cv_cm, labels=[0, 1, 2], title="CV Confusion", artifact_name=f"cv_cm_trial_{trial.number}")
                return cv_metrics.get("roc_auc_macro", 0.0)

        mlflow.log_param("train_rows", int(len(train_df)))
        mlflow.log_param("test_rows", int(len(test_df)))
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_dict(train_df[LABEL_COL].value_counts().to_dict(), "train_label_counts.json")
        mlflow.log_dict(test_df[LABEL_COL].value_counts().to_dict(), "test_label_counts.json")
        mlflow.log_dict(train_df[AGG_COL].value_counts().to_dict(), "train_label_counts_agg.json")
        mlflow.log_dict(test_df[AGG_COL].value_counts().to_dict(), "test_label_counts_agg.json")
        if os.path.exists(os.path.join("data", "diabetes_train.parquet")):
            mlflow.log_artifact(os.path.join("data", "diabetes_train.parquet"), artifact_path="data")
        if os.path.exists(os.path.join("data", "diabetes_test.parquet")):
            mlflow.log_artifact(os.path.join("data", "diabetes_test.parquet"), artifact_path="data")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        best_params = study.best_params
        best_params.update({"objective": "multiclass", "num_class": 3, "n_jobs": -1, "random_state": RANDOM_STATE})
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        best_model = LGBMClassifier(**best_params)
        best_model.fit(X_train, y_train)

        y_pred_test = best_model.predict(X_test)
        y_proba_test = best_model.predict_proba(X_test)
        test_metrics, test_cm = compute_metrics(y_test, y_pred_test, y_proba_test)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_dict({"confusion_matrix": test_cm.tolist()}, "test_confusion_matrix.json")

        plot_and_log_confusion(test_cm, labels=[0, 1, 2], title="Test Confusion", artifact_name="test_confusion")
        log_classification_report(y_test, y_pred_test, artifact_name="test_classification_report.txt")

        mlflow.lightgbm.log_model(best_model, name="model")


if __name__ == "__main__":
    main()