# Hyperparameter Tuning and Model Selection Methodology

## Overview

This document describes the methodology used for hyperparameter optimization and model selection in the diabetes prediction project using Optuna and MLflow.

## Dataset Preparation

### Training Set (Balanced)
- **Composition:** Three classes with equal representation (~4,320 samples each)
  - Class 0: Not diabetic
  - Class 1: T2D (Type 2 Diabetes)
  - Class 2: Other (includes T1D, Possible-T2D, Borderline, Excluded, Skipped)
- **Purpose:** Balanced training set ensures the model learns all classes equally without bias toward the majority class

### Test Set (Imbalanced, Original Distribution)
- **Composition:** Preserves the original class distribution from the dataset:
  - Not diabetic: ~83%
  - T2D: ~5%
  - T1D: ~0.2%
  - Possible-T2D: ~0.5%
  - Borderline: ~1%
  - Excluded: ~0.4%
  - Skipped: ~10%
- **Purpose:** Realistic evaluation that reflects real-world class imbalance

## Hyperparameter Optimization

### Framework
- **Optimization Tool:** Optuna
- **Algorithm:** Tree-structured Parzen Estimator (TPE)
- **Number of Trials:** 20
- **Cross-Validation:** 10-fold stratified CV on the balanced training set

**What is a Trial?**
- Each trial represents one complete model training run with a specific hyperparameter configuration
- 20 trials = training the model 20 times with different hyperparameter combinations
- Each trial involves 10-fold cross-validation, so 20 trials × 10 folds = 200 total model training runs

**Why Not Try All Combinations?**
- Hyperparameters include continuous values (e.g., `learning_rate` from 0.01 to 0.2)
- There are infinite possible combinations in the search space
- Exhaustive grid search would be computationally prohibitive

**How Optuna Works:**
1. Trial 0: Randomly samples initial hyperparameters → evaluates performance
2. Trial 1-19: Uses Bayesian optimization (TPE algorithm) to intelligently suggest next hyperparameters based on previous trial results
3. Optuna learns which hyperparameter regions produce better results and focuses exploration there
4. After all trials complete, selects the hyperparameters that achieved the highest cross-validation ROC-AUC

**Adjusting Number of Trials:**
- Fewer trials (e.g., 10): Faster, but may miss optimal hyperparameters
- More trials (e.g., 50-100): Better chance of finding optimal values, but longer runtime
- Current setting (20): Balances exploration and computational cost
- To change: Modify `n_trials=20` in `src/models/optuna_multiclass_tuning.py`

### Hyperparameters Tuned
1. **n_estimators** (200–800): Number of boosting iterations
2. **learning_rate** (0.01–0.2, log scale): Step size shrinkage to prevent overfitting
3. **max_depth** (3–12): Maximum tree depth for base learners
4. **num_leaves** (16–128): Maximum number of leaves in one tree
5. **min_child_samples** (5–50): Minimum number of samples in a leaf
6. **reg_alpha** (1e-8–1.0, log scale): L1 regularization term
7. **reg_lambda** (1e-8–1.0, log scale): L2 regularization term

### Optimization Metric

**Primary Metric:** `roc_auc_macro` (Macro-averaged ROC-AUC from 10-fold CV)

**Rationale:**
- The training set is balanced with equal class representation
- Macro averaging treats all classes equally, which is appropriate for balanced data
- ROC-AUC is robust to class imbalance within folds and measures the model's ability to discriminate between classes
- Maximizing macro ROC-AUC ensures the model performs well across all three classes (Not diabetic, T2D, Other)

**Implementation:**
```python
def objective(trial):
    # ... train model with suggested hyperparameters
    cv_metrics, cv_cm = cross_validate_model(X_train, y_train, params)
    return cv_metrics.get("roc_auc_macro", 0.0)  # Optuna maximizes this
```

## Final Model Training and Evaluation

### Model Training
After Optuna identifies the best hyperparameters:
1. The hyperparameters with the highest `cv_roc_auc_macro` are selected
2. A final LightGBM model is trained on the **entire balanced training set** using these optimal hyperparameters
3. The model is logged to MLflow for reproducibility

### Final Evaluation (Held-Out Test Set)

The final model is evaluated on the **imbalanced held-out test set** to measure real-world performance.

**Metrics Logged:**

#### For Research Papers and Reporting (Use These):
- **test_roc_auc_weighted**: Weighted ROC-AUC (accounts for class imbalance)
- **test_pr_auc_weighted**: Weighted PR-AUC (precision-recall, better for imbalanced data)
- **test_recall_weighted**: Weighted recall
- **test_precision_weighted**: Weighted precision
- **test_f1_weighted**: Weighted F1 score
- **test_specificity_macro**: Macro-averaged specificity across all classes

#### Per-Class Performance:
- **test_precision_class_0**: Precision for "Not diabetic"
- **test_recall_class_0**: Recall for "Not diabetic"
- **test_f1_class_0**: F1 score for "Not diabetic"
- **test_precision_class_1**: Precision for "T2D"
- **test_recall_class_1**: Recall for "T2D" (critical for diabetes detection)
- **test_f1_class_1**: F1 score for "T2D"
- **test_precision_class_2**: Precision for "Other"
- **test_recall_class_2**: Recall for "Other"
- **test_f1_class_2**: F1 score for "Other"

#### Additional Metrics:
- **test_accuracy**: Overall accuracy
- **test_roc_auc_macro**: Macro-averaged ROC-AUC (equal class weighting)
- **test_pr_auc_macro**: Macro-averaged PR-AUC
- **Confusion Matrix**: Saved as JSON and PNG for detailed error analysis

### Why Weighted Metrics for Test Evaluation?

**Weighted averaging** multiplies each class metric by the number of samples in that class, then divides by the total number of samples. This:
- Reflects the true class distribution in the real world
- Gives more importance to the majority class ("Not diabetic") which is clinically relevant
- Provides a realistic estimate of model performance in deployment
- Is the standard for reporting on imbalanced datasets in research papers

**Macro averaging** treats all classes equally regardless of their frequency. While useful for understanding per-class performance, it can be misleading when class distributions are heavily imbalanced.

## Reproducibility

All experiments are tracked in MLflow with:
- Exact hyperparameters for each trial
- Cross-validation metrics per trial
- Final test metrics in the parent run
- Trained model artifacts
- Datasets used for training and testing
- Confusion matrices and classification reports

**MLflow Experiment Name:** `diabetes-optuna-multiclass`

**To reproduce:**
```bash
poetry run python -m src.models.optuna_multiclass_tuning
mlflow ui  # View results at http://localhost:5000
```

## Model Loading and Inference

The best model can be loaded from MLflow:

```python
import mlflow.lightgbm

mlflow.set_tracking_uri("file:./mlruns")
latest_run = mlflow.search_runs(
    experiment_names=["diabetes-optuna-multiclass"], 
    max_results=1
).iloc[0]
model = mlflow.lightgbm.load_model(f"runs:/{latest_run.run_id}/model")

# Predictions
y_pred = model.predict(X)  # Class labels (0, 1, 2)
y_proba = model.predict_proba(X)  # Probabilities for each class
```

## Summary for Research Papers

**Model Selection:**
- Hyperparameters optimized using 10-fold cross-validation on a balanced training set (n=~12,960)
- Optimization metric: Macro-averaged ROC-AUC
- Optimization framework: Optuna with 20 trials
- Best model selected based on highest cross-validation macro ROC-AUC

**Model Evaluation:**
- Final model evaluated on held-out test set preserving original class distribution
- Primary metrics for reporting: Weighted ROC-AUC, Weighted PR-AUC, Weighted Recall
- Per-class metrics reported for clinical interpretability
- Confusion matrix provided for detailed error analysis

**Clinical Relevance:**
- Recall for T2D (Class 1) is critical for minimizing false negatives
- Weighted metrics reflect real-world deployment scenarios with class imbalance
- Three-class formulation (Not diabetic, T2D, Other) prevents misclassification of borderline/uncertain cases into binary categories

---

**Generated:** December 31, 2025  
**Script:** `src/models/optuna_multiclass_tuning.py`  
**Data Preparation:** `src/data/prepare_diabetes_data.py`
