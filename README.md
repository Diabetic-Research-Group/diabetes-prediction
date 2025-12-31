# Diabetes Prediction

Entire codebase for predicting diabetes with labs and lifestyle factors.

## Project Structure

```
diabetes-prediction/
├── data/
│   ├── raw/                 # Original, immutable data
│   ├── processed/           # Cleaned and preprocessed data
│   └── external/            # External data sources
├── notebooks/               # Jupyter notebooks for exploration and analysis
├── src/                     # Source code for the project
│   ├── data/               # Data processing scripts
│   ├── features/           # Feature engineering scripts
│   └── models/             # Model training and prediction scripts
├── models/                  # Trained model files
├── reports/                 # Generated analysis and reports
│   └── figures/            # Figures and visualizations
├── tests/                   # Unit and integration tests
├── config/                  # Configuration files
├── requirements.txt         # Python dependencies
├── setup.py                # Package installation configuration
└── README.md               # This file
```

## Prerequisites

Ensure you have the following installed:

- Python 3.13+
- Poetry
- Git

## Getting Started

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Diabetic-Research-Group/diabetes-prediction.git
    cd diabetes-prediction
    ```

2. Install dependencies using Poetry:

    ```bash
    poetry install
    ```

### Usage

1. Place raw data in `data/raw/`
2. Use notebooks in `notebooks/` for exploratory analysis
3. Develop reusable code in `src/`
4. Save trained models to `models/`
5. Generate reports and figures in `reports/`

## Hyperparameter Tuning & Model Training

### Step 1: Prepare Data Splits

Run the data preparation script to create balanced training and test splits:

```bash
    poetry run python -m src.data.prepare_diabetes_data
```

This will:

- Load the raw dataset from Hugging Face (or use local parquet if available)
- Create a balanced training set with ~4,320 samples per class (Not diabetic, T2D, Other)
- Create a held-out test set preserving the original class distribution
- Save datasets as parquet files to `data/diabetes_train.parquet` and `data/diabetes_test.parquet`
- Log class distributions and data info

**Note:** Update the Hugging Face URLs in [src/data/prepare_diabetes_data.py](src/data/prepare_diabetes_data.py) after uploading datasets.

### Step 2: Run Hyperparameter Tuning with Optuna

Execute the Optuna-based tuning script to optimize LightGBM hyperparameters:

```bash
poetry run python -m src.models.optuna_multiclass_tuning
```

This will:
- Use 10-fold stratified cross-validation on the balanced training set to evaluate each trial
- Run 20 Optuna trials, tuning 7 key hyperparameters:
  - `n_estimators` (200–800)
  - `learning_rate` (0.01–0.2, log scale)
  - `max_depth` (3–12)
  - `num_leaves` (16–128)
  - `min_child_samples` (5–50)
  - `reg_alpha` (L1, 1e-8–1.0, log scale)
  - `reg_lambda` (L2, 1e-8–1.0, log scale)
- Log CV metrics per trial to MLflow (under nested runs)
- Select the best model based on macro ROC-AUC from CV
- Train the best model on the full training set
- Evaluate on the held-out test set and log test metrics to MLflow (parent run)
- Log confusion matrices (PNG + JSON), classification reports, datasets, and the trained model

**Metrics Logged:**
- Per-trial CV metrics: `cv_accuracy`, `cv_roc_auc_macro`, `cv_pr_auc_macro`, `cv_recall_macro`, `cv_specificity_macro`, per-class precision/recall/F1
- Final test metrics: `test_*` variants (use `test_roc_auc_weighted` and `test_pr_auc_weighted` for imbalanced evaluation)
- Confusion matrices and classification reports as artifacts

### Step 3: View Results in MLflow

Launch the MLflow UI to visualize tuning results and metrics:

```bash
poetry run mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

**Navigation:**
- **Experiment:** "diabetes-optuna-multiclass"
- **Parent Run:** "optuna_tuning" – contains:
  - Dataset info (row counts, class distributions)
  - Best hyperparameters (`best_*` params)
  - Final test metrics (`test_*`, use weighted variants for imbalanced evaluation)
  - Confusion matrix visualization (`test_confusion.png`)
  - Classification report (`test_classification_report.txt`)
  - Datasets folder with train/test parquet files
  - Trained model (logged via MLflow)
- **Nested Runs:** Each trial (trial_0, trial_1, ..., trial_19) contains:
  - 10-fold CV metrics (`cv_*`)
  - CV confusion matrix (`cv_cm_trial_*.json`)
  - Non-numeric value warnings if any

### Step 4: Use the Trained Model

The best model is automatically trained and logged to MLflow. Load it in Python:

```python
import mlflow.lightgbm
import pandas as pd

# Set tracking URI to match your MLflow setup
mlflow.set_tracking_uri("file:./mlruns")

# Find the experiment and run
experiment = mlflow.get_experiment_by_name("diabetes-optuna-multiclass")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], 
                          filter_string="tags.mlflow.runName = 'optuna_tuning'")
best_run_id = runs.iloc[0].run_id  # Most recent optuna_tuning parent run

# Load the trained model
model = mlflow.lightgbm.load_model(f"runs:/{best_run_id}/model")

# Make predictions on new data
# (Assuming X is a DataFrame with the same feature columns used in training)
y_pred = model.predict(X)
y_proba = model.predict_proba(X)
```

**Or, load from latest run directly:**

```python
import mlflow

mlflow.set_tracking_uri("file:./mlruns")
latest_run = mlflow.search_runs(experiment_names=["diabetes-optuna-multiclass"], 
                                max_results=1).iloc[0]
model = mlflow.lightgbm.load_model(f"runs:/{latest_run.run_id}/model")
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

Format code with Black:
```bash
black src/ tests/
```

Check code style:
```bash
flake8 src/ tests/
```

## License

See LICENSE file for details.
