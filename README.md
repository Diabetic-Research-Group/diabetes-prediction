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

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Diabetic-Research-Group/diabetes-prediction.git
cd diabetes-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

### Usage

1. Place raw data in `data/raw/`
2. Use notebooks in `notebooks/` for exploratory analysis
3. Develop reusable code in `src/`
4. Save trained models to `models/`
5. Generate reports and figures in `reports/`

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
