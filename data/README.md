# Data Directory

This directory contains all data used in the diabetes prediction project.

## Structure

- `raw/`: Original, immutable data dump. Never modify files in this directory.
- `processed/`: Cleaned and preprocessed data ready for modeling.
- `external/`: External data sources and reference datasets.

## Usage

Place raw datasets in the `raw/` directory. All data processing scripts should read from `raw/` and write to `processed/`.
