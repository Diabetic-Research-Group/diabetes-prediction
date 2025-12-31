from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from datasets import load_dataset
from sklearn.utils import check_random_state
import datasets


LABEL_COL = "Diabetes_Type"
AGG_COL = "label_three_class"

# Placeholders: replace after uploading to Hugging Face.
HF_TRAIN_URL = "rtweera/diabetes_prediction_train"
HF_TEST_URL = "rtweera/diabetes_prediction_test"


@dataclass
class SplitSummary:
    train_counts: Dict[str, int]
    test_counts: Dict[str, int]
    train_counts_agg: Dict[str, int]
    test_counts_agg: Dict[str, int]
    original_counts: Dict[str, int]
    original_percentages: Dict[str, float]
    train_path: str
    test_path: str


def _map_to_three_classes(raw_label: str) -> str:
    if raw_label == "T2D":
        return "T2D"
    if raw_label == "Not diabetic":
        return "Not diabetic"
    return "Other"


def _compute_distribution(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


def _sample_balanced_train(
    df: pd.DataFrame,
    random_state: int,
    t2d_train_fraction: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    rng = check_random_state(random_state)

    t2d_df = df[df[LABEL_COL] == "T2D"]
    not_df = df[df[LABEL_COL] == "Not diabetic"]
    other_df = df[~df[LABEL_COL].isin(["T2D", "Not diabetic"])]

    t2d_train_size = int(round(len(t2d_df) * t2d_train_fraction))
    t2d_train = t2d_df.sample(n=min(t2d_train_size, len(t2d_df)), random_state=rng)
    t2d_left = t2d_df.drop(t2d_train.index)

    target_per_class = len(t2d_train)
    not_train = not_df.sample(n=min(target_per_class, len(not_df)), random_state=rng)
    other_train = other_df.sample(n=min(target_per_class, len(other_df)), random_state=rng)

    train_df = pd.concat([t2d_train, not_train, other_train]).sample(frac=1.0, random_state=rng)
    remaining_df = df.drop(train_df.index)
    return train_df, remaining_df, len(t2d_left)


def _build_test_with_distribution(
    remaining_df: pd.DataFrame,
    original_percentages: Dict[str, float],
    t2d_left_count: int,
    random_state: int,
) -> pd.DataFrame:
    rng = check_random_state(random_state)

    perc_t2d = original_percentages.get("T2D", 0.0)
    desired_test_total = int(round(t2d_left_count / perc_t2d)) if perc_t2d > 0 else len(remaining_df)
    test_parts: List[pd.DataFrame] = []

    for label, perc in original_percentages.items():
        pool = remaining_df[remaining_df[LABEL_COL] == label]
        desired = int(round(desired_test_total * perc))
        take = min(len(pool), desired)
        if take > 0:
            test_parts.append(pool.sample(n=take, random_state=rng))

    test_df = pd.concat(test_parts) if test_parts else pd.DataFrame(columns=remaining_df.columns)

    # Guarantee we keep all leftover T2D for the held-out set
    t2d_remaining = remaining_df[remaining_df[LABEL_COL] == "T2D"]
    test_df = pd.concat([test_df, t2d_remaining]).drop_duplicates()

    # If we are short due to rounding, top up with a random sample from leftovers
    remaining_after = remaining_df.drop(test_df.index)
    needed = max(desired_test_total - len(test_df), 0)
    if needed > 0 and len(remaining_after) > 0:
        extra = remaining_after.sample(n=min(needed, len(remaining_after)), random_state=rng)
        test_df = pd.concat([test_df, extra])

    return test_df.sample(frac=1.0, random_state=rng)


def build_train_test_splits(
    df: pd.DataFrame,
    label_col: str = LABEL_COL,
    random_state: int = 42,
    t2d_train_fraction: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, SplitSummary]:
    df = df[df[label_col].notna()].copy()
    df[LABEL_COL] = df[label_col]

    original_counts = df[LABEL_COL].value_counts().to_dict()
    original_percentages = _compute_distribution(original_counts)

    train_df, remaining_df, t2d_left_count = _sample_balanced_train(
        df=df, random_state=random_state, t2d_train_fraction=t2d_train_fraction
    )
    test_df = _build_test_with_distribution(
        remaining_df=remaining_df,
        original_percentages=original_percentages,
        t2d_left_count=t2d_left_count,
        random_state=random_state,
    )

    for part in (train_df, test_df):
        part[AGG_COL] = part[LABEL_COL].apply(_map_to_three_classes)

    train_counts = train_df[LABEL_COL].value_counts().to_dict()
    test_counts = test_df[LABEL_COL].value_counts().to_dict()
    train_counts_agg = train_df[AGG_COL].value_counts().to_dict()
    test_counts_agg = test_df[AGG_COL].value_counts().to_dict()

    summary = SplitSummary(
        train_counts=train_counts,
        test_counts=test_counts,
        train_counts_agg=train_counts_agg,
        test_counts_agg=test_counts_agg,
        original_counts=original_counts,
        original_percentages=original_percentages,
        train_path="",
        test_path="",
    )
    return train_df, test_df, summary


def save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_dir: str = "data",
    train_name: str = "diabetes_train.parquet",
    test_name: str = "diabetes_test.parquet",
) -> Tuple[str, str]:
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, train_name)
    test_path = os.path.join(data_dir, test_name)
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    return train_path, test_path


def load_or_prepare_data(
    dataset_name: str = "rtweera/nhanes-data-converted",
    split: str = "train",
    random_state: int = 42,
    force_rebuild: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, SplitSummary]:
    train_path = os.path.join("data", "diabetes_train.parquet")
    test_path = os.path.join("data", "diabetes_test.parquet")

    if not force_rebuild and os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        summary = SplitSummary(
            train_counts=train_df[LABEL_COL].value_counts().to_dict(),
            test_counts=test_df[LABEL_COL].value_counts().to_dict(),
            train_counts_agg=train_df[AGG_COL].value_counts().to_dict(),
            test_counts_agg=test_df[AGG_COL].value_counts().to_dict(),
            original_counts={},
            original_percentages={},
            train_path=train_path,
            test_path=test_path,
        )
        return train_df, test_df, summary

    raw = load_dataset(dataset_name, split=split)
    df = raw.to_pandas()
    train_df, test_df, summary = build_train_test_splits(df, random_state=random_state)
    train_path, test_path = save_splits(train_df, test_df)
    summary.train_path = train_path
    summary.test_path = test_path
    return train_df, test_df, summary


def load_from_huggingface_or_local(train_path: str | None = None, test_path: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience loader that prefers local parquet files but can fall back to
    Hugging Face-hosted parquet URLs when provided.
    """

    if train_path is None:
        train_path = os.path.join("data", "diabetes_train.parquet")
    if test_path is None:
        test_path = os.path.join("data", "diabetes_test.parquet")

    if os.path.exists(train_path) and os.path.exists(test_path):
        return pd.read_parquet(train_path), pd.read_parquet(test_path)

    # Fallback to Hugging Face
    train_dataset = datasets.load_dataset(HF_TRAIN_URL, split="train")
    test_dataset = datasets.load_dataset(HF_TEST_URL, split="train")
    train_df = train_dataset.to_pandas()
    test_df = test_dataset.to_pandas()
    if type(train_df) is not pd.DataFrame or type(test_df) is not pd.DataFrame:
        raise ValueError(f"\
            Failed to load datasets from Hugging Face. Type mismatch.\
            Required: pandas.DataFrame but got train as {type(train_df)} and test as {type(test_df)}\
        ")
    return train_df, test_df


def main():
    train_df, test_df, summary = load_or_prepare_data(force_rebuild=True)
    print("Prepared splits saved to:")
    print(f"  Train: {summary.train_path} -> {len(train_df)} rows")
    print(f"  Test : {summary.test_path} -> {len(test_df)} rows")
    print("Original label distribution:")
    for label, perc in summary.original_percentages.items():
        print(f"  {label}: {perc:.3%}")


if __name__ == "__main__":
    main()