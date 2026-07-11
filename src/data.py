"""Loading, merging, and splitting the pathway-enriched drug response data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

META_COLS = [
    "model_id", "model_name", "DRUG_ID", "DRUG_NAME", "DATASET",
    "NLME_RESULT_ID", "NLME_CURVE_ID", "CELL_LINE_NAME", "SANGER_MODEL_ID",
    "CANCER_TYPE", "PUTATIVE_TARGET", "PATHWAY_NAME", "MIN_CONC",
    "MAX_CONC", "LN_IC50", "AUC", "RMSE", "Z_SCORE",
]

DROP_COLS = ["DATASET", "NLME_RESULT_ID", "NLME_CURVE_ID", "CELL_LINE_NAME", "SANGER_MODEL_ID"]


def build_pathway_enriched_data(required_data_path: Path, gsva_scores_path: Path) -> pd.DataFrame:
    """Merge per-cell-line metadata/IC50 data with GSVA pathway enrichment scores."""
    required_data = pd.read_parquet(required_data_path)
    required_data_cleaned = required_data.dropna(axis=1)  # replicate R pipeline

    gsva_scores = pd.read_csv(gsva_scores_path)
    gsva_t = gsva_scores.transpose()
    gsva_t.columns = gsva_t.iloc[0]
    gsva_t = gsva_t.drop(gsva_t.index[0])
    gsva_t.reset_index(inplace=True)

    required_columns = required_data_cleaned[META_COLS]
    merged = pd.merge(required_columns, gsva_t, left_on="model_id", right_on="index")
    merged = merged.drop(columns=DROP_COLS)
    return merged


def split_by_cell_line(df: pd.DataFrame, train_size=0.7, val_size=0.1, random_state=42):
    """Group-split one drug's data by model_id (cell line) so no cell line
    leaks across train/val/test."""
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(df, groups=df["model_id"]))
    train = df.iloc[train_idx]
    temp = df.iloc[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_size / (1 - train_size), random_state=random_state)
    val_idx, test_idx = next(gss2.split(temp, groups=temp["model_id"]))
    val = temp.iloc[val_idx]
    test = temp.iloc[test_idx]

    return train, val, test


def make_splits(df: pd.DataFrame, drug_col: str = "DRUG_ID", **split_kwargs):
    """Apply split_by_cell_line independently per drug, then recombine."""
    train_parts, val_parts, test_parts = [], [], []
    for _, drug_df in df.groupby(drug_col):
        train, val, test = split_by_cell_line(drug_df, **split_kwargs)
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    _assert_no_leakage(df, train_df, val_df, test_df, drug_col)
    return train_df, val_df, test_df


def _assert_no_leakage(df, train_df, val_df, test_df, drug_col):
    for drug_id in df[drug_col].unique():
        train_ids = set(train_df.loc[train_df[drug_col] == drug_id, "model_id"])
        val_ids = set(val_df.loc[val_df[drug_col] == drug_id, "model_id"])
        test_ids = set(test_df.loc[test_df[drug_col] == drug_id, "model_id"])
        assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                 outdir: Path, suffix: str = "C2") -> None:
    outdir = Path(outdir)
    train_df.to_csv(outdir / f"train_{suffix}.csv", index=False)
    val_df.to_csv(outdir / f"val_{suffix}.csv", index=False)
    test_df.to_csv(outdir / f"test_{suffix}.csv", index=False)


def load_splits(datadir: Path, suffix: str = "C2"):
    datadir = Path(datadir)
    train_df = pd.read_csv(datadir / f"train_{suffix}.csv")
    val_df = pd.read_csv(datadir / f"val_{suffix}.csv")
    test_df = pd.read_csv(datadir / f"test_{suffix}.csv")
    return train_df, val_df, test_df
