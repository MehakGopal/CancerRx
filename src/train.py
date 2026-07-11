"""Fit model registries per-drug and log metrics to the persistent results store."""
from __future__ import annotations

import lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from src import results_store
from src.models import EARLY_STOPPING_ROUNDS

TARGET_COL = "LN_IC50"
DRUG_ID_COL = "DRUG_ID"
DRUG_NAME_COL = "DRUG_NAME"


def _fit_one(model_name, model, X_train, y_train, X_val, y_val):
    n_trees_used = None
    if model_name == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        n_trees_used = model.best_iteration + 1
    elif model_name == "lightgbm":
        model.fit(
            X_train, y_train, eval_set=[(X_val, y_val)],
            callbacks=[lightgbm.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        n_trees_used = model.best_iteration_
    else:
        model.fit(X_train, y_train)
    return model, n_trees_used


def fit_and_evaluate(models: dict, train_df: pd.DataFrame, val_df: pd.DataFrame,
                     feature_prefix:str,feature_cols: list[str] | None = None, target_col: str = TARGET_COL,
                      drug_id_col: str = DRUG_ID_COL, drug_name_col: str = DRUG_NAME_COL):
    """Fit every model in `models` for every drug in train_df, evaluate on val_df.

    Returns (results: list[dict], fitted_models: dict[(drug_id, model_name), estimator]).
    """
    if feature_cols is None:
        feature_cols = [c for c in train_df.columns if c.startswith(feature_prefix)]

    results = []
    fitted_models = {}

    for drug_id, drug_train in train_df.groupby(drug_id_col):
        drug_name = drug_train[drug_name_col].iloc[0]
        drug_val = val_df[val_df[drug_id_col] == drug_id]

        X_train, y_train = drug_train[feature_cols], drug_train[target_col]
        X_val, y_val = drug_val[feature_cols], drug_val[target_col]

        for model_name, model in models.items():
            fitted, n_trees_used = _fit_one(model_name, model, X_train, y_train, X_val, y_val)
            preds = fitted.predict(X_val)

            results.append({
                "DRUG_ID": drug_id, "DRUG_NAME": drug_name, "model": model_name,
                "n_train": len(X_train), "n_val": len(X_val), "n_trees_used": n_trees_used,
                "rmse": np.sqrt(mean_squared_error(y_val, preds)),
                "r2": r2_score(y_val, preds),
            })
            fitted_models[(drug_id, model_name)] = fitted

    return results, fitted_models


def run_stage(models: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, stage: str,
              feature_prefix: str, **kwargs):
    """Fit+evaluate a model registry and persist the results, tagged with `stage`.

    Returns (results_df for just this run, fitted_models, run_id). The full
    history of every run ever made lives in results/metrics.csv regardless of
    whether this notebook/process is still alive.
    """
    results, fitted_models = fit_and_evaluate(models, train_df, val_df, feature_prefix=feature_prefix, **kwargs)
    run_id = results_store.append_results(results, stage=stage, feature_prefix=feature_prefix)
    results_df = pd.DataFrame(results).sort_values(["DRUG_ID", "rmse"])
    results_df.insert(0, "feature_prefix", feature_prefix)
    return results_df, fitted_models, run_id
