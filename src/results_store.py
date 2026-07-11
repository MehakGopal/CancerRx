"""Append-only store for model evaluation metrics.

Every call to `append_results` adds new rows to results/metrics.csv (creating it
on first use) instead of overwriting — score history survives notebook restarts
and accumulates across runs.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_METRICS_PATH = Path(__file__).resolve().parent.parent / "results" / "metrics.csv"
DEFAULT_BEST_MODELS_PATH = Path(__file__).resolve().parent.parent / "results" / "best_models.csv"

def append_results(results: list[dict], stage: str, feature_prefix: str,
                    path: Path = DEFAULT_METRICS_PATH) -> str:
    """Append a batch of per-model results to the metrics log, tagged with a run_id.

    Returns the run_id so callers can pull this run's rows back out later.
    """
    run_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now(timezone.utc).isoformat()

    batch = pd.DataFrame(results)
    batch.insert(0, "run_id", run_id)
    batch.insert(1, "timestamp", timestamp)
    batch.insert(2, "stage", stage)
    batch.insert(3, "feature_prefix", feature_prefix)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    batch.to_csv(path, mode="a", header=write_header, index=False)

    return run_id


def load_results(path: Path = DEFAULT_METRICS_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_run(run_id: str, path: Path = DEFAULT_METRICS_PATH) -> pd.DataFrame:
    return load_results(path).query("run_id == @run_id")

def best_models(results: pd.DataFrame, path: Path = DEFAULT_BEST_MODELS_PATH) -> pd.DataFrame:
    """Append the best model per drug (by r2) to the best-models log, timestamped.

    Never overwrites -- every call adds a new snapshot so you can see how the
    best model per drug changes across runs.
    """
    best_per_drug = results.loc[results.groupby("DRUG_ID")["r2"].idxmax()].copy()
    best_per_drug.insert(0, "timestamp", datetime.now(timezone.utc).isoformat())

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    best_per_drug.to_csv(path, mode="a", header=write_header, index=False)

    return best_per_drug