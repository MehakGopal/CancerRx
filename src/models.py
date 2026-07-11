"""Model registries for drug-sensitivity regression."""
from __future__ import annotations

from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 20
BOOSTING_MODELS = {"xgboost", "lightgbm"}


def baseline_models() -> dict:
    return {
        "elasticnet": make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=[.5, .7, .9, .95, .99, 1],
                eps=1e-2,
                cv=5,
                random_state=RANDOM_STATE,
                max_iter=5000,
            ),
        ),
        "dummy_regressor": make_pipeline(StandardScaler(), DummyRegressor(strategy="mean")),
    }


def candidate_models() -> dict:
    return {
        "decision_tree": DecisionTreeRegressor(
            max_depth=5,              # unconstrained -> memorized 650 rows
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,              # unconstrained trees inside RF still overfit individually
            min_samples_leaf=5,
            max_features="sqrt",        # default 1.0 -> all features per tree = overfit
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "svm": make_pipeline(StandardScaler(), SVR(C=1.0, gamma="scale")),
        "mlp": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 32),
            alpha=1.0,                 # 0 -> no regularization on 50-dim/650-row data
            early_stopping=True,       # missing -> ran full 2000 iters and overfit
            validation_fraction=0.15,
            n_iter_no_change=20,
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=3,               # unconstrained -> deep trees + high default lr = instant overfit
            learning_rate=0.03,        # default 0.3 -> validation loss blew up after tree 1
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            eval_metric="rmse",
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=300,
            max_depth=4,
            num_leaves=15,              # default 31 leaves too many splits for 650 rows
            min_child_samples=20,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }
