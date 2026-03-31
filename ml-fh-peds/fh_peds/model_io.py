"""
Model serialisation for fh-peds training runs.

The canonical on-disk format is a single ``model.json`` file that bundles
the trained weights, intercept, preprocessing statistics, hyperparameters,
and dataset provenance.  This format is consumed directly by the pure-Python
inference module (``fh_peds.inference``).

Functions
---------
save_model_json
    Serialise a trained sklearn LogisticRegression to ``model.json``.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from fh_peds.utils import BINARY_CATEGORICAL_COLUMNS
from fh_peds.utils import CLASS_NAMES
from fh_peds.utils import DATA_INFO
from fh_peds.utils import MULTI_CATEGORICAL_COLUMNS
from fh_peds.utils import X_COLUMNS_RAW
from fh_peds.utils import Y_COLUMN


def save_model_json(
    model: LogisticRegression,
    scaling_info: dict,
    results_dir: Path,
    *,
    timestamp: str,
    size_test_split: float,
    random_state: int,
    param_grid: dict,
) -> Path:
    """Serialise a trained :class:`~sklearn.linear_model.LogisticRegression`
    to ``<results_dir>/model.json``.

    The file contains model weights, intercept, preprocessing statistics
    (z-score parameters), hyperparameters, and dataset provenance metadata.
    It is intentionally self-contained so that inference can run without any
    sklearn dependency.

    Parameters
    ----------
    model:
        Fitted logistic regression estimator.
    scaling_info:
        Mapping of ``{column: {"mean": float, "std": float}}`` as returned by
        :func:`fh_peds.utils.impute_and_scale_data`.
    results_dir:
        Directory into which ``model.json`` is written.
    timestamp:
        ISO-style timestamp string used for provenance metadata.
    size_test_split:
        Fraction of SLO cohort reserved for the test split.
    random_state:
        Random seed used for the train/test split.
    param_grid:
        Hyperparameter grid passed to GridSearchCV.

    Returns
    -------
    model_json_path:
        Absolute path to the written ``model.json`` file.
    """
    assert model.coef_.shape == (1, len(model.feature_names_in_))

    training_cohorts = [
        {
            "cohort": cohort,
            "version": version,
            "file_name": DATA_INFO[(cohort, version)]["file_name"],
            "sheet_name": DATA_INFO[(cohort, version)]["sheet_name"],
            "role": "train_val" if cohort == "slo" else "test_external",
        }
        for cohort, version in [("slo", "final"), ("por", "final")]
    ]

    model_json = {
        "metadata": {
            "timestamp": timestamp,
            "description": "L2-regularised logistic regression for FH pediatric screening",
            "label_column": Y_COLUMN,
            "class_names": CLASS_NAMES,
            "training_cohorts": training_cohorts,
            "train_val_split": {
                "cohort": "slo",
                "test_size": size_test_split,
                "random_state": random_state,
                "stratify_by": Y_COLUMN,
            },
            "hyperparameter_search": {
                "method": "GridSearchCV",
                "cv_folds": 3,
                "scoring": "roc_auc",
                "param_grid": param_grid,
            },
        },
        "hyperparameters": {
            "C": model.C,
            "penalty": model.penalty,
            "class_weight": model.class_weight,
            "fit_intercept": model.fit_intercept,
            "max_iter": model.max_iter,
            "random_state": model.random_state,
        },
        "features": {
            "input_fields": X_COLUMNS_RAW,
            "model_fields": list(model.feature_names_in_),
            "binary_categorical": BINARY_CATEGORICAL_COLUMNS,
            "multi_categorical": MULTI_CATEGORICAL_COLUMNS,
            "continuous_normalized": list(scaling_info.keys()),
        },
        "preprocessing": scaling_info,
        "weights": {
            feature: float(weight)
            for feature, weight in zip(model.feature_names_in_, model.coef_[0])
        },
        "intercept": float(model.intercept_[0]),
    }

    model_json_path = results_dir / "model.json"
    with open(model_json_path, "w") as f:
        json.dump(model_json, f, indent=4)

    return model_json_path


def save_predictions(
    data_raw: pd.DataFrame,
    data: pd.DataFrame,
    model: LogisticRegression,
    results_dir: Path,
) -> Path:
    """Save per-sample predicted probabilities alongside the raw data.

    Parameters
    ----------
    data_raw:
        Original (unscaled) dataset with the same index as ``data``.
    data:
        Preprocessed dataset containing the model feature columns.
    model:
        Fitted logistic regression estimator.
    results_dir:
        Directory into which ``model_split_probability.xlsx`` is written.

    Returns
    -------
    xlsx_path:
        Absolute path to the written Excel file.
    """
    from fh_peds.utils import X_COLUMNS

    assert (data_raw.index == data.index).all()

    out = data_raw.copy()
    out["split"] = data["split"]
    out["predicted_probability"] = pd.Series(
        model.predict_proba(data[X_COLUMNS])[:, 1], index=data.index
    )

    xlsx_path = results_dir / "model_split_probability.xlsx"
    out.to_excel(xlsx_path)
    return xlsx_path
