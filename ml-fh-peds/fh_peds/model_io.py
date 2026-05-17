import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from fh_peds.constants import BINARY_CATEGORICAL_COLUMNS
from fh_peds.constants import CLASS_NAMES
from fh_peds.constants import DATA_INFO
from fh_peds.constants import MULTI_CATEGORICAL_COLUMNS
from fh_peds.constants import X_COLUMNS
from fh_peds.constants import X_COLUMNS_RAW
from fh_peds.constants import Y_COLUMN


def save_model_json(
    model: LogisticRegression,
    scaling_info: dict,
    results_dir: Path,
    *,
    timestamp: str,
    size_test_split: float,
    random_state: int,
    param_grid: dict,
    operating_point: dict,
) -> Path:
    """Serialise the trained model to ``model.json`` with weights, preprocessing stats, and provenance metadata.

    ``operating_point`` describes the clinical decision threshold and is
    consumed at the website's runtime to turn a raw probability into a
    diagnose / no-diagnose verdict. See :func:`fh_peds.plotting.find_operating_point`.
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
        "operating_point": operating_point,
        "weights": {
            feature: float(weight)
            for feature, weight in zip(
                model.feature_names_in_, model.coef_[0], strict=False
            )
        },
        "intercept": float(model.intercept_[0]),
    }

    model_json_path = results_dir / "model.json"
    with open(model_json_path, "w") as f:
        json.dump(model_json, f, indent=4)

    return model_json_path


def save_inference_samples(
    *,
    data_raw: pd.DataFrame,
    data: pd.DataFrame,
    model: LogisticRegression,
    results_dir: Path,
) -> Path:
    """Save every sample as a raw-input/probability pair to ``inference_samples.json``.

    Each entry in the output array has the form::

        {"input": {<raw field>: <value>, ...}, "probability": <float>}

    where ``input`` contains the original (unscaled) feature values and
    ``probability`` is the positive-class probability predicted by ``model``.
    This file is consumed by ``tests/test_inference.py`` to verify that the
    pure-Python inference module reproduces sklearn's output exactly.
    """
    assert (data_raw.index == data.index).all()

    probabilities = model.predict_proba(data[X_COLUMNS])[:, 1]

    records = [
        {
            "input": json.loads(data_raw.loc[sample_id, X_COLUMNS_RAW].to_json()),
            "probability": float(prob),
        }
        for sample_id, prob in zip(data_raw.index, probabilities, strict=False)
    ]

    inference_samples_path = results_dir / "inference_samples.json"
    with open(inference_samples_path, "w") as f:
        json.dump(records, f, indent=4)

    return inference_samples_path


def save_metrics_json(metrics: list[dict], results_dir: Path) -> Path:
    """Serialise per-split evaluation metrics to ``metrics.json``.

    Parameters
    ----------
    metrics:
        List of per-split metric dicts produced during the evaluation loop.
        Each entry has the shape::

            {
                "split":   str,          # e.g. "train_val" or "test"
                "cohort":  str,          # e.g. "slo" or "por"
                "version": str,          # e.g. "final"
                "auc":     float,
                "support": int,          # total number of samples
                "accuracy": float,
                "classes": {
                    "<class_name>": {
                        "precision": float,
                        "recall":    float,
                        "f1-score":  float,
                        "support":   int,
                    },
                    ...
                },
                "macro avg":    { ... },
                "weighted avg": { ... },
            }
    results_dir:
        Directory where ``metrics.json`` will be written.

    Returns:
    -------
    Path
        Absolute path to the written file.
    """
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    return metrics_path


def save_predictions(
    *,
    data_raw: pd.DataFrame,
    data: pd.DataFrame,
    model: LogisticRegression,
    results_dir: Path,
) -> Path:
    """Save per-sample predicted probabilities alongside the raw data to ``model_split_probability.xlsx``."""
    assert (data_raw.index == data.index).all()

    out = data_raw.copy()
    out["split"] = data["split"]
    out["predicted_probability"] = pd.Series(
        model.predict_proba(data[X_COLUMNS])[:, 1], index=data.index
    )

    xlsx_path = results_dir / "model_split_probability.xlsx"
    out.to_excel(xlsx_path)
    return xlsx_path
