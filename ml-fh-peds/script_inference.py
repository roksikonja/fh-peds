"""
Extract trained model weights, intercept, and scaling info and print them
so they can be copied into inference.py.

Run this script after script.py has been executed and a trained model is
available.  Because this is a development utility the script re-runs the
full training pipeline internally rather than loading a persisted model.
"""

import json
import math
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils import BINARY_CATEGORICAL_COLUMNS
from utils import MULTI_CATEGORICAL_COLUMNS
from utils import X_COLUMNS
from utils import X_COLUMNS_RAW
from utils import Y_COLUMN
from utils import impute_and_scale_data
from utils import read_data
from utils import train_model_and_cv

# ---------------------------------------------------------------------------
# Configuration (must match script.py)
# ---------------------------------------------------------------------------

RECOMPUTE = True
RANDOM_STATE = 3
SIZE_TEST_SPLIT = 0.4

BASE_MODEL = LogisticRegression(random_state=0, penalty="l2", max_iter=100)
PARAM_GRID = {
    "class_weight": ["balanced", None],
    "C": [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0, 4.0, 16.0, 64.0, 128.0],
    "fit_intercept": [True, False],
}

project_dir = Path(__file__).parent
data_dir = project_dir / "data"

# ---------------------------------------------------------------------------
# Train (mirrors script.py steps 1-3)
# ---------------------------------------------------------------------------

data_slo = read_data(data_dir=data_dir, cohort="slo", version="final", recompute=RECOMPUTE)
data_por = read_data(data_dir=data_dir, cohort="por", version="final", recompute=RECOMPUTE)
data_raw = pd.concat([data_slo, data_por], axis=0)

data, scaling_info = impute_and_scale_data(
    data_raw, mask_predicate=lambda row: row["cohort"] == "slo"
)

data["split"] = "test"
indices_train_val, _ = train_test_split(
    data[data["cohort"] == "slo"].index,
    test_size=SIZE_TEST_SPLIT,
    random_state=RANDOM_STATE,
    stratify=data[data["cohort"] == "slo"][Y_COLUMN],
)
data.loc[indices_train_val, "split"] = "train_val"

model, _ = train_model_and_cv(
    model=BASE_MODEL,
    param_grid=PARAM_GRID,
    X=data[data["split"] == "train_val"][X_COLUMNS],
    y=data[data["split"] == "train_val"][Y_COLUMN],
    cv=3,
    scoring="roc_auc",
)

# ---------------------------------------------------------------------------
# Extract and print values to copy into inference.py
# ---------------------------------------------------------------------------

WEIGHTS = {
    feature_name: float(weight)
    for feature_name, weight in zip(model.feature_names_in_, model.coef_[0])
}
INTERCEPT = float(model.intercept_[0])

print("# --- paste into inference.py ---\n")
print(f"INTERCEPT = {INTERCEPT!r}")
print(f"WEIGHTS = {json.dumps(WEIGHTS, indent=4)}")
print(f"\nPREPROCESSING_INFO = {json.dumps(scaling_info, indent=4)}")

# ---------------------------------------------------------------------------
# Quick inline validation (no sklearn required after copy-paste)
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def preprocess_sample_inline(raw_sample: dict) -> dict:
    skip_columns = {"cohort", "predicted_probability", "version", "gen_conf_fh", "split"}
    assert set(raw_sample) - skip_columns == set(X_COLUMNS_RAW)

    sample = {}
    for feature_name, feature_value in raw_sample.items():
        if feature_name in skip_columns:
            continue
        if feature_name in BINARY_CATEGORICAL_COLUMNS:
            sample[feature_name] = float(feature_value)
        elif feature_name in MULTI_CATEGORICAL_COLUMNS:
            assert isinstance(feature_value, int)
            for value in [1, 2, 3]:
                sample[f"{feature_name}_{value}"] = 1.0 if feature_value == value else 0.0
        elif feature_name in scaling_info:
            assert isinstance(feature_value, float)
            mean = scaling_info[feature_name]["mean"]
            std = scaling_info[feature_name]["std"]
            sample[feature_name] = float((feature_value - mean) / std)
        else:
            assert False, feature_name

    assert len(set(sample) - set(X_COLUMNS)) == 0, set(sample) - set(X_COLUMNS)
    assert len(set(X_COLUMNS) - set(sample)) == 0, set(X_COLUMNS) - set(sample)
    return sample


def model_fn_inline(sample: dict) -> float:
    weighted_sum = INTERCEPT
    for feature_name, weight_value in WEIGHTS.items():
        weighted_sum += weight_value * sample[feature_name]
    return _sigmoid(weighted_sum)


print("\n# --- validation ---")
raw_sample = data_raw.iloc[0][X_COLUMNS_RAW].to_dict()
inline_prob = model_fn_inline(preprocess_sample_inline(raw_sample))
sklearn_prob = float(model.predict_proba(data.iloc[[0]][X_COLUMNS])[0, 1])
print(f"Inline:  {inline_prob:.6f}")
print(f"sklearn: {sklearn_prob:.6f}")
assert abs(inline_prob - sklearn_prob) < 1e-6, "Mismatch between inline and sklearn!"
print("OK — values match.")
