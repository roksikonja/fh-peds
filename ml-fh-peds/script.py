"""
FH Pediatric Screening — training and evaluation pipeline.

Each run creates a timestamped results directory under results/ that contains:
  - stdout.log                      full console output
  - model.json                      model weights + metadata
  - specificity_sensitivity.xlsx
  - model_split_probability.xlsx
  - specificity_sensitivity_curve.png
  - precision_recall_curve.png
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inference import model_fn
from inference import preprocess_sample
from matplotlib import style
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from utils import BINARY_CATEGORICAL_COLUMNS
from utils import check_dicts_close
from utils import CLASS_NAMES
from utils import compute_metrics
from utils import DATA_INFO
from utils import filter_by_metadata
from utils import impute_and_scale_data
from utils import MULTI_CATEGORICAL_COLUMNS
from utils import read_data
from utils import train_model_and_cv
from utils import X_COLUMNS
from utils import X_COLUMNS_RAW
from utils import Y_COLUMN


# ---------------------------------------------------------------------------
# Configuration
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

style.use("seaborn-v0_8")

# ---------------------------------------------------------------------------
# Directory and logging setup
# ---------------------------------------------------------------------------

project_dir = Path(__file__).parent
data_dir = project_dir / "data"
data_dir.mkdir(exist_ok=True, parents=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = project_dir / "results" / timestamp
results_dir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(results_dir / "stdout.log")],
)
log = logging.getLogger(__name__)

log.info(f"Results directory: {results_dir}")

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

data_slo = read_data(
    data_dir=data_dir, cohort="slo", version="final", recompute=RECOMPUTE
)
data_por = read_data(
    data_dir=data_dir, cohort="por", version="final", recompute=RECOMPUTE
)

data_raw = pd.concat([data_slo, data_por], axis=0)

# ---------------------------------------------------------------------------
# 2. Impute and scale
# ---------------------------------------------------------------------------

data, scaling_info = impute_and_scale_data(
    data_raw, mask_predicate=lambda row: row["cohort"] == "slo"
)

# ---------------------------------------------------------------------------
# 3. Train / test split and model training
# ---------------------------------------------------------------------------

data["split"] = "test"
indices_train_val, _ = train_test_split(
    data[data["cohort"] == "slo"].index,
    test_size=SIZE_TEST_SPLIT,
    random_state=RANDOM_STATE,
    stratify=data[data["cohort"] == "slo"][Y_COLUMN],
)
data.loc[indices_train_val, "split"] = "train_val"

model, df_cv = train_model_and_cv(
    model=BASE_MODEL,
    param_grid=PARAM_GRID,
    X=data[data["split"] == "train_val"][X_COLUMNS],
    y=data[data["split"] == "train_val"][Y_COLUMN],
    cv=3,
    scoring="roc_auc",
)

log.info(f"\nBest model: {model}")
log.info(f"\nCV results (top 5):\n{df_cv.head()}")

# ---------------------------------------------------------------------------
# 4. Evaluation — classification report + AUC
# ---------------------------------------------------------------------------

log.info("\n" + "=" * 60)
log.info("Evaluation")
log.info("=" * 60)

for cohort, version, split in [
    ("slo", "final", "train_val"),
    ("slo", "final", "test"),
    ("por", "final", "test"),
]:
    log.info(f"\nSplit: {split!r}, cohort: {cohort!r}, version: {version!r}")
    data_subset = filter_by_metadata(data, cohort=cohort, version=version, split=split)

    y_true = data_subset[Y_COLUMN]
    y_pred = model.predict(data_subset[X_COLUMNS])
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    log.info(report)
    auc = roc_auc_score(
        y_true=data_subset[Y_COLUMN],
        y_score=model.predict_proba(data_subset[X_COLUMNS])[:, 1],
    )
    log.info(f"AUC: {auc}")

# ---------------------------------------------------------------------------
# 5. Specificity-Sensitivity curve
# ---------------------------------------------------------------------------

metrics = []
for t in np.linspace(0.01, 1.00, 100):
    recall_pos_slo, recall_neg_slo, precision_pos_slo = compute_metrics(
        filter_by_metadata(data, cohort="slo", version="final", split="test"),
        model=model,
        threshold=t,
    )
    recall_pos_por, recall_neg_por, precision_pos_por = compute_metrics(
        filter_by_metadata(data, cohort="por", version="final", split="test"),
        model=model,
        threshold=t,
    )
    metrics.append(
        {
            "threshold": t,
            "specificity (slo/test)": recall_neg_slo,
            "sensitivity/recall (slo/test)": recall_pos_slo,
            "precision (slo/test)": precision_pos_slo,
            "specificity (por/test)": recall_neg_por,
            "sensitivity/recall (por/test)": recall_pos_por,
            "precision (por/test)": precision_pos_por,
        }
    )

metrics_df = pd.DataFrame(metrics)

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Specificity-Sensitivity Curve")
ax.set_xlabel("Sensitivity")
ax.set_ylabel("Specificity")
ax.plot(
    metrics_df["sensitivity/recall (slo/test)"],
    metrics_df["specificity (slo/test)"],
    color="tab:green",
    label="SLO (Test split)",
)
ax.plot(
    metrics_df["sensitivity/recall (por/test)"],
    metrics_df["specificity (por/test)"],
    color="tab:red",
    label="POR (Test split)",
)
plt.legend(loc="lower left")
fig.savefig(
    results_dir / "specificity_sensitivity_curve.png", dpi=150, bbox_inches="tight"
)
plt.close(fig)

metrics_df.to_excel(results_dir / "specificity_sensitivity.xlsx", index=False)
log.info(f"\nSaved: {results_dir / 'specificity_sensitivity_curve.png'}")
log.info(f"Saved: {results_dir / 'specificity_sensitivity.xlsx'}")

# ---------------------------------------------------------------------------
# 6. Model coefficients
# ---------------------------------------------------------------------------

assert model.coef_.shape == (1, len(model.feature_names_in_))

coef_df = pd.DataFrame(
    {"feature_name": model.feature_names_in_, "weight": model.coef_[0]}
).sort_values("weight", ascending=False)

log.info(f"\nModel coefficients:\n{coef_df.to_string(index=False)}")

# ---------------------------------------------------------------------------
# 7. Save model weights and metadata as JSON
# ---------------------------------------------------------------------------

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
            "test_size": SIZE_TEST_SPLIT,
            "random_state": RANDOM_STATE,
            "stratify_by": Y_COLUMN,
        },
        "hyperparameter_search": {
            "method": "GridSearchCV",
            "cv_folds": 3,
            "scoring": "roc_auc",
            "param_grid": PARAM_GRID,
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
        for feature, weight in zip(
            model.feature_names_in_, model.coef_[0], strict=False
        )
    },
    "intercept": float(model.intercept_[0]),
}

model_json_path = results_dir / "model.json"
with open(model_json_path, "w") as f:
    json.dump(model_json, f, indent=4)
log.info(f"\nSaved: {model_json_path}")

# ---------------------------------------------------------------------------
# 8. Save predictions alongside raw data
# ---------------------------------------------------------------------------

assert (data_raw.index == data.index).all()

data_raw["split"] = data["split"]
data_raw["predicted_probability"] = pd.Series(
    model.predict_proba(data[X_COLUMNS])[:, 1], index=data.index
)
data_raw.to_excel(results_dir / "model_split_probability.xlsx")
log.info(f"\nSaved: {results_dir / 'model_split_probability.xlsx'}")

# ---------------------------------------------------------------------------
# 9. Precision-Recall curve
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title("Precision-Recall Curve")
for cohort, version, split, kwargs in [
    ("slo", "final", "train_val", {"color": "tab:green", "alpha": 0.5}),
    ("slo", "final", "test", {"color": "tab:green", "alpha": 1.0}),
    ("por", "final", "test", {"color": "tab:red", "alpha": 0.5}),
]:
    log.info(f"Precision-Recall — split: {split!r}, {cohort}/{version}")
    data_subset = filter_by_metadata(data, cohort=cohort, version=version, split=split)
    y_true = data_subset[Y_COLUMN]
    y_pred = model.predict_proba(data_subset[X_COLUMNS])[:, 1]
    PrecisionRecallDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        name=f"{split}, {cohort}/{version}",
        plot_chance_level=False,
        drop_intermediate=True,
        ax=ax,
        **kwargs,
    )

fig.savefig(results_dir / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(f"Saved: {results_dir / 'precision_recall_curve.png'}")

# ---------------------------------------------------------------------------
# 10. Validate inference module against sklearn model
# ---------------------------------------------------------------------------

log.info("\n" + "=" * 60)
log.info("Validating inference module")
log.info("=" * 60)

discrepancies = 0
for sample_id in data_raw.index:
    sample_raw = json.loads(data_raw.loc[sample_id, X_COLUMNS_RAW].to_json())
    sample = preprocess_sample(sample_raw, debug=False)
    expected_sample = data.loc[sample_id, X_COLUMNS].to_dict()
    for feature_name_bin in BINARY_CATEGORICAL_COLUMNS:
        expected_sample[feature_name_bin] = float(expected_sample[feature_name_bin])

    check_dicts_close(sample, expected_sample)

    probability = model_fn(sample)
    expected_probability = float(
        model.predict_proba(data.loc[[sample_id], X_COLUMNS])[0, 1]
    )

    if abs(probability - expected_probability) >= 0.7:
        log.warning(
            f"  MISMATCH sample_id={sample_id}: inference={probability:.4f}, sklearn={expected_probability:.4f}"
        )
        discrepancies += 1

if discrepancies == 0:
    log.info("  All samples match (tolerance 0.7). Inference module is consistent.")
else:
    log.warning(f"  {discrepancies} sample(s) exceeded tolerance.")

# ---------------------------------------------------------------------------
# 11. Export sample fixtures for testing
# ---------------------------------------------------------------------------

test_indices = data_raw.sample(5, random_state=10).index
data_raw.loc[test_indices, X_COLUMNS_RAW].to_json(
    data_dir / "sample.raw.json", orient="records", indent=4
)
data.loc[test_indices, X_COLUMNS].to_json(
    data_dir / "sample.json", orient="records", indent=4
)
log.info(
    f"\nSaved sample fixtures: {data_dir / 'sample.raw.json'}, {data_dir / 'sample.json'}"
)

log.info(f"\nDone. All outputs written to: {results_dir}")
