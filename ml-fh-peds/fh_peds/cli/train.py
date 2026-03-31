import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from fh_peds.logging import setup_run
from fh_peds.model_io import save_inference_samples
from fh_peds.model_io import save_model_json
from fh_peds.model_io import save_predictions
from fh_peds.plotting import plot_precision_recall
from fh_peds.plotting import plot_specificity_sensitivity
from fh_peds.constants import CLASS_NAMES
from fh_peds.constants import X_COLUMNS
from fh_peds.constants import Y_COLUMN
from fh_peds.data import filter_by_metadata
from fh_peds.data import impute_and_scale_data
from fh_peds.data import read_data
from fh_peds.data import train_model_and_cv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fh-peds train",
        description="Train and evaluate the FH pediatric screening model.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the source Excel files and cache pickles. "
            "Defaults to <project_dir>/data."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Base directory under which a timestamped results sub-directory is "
            "created. Defaults to <project_dir>."
        ),
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        default=False,
        help="Force re-reading source Excel files even when a cache pickle exists.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.4,
        dest="size_test_split",
        help="Fraction of the SLO cohort to reserve for the test split (default: 0.4).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=3,
        help="Random seed for the train/test split (default: 3).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent.parent
    data_dir = args.data_dir or project_dir / "data"
    base_dir = args.results_dir or project_dir

    data_dir.mkdir(exist_ok=True, parents=True)
    results_dir, log = setup_run(base_dir)

    RANDOM_STATE: int = args.random_state
    SIZE_TEST_SPLIT: float = args.size_test_split

    BASE_MODEL = LogisticRegression(random_state=0, penalty="l2", max_iter=100)
    PARAM_GRID = {
        "class_weight": ["balanced", None],
        "C": [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0, 4.0, 16.0, 64.0, 128.0],
        "fit_intercept": [True, False],
    }

    data_slo = read_data(
        data_dir=data_dir, cohort="slo", version="final", recompute=args.recompute
    )
    data_por = read_data(
        data_dir=data_dir, cohort="por", version="final", recompute=args.recompute
    )
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

    assert model.coef_.shape == (1, len(model.feature_names_in_))

    coef_df = pd.DataFrame(
        {
            "feature_name": model.feature_names_in_,
            "weight": model.coef_[0],
        }
    ).sort_values("weight", ascending=False)

    log.info(f"\nModel coefficients:\n{coef_df.to_string(index=False)}")

    plot_specificity_sensitivity(data, model, results_dir)
    log.info(f"\nSaved: {results_dir / 'specificity_sensitivity_curve.png'}")
    log.info(f"Saved: {results_dir / 'specificity_sensitivity.xlsx'}")

    timestamp = results_dir.name

    model_json_path = save_model_json(
        model,
        scaling_info,
        results_dir,
        timestamp=timestamp,
        size_test_split=SIZE_TEST_SPLIT,
        random_state=RANDOM_STATE,
        param_grid=PARAM_GRID,
    )
    log.info(f"\nSaved: {model_json_path}")

    xlsx_path = save_predictions(data_raw, data, model, results_dir)
    log.info(f"\nSaved: {xlsx_path}")

    plot_precision_recall(data, model, results_dir)
    log.info(f"Saved: {results_dir / 'precision_recall_curve.png'}")

    inference_samples_path = save_inference_samples(data_raw, data, model, results_dir)
    log.info(f"\nSaved: {inference_samples_path}")

    log.info(f"\nDone. All outputs written to: {results_dir}")
