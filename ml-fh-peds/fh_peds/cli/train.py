from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from matplotlib import style
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from fh_peds.constants import CLASS_NAMES
from fh_peds.constants import X_COLUMNS
from fh_peds.constants import Y_COLUMN
from fh_peds.data import filter_by_metadata
from fh_peds.data import impute_and_scale_data
from fh_peds.data import read_data
from fh_peds.data import train_model_and_cv
from fh_peds.logging import setup_logging
from fh_peds.model_io import save_inference_samples
from fh_peds.model_io import save_metrics_json
from fh_peds.model_io import save_model_json
from fh_peds.model_io import save_predictions
from fh_peds.plotting import plot_precision_recall
from fh_peds.plotting import find_operating_point
from fh_peds.plotting import plot_specificity_sensitivity


def main(
    data_dir: Annotated[Path, typer.Option(help="Data sources directory.")],
    results_dir: Annotated[Path, typer.Option(help="Base results directory.")],
    recompute: Annotated[
        bool, typer.Option(help="Re-read source Excel files, ignoring cache.")
    ] = False,
    test_split: Annotated[
        float, typer.Option(help="Fraction of the SLO cohort held out for testing.")
    ] = 0.4,
    random_state: Annotated[
        int, typer.Option(help="Random seed for the train/test split.")
    ] = 3,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_dir = data_dir.resolve()
    results_dir = (results_dir / timestamp).resolve()

    data_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)

    style.use("seaborn-v0_8")
    log = setup_logging(results_dir)
    log.info(f"Data directory: {data_dir}")
    log.info(f"Results directory: {results_dir}")

    BASE_MODEL = LogisticRegression(random_state=0, max_iter=100, l1_ratio=0.0)
    PARAM_GRID = {
        "class_weight": ["balanced", None],
        "C": [
            1 / 128,
            1 / 64,
            1 / 32,
            1 / 16,
            1 / 8,
            1 / 4,
            1.0,
            4.0,
            16.0,
            64.0,
            128.0,
        ],
        "fit_intercept": [True, False],
    }

    data_slo = read_data(
        data_dir=data_dir, cohort="slo", version="final", recompute=recompute
    )
    data_por = read_data(
        data_dir=data_dir, cohort="por", version="final", recompute=recompute
    )
    data_raw = pd.concat([data_slo, data_por], axis=0)

    data, scaling_info = impute_and_scale_data(
        data_raw, mask_predicate=lambda row: row["cohort"] == "slo"
    )

    data["split"] = "test"
    indices_train_val, _ = train_test_split(
        data[data["cohort"] == "slo"].index,
        test_size=test_split,
        random_state=random_state,
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

    all_metrics: list[dict] = []

    for cohort, version, split in [
        ("slo", "final", "train_val"),
        ("slo", "final", "test"),
        ("por", "final", "test"),
    ]:
        log.info(f"\nSplit: {split!r}, cohort: {cohort!r}, version: {version!r}")
        data_subset = filter_by_metadata(
            data, cohort=cohort, version=version, split=split
        )

        y_true = data_subset[Y_COLUMN]
        y_pred = model.predict(data_subset[X_COLUMNS])
        y_prob = model.predict_proba(data_subset[X_COLUMNS])[:, 1]

        report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
        report_dict = classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
        )
        log.info(report_str)

        auc = roc_auc_score(y_true=y_true, y_score=y_prob)
        log.info(f"AUC: {auc}")

        entry: dict = {
            "split": split,
            "cohort": cohort,
            "version": version,
            "auc": auc,
            "accuracy": accuracy_score(y_true, y_pred),
            "support": int(len(y_true)),
        }
        # Attach per-class and aggregate rows from the report dict,
        # excluding the plain "accuracy" scalar sklearn adds.
        for key, value in report_dict.items():
            if key == "accuracy":
                continue
            entry[key] = value

        all_metrics.append(entry)

    metrics_path = save_metrics_json(all_metrics, results_dir)
    log.info(f"\nSaved: {metrics_path}")

    assert model.coef_.shape == (1, len(model.feature_names_in_))

    coef_df = pd.DataFrame(
        {"feature_name": model.feature_names_in_, "weight": model.coef_[0]}
    ).sort_values("weight", ascending=False)

    log.info(f"\nModel coefficients:\n{coef_df.to_string(index=False)}")

    metrics_df = plot_specificity_sensitivity(data, model, results_dir)
    log.info(f"\nSaved: {results_dir / 'specificity_sensitivity_curve.png'}")
    log.info(f"Saved: {results_dir / 'specificity_sensitivity.xlsx'}")

    operating_point = find_operating_point(metrics_df, target_specificity=0.98)
    log.info(
        f"\nOperating point @ {operating_point['target_specificity']:.0%} specificity "
        f"({operating_point['cohort']}): threshold = {operating_point['threshold']:.2f} "
        f"(sens = {operating_point['achieved_sensitivity']:.3f}, "
        f"spec = {operating_point['achieved_specificity']:.3f})"
    )

    model_json_path = save_model_json(
        model,
        scaling_info,
        results_dir,
        timestamp=timestamp,
        size_test_split=test_split,
        random_state=random_state,
        param_grid=PARAM_GRID,
        operating_point=operating_point,
    )
    log.info(f"\nSaved: {model_json_path}")

    xlsx_path = save_predictions(
        data_raw=data_raw, data=data, model=model, results_dir=results_dir
    )
    log.info(f"\nSaved: {xlsx_path}")

    plot_precision_recall(data, model, results_dir)
    log.info(f"Saved: {results_dir / 'precision_recall_curve.png'}")

    inference_samples_path = save_inference_samples(
        data_raw=data_raw, data=data, model=model, results_dir=results_dir
    )
    log.info(f"\nSaved: {inference_samples_path}")

    log.info(f"\nDone. All outputs written to: {results_dir}")
