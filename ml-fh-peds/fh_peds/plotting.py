"""
Plotting utilities for fh-peds training runs.

Functions
---------
plot_specificity_sensitivity
    Plot specificity vs. sensitivity curves for multiple cohort/split pairs.

plot_precision_recall
    Plot precision-recall curves for multiple cohort/split pairs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay

from fh_peds.utils import compute_metrics
from fh_peds.utils import filter_by_metadata

style.use("seaborn-v0_8")


def plot_specificity_sensitivity(
    data: pd.DataFrame,
    model: LogisticRegression,
    output_path: Path,
    *,
    n_thresholds: int = 100,
) -> pd.DataFrame:
    """Compute and plot the specificity-sensitivity curve for the SLO test and
    POR test splits.

    The curve is computed by sweeping the classification threshold from 0.01 to
    1.00 and recording specificity (recall of the negative class) and
    sensitivity (recall of the positive class) at each step.

    Parameters
    ----------
    data:
        Full preprocessed dataset with ``cohort``, ``version``, and ``split``
        metadata columns.
    model:
        Trained sklearn estimator with a ``predict_proba`` method.
    output_path:
        Directory into which ``specificity_sensitivity_curve.png`` and
        ``specificity_sensitivity.xlsx`` are written.
    n_thresholds:
        Number of evenly-spaced threshold values between 0.01 and 1.00.

    Returns
    -------
    metrics_df:
        DataFrame with one row per threshold containing specificity,
        sensitivity, and precision for both cohorts.
    """
    metrics = []
    for t in np.linspace(0.01, 1.00, n_thresholds):
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

    png_path = output_path / "specificity_sensitivity_curve.png"
    xlsx_path = output_path / "specificity_sensitivity.xlsx"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    metrics_df.to_excel(xlsx_path, index=False)

    return metrics_df


def plot_precision_recall(
    data: pd.DataFrame,
    model: LogisticRegression,
    output_path: Path,
) -> None:
    """Plot precision-recall curves for the SLO train/val, SLO test, and POR
    test splits and save the figure.

    Parameters
    ----------
    data:
        Full preprocessed dataset with ``cohort``, ``version``, and ``split``
        metadata columns.
    model:
        Trained sklearn estimator with a ``predict_proba`` method.
    output_path:
        Directory into which ``precision_recall_curve.png`` is written.
    """
    from fh_peds.utils import X_COLUMNS
    from fh_peds.utils import Y_COLUMN

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("Precision-Recall Curve")

    for cohort, version, split, kwargs in [
        ("slo", "final", "train_val", {"color": "tab:green", "alpha": 0.5}),
        ("slo", "final", "test", {"color": "tab:green", "alpha": 1.0}),
        ("por", "final", "test", {"color": "tab:red", "alpha": 0.5}),
    ]:
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

    png_path = output_path / "precision_recall_curve.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
