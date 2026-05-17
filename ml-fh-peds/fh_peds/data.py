import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

from fh_peds.constants import BINARY_CATEGORICAL_COLUMNS
from fh_peds.constants import COLUMN_DTYPES_RAW
from fh_peds.constants import COLUMNS_RAW
from fh_peds.constants import DATA_INFO
from fh_peds.constants import MULTI_CATEGORICAL_COLUMNS
from fh_peds.constants import X_COLUMNS
from fh_peds.constants import X_COLUMNS_RAW
from fh_peds.constants import Y_COLUMN
from fh_peds.constants import Cohort

log = logging.getLogger("fh_peds")


def _read_data(sheet_path: Path, *, sheet_name: str, column_map: dict[str, str]) -> pd.DataFrame:
    df_raw = pd.read_excel(sheet_path, sheet_name=sheet_name, index_col=0)
    log.info(f"  - Loaded raw data from '{sheet_path}' with {len(df_raw.columns)} columns ...")

    df = df_raw.rename(columns=column_map)

    s = "\n"
    assert set(df.columns) == set(
        column_map.values()
    ), f"Columns do not match the map.\n{s.join(map(repr, df_raw.columns))}"

    columns_redundant = list(set(df.columns) - set(COLUMNS_RAW))
    if len(columns_redundant) > 0:
        log.info(f"  - Removed redundant columns: {', '.join(map(repr, columns_redundant))}")

    columns_missing = list(set(COLUMNS_RAW) - set(df.columns))
    if len(columns_missing) > 0:
        log.info(f"  - Added missing columns: {', '.join(map(repr, columns_missing))}")

    df = df.reindex(columns=COLUMNS_RAW)
    assert list(df.columns) == COLUMNS_RAW

    assert not df[Y_COLUMN].isna().any(), "Labels contain missing values."
    assert (
        df[Y_COLUMN] == df[Y_COLUMN].astype(int)
    ).all(), "Labels have non-discrete/non-integer values."
    df = df.astype(COLUMN_DTYPES_RAW)

    for column in BINARY_CATEGORICAL_COLUMNS:
        assert df[column].dtype == np.int64, f"Column: '{column}', {df[column].dtype}"
        assert (df[column] >= 0).all() & (df[column] < 2).all(), f"Column: '{column}'"

    for column in MULTI_CATEGORICAL_COLUMNS:
        assert df[column].dtype == np.int64, f"Column: '{column}', {df[column].dtype}"
        assert (df[column] >= 0).all() & (df[column] < 4).all(), f"Column: '{column}'"

    log.info("  - Standardized column names, ordering and data types ...")
    return df


def read_data(
    *, data_dir: Path, cohort: Cohort, version: str, recompute: bool = False
) -> pd.DataFrame:
    log.info(f"Data: Cohort '{cohort}' and version '{version}'")

    cache_path = data_dir / f"cache_cohort_{cohort}_{version}.pkl"
    if not recompute and cache_path.exists():
        log.info(f"- Reading cached data from '{cache_path}' ...")
        df = pd.read_pickle(cache_path)
        return df

    data_info = DATA_INFO[(cohort, version)]
    df = _read_data(
        sheet_path=data_dir / data_info["file_name"],
        sheet_name=data_info["sheet_name"],
        column_map=data_info["column_map"],
    )
    df["cohort"] = cohort
    df["version"] = version

    df.to_pickle(cache_path)
    return df


def impute_and_scale_data(
    data_raw: pd.DataFrame, mask_predicate: Callable[[pd.Series], bool]
) -> tuple[pd.DataFrame, dict]:
    log.info("Imputing and feature scaling ...")
    encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")

    info = {}
    data = pd.DataFrame(index=data_raw.index)
    for column in data_raw.columns:
        if column in BINARY_CATEGORICAL_COLUMNS:
            data[column] = data_raw[column].copy()
            log.info(f"  - Column '{column}' is binary")
        elif column in MULTI_CATEGORICAL_COLUMNS:
            data_multi_column = encoder.fit_transform(data_raw[[column]])
            for column_binary in data_multi_column.columns:
                if column_binary.endswith("_0"):
                    continue
                data[column_binary] = data_multi_column[column_binary]

            log.info(
                f"  - Column '{column}' is multi-categorical: "
                f"{', '.join(filter(lambda c: not c.endswith('_0'), data_multi_column.columns))}"
            )
        elif column in X_COLUMNS_RAW:
            mask = data_raw.apply(mask_predicate, axis=1)

            mean = data_raw[mask][column].mean(skipna=True)
            std = data_raw[mask][column].std(skipna=True)

            data[column] = (data_raw[column].copy().fillna(value=mean) - mean) / std
            info[column] = {"mean": mean, "std": std}
            log.info(f"  - Column '{column}' normalized: {mean:.2f} ± {std:.2f}")
        else:
            data[column] = data_raw[column]
            log.info(f"  - Column '{column}' is metadata")

    assert data.isna().sum().sum() == 0
    assert list(data.columns) == X_COLUMNS + ["gen_conf_fh", "cohort", "version"]

    return data, info


def train_model_and_cv(
    model: LogisticRegression,
    param_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv: int = 3,
    scoring: str = "roc_auc",
) -> tuple[LogisticRegression, pd.DataFrame]:
    cv_search = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=True,
        verbose=0,
        return_train_score=True,
        error_score="raise",
        n_jobs=-1,
    ).fit(X, y)

    df_cv = pd.DataFrame(cv_search.cv_results_).sort_values(by="rank_test_score")
    columns = [
        column
        for column in df_cv.columns
        if column.endswith("_time") or column.startswith("params") or column.startswith("split")
    ]
    df_cv = df_cv.drop(columns=columns)

    return cv_search.best_estimator_, df_cv


def filter_by_metadata(
    df: pd.DataFrame, *, cohort: str | None, version: str | None, split: str | None
) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    if cohort is not None:
        mask = mask & (df["cohort"] == cohort)
    if version is not None:
        mask = mask & (df["version"] == version)
    if split is not None:
        mask = mask & (df["split"] == split)
    return df[mask]


def compute_metrics(
    data_subset: pd.DataFrame, *, model: LogisticRegression, threshold: float
) -> tuple[float, float, float]:
    y_pred = (model.predict_proba(data_subset[X_COLUMNS])[:, 1] > threshold).astype(int)
    recall_pos = recall_score(y_true=data_subset[Y_COLUMN], y_pred=y_pred, pos_label=1)
    recall_neg = recall_score(y_true=data_subset[Y_COLUMN], y_pred=y_pred, pos_label=0)
    precision_pos = precision_score(
        y_true=data_subset[Y_COLUMN], y_pred=y_pred, pos_label=1, zero_division=np.nan
    )
    return recall_pos, recall_neg, precision_pos
