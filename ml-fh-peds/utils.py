from pathlib import Path
from typing import Any
from typing import Callable
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder


Cohort = Literal["slo", "por"]


DATA_INFO: dict[tuple[Cohort, str], dict[str, Any]] = {
    ("slo", "2.0"): {
        "file_name": "New score 2.0 - SLO -5Dec2024.xlsx",
        "sheet_name": "Sheet1",
        "column_map": {
            "AGE [year]": "age",
            "GENDER [0=Female, 1=Male]": "gender",
            "Family history of high cholesterol [0=negative; 1=first degree relative; 2=second degree relative; 3 = first and second degree relative]": "fh_high_cholesterol",
            "Family history of premature CAD [0=negative; 1=first degree relative; 2=second degree relative; 3 = first and second degree relative]": "fh_premature_cad",
            "Family history of PAD and CVI [0=negative; 1=first degree relative; 2=second degree relative; 3 = first and second degree relative]": "fh_pad_cvi",
            "Family history of Xantoma/Xantelasma [0=negative; 1=positive]": "fh_xant",
            "Family history of arcus senilis [0=negative; 1=positive]": "fh_acrus_senilis",
            "HDL cholesterol [mmol/L]": "hdl_cholesterol",
            "LDL cholesterol [mmol/L]": "ldl_cholesterol",
            "Total cholesterol [mmol/L]": "total_cholesterol",
            "TAG [mmol/L]": "tag",
            "Lp(a) [mg/L]": "lp_a",
            "BMI Z score": "bmi_z_score",
            # Label
            "Genetically confirmed FH [0= negative; 1= positive]": "gen_conf_fh",
        },
    },
    ("slo", "final"): {
        "file_name": "New score 2.0 - SLO -5Dec2024-final.xlsx",
        "sheet_name": "Sheet1",
        "column_map": {
            "AGE [year]": "age",
            "GENDER [0=Female, 1=Male]": "gender",
            "Family history of high cholesterol [0=negative; 1=first degree relative; 2=second degree relative; 3 = first and second degree relative]": "fh_high_cholesterol",
            "Family history of premature CAD [0=negative; 1=first degree relative; 2=second degree relative; 3 = first and second degree relative]": "fh_premature_cad",
            "Family history of PAD and CVI [0=negative; 1=first degree relative; 2=second degree relative; 3 = first and second degree relative]": "fh_pad_cvi",
            "Family history of Xantoma/Xantelasma [0=negative; 1=positive]": "fh_xant",
            "Family history of arcus senilis [0=negative; 1=positive]": "fh_acrus_senilis",
            "HDL cholesterol [mmol/L]": "hdl_cholesterol",
            "LDL cholesterol [mmol/L]": "ldl_cholesterol",
            "Total cholesterol [mmol/L]": "total_cholesterol",
            "TAG [mmol/L]": "tag",
            "Lp(a) [mg/L]": "lp_a",
            "BMI Z score": "bmi_z_score",
            # Label
            "Genetically confirmed FH [0= negative; 1= positive]": "gen_conf_fh",
        },
    },
    ("por", "2.0"): {
        "file_name": "Portuguese registry 2.0.xlsx",
        "sheet_name": "Sheet2",
        "column_map": {
            "Age [year]": "age",
            "Gender (F=0, M=1)": "gender",
            "Family history of high cholesterol [0=negative; 1=first degree relative; 2=second degree relative; 3 = first and second degree relative]": "fh_high_cholesterol",
            "Family history of premature CAD [0=negative; 1=first degree relative; 2=second degree relative; 3 = first and second degree relative]": "fh_premature_cad",
            "Family history of PAD and CVI [0=negative; 1=first degree relative; 2=second degree relative; 3 = first and second degree relative]": "fh_pad_cvi",
            "Family history of Xantoma/Xantelasma [0=negative; 1=positive]": "fh_xant",
            "Family history of arcus senilis [0=negative; 1=positive]": "fh_acrus_senilis",
            "HDL cholesterol [mmol/L]": "hdl_cholesterol",
            "LDL cholesterol [mmol/L]": "ldl_cholesterol",
            "TAG [mmol/L]": "tag",
            "BMI Z Score": "bmi_z_score",
            # Label
            "Genetically confirmed FH [0= negative; 1= positive]": "gen_conf_fh",
        },
    },
    ("por", "3.0"): {
        "file_name": "Portuguese registry 3.0 + Lp(a).xlsx",
        "sheet_name": "Sheet2",
        "column_map": {
            "Age at diagnosis": "age",
            "Gender (F=0, M=1)": "gender",
            "High cholesterol in family (0-no, 1-1st degree; 2-second degree; 3-both)": "fh_high_cholesterol",
            "History of premature heart disease: AMI, CABG, PCI men aged <55 years, women aged <60 years (0 - no, 1-1st degree, 2-second degree, 3-both)": "fh_premature_cad",
            "Vascular disease": "fh_pad_cvi",
            "Tedious\xa0xanthoma \n(0-no, 1-yes)": "fh_xant",
            "Arcus cornealis \n(0-no, 1-yes)": "fh_acrus_senilis",
            "HDL": "hdl_cholesterol",
            "LDL": "ldl_cholesterol",
            "TAG": "tag",
            "LPA": "lp_a",
            "BMI Z Score": "bmi_z_score",
            # Label
            "FH (0-negative, 1-positive)": "gen_conf_fh",
            # Unknown
            "DER": "DER",
        },
    },
    ("por", "final"): {
        "file_name": "Portuguese registry 3.1-final.xlsx",
        "sheet_name": "Sheet2",
        "column_map": {
            "AGE [year]": "age",
            "GENDER [0=Female, 1=Male]": "gender",
            "Family history of high cholesterol [0=negative; 1=first degree relative; 2=second degree relative; 3 = first and second degree relative]": "fh_high_cholesterol",
            "Family history of premature CAD [0=negative; 1=first degree relative; 2=second degree relative; 3 = first and second degree relative]": "fh_premature_cad",
            "Family history of PAD and CVI [0=negative; 1=first degree relative; 2=second degree relative; 3 = first and second degree relative]": "fh_pad_cvi",
            "Family history of Xantoma/Xantelasma [0=negative; 1=positive]": "fh_xant",
            "Family history of arcus senilis [0=negative; 1=positive]": "fh_acrus_senilis",
            "HDL cholesterol [mmol/L]": "hdl_cholesterol",
            "LDL cholesterol [mmol/L]": "ldl_cholesterol",
            "Total cholesterol [mmol/L]": "total_cholesterol",
            "TAG [mmol/L]": "tag",
            "Lp(a) [mg/L]": "lp_a",
            "BMI Z score": "bmi_z_score",
            # Label
            "Genetically confirmed FH [0= negative; 1= positive]": "gen_conf_fh",
        },
    },
}

X_COLUMN_ORDER = [
    "age",
    "gender",
    "fh_high_cholesterol",
    "fh_premature_cad",
    "fh_pad_cvi",
    "fh_xant",
    "fh_acrus_senilis",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "total_cholesterol",
    "tag",
    "bmi_z_score",
    "lp_a",
]
Y_COLUMN = "gen_conf_fh"

COLUMN_ORDER = X_COLUMN_ORDER + [Y_COLUMN]

COLUMN_DTYPES = {
    "age": np.dtype("float64"),
    "gender": np.dtype("int64"),
    "fh_high_cholesterol": np.dtype("int64"),
    "fh_premature_cad": np.dtype("int64"),
    "fh_pad_cvi": np.dtype("int64"),
    "fh_xant": np.dtype("int64"),
    "fh_acrus_senilis": np.dtype("int64"),
    "hdl_cholesterol": np.dtype("float64"),
    "ldl_cholesterol": np.dtype("float64"),
    "total_cholesterol": np.dtype("float64"),
    "tag": np.dtype("float64"),
    "bmi_z_score": np.dtype("float64"),
    "lp_a": np.dtype("float64"),
    "gen_conf_fh": np.dtype("int64"),
}

BINARY_CATEGORICAL_COLUMNS = [
    "gender",  # 2
    "fh_xant",  # 2
    "fh_acrus_senilis",  # 2
    # "gen_conf_fh",  # 2
]
MULTI_CATEGORICAL_COLUMNS = [
    "fh_high_cholesterol",  # 4
    "fh_premature_cad",  # 4
    "fh_pad_cvi",  # 4
]

CLASS_NAMES = ["negative", "positive"]


X_COLUMNS = [
    "age",
    "gender",
    # "fh_high_cholesterol_0",
    "fh_high_cholesterol_1",
    "fh_high_cholesterol_2",
    "fh_high_cholesterol_3",
    # "fh_premature_cad_0",
    "fh_premature_cad_1",
    "fh_premature_cad_2",
    "fh_premature_cad_3",
    # "fh_pad_cvi_0",
    "fh_pad_cvi_1",
    "fh_pad_cvi_2",
    "fh_pad_cvi_3",
    "fh_xant",
    "fh_acrus_senilis",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "total_cholesterol",
    "tag",
    "bmi_z_score",
    "lp_a",
]


def _read_data(
    sheet_path: Path, *, sheet_name: str, column_map: dict[str, str]
) -> pd.DataFrame:
    # Read Excel file.
    df_raw = pd.read_excel(sheet_path, sheet_name=sheet_name, index_col=0)
    print(
        f"  - Loaded raw data from '{sheet_path}' with {len(df_raw.columns)} "
        f"columns ..."
    )

    # Standardize column names.
    df = df_raw.rename(columns=column_map)

    s = "\n"
    assert set(df.columns) == set(
        column_map.values()
    ), f"Columns do not match the map.\n{s.join(map(repr, df_raw.columns))}"

    columns_redundant = list(set(df.columns) - set(COLUMN_ORDER))
    if len(columns_redundant) > 0:
        print(
            f"  - Removed redundant columns: {', '.join(map(repr, columns_redundant))}"
        )

    columns_missing = list(set(COLUMN_ORDER) - set(df.columns))
    if len(columns_missing) > 0:
        print(f"  - Added missing columns: {', '.join(map(repr, columns_missing))}")

    # Standardize column ordering and ensure that all columns are contained.
    df = df.reindex(columns=COLUMN_ORDER)
    assert list(df.columns) == COLUMN_ORDER

    # Standardize column dtypes.
    assert not df[Y_COLUMN].isna().any(), "Labels contain missing values."
    assert (
        df[Y_COLUMN] == df[Y_COLUMN].astype(int)
    ).all(), "Labels have non-discrete/non-integer values."
    df = df.astype(COLUMN_DTYPES)

    # Check categorical columns.
    for column in BINARY_CATEGORICAL_COLUMNS:
        assert df[column].dtype == np.int64, f"Column: '{column}', {df[column].dtype}"
        assert (0 <= df[column]).all() & (df[column] < 2).all(), f"Column: '{column}'"

    for column in MULTI_CATEGORICAL_COLUMNS:
        assert df[column].dtype == np.int64, f"Column: '{column}', {df[column].dtype}"
        assert (0 <= df[column]).all() & (df[column] < 4).all(), f"Column: '{column}'"

    print("  - Standardized column names, ordering and data types ...\n")
    df.info()
    print("\n")
    return df


def read_data(
    *, data_dir: Path, cohort: Cohort, version: str, recompute: bool = False
) -> pd.DataFrame:
    print(f"Data: Cohort '{cohort}' and version '{version}'")

    cache_path = data_dir / f"cache_cohort_{cohort}_{version}.pkl"
    if not recompute and cache_path.exists():
        print(f"- Reading cached data from '{cache_path}' ...")
        df = pd.read_pickle(cache_path)
        df.info()
        print("\n")
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
    print("Imputing and feature scaling ...")
    encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")

    info = {}
    data = pd.DataFrame(index=data_raw.index)
    for column in data_raw.columns:
        if column in BINARY_CATEGORICAL_COLUMNS:
            data[column] = data_raw[column].copy()
            print(f"  - Column '{column}' is binary")
        elif column in MULTI_CATEGORICAL_COLUMNS:
            data_multi_column = encoder.fit_transform(data_raw[[column]])
            for column_binary in data_multi_column.columns:
                if column_binary.endswith("_0"):
                    continue
                    
                data[column_binary] = data_multi_column[column_binary]

            print(
                f"  - Column '{column}' is multi-categorical: "
                f"{', '.join(filter(lambda c: not c.endswith('_0'), data_multi_column.columns))}"
            )
        elif column in X_COLUMN_ORDER:
            mask = data_raw.apply(mask_predicate, axis=1)

            mean = data_raw[mask][column].mean(skipna=True)
            std = data_raw[mask][column].std(skipna=True)

            data[column] = (data_raw[column].copy().fillna(value=mean) - mean) / std
            # data[column] = data_raw[column].copy().fillna(value=mean)
            info[column] = {"mean": mean, "std": std}
            print(f"  - Column '{column}' normalized: {mean:.2f} + {std:.2f}")
        else:
            data[column] = data_raw[column]
            print(f"  - Column '{column}' is metadata")

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
    cv = GridSearchCV(
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

    df_cv = pd.DataFrame(cv.cv_results_).sort_values(by="rank_test_score")
    columns = [
        column
        for column in df_cv.columns
        if column.endswith("_time")
        or column.startswith("params")
        or column.startswith("split")
    ]
    df_cv = df_cv.drop(columns=columns)

    return cv.best_estimator_, df_cv


import numpy as np


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
    recall_pos = recall_score(
        y_true=data_subset[Y_COLUMN],
        y_pred=(model.predict_proba(data_subset[X_COLUMNS])[:, 1] > threshold).astype(
            int
        ),
        pos_label=1,
    )
    recall_neg = recall_score(
        y_true=data_subset[Y_COLUMN],
        y_pred=(model.predict_proba(data_subset[X_COLUMNS])[:, 1] > threshold).astype(
            int
        ),
        pos_label=0,
    )
    precision_pos = precision_score(
        y_true=data_subset[Y_COLUMN],
        y_pred=(model.predict_proba(data_subset[X_COLUMNS])[:, 1] > threshold).astype(
            int
        ),
        pos_label=1,
        zero_division=np.nan,
    )
    return recall_pos, recall_neg, precision_pos
