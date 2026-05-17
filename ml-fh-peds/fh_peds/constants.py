from typing import Any
from typing import Literal

import numpy as np

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
            "FH (0-negative, 1-positive)": "gen_conf_fh",
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
            "Genetically confirmed FH [0= negative; 1= positive]": "gen_conf_fh",
        },
    },
}

Y_COLUMN = "gen_conf_fh"

X_COLUMNS_RAW = [
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
COLUMNS_RAW = X_COLUMNS_RAW + [Y_COLUMN]

COLUMN_DTYPES_RAW = {
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
    "gender",
    "fh_xant",
    "fh_acrus_senilis",
]
MULTI_CATEGORICAL_COLUMNS = [
    "fh_high_cholesterol",
    "fh_premature_cad",
    "fh_pad_cvi",
]

CLASS_NAMES = ["negative", "positive"]

X_COLUMNS = [
    "age",
    "gender",
    "fh_high_cholesterol_1",
    "fh_high_cholesterol_2",
    "fh_high_cholesterol_3",
    "fh_premature_cad_1",
    "fh_premature_cad_2",
    "fh_premature_cad_3",
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
