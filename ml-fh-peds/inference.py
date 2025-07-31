import json
import math


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

BINARY_CATEGORICAL_COLUMNS = ["gender", "fh_xant", "fh_acrus_senilis"]
MULTI_CATEGORICAL_COLUMNS = ["fh_high_cholesterol", "fh_premature_cad", "fh_pad_cvi"]

# Preprocessing
PREPROCESSING_INFO = {
    "age": {"mean": 7.314633123689728, "std": 2.607213078684607},
    "hdl_cholesterol": {"mean": 1.5362264150943397, "std": 0.37043406815321883},
    "ldl_cholesterol": {"mean": 3.78845283018868, "std": 1.1593288141513893},
    "total_cholesterol": {"mean": 5.767471698113208, "std": 1.1603039079279096},
    "tag": {"mean": 1.0961132075471698, "std": 0.7718188209096246},
    "bmi_z_score": {"mean": 0.28694020169346707, "std": 1.3078173489609493},
    "lp_a": {"mean": 310.6692307692308, "std": 332.1171687025873},
}


INTERCEPT = -8.330929160368113
WEIGHTS = {
    "age": 0.00667784927430266,
    "gender": 0.32143804346854776,
    "fh_high_cholesterol_1": 0.34873544396742856,
    "fh_high_cholesterol_2": 0.02078297866340733,
    "fh_high_cholesterol_3": 0.643902650173691,
    "fh_premature_cad_1": 0.05226457867177578,
    "fh_premature_cad_2": 0.12812175064921372,
    "fh_premature_cad_3": 0.03427669157864839,
    "fh_pad_cvi_1": 0.02237060945544122,
    "fh_pad_cvi_2": -0.364993539338868,
    "fh_pad_cvi_3": 0.00978393164164006,
    "fh_xant": -0.21087954347939916,
    "fh_acrus_senilis": 0.2413157398789852,
    "hdl_cholesterol": -0.8760712481075249,
    "ldl_cholesterol": 1.3552895085240824,
    "total_cholesterol": 1.1470281822853394,
    "tag": -0.5361077381018685,
    "bmi_z_score": -0.05708627050403221,
    "lp_a": -0.4028915834363285,
}


def preprocess_sample(raw_sample: dict, debug: bool = True) -> dict:
    assert set(raw_sample) == set(X_COLUMNS_RAW)

    if debug:
        print(json.dumps(raw_sample, indent=4))

    sample = {}
    for feature_name, feature_value in raw_sample.items():
        if feature_name in BINARY_CATEGORICAL_COLUMNS:
            assert isinstance(feature_value, (int, type(None)))
            if feature_value is None:
                feature_value = 0

            sample[feature_name] = float(feature_value)
        elif feature_name in MULTI_CATEGORICAL_COLUMNS:
            assert isinstance(feature_value, (int, type(None)))
            if feature_value is None:
                feature_value = 0

            for value in [1, 2, 3]:
                sample[f"{feature_name}_{value}"] = (
                    1.0 if feature_value == value else 0.0
                )
        elif feature_name in PREPROCESSING_INFO:
            assert isinstance(feature_value, (float, type(None)))

            mean = PREPROCESSING_INFO[feature_name]["mean"]
            std = PREPROCESSING_INFO[feature_name]["std"]

            if feature_value is None:
                feature_value = mean

            sample[feature_name] = float((feature_value - mean) / std)
        else:
            assert False, feature_name

    if debug:
        print(json.dumps(sample, indent=4))

    assert len(set(sample) - set(X_COLUMNS)) == 0, set(sample) - set(X_COLUMNS)
    assert len(set(X_COLUMNS) - set(sample)) == 0, set(X_COLUMNS) - set(sample)

    return sample


def model_fn(sample: dict) -> float:
    assert set(X_COLUMNS) == set(sample)
    assert set(WEIGHTS) == set(sample)

    def _sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    # Logistic regression
    weighted_sum = INTERCEPT
    for feature_name, weight_value in WEIGHTS.items():
        feature_value = sample[feature_name]
        weighted_sum += weight_value * feature_value

    return _sigmoid(weighted_sum)
