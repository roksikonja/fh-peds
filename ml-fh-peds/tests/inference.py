"""
Pure-Python inference for the FH-peds logistic regression model.

Dependencies: pydantic>=2

Usage
-----
Point the MODEL_DIR environment variable at a directory containing model.json,
then import and call ``load_model`` / ``predict_probability``:

    from tests.inference import load_model, predict_probability

    model = load_model()          # reads $MODEL_DIR/model.json
    prob = predict_probability(model, {"age": 7.0, "gender": 1, ...})

If MODEL_DIR is not set, ``<repo_root>/data/model.json`` is used as a
convenience default.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Annotated
from typing import Literal

from pydantic import BaseModel
from pydantic import BeforeValidator
from pydantic import confloat
from pydantic import TypeAdapter


def _to_int(v: object) -> int:
    """Coerce numeric values (e.g. 0.0, 1.0) to int before Literal validation."""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v  # type: ignore[return-value]


_IntLike = Annotated[int, BeforeValidator(_to_int)]


# ---------------------------------------------------------------------------
# Pydantic schema for raw input validation
# ---------------------------------------------------------------------------


class RawSample(BaseModel):
    age: confloat(ge=0.0, le=18.0)
    gender: Annotated[Literal[0, 1], BeforeValidator(_to_int)]
    fh_high_cholesterol: Annotated[Literal[0, 1, 2, 3], BeforeValidator(_to_int)]
    fh_premature_cad: Annotated[Literal[0, 1, 2, 3], BeforeValidator(_to_int)]
    fh_pad_cvi: Annotated[Literal[0, 1, 2, 3], BeforeValidator(_to_int)]
    fh_xant: Annotated[Literal[0, 1], BeforeValidator(_to_int)]
    fh_acrus_senilis: Annotated[Literal[0, 1], BeforeValidator(_to_int)]
    hdl_cholesterol: confloat(ge=0.0)
    ldl_cholesterol: confloat(ge=0.0)
    total_cholesterol: confloat(ge=0.0)
    tag: confloat(ge=0.0)
    lp_a: confloat(ge=0.0) | None
    bmi_z_score: confloat(ge=-50.0, le=50.0) | None


# ---------------------------------------------------------------------------
# Model schema (subset of model.json that inference needs)
# ---------------------------------------------------------------------------


class PreprocessingStats(BaseModel):
    mean: float
    std: float


class ModelFeatures(BaseModel):
    input_fields: list[str]
    model_fields: list[str]
    binary_categorical: list[str]
    multi_categorical: list[str]
    continuous_normalized: list[str]


class ModelJSON(BaseModel):
    features: ModelFeatures
    preprocessing: dict[str, PreprocessingStats]
    weights: dict[str, float]
    intercept: float


# ---------------------------------------------------------------------------
# Load model.json
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent


def load_model(model_dir: str | Path | None = None) -> ModelJSON:
    """Load and validate ``model.json``.

    Resolution order for the directory:
    1. ``model_dir`` argument (if provided)
    2. ``MODEL_DIR`` environment variable
    3. ``<repo_root>/data/`` as a convenience default

    Raises ``FileNotFoundError`` if no ``model.json`` is found.
    """
    if model_dir is not None:
        model_path = Path(model_dir) / "model.json"
    else:
        env_dir = os.environ.get("MODEL_DIR")
        if env_dir:
            model_path = Path(env_dir) / "model.json"
        else:
            model_path = _HERE.parent / "data" / "model.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"model.json not found at {model_path}. "
            "Set the MODEL_DIR environment variable to point at a results directory."
        )

    with open(model_path) as fh:
        raw = json.load(fh)

    return TypeAdapter(ModelJSON).validate_python(raw)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def preprocess_sample(model: ModelJSON, raw: dict) -> dict[str, float]:
    """Validate and preprocess a raw input dict into model feature space."""
    TypeAdapter(RawSample).validate_python(raw)

    sample: dict[str, float] = {}

    for field in model.features.input_fields:
        value = raw[field]

        if field in model.features.binary_categorical:
            sample[field] = float(value if value is not None else 0)

        elif field in model.features.multi_categorical:
            value = value if value is not None else 0
            for level in [1, 2, 3]:
                sample[f"{field}_{level}"] = 1.0 if value == level else 0.0

        elif field in model.features.continuous_normalized:
            stats = model.preprocessing[field]
            if value is None:
                value = stats.mean
            sample[field] = (float(value) - stats.mean) / stats.std

        else:
            raise ValueError(f"Unknown field: {field!r}")

    return sample


def predict_probability(model: ModelJSON, raw: dict) -> float:
    """Return the model's FH probability for a raw input dict."""
    sample = preprocess_sample(model, raw)

    weighted_sum = model.intercept
    for feature, weight in model.weights.items():
        weighted_sum += weight * sample[feature]

    return 1.0 / (1.0 + math.exp(-weighted_sum))
