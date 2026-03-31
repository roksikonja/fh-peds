"""
Pytest tests for the FH-peds inference module.

Dependencies: pydantic>=2, pytest

Usage
-----
Point the MODEL_DIR environment variable at a directory containing model.json
and test_fixtures.json, then run:

    MODEL_DIR=/path/to/results/2026-01-01_12-00-00 pytest tests/ -v

If MODEL_DIR is not set, ``<repo_root>/data/`` is tried as a convenience
default.

test_fixtures.json format (produced by training.py)
----------------------------------------------------
A JSON array of objects, each with:
  "input"       — raw input dict (same schema as RawSample)
  "probability" — expected output probability (float, from sklearn)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.inference import ModelJSON
from tests.inference import load_model
from tests.inference import predict_probability

_HERE = Path(__file__).parent

_TOL = 1e-6  # inference.py must reproduce sklearn probabilities exactly


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


def _results_dir() -> Path:
    env_dir = os.environ.get("MODEL_DIR")
    return Path(env_dir) if env_dir else _HERE.parent / "data"


@pytest.fixture(scope="session")
def model() -> ModelJSON:
    try:
        return load_model()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def test_fixtures() -> list[dict]:
    fixtures_path = _results_dir() / "test_fixtures.json"
    if not fixtures_path.exists():
        pytest.skip(f"test_fixtures.json not found at {fixtures_path}.")
    with open(fixtures_path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_predict_probability(model: ModelJSON, test_fixtures: list[dict]) -> None:
    """inference.py must reproduce the sklearn probability for every fixture."""
    mismatches = []
    for i, fixture in enumerate(test_fixtures):
        prob = predict_probability(model, fixture["input"])
        expected = fixture["probability"]
        if abs(prob - expected) > _TOL:
            mismatches.append(
                f"  fixture {i}: expected={expected:.8f}, got={prob:.8f}, "
                f"diff={abs(prob - expected):.2e}"
            )
    assert not mismatches, "Probability mismatches:\n" + "\n".join(mismatches)


def test_model_fields_consistent(model: ModelJSON) -> None:
    """model.json must declare the same features in weights and model_fields."""
    assert set(model.weights) == set(model.features.model_fields), (
        "Mismatch between model.json 'weights' keys and 'features.model_fields'"
    )


def test_preprocessing_covers_continuous_fields(model: ModelJSON) -> None:
    """Every continuous_normalized field must have a preprocessing entry."""
    missing = set(model.features.continuous_normalized) - set(model.preprocessing)
    assert not missing, f"Missing preprocessing stats for: {missing}"
