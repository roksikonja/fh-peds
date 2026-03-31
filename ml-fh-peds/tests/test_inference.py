from __future__ import annotations

import json
import os
from pathlib import Path

from tests.inference import load_model
from tests.inference import predict_probability


_TOL = 1e-6


def _get_model():
    results_dir = Path(os.environ["MODEL_DIR"])
    return load_model(results_dir)


def _get_samples() -> list[dict]:
    results_dir = Path(os.environ["MODEL_DIR"])
    with open(results_dir / "inference_samples.json") as fh:
        return json.load(fh)


def test_model_inference() -> None:
    model = _get_model()
    samples = _get_samples()
    mismatches = []
    for i, fixture in enumerate(samples):
        prob = predict_probability(model, fixture["input"])
        expected = fixture["probability"]
        if abs(prob - expected) > _TOL:
            mismatches.append(
                f"  fixture {i}: expected={expected:.8f}, got={prob:.8f}, "
                f"diff={abs(prob - expected):.2e}"
            )
    assert not mismatches, "Probability mismatches:\n" + "\n".join(mismatches)
