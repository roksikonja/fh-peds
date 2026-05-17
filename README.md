# fh-peds

ML diagnostic score for familial hypercholesterolaemia in children and adolescents.

---

## Website

A static calculator served from the project root (`index.html`, `about.html`). No build step — open `index.html` directly in a browser or serve it with any static file server:

```bash
npx serve .
```

The calculator logic is split across four files in `website/`:

| File | Responsibility |
|---|---|
| `bmi_zscore_table.js` | UK90 BMI z-score lookup table + `bmiToZScore()` |
| `preprocessing.js` | Unit conversions, `formSampleToRawSample`, `validateField` |
| `model.js` | Model weights, `preprocessSample`, `calculateMLFHPEDS` |
| `plotting.js` | Canvas 2D precision-recall and feature-weights charts |
| `main.js` | Form wiring and page initialisation |

---

## Data assets

### `data/BMI-SDS-LMS.xlsx`

UK90 reference table for converting raw BMI (kg/m²) to an age- and sex-adjusted Z-score (SDS). Two sheets:

| Sheet | Sex |
|---|---|
| `Male=1` | Male (gender = 1) |
| `Female=2` | Female (gender = 0) |

Grid layout: rows = BMI 10–50 in 0.05 steps (801 values), columns = age 0–18 in 0.05-year steps (361 values). Each cell is the Z-score for that (BMI, age) pair.

### `website/bmi_zscore_table.js` — auto-generated

Compiled from `BMI-SDS-LMS.xlsx` by `data/bmi_zscore_to_js.py`. Contains two flat `Int16Array` constants (`BMI_ZSCORE_MALE`, `BMI_ZSCORE_FEMALE`) storing Z-scores × 1000 (3 decimal-place precision, max error 0.0005 Z), plus a `bmiToZScore(bmi, age, gender)` function that performs bilinear interpolation on the grid.

**Do not edit by hand.** Regenerate whenever `BMI-SDS-LMS.xlsx` changes:

```bash
cd data
../ml-fh-peds/venv/bin/python bmi_zscore_to_js.py
```

The script validates grid dimensions and step sizes, runs sanity checks against known reference values, and exits with an error if anything looks wrong.

---

## ML

### Installation

```bash
uv venv --python 3.12
source .venv/bin/activate
cd ml-fh-peds
uv pip install -e ".[test]"
```

### Usage

```bash
fh train
fh dashboard /path/to/results
```

### Tests

**Python (pytest):**

```bash
cd ml-fh-peds
pytest
```

**JavaScript inference (requires a trained model run):**

```bash
MODEL_DIR=/path/to/results/run_id node ml-fh-peds/tests/test_inference.js
```

Loads `model.json` and `inference_samples.json` from `MODEL_DIR`. The test loads `website/bmi_zscore_table.js`, `website/preprocessing.js`, and `website/model.js` directly via Node's `vm` module and verifies the JS inference matches the Python model within tolerance `1e-6`.

