# fh-peds

ML diagnostic score for familial hypercholesterolaemia in children and adolescents.

---

## Website

The website is an [Astro](https://astro.build) static site, ready to deploy to **Cloudflare Pages**. Page content is rendered to plain HTML at build time; the only client-side JavaScript is the calculator's form logic plus the shared inference modules in `public/js/`.

### Local development

```bash
npm install
npm run dev       # http://localhost:4321
```

### Production build

```bash
npm run build     # outputs to dist/
npm run preview   # serves dist/ for local verification
```

### Deploy to Cloudflare Pages

Connect the repo and set:

| Setting          | Value           |
| ---------------- | --------------- |
| Framework preset | Astro           |
| Build command    | `npm run build` |
| Build output dir | `dist`          |
| Node version     | `20` or newer   |

No environment variables are required for the deploy.

### Project layout

```
src/
├── content/descriptions/   per-field markdown (math via $...$ and $$...$$)
├── components/             Nav, Hero, DescriptionSidebar, CalculatorForm
├── layouts/Layout.astro    page shell (head + nav)
├── pages/{index,model,about}.astro
├── scripts/calculator.ts   client-side form wiring
└── styles/global.css

public/
├── js/                     SINGLE SOURCE OF TRUTH for inference JS
│   ├── bmi_zscore_table.js UK90 LMS parameters + bmiToZScore()
│   ├── preprocessing.js    unit conversions, formSampleToRawSample, validateField
│   ├── model.js            model weights, preprocessSample, calculateMLFHPEDS
│   └── plotting.js         Canvas 2D charts (model page)
└── logos/                  partner logos
```

The four files under `public/js/` are also loaded directly by the Node test harness (`ml-fh-peds/tests/test_inference.js`) via Node's `vm` module — no bundling, no duplication.

### Field descriptions

Each form field has a markdown file in `src/content/descriptions/` (e.g. `ldl_cholesterol.md`, `bmi.md`). They are pre-rendered at build time with **remark-math + rehype-katex**, so LaTeX math (e.g. the Cole & Green Z-score formula in `bmi.md`) ships as plain HTML — no runtime math library. The compiled HTML for every field is inlined into the calculator page in hidden `<template>` elements and swapped into the sidebar on field focus.

---

## Data assets

### `data/BMI-SDS-LMS.xlsx`

UK90 reference table for converting raw BMI (kg/m²) to an age- and sex-adjusted Z-score (SDS). Two sheets:

| Sheet      | Sex                 |
| ---------- | ------------------- |
| `Male=1`   | Male (gender = 1)   |
| `Female=2` | Female (gender = 0) |

Grid layout: rows = BMI 10–50 in 0.05 steps (801 values), columns = age 0–18 in 0.05-year steps (361 values). Each cell is the Z-score for that (BMI, age) pair.

### `public/js/bmi_zscore_table.js` — auto-generated

Compiled from `BMI-SDS-LMS.xlsx` by `data/bmi_zscore_to_js.py`. Fits the underlying UK90 LMS parameters `L`, `M`, `S` at each 0.05-year age knot (361 per sex) and writes them as `Float32Array`s, together with a `bmiToZScore(bmi, age, gender)` function that evaluates the Cole & Green formula with linear interpolation between adjacent knots.

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

The official model used by the test suites lives in `models/20260331_220132/`. Experimental training runs go to `ml-fh-peds/results/` (gitignored).

**Python (pytest):**

```bash
cd ml-fh-peds
MODEL_DIR=$(git rev-parse --show-toplevel)/models/20260331_220132 pytest
```

**JavaScript inference:**

```bash
MODEL_DIR=$(git rev-parse --show-toplevel)/models/20260331_220132 node ml-fh-peds/tests/test_inference.js
```

Loads `model.json` and `inference_samples.json` from `MODEL_DIR`. The JS test loads `public/js/bmi_zscore_table.js`, `public/js/preprocessing.js`, and `public/js/model.js` directly via Node's `vm` module and verifies the JS inference matches the Python model within tolerance `1e-6`.

The suite also validates the fitted UK90 BMI LMS parameters against the canonical `sitar::uk90` reference (Tim Cole's R package — the authoritative source of UK90 LMS values). On first run, the test downloads `uk90.rda` from <https://github.com/statist7/sitar> and caches it under `ml-fh-peds/tests/.cache/` (gitignored); subsequent runs read from the cache. The parser is hand-rolled in pure Node — see `ml-fh-peds/tests/uk90_rda_loader.js`. Tolerance: `|Δ| ≤ 0.005` on L, M, and S at seven sentinel ages (2, 5, 7.5, 10, 12, 15, 17 years) × both sexes.

---

## Development tooling

### Pre-commit hooks

Static QA (ruff lint+format for Python, prettier for JS/HTML/CSS/MD/JSON/YAML, eslint for JS) runs on every commit. Tests are not part of the hook — they run in CI.

```bash
uv tool install pre-commit
```
