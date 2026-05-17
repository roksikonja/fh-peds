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

### Project layout

- `src/` — Astro pages, components, layouts, calculator script, global styles, and per-field markdown descriptions under `src/content/descriptions/`.
- `public/js/` — single source of truth for the inference pipeline (`bmi_zscore_table.js`, `preprocessing.js`, `model.js`) plus `plotting.js` for the model page. These files are loaded both by the site (as classic `<script>` tags) and by the Node test harness via `vm`, so there is no bundling or duplication.

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

The official model used by the test suites lives in `data/models/20260331_220132/`. Experimental training runs go to `ml-fh-peds/results/` (gitignored).

**Python (pytest):**

```bash
cd ml-fh-peds
MODEL_DIR=$(git rev-parse --show-toplevel)/data/models/20260331_220132 pytest
```

**JavaScript inference:**

```bash
MODEL_DIR=$(git rev-parse --show-toplevel)/data/models/20260331_220132 node ml-fh-peds/tests/test_inference.js
```

Loads `model.json` and `inference_samples.json` from `MODEL_DIR` and verifies the JS inference matches the Python model within tolerance `1e-6`. The suite also validates the fitted UK90 BMI LMS parameters against the canonical `sitar::uk90` reference (Tim Cole's R package). On first run the test downloads `uk90.rda` and caches it under `ml-fh-peds/tests/.cache/` (gitignored).

---

## Development tooling

### Pre-commit hooks

Static QA (ruff lint+format for Python, prettier for JS/HTML/CSS/MD/JSON/YAML, eslint for JS) runs on every commit. Tests are not part of the hook — they run in CI.

```bash
uv tool install pre-commit
```
