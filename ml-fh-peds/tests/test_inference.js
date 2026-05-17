/**
 * test_inference.js
 *
 * Pure-JS port of test_inference.py.
 *
 * Loads model.json and inference_samples.json from the directory pointed to by
 * the MODEL_DIR environment variable, then verifies that the JS inference
 * implementation reproduces every expected probability within tolerance 1e-6.
 *
 * Also validates the fitted UK90 BMI LMS parameters in
 * public/js/bmi_zscore_table.js against the canonical sitar::uk90 reference
 * (Tim Cole's R package — see uk90_rda_loader.js).
 *
 * Inference modules are loaded via Node's vm module so they run in this global
 * scope exactly as they would in a browser — no bundler required.
 * Loaded in dependency order:
 *   bmi_zscore_table.js  →  preprocessing.js  →  model.js
 *
 * Usage:
 *   MODEL_DIR=/path/to/results/20260331_220132 node tests/test_inference.js
 *
 * Exit code 0 = all tests pass.  Exit code 1 = failures or error.
 */

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

import { loadUK90BMI, interpRef } from './uk90_rda_loader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/* ── Resolve MODEL_DIR ──────────────────────────────────────── */

const modelDir = process.env.MODEL_DIR;
if (!modelDir) {
  console.error('Error: MODEL_DIR environment variable is not set.');
  console.error('Usage: MODEL_DIR=/path/to/results/run_id node tests/test_inference.js');
  process.exit(1);
}

function loadJSON(file) {
  const full = path.join(modelDir, file);
  if (!fs.existsSync(full)) {
    console.error(`Error: file not found: ${full}`);
    process.exit(1);
  }
  return JSON.parse(fs.readFileSync(full, 'utf8'));
}

/* ── Load artefacts ─────────────────────────────────────────── */

const model = loadJSON('model.json');
const samples = loadJSON('inference_samples.json');

/* ── Load inference modules into this global scope ──────────── */

// The inference JS lives in `public/js/` — single source of truth, shared
// between the Astro site (served as static assets) and these Node tests.
const JS_DIR = path.resolve(__dirname, '../../public/js');

function loadModule(filename) {
  const fullPath = path.join(JS_DIR, filename);
  if (!fs.existsSync(fullPath)) {
    console.error(`Error: ${filename} not found at ${fullPath}`);
    process.exit(1);
  }
  vm.runInThisContext(fs.readFileSync(fullPath, 'utf8'), { filename: fullPath });
}

// Dependency order matters: bmi table first, then preprocessing (uses
// bmiToZScore), then model (uses MODEL / preprocessSample).
loadModule('bmi_zscore_table.js');
loadModule('preprocessing.js');
loadModule('model.js');

/* ── predictProbability (reads from loaded model.json artefact) ─
   Mirrors predict_probability() in inference.py but uses the
   model.json loaded from MODEL_DIR rather than the hardcoded
   MODEL constant in model.js, so the fixture sweep is always
   consistent with the artefact under test.
──────────────────────────────────────────────────────────────── */

function preprocessSampleFromArtefact(raw) {
  const { features, preprocessing } = model;
  const sample = {};

  for (const field of features.input_fields) {
    const rawVal = raw[field] !== null && raw[field] !== undefined ? raw[field] : null;

    if (features.binary_categorical.includes(field)) {
      sample[field] = rawVal !== null ? parseFloat(rawVal) : 0.0;
    } else if (features.multi_categorical.includes(field)) {
      const v = rawVal !== null ? Math.round(parseFloat(rawVal)) : 0;
      sample[`${field}_1`] = v === 1 ? 1.0 : 0.0;
      sample[`${field}_2`] = v === 2 ? 1.0 : 0.0;
      sample[`${field}_3`] = v === 3 ? 1.0 : 0.0;
    } else if (features.continuous_normalized.includes(field)) {
      const stats = preprocessing[field];
      const v = rawVal !== null ? parseFloat(rawVal) : stats.mean;
      sample[field] = (v - stats.mean) / stats.std;
    } else {
      throw new Error(`Unknown field: ${field}`);
    }
  }

  return sample;
}

function predictProbability(raw) {
  const sample = preprocessSampleFromArtefact(raw);
  let weightedSum = model.intercept;
  for (const [feature, weight] of Object.entries(model.weights)) {
    weightedSum += weight * sample[feature];
  }
  return 1.0 / (1.0 + Math.exp(-weightedSum));
}

/* ── Test helpers ────────────────────────────────────────────── */

const TOL = 1e-6;
const CHOL_TOL = 1e-9; // unit-conversion round-trip tolerance

let passed = 0;
let failed = 0;

function assert(condition, message) {
  if (condition) {
    console.log(`  ✓ ${message}`);
    passed++;
  } else {
    console.error(`  ✗ ${message}`);
    failed++;
  }
}

function assertClose(got, expected, tol, message) {
  const diff = Math.abs(got - expected);
  if (diff <= tol) {
    console.log(`  ✓ ${message} (got ${got}, diff ${diff.toExponential(2)})`);
    passed++;
  } else {
    console.error(
      `  ✗ ${message} — expected ${expected}, got ${got}, diff ${diff.toExponential(2)}`
    );
    failed++;
  }
}

function assertNull(got, message) {
  assert(got === null, `${message} is null`);
}

/* ── Tests: bmiToZScore ──────────────────────────────────────── */

function testBmiToZScore() {
  console.log('\nbmiToZScore');

  // Reference values are exact on-grid lookups from BMI-SDS-LMS.xlsx,
  // so expected == stored_int / 1000 with no interpolation error.
  // Tolerance = 0.001 (one Int16 step) to absorb bilinear interpolation
  // at on-grid points where neighbouring cells differ minimally.
  const TOL_Z = 0.001;

  // ── 1. On-grid reference points ──────────────────────────────
  console.log('\n  1. On-grid reference values (from BMI-SDS-LMS.xlsx)');
  {
    // Male, BMI=17.5, age=7.3 → Z=1.120
    assertClose(bmiToZScore(17.5, 7.3, 1), 1.12, TOL_Z, 'male   BMI=17.5 age=7.3  → Z=1.120');
    // Male, BMI=15.6, age=7.3 → Z=-0.006  (near median)
    assertClose(bmiToZScore(15.6, 7.3, 1), -0.006, TOL_Z, 'male   BMI=15.6 age=7.3  → Z=-0.006');
    // Female, BMI=20.0, age=12.0 → Z=0.719
    assertClose(bmiToZScore(20.0, 12.0, 0), 0.719, TOL_Z, 'female BMI=20.0 age=12.0 → Z=0.719');
    // Male, BMI=10.0, age=5.0 → Z=-7.927  (very low BMI)
    assertClose(bmiToZScore(10.0, 5.0, 1), -7.927, TOL_Z, 'male   BMI=10.0 age=5.0  → Z=-7.927');
    // Female, BMI=25.0, age=15.0 → Z=1.533
    assertClose(bmiToZScore(25.0, 15.0, 0), 1.533, TOL_Z, 'female BMI=25.0 age=15.0 → Z=1.533');
  }

  // ── 2. Sex symmetry: male ≠ female at same BMI/age ───────────
  console.log('\n  2. Male and female give different Z at same BMI/age');
  {
    const zM = bmiToZScore(20.0, 10.0, 1);
    const zF = bmiToZScore(20.0, 10.0, 0);
    assert(Math.abs(zM - zF) > 0.01, `male Z=${zM.toFixed(3)} ≠ female Z=${zF.toFixed(3)}`);
  }

  // ── 3. Monotone in BMI: higher BMI → higher Z ────────────────
  console.log('\n  3. Z increases monotonically with BMI (age=10, male)');
  {
    const bmis = [12, 15, 18, 22, 28, 35];
    const zs = bmis.map((b) => bmiToZScore(b, 10.0, 1));
    for (let i = 1; i < zs.length; i++) {
      assert(
        zs[i] > zs[i - 1],
        `BMI ${bmis[i - 1]}→${bmis[i]}: Z ${zs[i - 1].toFixed(3)}→${zs[i].toFixed(3)}`
      );
    }
  }

  // ── 4. Clamping: out-of-range inputs don't throw ─────────────
  console.log('\n  4. Clamping (BMI < 10, BMI > 50, age > 18)');
  {
    assert(isFinite(bmiToZScore(5.0, 10.0, 1)), 'BMI=5  (below min) → finite');
    assert(isFinite(bmiToZScore(60.0, 10.0, 1)), 'BMI=60 (above max) → finite');
    assert(isFinite(bmiToZScore(20.0, 25.0, 0)), 'age=25 (above max) → finite');
  }

  // ── 5. Round-trip: bmi_unit='index' converts via bmiToZScore ─
  console.log('\n  5. Round-trip: bmi index → z-score via formSampleToRawSample');
  {
    const bmi = 17.5,
      age = 7.3,
      gender = 1;
    const expectedZ = bmiToZScore(bmi, age, gender);
    const raw = formSampleToRawSample({
      bmi: String(bmi),
      bmi_unit: 'index',
      age: String(age),
      gender: String(gender),
    });
    assertClose(
      raw.bmi_z_score,
      expectedZ,
      TOL_Z,
      `bmi index converted to z-score (${expectedZ.toFixed(3)})`
    );
  }

  // ── 6. bmi_unit='z-score' passes value through unchanged ─────
  console.log('\n  6. bmi_unit=z-score passes value directly to bmi_z_score');
  {
    const z = 1.42;
    const raw = formSampleToRawSample({ bmi: String(z), bmi_unit: 'z-score' });
    assertClose(raw.bmi_z_score, z, CHOL_TOL, `bmi z-score passed through (${z})`);
  }

  // ── 7. bmi index without age/gender → null ───────────────────
  console.log('\n  7. bmi index without age/gender → bmi_z_score is null');
  {
    const raw = formSampleToRawSample({ bmi: '18.0', bmi_unit: 'index' });
    assertNull(raw.bmi_z_score, 'bmi_z_score when age/gender missing');
  }
}

/* ── Tests: UK90 LMS values vs sitar::uk90 reference ────────── */

/**
 * Validate that the fitted L/M/S parameters in bmi_zscore_table.js agree
 * with the canonical UK90 reference (sitar::uk90, maintained by Tim Cole)
 * within strict tolerance.
 *
 * Provenance chain being checked:
 *   sitar::uk90 (Cole et al., R package)
 *     ↓  [reference: ground truth]
 *   data/BMI-SDS-LMS.xlsx (dense Z-score grid, presumably from LMSgrowth)
 *     ↓  [data/bmi_zscore_to_js.py: refit LMS from grid]
 *   public/js/bmi_zscore_table.js (BMI_LMS_MALE / BMI_LMS_FEMALE)
 *
 * If this test passes, we have evidence that the deployed LMS values
 * match the published UK90 reference end-to-end.
 */
const UK90_LMS_TOL = 0.005; // |Δ| in L, M, or S
const UK90_TEST_AGES = [2.0, 5.0, 7.5, 10.0, 12.0, 15.0, 17.0];

async function testUK90LMSReference() {
  console.log('\nUK90 LMS values vs sitar::uk90 reference');
  console.log(`  (tolerance |Δ| ≤ ${UK90_LMS_TOL} on L, M, S)`);

  let ref;
  try {
    ref = await loadUK90BMI();
  } catch (err) {
    console.error(`  ✗ failed to load sitar::uk90 reference: ${err.message}`);
    failed++;
    return;
  }

  console.log(`  · reference rows: ${ref.male.length} male, ${ref.female.length} female`);

  const cases = [
    { label: 'male', sex: 'male', lms: BMI_LMS_MALE },
    { label: 'female', sex: 'female', lms: BMI_LMS_FEMALE },
  ];

  // The fitted table now stores LMS values on sitar's native (non-uniform)
  // age knots; every sentinel age in UK90_TEST_AGES is an exact knot, so a
  // direct index lookup against `years` gives the stored value with no
  // interpolation. (The earlier `age/0.05` formula assumed a uniform grid
  // and silently returned wrong / out-of-range entries.)
  const knotIndex = (lms, age) => {
    const y = lms.years;
    for (let i = 0; i < y.length; i++) {
      if (Math.abs(y[i] - age) < 1e-6) return i;
    }
    return -1;
  };

  for (const age of UK90_TEST_AGES) {
    console.log(`\n  age = ${age.toFixed(2)} y`);
    for (const c of cases) {
      const i = knotIndex(c.lms, age);
      if (i < 0) {
        console.error(`  ✗ ${c.label}: no knot at age ${age} in stored table`);
        failed++;
        continue;
      }
      const Lf = c.lms.L[i],
        Mf = c.lms.M[i],
        Sf = c.lms.S[i];
      const Lr = interpRef(ref[c.sex], age, 'L');
      const Mr = interpRef(ref[c.sex], age, 'M');
      const Sr = interpRef(ref[c.sex], age, 'S');

      if (Lr === null || Mr === null || Sr === null) {
        console.error(`  ✗ ${c.label}: reference undefined at age ${age}`);
        failed++;
        continue;
      }

      assertClose(Lf, Lr, UK90_LMS_TOL, `${c.label.padEnd(6)} L (ref ${Lr.toFixed(4)})`);
      assertClose(Mf, Mr, UK90_LMS_TOL, `${c.label.padEnd(6)} M (ref ${Mr.toFixed(4)})`);
      assertClose(Sf, Sr, UK90_LMS_TOL, `${c.label.padEnd(6)} S (ref ${Sr.toFixed(5)})`);
    }
  }
}

/* ── Tests: formSampleToRawSample ───────────────────────────── */

function testFormSampleToRawSample() {
  console.log('\nformSampleToRawSample');

  // ── 1. All fields blank → all null ──────────────────────────
  console.log('\n  1. All blank fields → all null');
  {
    const raw = formSampleToRawSample({});
    for (const field of [
      'age',
      'gender',
      'fh_high_cholesterol',
      'fh_premature_cad',
      'fh_pad_cvi',
      'fh_xant',
      'fh_acrus_senilis',
      'hdl_cholesterol',
      'ldl_cholesterol',
      'total_cholesterol',
      'tag',
      'lp_a',
      'bmi_z_score',
    ]) {
      assertNull(raw[field], field);
    }
  }

  // ── 2. Pass-through: all units already model-native ─────────
  console.log('\n  2. Pass-through (mmol/L, mg/L, z-score)');
  {
    const raw = formSampleToRawSample({
      age: '8.5',
      gender: '1',
      fh_high_cholesterol: '2',
      fh_premature_cad: '1',
      fh_pad_cvi: '0',
      fh_xant: '0',
      fh_acrus_senilis: '1',
      hdl_cholesterol: '1.4',
      hdl_cholesterol_unit: 'mmol/L',
      ldl_cholesterol: '4.2',
      ldl_cholesterol_unit: 'mmol/L',
      total_cholesterol: '6.1',
      total_cholesterol_unit: 'mmol/L',
      tag: '1.1',
      tag_unit: 'mmol/L',
      lp_a: '200.0',
      lp_a_unit: 'mg/L',
      bmi: '0.5',
      bmi_unit: 'z-score',
    });
    assertClose(raw.age, 8.5, CHOL_TOL, 'age');
    assert(raw.gender === 1, 'gender');
    assert(raw.fh_high_cholesterol === 2, 'fh_high_cholesterol');
    assert(raw.fh_premature_cad === 1, 'fh_premature_cad');
    assert(raw.fh_pad_cvi === 0, 'fh_pad_cvi');
    assert(raw.fh_xant === 0, 'fh_xant');
    assert(raw.fh_acrus_senilis === 1, 'fh_acrus_senilis');
    assertClose(raw.hdl_cholesterol, 1.4, CHOL_TOL, 'hdl_cholesterol mmol/L');
    assertClose(raw.ldl_cholesterol, 4.2, CHOL_TOL, 'ldl_cholesterol mmol/L');
    assertClose(raw.total_cholesterol, 6.1, CHOL_TOL, 'total_cholesterol mmol/L');
    assertClose(raw.tag, 1.1, CHOL_TOL, 'tag mmol/L');
    assertClose(raw.lp_a, 200.0, CHOL_TOL, 'lp_a mg/L');
    assertClose(raw.bmi_z_score, 0.5, CHOL_TOL, 'bmi z-score pass-through');
  }

  // ── 3. Cholesterol mg/dL → mmol/L ───────────────────────────
  console.log('\n  3. Cholesterol mg/dL → mmol/L');
  {
    const hdl_mgdl = 54.138; // → 1.4 mmol/L
    const ldl_mgdl = 162.414; // → 4.2 mmol/L
    const tc_mgdl = 235.887; // → 6.1 mmol/L
    const raw = formSampleToRawSample({
      hdl_cholesterol: String(hdl_mgdl),
      hdl_cholesterol_unit: 'mg/dL',
      ldl_cholesterol: String(ldl_mgdl),
      ldl_cholesterol_unit: 'mg/dL',
      total_cholesterol: String(tc_mgdl),
      total_cholesterol_unit: 'mg/dL',
    });
    assertClose(raw.hdl_cholesterol, hdl_mgdl / CHOL_MGDL_PER_MMOLL, CHOL_TOL, 'hdl mg/dL→mmol/L');
    assertClose(raw.ldl_cholesterol, ldl_mgdl / CHOL_MGDL_PER_MMOLL, CHOL_TOL, 'ldl mg/dL→mmol/L');
    assertClose(raw.total_cholesterol, tc_mgdl / CHOL_MGDL_PER_MMOLL, CHOL_TOL, 'tc  mg/dL→mmol/L');
  }

  // ── 4. TAG mg/dL → mmol/L ───────────────────────────────────
  console.log('\n  4. TAG mg/dL → mmol/L');
  {
    const tag_mgdl = 97.427; // → 1.1 mmol/L
    const raw = formSampleToRawSample({
      tag: String(tag_mgdl),
      tag_unit: 'mg/dL',
    });
    assertClose(raw.tag, tag_mgdl / TAG_MGDL_PER_MMOLL, CHOL_TOL, 'tag mg/dL→mmol/L');
  }

  // ── 5. Lp(a) nmol/L → mg/L ──────────────────────────────────
  console.log('\n  5. Lp(a) nmol/L → mg/L');
  {
    // 125 nmol/L × 4.0 = 500 mg/L
    const raw = formSampleToRawSample({ lp_a: '125', lp_a_unit: 'nmol/L' });
    assertClose(raw.lp_a, 500.0, CHOL_TOL, 'lp_a nmol/L→mg/L (125→500)');

    // 0 nmol/L → 0 mg/L
    const raw0 = formSampleToRawSample({ lp_a: '0', lp_a_unit: 'nmol/L' });
    assertClose(raw0.lp_a, 0.0, CHOL_TOL, 'lp_a nmol/L→mg/L (0→0)');
  }

  // ── 6. Missing lipid fields stay null (not converted) ────────
  console.log('\n  6. Blank lipid fields stay null');
  {
    const raw = formSampleToRawSample({
      hdl_cholesterol: '',
      hdl_cholesterol_unit: 'mg/dL',
      ldl_cholesterol: null,
      ldl_cholesterol_unit: 'mg/dL',
      lp_a: '',
      lp_a_unit: 'nmol/L',
    });
    assertNull(raw.hdl_cholesterol, 'hdl blank string');
    assertNull(raw.ldl_cholesterol, 'ldl null');
    assertNull(raw.lp_a, 'lp_a blank');
  }

  // ── 7. Default units when unit fields are absent ─────────────
  console.log('\n  7. Default units (no unit field provided)');
  {
    // Cholesterol defaults to mmol/L → value passes through unchanged
    const raw = formSampleToRawSample({ hdl_cholesterol: '1.6' });
    assertClose(raw.hdl_cholesterol, 1.6, CHOL_TOL, 'hdl default mmol/L');

    // Lp(a) defaults to mg/L → value passes through unchanged
    const raw2 = formSampleToRawSample({ lp_a: '310' });
    assertClose(raw2.lp_a, 310, CHOL_TOL, 'lp_a default mg/L');

    // BMI defaults to index → null without age/gender
    const raw3 = formSampleToRawSample({ bmi: '20.0' });
    assertNull(raw3.bmi_z_score, 'bmi default index without age/gender');
  }

  // ── 8. Round-trip: formSample → rawSample → probability ──────
  // Use inference_samples fixture 2 (probability 0.5081666923035468).
  // The fixture stores values already in model-native units, so we
  // feed them as mmol/L / mg/L and expect the same probability.
  console.log('\n  8. Round-trip: formSample → probability (fixture 2)');
  {
    const fixture = samples[2];
    const inp = fixture.input;
    const formSample = {
      age: String(inp.age),
      gender: String(inp.gender),
      fh_high_cholesterol: String(inp.fh_high_cholesterol),
      fh_premature_cad: String(inp.fh_premature_cad),
      fh_pad_cvi: String(inp.fh_pad_cvi),
      fh_xant: String(inp.fh_xant),
      fh_acrus_senilis: String(inp.fh_acrus_senilis),
      hdl_cholesterol: String(inp.hdl_cholesterol),
      hdl_cholesterol_unit: 'mmol/L',
      ldl_cholesterol: String(inp.ldl_cholesterol),
      ldl_cholesterol_unit: 'mmol/L',
      total_cholesterol: String(inp.total_cholesterol),
      total_cholesterol_unit: 'mmol/L',
      tag: String(inp.tag),
      tag_unit: 'mmol/L',
      lp_a: inp.lp_a !== null ? String(inp.lp_a) : null,
      lp_a_unit: 'mg/L',
      // bmi_z_score from fixture is already a z-score — pass directly
      bmi: inp.bmi_z_score !== null ? String(inp.bmi_z_score) : null,
      bmi_unit: 'z-score',
    };
    const raw = formSampleToRawSample(formSample);
    const prob = predictProbability(raw);
    assertClose(
      prob,
      fixture.probability,
      TOL,
      `round-trip probability (expected ${fixture.probability.toFixed(8)})`
    );
  }
}

/* ── Tests: model inference (all fixtures) ───────────────────── */

function testModelInference() {
  const mismatches = [];

  for (let i = 0; i < samples.length; i++) {
    const { input, probability: expected } = samples[i];
    const got = predictProbability(input);
    const diff = Math.abs(got - expected);

    if (diff > TOL) {
      mismatches.push(
        `  fixture ${i}: expected=${expected.toFixed(8)}, ` +
          `got=${got.toFixed(8)}, diff=${diff.toExponential(2)}`
      );
    }
  }

  return mismatches;
}

/* ── Run & report ────────────────────────────────────────────── */

async function main() {
  console.log(`Model:   ${path.join(modelDir, 'model.json')}`);
  console.log(`Samples: ${path.join(modelDir, 'inference_samples.json')}`);

  // bmiToZScore unit tests
  testBmiToZScore();

  // UK90 LMS values vs sitar::uk90 reference (network on first run, cached after)
  await testUK90LMSReference();

  // formSampleToRawSample unit tests
  testFormSampleToRawSample();

  // Full inference fixture sweep
  console.log(`\nModel inference (${samples.length} fixtures, tolerance ${TOL})`);
  const mismatches = testModelInference();
  if (mismatches.length === 0) {
    console.log(`  ✓ All ${samples.length} fixtures passed.`);
    passed += samples.length;
  } else {
    mismatches.forEach((m) => console.error(m));
    failed += mismatches.length;
  }

  // Summary
  console.log(`\n${'─'.repeat(50)}`);
  if (failed === 0) {
    console.log(`✓ All ${passed} tests passed.`);
    process.exit(0);
  } else {
    console.error(`✗ ${failed} failed, ${passed} passed.`);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(`Fatal: ${err.stack || err.message}`);
  process.exit(1);
});
