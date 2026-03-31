/**
 * test_inference.js
 *
 * Pure-JS port of test_inference.py.
 *
 * Loads model.json and inference_samples.json from the directory pointed to by
 * the MODEL_DIR environment variable, then verifies that the JS inference
 * implementation reproduces every expected probability within tolerance 1e-6.
 *
 * Usage:
 *   MODEL_DIR=/path/to/results/20260331_220132 node tests/test_inference.js
 *
 * Exit code 0 = all tests pass.  Exit code 1 = failures or error.
 */

'use strict';

const fs   = require('fs');
const path = require('path');

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

const model   = loadJSON('model.json');
const samples = loadJSON('inference_samples.json');

/* ── Inference (mirrors tests/inference.py) ─────────────────── */

/**
 * Preprocess a raw input dict into model feature space.
 *
 * Mirrors preprocess_sample() in inference.py:
 *   - binary_categorical  → float(value) if present, else 0.0
 *   - multi_categorical   → one-hot for levels 1, 2, 3
 *   - continuous_normalized → (value - mean) / std
 *                             missing values imputed with training mean
 *
 * @param {Object} raw   Keys from model.features.input_fields; values may be
 *                       null/undefined for missing inputs.
 * @returns {Object}     Flat feature dict ready for dot-product.
 */
function preprocessSample(raw) {
  const { features, preprocessing } = model;
  const sample = {};

  for (const field of features.input_fields) {
    // Treat null / undefined / missing as absent
    const rawVal = (raw[field] !== null && raw[field] !== undefined)
      ? raw[field]
      : null;

    if (features.binary_categorical.includes(field)) {
      sample[field] = rawVal !== null ? parseFloat(rawVal) : 0.0;

    } else if (features.multi_categorical.includes(field)) {
      const v = rawVal !== null ? Math.round(parseFloat(rawVal)) : 0;
      sample[`${field}_1`] = v === 1 ? 1.0 : 0.0;
      sample[`${field}_2`] = v === 2 ? 1.0 : 0.0;
      sample[`${field}_3`] = v === 3 ? 1.0 : 0.0;

    } else if (features.continuous_normalized.includes(field)) {
      const stats = preprocessing[field];
      const v = rawVal !== null ? parseFloat(rawVal) : stats.mean; // impute
      sample[field] = (v - stats.mean) / stats.std;

    } else {
      throw new Error(`Unknown field: ${field}`);
    }
  }

  return sample;
}

/**
 * Return the model's FH probability for a raw input dict.
 * Mirrors predict_probability() in inference.py.
 */
function predictProbability(raw) {
  const sample = preprocessSample(raw);

  let weightedSum = model.intercept;
  for (const [feature, weight] of Object.entries(model.weights)) {
    weightedSum += weight * sample[feature];
  }

  return 1.0 / (1.0 + Math.exp(-weightedSum));
}

/* ── Test runner ─────────────────────────────────────────────── */

const TOL = 1e-6;

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

console.log(`Model:   ${path.join(modelDir, 'model.json')}`);
console.log(`Samples: ${path.join(modelDir, 'inference_samples.json')}`);
console.log(`Running ${samples.length} inference fixtures (tolerance ${TOL})…\n`);

const mismatches = testModelInference();

if (mismatches.length === 0) {
  console.log(`✓ All ${samples.length} fixtures passed.`);
  process.exit(0);
} else {
  console.error(`✗ ${mismatches.length} mismatch(es):\n`);
  mismatches.forEach(m => console.error(m));
  process.exit(1);
}
