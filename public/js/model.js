/* ============================================================
   ML-FH-PeDS  —  Model definition & inference
   ============================================================ */

'use strict';

/* ── Model definition ────────────────────────────────────────
   Logistic regression trained on run 20260331_220132.
   Exact port of tests/inference.py — verified against all
   1665 inference fixtures at tolerance 1e-6 via
   tests/test_inference.js.

   Unit expectations (model-native space):
     age               years
     gender            0 = Female, 1 = Male
     fh_*              0–3 (or 0–1 for binary)
     hdl / ldl / tc    mmol/L
     tag               mmol/L
     bmi_z_score       z-score (passed through as-is)
     lp_a              mg/L
──────────────────────────────────────────────────────────────*/

const MODEL = {
  intercept: -2.3688014720809085,

  preprocessing: {
    age: { mean: 7.314633123689728, std: 2.607213078684607 },
    hdl_cholesterol: { mean: 1.5362264150943397, std: 0.37043406815321883 },
    ldl_cholesterol: { mean: 3.78845283018868, std: 1.1593288141513893 },
    total_cholesterol: { mean: 5.767471698113208, std: 1.1603039079279096 },
    tag: { mean: 1.0961132075471698, std: 0.7718188209096246 },
    bmi_z_score: { mean: 0.28694020169346707, std: 1.3078173489609493 },
    lp_a: { mean: 310.6692307692308, std: 332.1171687025873 },
  },

  weights: {
    age: 0.00667784927430266,
    gender: 0.32143804346854776,
    fh_high_cholesterol_1: 0.34873544396742856,
    fh_high_cholesterol_2: 0.02078297866340733,
    fh_high_cholesterol_3: 0.643902650173691,
    fh_premature_cad_1: 0.05226457867177578,
    fh_premature_cad_2: 0.12812175064921372,
    fh_premature_cad_3: 0.03427669157864839,
    fh_pad_cvi_1: 0.02237060945544122,
    fh_pad_cvi_2: -0.364993539338868,
    fh_pad_cvi_3: 0.00978393164164006,
    fh_xant: -0.21087954347939916,
    fh_acrus_senilis: 0.2413157398789852,
    hdl_cholesterol: -0.8760712481075249,
    ldl_cholesterol: 1.3552895085240824,
    total_cholesterol: 1.1470281822853394,
    tag: -0.5361077381018685,
    bmi_z_score: -0.05708627050403221,
    lp_a: -0.4028915834363285,
  },

  binary_categorical: ['gender', 'fh_xant', 'fh_acrus_senilis'],
  multi_categorical: ['fh_high_cholesterol', 'fh_premature_cad', 'fh_pad_cvi'],
  continuous_normalized: [
    'age',
    'hdl_cholesterol',
    'ldl_cholesterol',
    'total_cholesterol',
    'tag',
    'bmi_z_score',
    'lp_a',
  ],
  input_fields: [
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
    'bmi_z_score',
    'lp_a',
  ],
};

/**
 * Preprocess a raw input dict into model feature space.
 * Mirrors preprocess_sample() in tests/inference.py exactly:
 *   - binary_categorical:   float(value) if present, else 0.0
 *   - multi_categorical:    one-hot for levels 1, 2, 3
 *   - continuous_normalized: z-score; missing → impute with training mean
 *
 * @param {Object} raw  Values in model-native units; null/'' = missing.
 * @returns {Object}    Flat feature dict ready for weighted sum.
 */
function preprocessSample(raw) {
  const sample = {};

  for (const field of MODEL.input_fields) {
    const value =
      raw[field] !== null && raw[field] !== undefined && raw[field] !== '' ? raw[field] : null;

    if (MODEL.binary_categorical.includes(field)) {
      sample[field] = value !== null ? parseFloat(value) : 0.0;
    } else if (MODEL.multi_categorical.includes(field)) {
      const v = value !== null ? Math.round(parseFloat(value)) : 0;
      sample[field + '_1'] = v === 1 ? 1.0 : 0.0;
      sample[field + '_2'] = v === 2 ? 1.0 : 0.0;
      sample[field + '_3'] = v === 3 ? 1.0 : 0.0;
    } else {
      // continuous_normalized
      const stats = MODEL.preprocessing[field];
      const v = value !== null ? parseFloat(value) : stats.mean;
      sample[field] = (v - stats.mean) / stats.std;
    }
  }

  return sample;
}

/**
 * Return the model's FH probability ∈ [0, 1] for a raw input dict.
 * Mirrors predict_probability() in tests/inference.py.
 *
 * @param {Object} raw  Model-native-unit values; null/'' accepted for missing.
 */
function calculateMLFHPEDS(raw) {
  const sample = preprocessSample(raw);
  let ws = MODEL.intercept;
  for (const [feature, weight] of Object.entries(MODEL.weights)) {
    ws += weight * sample[feature];
  }
  return 1.0 / (1.0 + Math.exp(-ws));
}
