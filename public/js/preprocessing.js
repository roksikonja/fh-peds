/* ============================================================
   ML-FH-PeDS  —  Preprocessing & validation
   ============================================================ */

'use strict';

/* ── Unit conversion constants ───────────────────────────────
   Cholesterol (TC, HDL-C, LDL-C): model expects mmol/L
     mg/dL → mmol/L : divide by 38.67

   Triglycerides (TAG):             model expects mmol/L
     mg/dL → mmol/L : divide by 88.57

   Lp(a):                           model expects mg/L
     nmol/L → mg/L  : multiply by 4.0
       (EAS consensus: 125 nmol/L ≈ 50 mg/dL = 500 mg/L)
──────────────────────────────────────────────────────────────*/

const CHOL_MGDL_PER_MMOLL = 38.67;
const TAG_MGDL_PER_MMOLL = 88.57;
const LPA_MGL_PER_NMOLL = 4.0;

/* ── formSampleToRawSample ───────────────────────────────────
   Converts the structured form-field object (string values +
   explicit unit selections) into a raw-sample dict ready for
   calculateMLFHPEDS / preprocessSample.

   FormSample shape
   ────────────────
   All numeric values are strings or null (blank = null).
   Unit fields default to the model-native unit so callers
   can omit them when the value is already in the right unit.

   {
     age:                      string | null   (years, 0–18)
     gender:                   string | null   ('0' Female / '1' Male)
     fh_high_cholesterol:      string | null   ('0'–'3')
     fh_premature_cad:         string | null   ('0'–'3')
     fh_pad_cvi:               string | null   ('0'–'3')
     fh_xant:                  string | null   ('0'–'1')
     fh_acrus_senilis:         string | null   ('0'–'1')
     hdl_cholesterol:          string | null
     hdl_cholesterol_unit:     'mmol/L'|'mg/dL'        (default 'mmol/L')
     ldl_cholesterol:          string | null
     ldl_cholesterol_unit:     'mmol/L'|'mg/dL'        (default 'mmol/L')
     total_cholesterol:        string | null
     total_cholesterol_unit:   'mmol/L'|'mg/dL'        (default 'mmol/L')
     tag:                      string | null
     tag_unit:                 'mmol/L'|'mg/dL'        (default 'mmol/L')
     lp_a:                     string | null
     lp_a_unit:                'mg/L'|'nmol/L'         (default 'mg/L')
     bmi:                      string | null   (numeric BMI value; interpretation
                                               depends on bmi_unit)
     bmi_unit:                 'index'|'z-score'       (default 'index')
                               'index'   → raw BMI in kg/m²; converted to z-score
                                           via bmiToZScore (requires age + gender)
                               'z-score' → value is already an age/sex-adjusted
                                           z-score; used as-is
   }

   RawSample shape (returned)
   ──────────────────────────
   {
     age, gender, fh_high_cholesterol, fh_premature_cad,
     fh_pad_cvi, fh_xant, fh_acrus_senilis   — number | null  (pass-through)
     hdl_cholesterol, ldl_cholesterol,
     total_cholesterol, tag                   — number | null  (mmol/L)
     lp_a                                     — number | null  (mg/L)
     bmi_z_score                              — number | null  (z-score)
   }
   null means "field absent" — preprocessSample will impute with training mean.
──────────────────────────────────────────────────────────────*/

/**
 * Parse a string form value to a float, returning null for blank/invalid.
 * @param {string|null} s
 * @returns {number|null}
 */
function _parseNum(s) {
  if (s === '' || s === null || s === undefined) return null;
  const n = parseFloat(s);
  return isNaN(n) ? null : n;
}

/**
 * Convert cholesterol / TAG value to mmol/L.
 * @param {number|null} value
 * @param {'mmol/L'|'mg/dL'} unit
 * @param {boolean} isTag  true → use TAG conversion factor
 * @returns {number|null}
 */
function _toMmol(value, unit, isTag) {
  if (value === null) return null;
  if (unit === 'mg/dL') return value / (isTag ? TAG_MGDL_PER_MMOLL : CHOL_MGDL_PER_MMOLL);
  return value; // already mmol/L
}

/**
 * Convert Lp(a) value to mg/L.
 * @param {number|null} value
 * @param {'mg/L'|'nmol/L'} unit
 * @returns {number|null}
 */
function _toLpaML(value, unit) {
  if (value === null) return null;
  if (unit === 'nmol/L') return value * LPA_MGL_PER_NMOLL;
  return value; // already mg/L
}

/**
 * Convert a FormSample (web-form values + unit selections) into a raw-sample
 * dict in model-native units, ready for calculateMLFHPEDS.
 *
 * Mirrors the logic of form_sample_to_raw_sample() in tests/inference.py.
 *
 * @param {Object} formSample  See FormSample shape above.
 * @returns {Object}           RawSample dict with null for absent fields.
 */
function formSampleToRawSample(formSample) {
  // Parse all numeric string values up front
  const age = _parseNum(formSample.age);
  const gender = _parseNum(formSample.gender);
  const fh_high_chol = _parseNum(formSample.fh_high_cholesterol);
  const fh_cad = _parseNum(formSample.fh_premature_cad);
  const fh_pad = _parseNum(formSample.fh_pad_cvi);
  const fh_xant = _parseNum(formSample.fh_xant);
  const fh_acrus = _parseNum(formSample.fh_acrus_senilis);
  const hdl = _parseNum(formSample.hdl_cholesterol);
  const ldl = _parseNum(formSample.ldl_cholesterol);
  const tc = _parseNum(formSample.total_cholesterol);
  const tag = _parseNum(formSample.tag);
  const lpa = _parseNum(formSample.lp_a);
  const bmi = _parseNum(formSample.bmi);
  const bmiUnit = formSample.bmi_unit || 'index';

  // Resolve BMI to z-score based on selected unit.
  // 'index'   → convert raw BMI (kg/m²) via bmiToZScore; requires age + gender.
  // 'z-score' → value is already a z-score; use directly.
  let bmiZ = null;
  if (bmi !== null) {
    if (bmiUnit === 'z-score') {
      bmiZ = bmi;
    } else if (age !== null && gender !== null) {
      bmiZ = bmiToZScore(bmi, age, gender);
    }
    // else: age or gender missing — cannot convert index → z-score; leave null
  }

  return {
    age,
    gender,
    fh_high_cholesterol: fh_high_chol,
    fh_premature_cad: fh_cad,
    fh_pad_cvi: fh_pad,
    fh_xant,
    fh_acrus_senilis: fh_acrus,
    hdl_cholesterol: _toMmol(hdl, formSample.hdl_cholesterol_unit || 'mmol/L', false),
    ldl_cholesterol: _toMmol(ldl, formSample.ldl_cholesterol_unit || 'mmol/L', false),
    total_cholesterol: _toMmol(tc, formSample.total_cholesterol_unit || 'mmol/L', false),
    tag: _toMmol(tag, formSample.tag_unit || 'mmol/L', true),
    lp_a: _toLpaML(lpa, formSample.lp_a_unit || 'mg/L'),
    bmi_z_score: bmiZ,
  };
}

/* ── Field validation ────────────────────────────────────────*/

const FIELD_CONSTRAINTS = {
  age: { min: 0, max: 18 },
  bmi_index: { min: 0, max: 50 }, // raw BMI kg/m²
  bmi_zscore: { min: -10, max: 10 }, // z-score (generous bounds)
  total_cholesterol: { min: 0 },
  hdl_cholesterol: { min: 0 },
  ldl_cholesterol: { min: 0 },
  tag: { min: 0 },
  lp_a: { min: 0 },
};

/**
 * @param {string} name       Form element name attribute.
 * @param {string|null} rawValue
 * @param {HTMLFormElement} [form]  Passed to resolve bmi_unit for the bmi field.
 */
function validateField(name, rawValue, form) {
  if (rawValue === '' || rawValue === null || rawValue === undefined) return true;
  let constraintKey = name;
  if (name === 'bmi' && form) {
    const unitEl = form.querySelector('[name="bmi_unit"]');
    constraintKey = unitEl && unitEl.value === 'z-score' ? 'bmi_zscore' : 'bmi_index';
  }
  const c = FIELD_CONSTRAINTS[constraintKey];
  if (!c) return true;
  const n = parseFloat(rawValue);
  if (isNaN(n)) return false;
  if (c.min !== undefined && n < c.min) return false;
  if (c.max !== undefined && n > c.max) return false;
  return true;
}

/* ── Browser globals ─────────────────────────────────────────
   The function declarations above are already global in classic
   script context, but `const` declarations are not — re-expose
   the unit conversion factors so callers (e.g. the FH-PeDS UI,
   which needs to convert mmol/L thresholds back to mg/dL for
   display) can read them without duplicating the constants. */

if (typeof window !== 'undefined') {
  window.UNIT_CONVERSIONS = {
    CHOL_MGDL_PER_MMOLL,
    TAG_MGDL_PER_MMOLL,
    LPA_MGL_PER_NMOLL,
  };
}
