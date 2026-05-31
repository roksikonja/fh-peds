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
  age: { min: 1, max: 18 },
  bmi_index: { min: 0, max: 50 }, // raw BMI kg/m²
  bmi_zscore: { min: -10, max: 10 }, // z-score (generous bounds)
  total_cholesterol: { min: 0 },
  hdl_cholesterol: { min: 0 },
  ldl_cholesterol: { min: 0 },
  tag: { min: 0 },
  lp_a: { min: 0 },
};

/**
 * Check whether a raw input string contains a comma, which (in this app)
 * is treated as an invalid decimal separator. Period (.) is required.
 * @param {string|null} s
 * @returns {boolean}
 */
function _hasDecimalComma(s) {
  return typeof s === 'string' && s.indexOf(',') !== -1;
}

/**
 * Detailed field validation.
 *
 * Returns an object describing whether the value is acceptable and, if not,
 * the reason. Reasons:
 *   - 'decimal_comma' : value contains a ',' (commas are not accepted)
 *   - 'not_a_number'  : value cannot be parsed as a number
 *   - 'range'         : value is outside the allowed min/max for the field
 *
 * Blank values are considered valid (the model imputes them).
 *
 * @param {string} name       Form element name attribute.
 * @param {string|null} rawValue
 * @param {HTMLFormElement} [form]  Passed to resolve bmi_unit for the bmi field.
 * @returns {{valid: boolean, reason: string|null}}
 */
function validateFieldDetailed(name, rawValue, form) {
  if (rawValue === '' || rawValue === null || rawValue === undefined) {
    return { valid: true, reason: null };
  }
  if (_hasDecimalComma(rawValue)) {
    return { valid: false, reason: 'decimal_comma' };
  }
  let constraintKey = name;
  if (name === 'bmi' && form) {
    const unitEl = form.querySelector('[name="bmi_unit"]');
    constraintKey = unitEl && unitEl.value === 'z-score' ? 'bmi_zscore' : 'bmi_index';
  }
  const c = FIELD_CONSTRAINTS[constraintKey];
  if (!c) return { valid: true, reason: null };
  const n = parseFloat(rawValue);
  if (isNaN(n)) return { valid: false, reason: 'not_a_number' };
  if (c.min !== undefined && n < c.min) return { valid: false, reason: 'range' };
  if (c.max !== undefined && n > c.max) return { valid: false, reason: 'range' };
  return { valid: true, reason: null };
}

/**
 * Backwards-compatible boolean wrapper around validateFieldDetailed.
 * @param {string} name
 * @param {string|null} rawValue
 * @param {HTMLFormElement} [form]
 * @returns {boolean}
 */
function validateField(name, rawValue, form) {
  return validateFieldDetailed(name, rawValue, form).valid;
}

/* ── Plausibility ranges (soft, unit-aware) ──────────────────
   Plausible ranges keyed by [fieldName][unit]. Distinct from
   FIELD_CONSTRAINTS (hard bounds): these drive a non-blocking
   advisory warning that nudges the user to verify the selected
   unit when the entered value is far outside the clinically
   expected range for that unit. The model and downstream
   calculation are unaffected.

   The unit-key strings here must match the <select> option
   values used in CalculatorForm.astro / FhpedsForm.astro
   ('mmol/L', 'mg/dL', 'mg/L', 'nmol/L', 'index', 'z-score').

   Ranges are clinical-plausibility ceilings, not algorithmic
   limits — values outside them are most likely caused by the
   user selecting the wrong unit (e.g. typing the mg/dL number
   while mmol/L is selected) or by a typo.
──────────────────────────────────────────────────────────────*/

const PLAUSIBLE_RANGES = {
  ldl_cholesterol: {
    'mmol/L': { min: 0, max: 20 },
    'mg/dL': { min: 40, max: 773 },
  },
  hdl_cholesterol: {
    'mmol/L': { min: 0, max: 3 },
    'mg/dL': { min: 5, max: 116 },
  },
  total_cholesterol: {
    'mmol/L': { min: 2, max: 25 },
    'mg/dL': { min: 77, max: 966 },
  },
  tag: {
    'mmol/L': { min: 0, max: 10 },
    'mg/dL': { min: 40, max: 885 },
  },
  lp_a: {
    'mg/L': { min: 0, max: 2500 },
    'nmol/L': { min: 0, max: 625 },
  },
  bmi: {
    index: { min: 10, max: 40 },
    'z-score': { min: -5, max: 5 },
  },
};

/**
 * Check whether a value is plausible for the selected unit.
 *
 * Returns `null` when the field has no plausibility table (e.g. age, sex,
 * family-history selects). Returns `{plausible: true}` when the value is
 * blank or unparseable (the hard validator handles those cases and the
 * plausibility layer stays silent so we don't double-warn).
 *
 * Otherwise returns `{plausible, min, max, unit, value}` so the caller can
 * format an informative message.
 *
 * @param {string} name        Form field name (matches <input name="…">)
 * @param {string|null} rawValue
 * @param {string|null} unit   Unit string from the matching <select>; if
 *                             missing or unknown the first table entry is
 *                             used as a defensive default.
 * @returns {{plausible: boolean, min: number, max: number, unit: string, value: number}
 *           | {plausible: true}
 *           | null}
 */
function checkPlausibility(name, rawValue, unit) {
  const table = PLAUSIBLE_RANGES[name];
  if (!table) return null;
  if (rawValue === '' || rawValue === null || rawValue === undefined) {
    return { plausible: true };
  }
  const n = parseFloat(rawValue);
  if (isNaN(n)) return { plausible: true };
  // Pick the matching unit row; fall back to the first row defensively so
  // an unknown unit string never crashes the warning layer.
  let row = unit && table[unit] ? table[unit] : null;
  let resolvedUnit = unit;
  if (!row) {
    const firstKey = Object.keys(table)[0];
    row = table[firstKey];
    resolvedUnit = firstKey;
  }
  const plausible = n >= row.min && n <= row.max;
  return {
    plausible,
    min: row.min,
    max: row.max,
    unit: resolvedUnit,
    value: n,
  };
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
  window.validateFieldDetailed = validateFieldDetailed;
  window.checkPlausibility = checkPlausibility;
  window.PLAUSIBLE_RANGES = PLAUSIBLE_RANGES;
}
