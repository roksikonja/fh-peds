/* ============================================================
   ML-FH-PeDS  —  Form setup & application entry point
   Depends on: bmi_zscore_table.js, preprocessing.js,
               model.js, plotting.js
   ============================================================ */

'use strict';

/* ── Form setup ──────────────────────────────────────────────*/

function setupForm() {
  const form      = document.getElementById('form-ml');
  const resultBox = document.getElementById('result-ml');
  const resetBtn  = document.getElementById('reset-ml');
  if (!form) return;

  /** Collect all form inputs into a FormSample, then convert to raw sample. */
  function readData() {
    const v = {};
    form.querySelectorAll('input[name], select[name]').forEach(el => {
      v[el.name] = el.value;
    });

    const formSample = {
      age:                     v.age,
      gender:                  v.gender,
      fh_high_cholesterol:     v.fh_high_cholesterol,
      fh_premature_cad:        v.fh_premature_cad,
      fh_pad_cvi:              v.fh_pad_cvi,
      fh_xant:                 v.fh_xant,
      fh_acrus_senilis:        v.fh_acrus_senilis,
      hdl_cholesterol:         v.hdl_cholesterol,
      hdl_cholesterol_unit:    v.hdl_cholesterol_unit   || 'mmol/L',
      ldl_cholesterol:         v.ldl_cholesterol,
      ldl_cholesterol_unit:    v.ldl_cholesterol_unit   || 'mmol/L',
      total_cholesterol:       v.total_cholesterol,
      total_cholesterol_unit:  v.total_cholesterol_unit || 'mmol/L',
      tag:                     v.tag,
      tag_unit:                v.tag_unit               || 'mmol/L',
      lp_a:                    v.lp_a,
      lp_a_unit:               v.lp_a_unit              || 'mg/L',
      bmi:                     v.bmi,
      bmi_unit:                v.bmi_unit               || 'index',
    };

    return formSampleToRawSample(formSample);
  }

  function markInvalid(el, invalid) {
    el.classList.toggle('field__input--invalid',  invalid);
    el.classList.toggle('field__select--invalid', invalid);
  }

  function runCalc() {
    const data = readData();
    let firstInvalid = null;
    let hasAnyValue  = false;

    form.querySelectorAll('input[name], select[name]').forEach(el => {
      if (el.name.endsWith('_unit')) return;
      const raw   = el.value;
      if (raw !== '') hasAnyValue = true;
      const valid = validateField(el.name, raw, form);
      markInvalid(el, raw !== '' && !valid);
      if (!valid && raw !== '' && !firstInvalid) firstInvalid = el.name;
    });

    if (firstInvalid)  { showError('Invalid input ' + firstInvalid); return; }
    if (!hasAnyValue)  { hideResult(); return; }

    // Enforce required fields: Age, Sex, LDL-C
    const REQUIRED = { age: 'Age', gender: 'Sex', ldl_cholesterol: 'LDL-C' };
    for (const [field, label] of Object.entries(REQUIRED)) {
      const el = form.querySelector(`[name="${field}"]`);
      if (!el || el.value === '') { showError(`${label} is required`); return; }
    }

    const prob = calculateMLFHPEDS(data);
    showProbability(prob);
  }

  function showProbability(prob) {
    resultBox.classList.remove('result-box--hidden', 'result-box--error');
    resultBox.textContent = 'Likelihood of FH: ' + (prob * 100).toFixed(1) + '%';
  }

  function showError(msg) {
    resultBox.classList.remove('result-box--hidden');
    resultBox.classList.add('result-box--error');
    resultBox.textContent = msg;
  }

  function hideResult() {
    resultBox.classList.add('result-box--hidden');
    resultBox.classList.remove('result-box--error');
    resultBox.textContent = '';
  }

  form.addEventListener('input',  runCalc);
  form.addEventListener('change', runCalc);

  resetBtn.addEventListener('click', () => {
    form.reset();
    form.querySelectorAll('input[name], select[name]').forEach(el =>
      el.classList.remove('field__input--invalid', 'field__select--invalid')
    );
    hideResult();
  });

  runCalc();
}

/* ── Init ────────────────────────────────────────────────────*/

document.addEventListener('DOMContentLoaded', () => {
  setupForm();
  drawPRChart();
  drawWeightsChart();

  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { drawPRChart(); drawWeightsChart(); }, 120);
  });
});
