/**
 * Client-side wiring for the calculator page.
 *
 * Responsibilities:
 *  - hook up form inputs to global inference functions (loaded via classic
 *    <script> tags from /js/*.js)
 *  - expand the matching accordion entry in the left description sidebar
 *    when a form field gains focus
 *  - show / hide the result placeholder in the right column
 *
 * Markdown content is pre-rendered at build time into <details> blocks by
 * DescriptionSidebar.astro, so there is **no runtime markdown / KaTeX work**
 * in the browser.
 */

import { setupSidebarSync } from './sidebarSync';
import { ML_SAMPLE_PATIENTS } from './samplePatients';

// The inference globals come from /public/js/*.js loaded as classic scripts
// before this module runs. Declare them for TypeScript only.
declare global {
  interface Window {
    formSampleToRawSample: (s: Record<string, unknown>) => Record<string, number | null>;
    calculateMLFHPEDS: (raw: Record<string, number | null>) => number;
    validateField: (name: string, value: string | null, form?: HTMLFormElement) => boolean;
    MODEL: {
      operating_point: {
        threshold: number;
        target_specificity: number;
        cohort: string;
      };
    };
  }
}

const REQUIRED: Record<string, string> = {
  age: 'Age',
  gender: 'Sex',
  ldl_cholesterol: 'LDL-C',
};

const DEFAULT_UNITS: Record<string, string> = {
  hdl_cholesterol_unit: 'mmol/L',
  ldl_cholesterol_unit: 'mmol/L',
  total_cholesterol_unit: 'mmol/L',
  tag_unit: 'mmol/L',
  lp_a_unit: 'mg/L',
  bmi_unit: 'index',
};

// Example patients used by the "Patient X" / "Patient Y" prefill buttons.
// Defined in src/scripts/samplePatients.ts so both calculator pages share
// the same canonical patient definitions. The values correspond to the
// inference_samples.json fixture (probabilities ≈ 0.98 for X and ≈ 0.40
// for Y) once the form is populated.
const SAMPLE_PATIENTS = ML_SAMPLE_PATIENTS;

export function setupCalculator(): void {
  const form = document.getElementById('form-ml') as HTMLFormElement | null;
  const resultBox = document.getElementById('result-ml');
  const resetBtn = document.getElementById('reset-ml');
  const resultPlaceholder = document.getElementById('result-placeholder');
  if (!form || !resultBox || !resetBtn) return;

  /* ── Description accordion (left column) ──────────────── */

  setupSidebarSync(form);

  /* ── Workflow diagram pulse ───────────────────────────── */

  // Any form change triggers a single left-to-right colour sweep across the
  // three workflow steps. CSS keyframes do the actual animation; here we
  // just toggle `data-pulse` to start a new run. Re-setting the attribute
  // mid-animation forces a clean restart so rapid typing doesn't drop the
  // visual feedback.
  const workflow = document.getElementById('workflow-diagram');
  function pulseWorkflow(): void {
    if (!workflow) return;
    workflow.removeAttribute('data-pulse');
    // Force a reflow so the browser re-evaluates the animation from scratch.
    void (workflow as SVGElement).getBoundingClientRect();
    workflow.setAttribute('data-pulse', '');
  }

  /* ── Calculator core ──────────────────────────────────── */

  function readFormSample(): Record<string, unknown> {
    const v: Record<string, string> = {};
    form!
      .querySelectorAll<HTMLInputElement | HTMLSelectElement>('input[name], select[name]')
      .forEach((el) => {
        v[el.name] = el.value;
      });
    // Apply unit defaults so the inference layer sees a canonical shape.
    const sample: Record<string, unknown> = { ...v };
    for (const [k, def] of Object.entries(DEFAULT_UNITS)) {
      sample[k] = v[k] || def;
    }
    return sample;
  }

  function markInvalid(el: Element, invalid: boolean): void {
    el.classList.toggle('field__input--invalid', invalid);
    el.classList.toggle('field__select--invalid', invalid);
  }

  function runCalc(): void {
    let firstInvalid: string | null = null;

    form!
      .querySelectorAll<HTMLInputElement | HTMLSelectElement>('input[name], select[name]')
      .forEach((el) => {
        if (el.name.endsWith('_unit')) return;
        const raw = el.value;
        const valid = window.validateField(el.name, raw, form!);
        markInvalid(el, raw !== '' && !valid);
        if (!valid && raw !== '' && !firstInvalid) firstInvalid = el.name;
      });

    if (firstInvalid) {
      showError('Invalid input: ' + firstInvalid);
      return;
    }

    // Count how many of the three required fields are filled. If any are
    // missing we render a progress ring instead of computing a probability.
    let requiredFilled = 0;
    for (const field of Object.keys(REQUIRED)) {
      const el = form!.querySelector<HTMLInputElement | HTMLSelectElement>(`[name="${field}"]`);
      if (el && el.value !== '') requiredFilled += 1;
    }
    if (requiredFilled < Object.keys(REQUIRED).length) {
      showProgress(requiredFilled, Object.keys(REQUIRED).length);
      return;
    }

    const formSample = readFormSample();
    const raw = window.formSampleToRawSample(formSample);
    const prob = window.calculateMLFHPEDS(raw);
    showProbability(prob);
  }

  function showProbability(prob: number): void {
    // Clinical decision threshold: see MODEL.operating_point in
    // public/js/model.js (mirrors model.json).
    const op = window.MODEL.operating_point;
    const isHigh = prob >= op.threshold;
    const verdictText = isHigh ? 'Yes' : 'No';
    const stateMod = isHigh ? 'result-box--high' : 'result-box--low';
    const probPct = prob * 100;
    const probLabel = probPct.toFixed(1);
    const thrPct = op.threshold * 100;
    const thrLabel = Math.round(thrPct).toString();
    const specPct = (op.target_specificity * 100).toFixed(0);

    resultBox!.className = `result-box ${stateMod}`;
    resultBox!.innerHTML =
      `<section class="result-block">` +
      `<h3 class="result-block__label">Familial Hypercholesterolemia</h3>` +
      `<p class="result-block__verdict">${verdictText}</p>` +
      `</section>` +
      `<section class="result-block">` +
      `<h3 class="result-block__label">Estimated Likelihood</h3>` +
      `<div class="result-gauge" ` +
      `role="img" aria-label="${probLabel}% likelihood; threshold ${thrLabel}%">` +
      `<div class="result-gauge__track">` +
      `<div class="result-gauge__fill" style="width: ${probPct}%"></div>` +
      `<div class="result-gauge__marker" style="left: ${probPct}%">` +
      `<span class="result-gauge__marker-pill">${probLabel}%</span>` +
      `</div>` +
      `<div class="result-gauge__tick" style="left: ${thrPct}%">` +
      `<span class="result-gauge__tick-label result-gauge__tick-label--top">` +
      `Threshold ${thrLabel}%` +
      `</span>` +
      `</div>` +
      `</div>` +
      `<div class="result-gauge__zones">` +
      `<span class="result-gauge__zone result-gauge__zone--low" ` +
      `style="width: ${thrPct}%">FH unlikely</span>` +
      `<span class="result-gauge__zone result-gauge__zone--high" ` +
      `style="width: ${100 - thrPct}%">FH likely</span>` +
      `</div>` +
      `</div>` +
      `<p class="result-block__hint">` +
      `The threshold (${thrLabel}%) for ML-FH-PeDS is selected such that the model ` +
      `achieves ${specPct}% specificity on the testing Slovenian cohort.` +
      `</p>` +
      `</section>`;
    if (resultPlaceholder) resultPlaceholder.hidden = true;
  }
  function showError(msg: string): void {
    resultBox!.className = 'result-box result-box--error';
    resultBox!.innerHTML =
      `<section class="result-block">` +
      `<h3 class="result-block__label">Cannot compute</h3>` +
      `<p class="result-block__verdict result-block__verdict--error">${msg}</p>` +
      `</section>`;
    if (resultPlaceholder) resultPlaceholder.hidden = true;
  }
  function showProgress(filled: number, total: number): void {
    // Pre-prediction state. Use the same horizontal bar as the likelihood
    // gauge so the layout is consistent: a neutral track with a blue fill
    // covering (filled / total) of the width. We deliberately omit the
    // threshold tick (no probability yet) and label the bar with the
    // required-field count rather than a percentage.
    const fillPct = (filled / total) * 100;
    resultBox!.className = 'result-box result-box--progress';
    resultBox!.innerHTML =
      `<section class="result-block">` +
      `<h3 class="result-block__label">Estimated Likelihood</h3>` +
      `<div class="result-gauge" ` +
      `role="img" aria-label="${filled} of ${total} required fields filled">` +
      `<div class="result-gauge__track result-gauge__track--bare">` +
      `<div class="result-gauge__fill" style="width: ${fillPct}%"></div>` +
      `</div>` +
      // Single centred label. The original "0/N … middle … N/N"
      // triplet was redundant once the bar itself shows the
      // progress, so the gauge keeps only the live count.
      `<div class="result-gauge__labels result-gauge__labels--single">` +
      `<span class="result-gauge__value">${filled}/${total} required fields</span>` +
      `</div>` +
      `</div>` +
      `</section>`;
    // Keep the placeholder visible underneath the bar as the explanatory
    // caption ("Fill in the required fields…").
    if (resultPlaceholder) resultPlaceholder.hidden = false;
  }
  function onFormChange(): void {
    pulseWorkflow();
    runCalc();
  }
  form.addEventListener('input', onFormChange);
  form.addEventListener('change', onFormChange);
  resetBtn.addEventListener('click', () => {
    form.reset();
    form
      .querySelectorAll('input[name], select[name]')
      .forEach((el) => el.classList.remove('field__input--invalid', 'field__select--invalid'));
    // Re-run the calculator so the empty-state progress ring (0/3) is shown,
    // matching the initial page-load state instead of leaving the panel blank.
    runCalc();
  });

  /* ── Patient prefill buttons (next to the "Form" title) ── */

  function prefill(values: Record<string, string>): void {
    form!
      .querySelectorAll<HTMLInputElement | HTMLSelectElement>('input[name], select[name]')
      .forEach((el) => {
        if (Object.prototype.hasOwnProperty.call(values, el.name)) {
          el.value = values[el.name];
        } else {
          el.value = '';
        }
        el.classList.remove('field__input--invalid', 'field__select--invalid');
      });
    runCalc();
  }

  document.querySelectorAll<HTMLButtonElement>('[data-prefill]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const key = btn.getAttribute('data-prefill');
      if (!key) return;
      const sample = SAMPLE_PATIENTS[key];
      if (sample) prefill(sample);
    });
  });

  runCalc();
}
