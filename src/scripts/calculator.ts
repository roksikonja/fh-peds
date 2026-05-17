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

/**
 * Example patients used by the "Patient X" / "Patient Y" prefill buttons.
 *
 * Values are taken from the canonical inference_samples.json fixture and are
 * expressed in the form's default units (mmol/L for cholesterols & TAG, mg/L
 * for Lp(a), z-score for BMI). The model probabilities they correspond to —
 * roughly 98 % for Patient X and ~40 % for Patient Y — are reproduced by the
 * inference layer once the form is populated.
 */
const SAMPLE_PATIENTS: Record<string, Record<string, string>> = {
  // High-likelihood case: moderately elevated LDL with first-degree FH of
  // high cholesterol and premature CAD, plus high Lp(a) —
  // inference_samples.json probability ≈ 0.98.
  patientX: {
    age: '14.3',
    gender: '1',
    ldl_cholesterol: '6.2',
    ldl_cholesterol_unit: 'mmol/L',
    fh_high_cholesterol: '1',
    fh_premature_cad: '1',
    fh_pad_cvi: '0',
    fh_xant: '0',
    fh_acrus_senilis: '0',
    total_cholesterol: '7.8',
    total_cholesterol_unit: 'mmol/L',
    hdl_cholesterol: '1.2',
    hdl_cholesterol_unit: 'mmol/L',
    tag: '0.7',
    tag_unit: 'mmol/L',
    lp_a: '591',
    lp_a_unit: 'mg/L',
    bmi: '1.07',
    bmi_unit: 'z-score',
  },
  // Intermediate-likelihood case: borderline LDL with second-degree FH of
  // high cholesterol — inference_samples.json probability ≈ 0.40.
  patientY: {
    age: '5.7',
    gender: '1',
    ldl_cholesterol: '4.3',
    ldl_cholesterol_unit: 'mmol/L',
    fh_high_cholesterol: '2',
    fh_premature_cad: '0',
    fh_pad_cvi: '0',
    fh_xant: '0',
    fh_acrus_senilis: '0',
    total_cholesterol: '6.0',
    total_cholesterol_unit: 'mmol/L',
    hdl_cholesterol: '1.3',
    hdl_cholesterol_unit: 'mmol/L',
    tag: '0.9',
    tag_unit: 'mmol/L',
    lp_a: '214',
    lp_a_unit: 'mg/L',
    bmi: '0.77',
    bmi_unit: 'z-score',
  },
};

export function setupCalculator(): void {
  const form = document.getElementById('form-ml') as HTMLFormElement | null;
  const resultBox = document.getElementById('result-ml');
  const resetBtn = document.getElementById('reset-ml');
  const resultPlaceholder = document.getElementById('result-placeholder');
  if (!form || !resultBox || !resetBtn) return;

  /* ── Description accordion (left column) ──────────────── */

  const sidebar = document.getElementById('desc-sidebar');

  // Map: field id → <details> element in the accordion.
  const descItems = new Map<string, HTMLDetailsElement>();
  if (sidebar) {
    sidebar.querySelectorAll<HTMLElement>('.desc-accordion__item').forEach((li) => {
      const id = li.getAttribute('data-desc-id');
      const details = li.querySelector<HTMLDetailsElement>('details');
      if (id && details) descItems.set(id, details);
    });
  }

  function setActive(fieldName: string): void {
    // Highlight the field in the form.
    form!.querySelectorAll<HTMLElement>('.field').forEach((f) => {
      f.classList.toggle('field--active', f.dataset.field === fieldName);
    });
    // Highlight + open the matching accordion entry (collapsing any other
    // previously-opened entry) and scroll it into view within the sidebar.
    descItems.forEach((details, id) => {
      const li = details.parentElement;
      const active = id === fieldName;
      li?.classList.toggle('desc-accordion__item--active', active);
      if (active) {
        details.open = true;
      } else if (details.open) {
        details.open = false;
      }
    });
  }

  // Wire field focus → expand matching accordion entry.
  form.querySelectorAll<HTMLElement>('.field').forEach((fieldEl) => {
    const name = fieldEl.dataset.field;
    if (!name) return;
    fieldEl
      .querySelectorAll<HTMLInputElement | HTMLSelectElement>('input, select')
      .forEach((inp) => {
        inp.addEventListener('focus', () => setActive(name));
      });
  });

  // Clicking an accordion summary should also highlight the matching form
  // field, so the visual link between left and centre columns is bi-directional.
  descItems.forEach((details, id) => {
    details.addEventListener('toggle', () => {
      if (!details.open) return;
      form!.querySelectorAll<HTMLElement>('.field').forEach((f) => {
        f.classList.toggle('field--active', f.dataset.field === id);
      });
      details.parentElement?.classList.add('desc-accordion__item--active');
    });
  });

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
      `<div class="result-gauge__tick" style="left: ${thrPct}%">` +
      `<span class="result-gauge__tick-label">${thrLabel}%</span>` +
      `</div>` +
      `</div>` +
      `<div class="result-gauge__labels">` +
      `<span>0%</span>` +
      `<span class="result-gauge__value">${probLabel}%</span>` +
      `<span>100%</span>` +
      `</div>` +
      `</div>` +
      `<p class="result-block__hint">` +
      `The threshold is selected such that the model achieves ${specPct}% ` +
      `specificity on the testing Slovenian cohort.` +
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
    // SVG progress ring: a neutral track with a blue arc covering
    // (filled / total) of the circumference. The arc starts at the top
    // (12 o'clock) and grows clockwise.
    const r = 38;
    const circ = 2 * Math.PI * r;
    const dash = (filled / total) * circ;
    const ringSvg =
      `<svg class="result-progress__ring" viewBox="0 0 100 100" ` +
      `role="img" aria-label="${filled} of ${total} required fields filled">` +
      `<circle class="result-progress__track" cx="50" cy="50" r="${r}"></circle>` +
      `<circle class="result-progress__fill" cx="50" cy="50" r="${r}" ` +
      `stroke-dasharray="${dash.toFixed(3)} ${circ.toFixed(3)}"></circle>` +
      `<text class="result-progress__count" x="50" y="50" ` +
      `text-anchor="middle" dominant-baseline="central">${filled}/${total}</text>` +
      `</svg>`;
    resultBox!.className = 'result-box result-box--progress';
    resultBox!.innerHTML = `<div class="result-progress">${ringSvg}</div>`;
    // Keep the placeholder visible underneath the ring as the explanatory
    // caption ("Fill in the required fields…").
    if (resultPlaceholder) resultPlaceholder.hidden = false;
  }
  function hideResult(): void {
    resultBox!.className = 'result-box result-box--hidden';
    resultBox!.textContent = '';
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
    hideResult();
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
