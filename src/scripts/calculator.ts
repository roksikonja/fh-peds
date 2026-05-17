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

  /* ── Workflow diagram state ───────────────────────────── */

  // Drive the highlight in WorkflowDiagram.astro by toggling a data attribute.
  // CSS handles the cross-fade between idle / filling / predicted via
  // `transition: … 600ms ease`.
  const workflow = document.getElementById('workflow-diagram');
  function setWorkflowState(state: '' | 'filling' | 'predicted'): void {
    if (!workflow) return;
    if (state === '') workflow.removeAttribute('data-state');
    else workflow.setAttribute('data-state', state);
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
    let hasAnyValue = false;

    form!
      .querySelectorAll<HTMLInputElement | HTMLSelectElement>('input[name], select[name]')
      .forEach((el) => {
        if (el.name.endsWith('_unit')) return;
        const raw = el.value;
        if (raw !== '') hasAnyValue = true;
        const valid = window.validateField(el.name, raw, form!);
        markInvalid(el, raw !== '' && !valid);
        if (!valid && raw !== '' && !firstInvalid) firstInvalid = el.name;
      });

    if (firstInvalid) {
      showError('Invalid input: ' + firstInvalid);
      return;
    }
    if (!hasAnyValue) {
      hideResult();
      return;
    }

    for (const [field, label] of Object.entries(REQUIRED)) {
      const el = form!.querySelector<HTMLInputElement | HTMLSelectElement>(`[name="${field}"]`);
      if (!el || el.value === '') {
        showError(`${label} is required`);
        return;
      }
    }

    const formSample = readFormSample();
    const raw = window.formSampleToRawSample(formSample);
    const prob = window.calculateMLFHPEDS(raw);
    showProbability(prob);
  }

  function showProbability(prob: number): void {
    resultBox!.classList.remove('result-box--hidden', 'result-box--error');
    resultBox!.innerHTML =
      `<span class="result-box__label">Likelihood of FH</span>` +
      `<span class="result-box__value">${(prob * 100).toFixed(1)}%</span>`;
    if (resultPlaceholder) resultPlaceholder.hidden = true;
    setWorkflowState('predicted');
  }
  function showError(msg: string): void {
    resultBox!.classList.remove('result-box--hidden');
    resultBox!.classList.add('result-box--error');
    resultBox!.innerHTML = `<span class="result-box__value">${msg}</span>`;
    if (resultPlaceholder) resultPlaceholder.hidden = true;
    // User is mid-flight providing inputs; keep "Measure" highlighted.
    setWorkflowState('filling');
  }
  function hideResult(): void {
    resultBox!.classList.add('result-box--hidden');
    resultBox!.classList.remove('result-box--error');
    resultBox!.textContent = '';
    if (resultPlaceholder) resultPlaceholder.hidden = false;
    setWorkflowState('');
  }

  form.addEventListener('input', runCalc);
  form.addEventListener('change', runCalc);
  resetBtn.addEventListener('click', () => {
    form.reset();
    form
      .querySelectorAll('input[name], select[name]')
      .forEach((el) => el.classList.remove('field__input--invalid', 'field__select--invalid'));
    hideResult();
  });

  runCalc();
}
