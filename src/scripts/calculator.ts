/**
 * Client-side wiring for the calculator page.
 *
 * Responsibilities:
 *  - hook up form inputs to global inference functions (loaded via classic
 *    <script> tags from /js/*.js)
 *  - swap the sidebar description when a field gains focus
 *  - toggle the mobile sidebar
 *
 * Markdown content is pre-rendered at build time into <template> elements
 * by DescriptionSidebar.astro, so there is **no runtime markdown / KaTeX
 * work** in the browser.
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
  if (!form || !resultBox || !resetBtn) return;

  /* ── Description sidebar ───────────────────────────────── */

  const sidebar = document.getElementById('desc-sidebar');
  const content = document.getElementById('desc-content');
  let activeField: string | null = null;

  // Build a map: field id → pre-rendered HTML (from <template> elements).
  const descTemplates = new Map<string, string>();
  if (sidebar) {
    sidebar.querySelectorAll<HTMLTemplateElement>('template[data-desc-id]').forEach((tpl) => {
      const id = tpl.getAttribute('data-desc-id');
      if (id) descTemplates.set(id, tpl.innerHTML);
    });
  }

  function renderDescription(targetEl: HTMLElement, html: string): void {
    // The compiled markdown starts with an <h1>. Extract it so we can render
    // it with a distinct style (large serif title).
    const tmp = document.createElement('div');
    tmp.innerHTML = html;
    const h1 = tmp.querySelector('h1');
    let titleHtml = '';
    if (h1) {
      titleHtml = `<h2 class="desc-panel__title">${h1.innerHTML}</h2>`;
      h1.remove();
    }
    targetEl.innerHTML = titleHtml + `<div class="desc-panel__body">${tmp.innerHTML}</div>`;
  }

  function setActive(fieldName: string): void {
    activeField = fieldName;
    form!.querySelectorAll<HTMLElement>('.field').forEach((f) => {
      f.classList.toggle('field--active', f.dataset.field === fieldName);
    });
    const html = descTemplates.get(fieldName);
    if (html && content) renderDescription(content, html);
  }

  // Show the intro (or first field) when the sidebar is first revealed.
  function showInitial(): void {
    if (!content || activeField) return;
    const initial = content.dataset.initial || 'intro';
    const html = descTemplates.get(initial) ?? descTemplates.values().next().value;
    if (html) renderDescription(content, html);
  }

  // Wire field focus → sidebar.
  form.querySelectorAll<HTMLElement>('.field').forEach((fieldEl) => {
    const name = fieldEl.dataset.field;
    if (!name) return;
    fieldEl
      .querySelectorAll<HTMLInputElement | HTMLSelectElement>('input, select')
      .forEach((inp) => {
        inp.addEventListener('focus', () => setActive(name));
      });
  });

  /* ── Mobile sidebar toggle (collapsed by default) ─────── */

  const toggleBtn = document.getElementById('desc-mobile-toggle') as HTMLButtonElement | null;
  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', () => {
      const expanded = toggleBtn.getAttribute('aria-expanded') === 'true';
      toggleBtn.setAttribute('aria-expanded', String(!expanded));
      if (expanded) {
        sidebar.setAttribute('hidden', '');
        toggleBtn.querySelector('span:first-child')!.textContent = 'Show field details';
      } else {
        sidebar.removeAttribute('hidden');
        toggleBtn.querySelector('span:first-child')!.textContent = 'Hide field details';
        showInitial();
      }
    });
  }

  // On desktop, sidebar visible by default.
  function syncSidebarVisibility(): void {
    if (!sidebar) return;
    if (window.matchMedia('(min-width: 981px)').matches) {
      sidebar.removeAttribute('hidden');
      showInitial();
    } else if (toggleBtn?.getAttribute('aria-expanded') !== 'true') {
      sidebar.setAttribute('hidden', '');
    }
  }
  syncSidebarVisibility();
  let rt: ReturnType<typeof setTimeout> | undefined;
  window.addEventListener('resize', () => {
    clearTimeout(rt);
    rt = setTimeout(syncSidebarVisibility, 120);
  });

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
    resultBox!.innerHTML = `<span class="result-box__label">Likelihood of FH</span>${(prob * 100).toFixed(1)}%`;
  }
  function showError(msg: string): void {
    resultBox!.classList.remove('result-box--hidden');
    resultBox!.classList.add('result-box--error');
    resultBox!.textContent = msg;
  }
  function hideResult(): void {
    resultBox!.classList.add('result-box--hidden');
    resultBox!.classList.remove('result-box--error');
    resultBox!.textContent = '';
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
