/**
 * Client-side wiring for the FH-PeDS calculator page (/fh-peds).
 *
 * Mirrors src/scripts/calculator.ts (the ML-FH-PeDS page) but renders the
 * semi-quantitative FH-PeDS scoring breakdown instead of the model's
 * probability:
 *
 *   1. Validate each field individually using validateField from
 *      /js/preprocessing.js (same min/max guards as the ML form).
 *   2. If LDL-C is missing (the single required field), show a progress
 *      ring counting how many required fields are filled.
 *   3. Otherwise convert the form sample into model-native units via
 *      formSampleToRawSample, then call calculateFHPEDS to obtain the
 *      per-component breakdown + total + interpretation bucket.
 *   4. Render a Verdict block ("Unlikely / Possible / Probable FH"), a
 *      summation block (per-component points with their matched
 *      criterion), and highlight the matching row + bucket in the
 *      scoring table below.
 */

import { setupSidebarSync } from './sidebarSync';
import { FHPEDS_SAMPLE_PATIENTS } from './samplePatients';

// Inference globals from /public/js/*.js — same shape as the ML page.
declare global {
  interface Window {
    formSampleToRawSample: (s: Record<string, unknown>) => Record<string, number | null>;
    validateField: (name: string, value: string | null, form?: HTMLFormElement) => boolean;
    FH_PEDS: {
      calculateFHPEDS: (raw: Record<string, number | null>) => FhpedsResult;
      bucketForScore: (score: number) => FhpedsBucket;
      BUCKETS: FhpedsBucket[];
      COMPONENT_ORDER: { id: string; label: string }[];
    };
    UNIT_CONVERSIONS: {
      /** mg/dL value of 1 mmol/L for cholesterol (LDL/HDL/TC). */
      CHOL_MGDL_PER_MMOLL: number;
      /** mg/dL value of 1 mmol/L for triglycerides. */
      TAG_MGDL_PER_MMOLL: number;
      /** mg/L value of 1 nmol/L for Lp(a). */
      LPA_MGL_PER_NMOLL: number;
    };
  }
}

interface FhpedsBucket {
  id: 'unlikely' | 'possible' | 'probable';
  label: string;
  min: number;
  max: number;
}
/**
 * Lipid band record. The primary thresholds (`min`/`max`) are always
 * in mmol/L and drive the scoring; the optional `minMg`/`maxMg`
 * fields carry the pre-rounded mg/dL values from the publication so
 * the UI can display values that match the reference table exactly.
 */
interface FhpedsBand {
  min: number;
  max: number;
  minMg?: number;
  maxMg?: number;
  points: number;
}
interface FhpedsComponent {
  id: 'ldl' | 'hdl' | 'tag' | 'bmi' | 'family_history';
  label: string;
  points: number;
  /**
   * Free-text matched-criterion description from the scorer. Empty
   * string for lipid components (LDL/HDL/TAG) — those carry a
   * structured `band` instead and the UI generates a unit-aware
   * description on the fly so the score breakdown displays only the
   * unit the user picked in the form.
   */
  matched: string;
  /** Lipid components only: structured band info for UI formatting. */
  band?: FhpedsBand | null;
  missing: boolean;
}
interface FhpedsResult {
  components: FhpedsComponent[];
  total: number;
  bucket: FhpedsBucket;
  missing: number;
}

// Required fields gate the calculation. LDL-C is the only field that
// actually contributes to the FH-PeDS score independent of other
// inputs; Age and Sex are needed for the BMI z-score lookup whenever
// the user enters BMI as a raw "index". We include all three in the
// progress UI so the right column shows "x/3" — matching the visual
// `*` markers on the form and the ML page's behaviour.
const REQUIRED: Record<string, string> = {
  age: 'Age',
  gender: 'Sex',
  ldl_cholesterol: 'LDL-C',
};

const DEFAULT_UNITS: Record<string, string> = {
  hdl_cholesterol_unit: 'mmol/L',
  ldl_cholesterol_unit: 'mmol/L',
  tag_unit: 'mmol/L',
  bmi_unit: 'index',
};

/* The scoring reference table (below the calculator shell) is
   intentionally static — it serves as a published-table reference
   for clinicians and does not react to the inputs. The only
   dynamic highlight on the page is the bucket cell in the
   Interpretation key inside the Prediction panel. */

/** Per-lipid units selected by the user in the form. */
interface LipidUnits {
  ldl: string;
  hdl: string;
  tag: string;
}

/**
 * Format a lipid component's matched-criterion text using the unit
 * the user selected in the form. Thresholds in the band record are
 * always in mmol/L; we convert to mg/dL when needed so the score
 * breakdown shows one unit only — the one already on screen in the
 * input row.
 */
function formatLipidMatched(component: FhpedsComponent, units: LipidUnits): string {
  if (!component.band) return '';
  let measurement: string;
  let userUnit: string;
  let perMmoll: number;
  switch (component.id) {
    case 'ldl':
      measurement = 'LDL-C';
      userUnit = units.ldl;
      perMmoll = window.UNIT_CONVERSIONS.CHOL_MGDL_PER_MMOLL;
      break;
    case 'hdl':
      measurement = 'HDL-C';
      userUnit = units.hdl;
      perMmoll = window.UNIT_CONVERSIONS.CHOL_MGDL_PER_MMOLL;
      break;
    case 'tag':
      measurement = 'TAG';
      userUnit = units.tag;
      perMmoll = window.UNIT_CONVERSIONS.TAG_MGDL_PER_MMOLL;
      break;
    default:
      return '';
  }
  // Resolve the threshold value the user should see for this
  // (band, unit) pair. For mmol/L we use the canonical scoring
  // thresholds directly. For mg/dL we prefer the publication's
  // pre-rounded value (`minMg` / `maxMg`) so the score-breakdown
  // numbers exactly match the reference table; we only fall back to
  // a runtime conversion if the band omits the pre-computed value.
  const { min, max, minMg, maxMg } = component.band;
  const resolveMin = (): number => (userUnit === 'mg/dL' ? (minMg ?? min * perMmoll) : min);
  const resolveMax = (): number => (userUnit === 'mg/dL' ? (maxMg ?? max * perMmoll) : max);
  const display = (v: number): string => {
    if (userUnit === 'mg/dL') return v.toFixed(1);
    // Trim trailing ".0" from whole-number thresholds (e.g. "3.0" →
    // "3") so mmol/L values read cleanly; otherwise keep one decimal.
    return Number(v.toFixed(2)).toString();
  };

  const minV = resolveMin();
  const maxV = resolveMax();
  if (maxV === Infinity) {
    return `${measurement} > ${display(minV)} ${userUnit}`;
  }
  if (minV === -Infinity) {
    return `${measurement} ≤ ${display(maxV)} ${userUnit}`;
  }
  return `${display(minV)} ${userUnit} < ${measurement} ≤ ${display(maxV)} ${userUnit}`;
}

/* ── Entry point ─────────────────────────────────────────────*/

export function setupFhpedsCalculator(): void {
  const form = document.getElementById('form-fhpeds') as HTMLFormElement | null;
  const resultBox = document.getElementById('result-fhpeds');
  const resetBtn = document.getElementById('reset-fhpeds');
  // The Score block in the Prediction panel has two mutually
  // exclusive modes, driven by `setInterpretationMode`:
  //   'progress'  some or all required fields still empty — show
  //               the gauge inside [data-progress-slot] plus the
  //               "Fill in the required fields…" caption. This is
  //               the default state on page load (0/3 filled).
  //   'score'     all required fields filled — show the static
  //               3-cell bucket key with the matching cell
  //               highlighted and the score injected.
  // The scoring reference table further down the page stays
  // static — it is a published-table reference, not live feedback.
  const interpretationBlock = document.getElementById('fhpeds-interpretation');
  const progressBlock = document.getElementById('fhpeds-progress');
  const progressSlot = progressBlock?.querySelector<HTMLElement>('[data-progress-slot]') ?? null;
  const interpretationKey = document.getElementById('fhpeds-key-list');
  if (!form || !resultBox || !resetBtn) return;

  setupSidebarSync(form);

  /* ── Highlight helpers ───────────────────────────────── */

  function clearHighlights(): void {
    if (!interpretationKey) return;
    interpretationKey.querySelectorAll('[data-bucket]').forEach((el) => {
      el.classList.remove('scoring-table__bucket--active');
    });
    // Wipe any score badges injected into a previously-active cell.
    interpretationKey.querySelectorAll<HTMLElement>('[data-score-slot]').forEach((el) => {
      el.textContent = '';
    });
  }

  function setInterpretationMode(mode: 'progress' | 'score'): void {
    if (interpretationBlock) interpretationBlock.hidden = false;
    if (progressBlock) progressBlock.hidden = mode !== 'progress';
    if (interpretationKey) interpretationKey.hidden = mode !== 'score';
  }

  function applyHighlights(bucketId: string | null, scoreLabel: string | null): void {
    clearHighlights();
    if (!bucketId || !interpretationKey) return;
    const cell = interpretationKey.querySelector(`[data-bucket="${bucketId}"]`);
    if (!cell) return;
    cell.classList.add('scoring-table__bucket--active');
    // Write the score badge into the active cell's slot, merging
    // the "+N points" total with the bucket key it belongs to.
    if (scoreLabel) {
      const slot = cell.querySelector<HTMLElement>('[data-score-slot]');
      if (slot) slot.textContent = `${scoreLabel} points`;
    }
  }

  /* ── Core ────────────────────────────────────────────── */

  function readFormSample(): Record<string, unknown> {
    const v: Record<string, string> = {};
    form!
      .querySelectorAll<HTMLInputElement | HTMLSelectElement>('input[name], select[name]')
      .forEach((el) => {
        v[el.name] = el.value;
      });
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

    // Required-field gate. LDL-C must be filled before we can render a
    // score — without it the entire FH-PeDS calculation is meaningless.
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
    // Pull the genetic-FH flag through manually — formSampleToRawSample
    // doesn't know about it (it's specific to FH-PeDS).
    const rawWithGenetic = {
      ...raw,
      fh_genetic: parseFhGenetic(formSample.fh_genetic),
    };
    const result = window.FH_PEDS.calculateFHPEDS(rawWithGenetic);

    // Capture the per-lipid unit picks so showScore() can format the
    // matched-criterion text using only the unit the user chose. We
    // do this here (rather than inside showScore) so that the
    // showScore signature stays decoupled from the form DOM.
    const units: LipidUnits = {
      ldl: String(formSample.ldl_cholesterol_unit ?? 'mmol/L'),
      hdl: String(formSample.hdl_cholesterol_unit ?? 'mmol/L'),
      tag: String(formSample.tag_unit ?? 'mmol/L'),
    };
    showScore(result, units);
  }

  function parseFhGenetic(v: unknown): number | null {
    if (v === '' || v === null || v === undefined) return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }

  function showScore(result: FhpedsResult, units: LipidUnits): void {
    const bucketMod = `result-box--bucket-${result.bucket.id}`;
    resultBox!.className = `result-box ${bucketMod}`;

    const totalLabel = formatSigned(result.total);

    // Component summation: "14 + (-2) + 4 + 0 + 2 = 18". Components
    // that are missing (no input) get rendered as "—" so the user can
    // see what they could still fill in to influence the score.
    const termsHtml = result.components
      .map((c) => {
        const ariaLabel = `${c.label}: ${c.missing ? 'missing' : formatSigned(c.points)} points`;
        const display = c.missing ? '—' : formatSigned(c.points);
        const stateClass = c.missing
          ? 'fhpeds-term--missing'
          : c.points !== 0
            ? 'fhpeds-term--scoring'
            : '';
        // For lipid components the matched-criterion text is generated
        // on the fly so we can show only the user-selected unit (the
        // scorer leaves c.matched empty for these). BMI and
        // family-history components still use the scorer's matched
        // string as-is.
        let matchedText = c.matched;
        if (!matchedText && (c.id === 'ldl' || c.id === 'hdl' || c.id === 'tag')) {
          matchedText = formatLipidMatched(c, units);
        }
        const matchedHtml = matchedText
          ? escapeHtml(matchedText)
          : c.missing
            ? 'not provided'
            : '—';
        return (
          `<li class="fhpeds-term ${stateClass}" data-component="${c.id}" aria-label="${escapeHtml(ariaLabel)}">` +
          `<span class="fhpeds-term__label">${escapeHtml(c.label)}</span>` +
          `<span class="fhpeds-term__points">${display}</span>` +
          `<span class="fhpeds-term__matched">${matchedHtml}</span>` +
          `</li>`
        );
      })
      .join('');

    // Sticky note about partial data — useful for clinicians who fill
    // out only the required field. The bucket itself is still computed
    // from whatever was provided.
    const missingHint =
      result.missing > 0
        ? `<p class="result-block__hint">${result.missing} of ${result.components.length} components have no input — fill them in to refine the score.</p>`
        : '';

    // Dynamic result content is now just the score breakdown. The
    // bucket verdict + score total live inside the active cell of
    // the static interpretation key above this block; updating that
    // is the applyHighlights call below.
    resultBox!.innerHTML =
      `<section class="result-block">` +
      `<h3 class="result-block__label">Score breakdown</h3>` +
      `<ul class="fhpeds-terms" role="list">${termsHtml}</ul>` +
      missingHint +
      `</section>`;

    // Score mode: hide the progress prompt + bar, reveal the bucket
    // cells, and highlight the matching one.
    setInterpretationMode('score');
    applyHighlights(result.bucket.id, totalLabel);
  }

  function showError(msg: string): void {
    resultBox!.className = 'result-box result-box--error';
    resultBox!.innerHTML =
      `<section class="result-block">` +
      `<h3 class="result-block__label">Cannot compute</h3>` +
      `<p class="result-block__verdict result-block__verdict--error">${escapeHtml(msg)}</p>` +
      `</section>`;
    // Treat invalid input the same as "not enough input yet" for the
    // Interpretation block: show the progress prompt, no bucket
    // highlighted. The error itself sits in #result-fhpeds below.
    setInterpretationMode('progress');
    clearHighlights();
    if (progressSlot) progressSlot.innerHTML = '';
  }

  function showProgress(filled: number, total: number): void {
    // Pre-score state. Same horizontal progress bar recipe used on
    // the ML-FH-PeDS page (see calculator.ts → showProgress) so the
    // two calculators look consistent. Layout: bar on top, then a
    // labels row reading "0/N    required fields    N/N", then the
    // "Fill in…" caption underneath (rendered by the Astro
    // template). The middle label is a static caption rather than
    // a duplicate of the left value.
    const fillPct = (filled / total) * 100;
    if (progressSlot) {
      progressSlot.innerHTML =
        `<div class="result-gauge" ` +
        `role="img" aria-label="${filled} of ${total} required fields filled">` +
        `<div class="result-gauge__track result-gauge__track--bare">` +
        `<div class="result-gauge__fill" style="width: ${fillPct}%"></div>` +
        `</div>` +
        // Single centred label with the live count, matching the
        // ML page's gauge ("0/3 required fields" → "1/3 required
        // fields" → …). The original "0/N … middle … N/N"
        // triplet was redundant once the bar itself shows the
        // progress, so we keep only this central label.
        `<div class="result-gauge__labels result-gauge__labels--single">` +
        `<span class="result-gauge__value">${filled}/${total} required fields</span>` +
        `</div>` +
        `</div>`;
    }
    // Clear the score breakdown — there's no score yet.
    resultBox!.className = 'result-box result-box--hidden';
    resultBox!.innerHTML = '';
    // The progress UI is shown the whole time the user is still
    // filling in required fields, including at 0/N on page load —
    // that's how the user knows what action to take. The bucket
    // key appears only once a score is available.
    setInterpretationMode('progress');
    clearHighlights();
  }

  /* ── Helpers ─────────────────────────────────────────── */

  function formatSigned(n: number): string {
    if (n > 0) return `+${n}`;
    if (n < 0) return `−${Math.abs(n)}`;
    return '0';
  }

  function escapeHtml(s: string): string {
    return s
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /* ── Event wiring ────────────────────────────────────── */

  function onFormChange(): void {
    runCalc();
  }
  form.addEventListener('input', onFormChange);
  form.addEventListener('change', onFormChange);
  resetBtn.addEventListener('click', () => {
    form.reset();
    form
      .querySelectorAll('input[name], select[name]')
      .forEach((el) => el.classList.remove('field__input--invalid', 'field__select--invalid'));
    runCalc();
  });

  /* ── Patient prefill chips ───────────────────────────── */

  // Loads a SamplePatient object into the form. Fields the patient
  // object does not specify get cleared so consecutive clicks on
  // Patient X then Patient Y don't leave stale values behind. Mirrors
  // the prefill helper on the ML calculator page.
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
      const sample = FHPEDS_SAMPLE_PATIENTS[key];
      if (sample) prefill(sample);
    });
  });

  // Initial render so the progress state appears on page load.
  runCalc();
}
