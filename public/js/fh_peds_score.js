/* ============================================================
   FH-PeDS  —  Semi-quantitative diagnostic score
   ============================================================ */

'use strict';

/* ── Scoring system ──────────────────────────────────────────
   Hard-coded transcription of the published FH-PeDS scoring
   table. Each component returns:
     {
       points:        number  // contribution to the running total
       missing:       boolean // true when the input was null/blank
                              //   and the row therefore could not
                              //   contribute. Distinguishing "missing"
                              //   from "0 points (no band matched)"
                              //   matters for the UI's progress state
                              //   and the per-component breakdown.

       // Lipid components (LDL-C, HDL-C, TAG) also return:
       band:          object | null  // the matched band record from
                              //   LDL_BANDS / HDL_BANDS / TAG_BANDS
                              //   (with { min, max, points }), or
                              //   null when the value sits below the
                              //   lowest band's minimum. The UI is
                              //   responsible for converting `min` /
                              //   `max` into the user's selected unit
                              //   and formatting the criterion line.

       // BMI and family-history components return:
       matched:       string  // human-readable description (unit-free
                              //   for BMI; descriptive for family
                              //   history). Empty string when no row
                              //   matched.
     }

   Units expected (model-native):
     ldl, hdl, tag    mmol/L     (page UI converts mg/dL → mmol/L)
     bmi_z_score     z-score
     fh_*            integer codes matching the form's <select> values

   Family history contributes ONLY ITS HIGHEST-SCORING category
   (footnote 3 of the published table). Any genetically-confirmed
   FH in a first-degree relative also rolls into that category at
   4 points.
──────────────────────────────────────────────────────────────*/

/* ── Cholesterol & TAG bands ─────────────────────────────────
   Bands are strict-lower / inclusive-upper to match the
   published "a < X ≤ b" notation. Thresholds in mmol/L drive the
   scoring; the optional minMg / maxMg fields carry the
   pre-rounded mg/dL values from the publication so the UI can
   display values that exactly match the reference table (rather
   than computing them from the conversion factor and getting a
   value off by ±0.5 mg/dL due to rounding). */

const LDL_BANDS = [
  { min: 6.5, max: Infinity, minMg: 251.3, maxMg: Infinity, points: 14 },
  { min: 4.8, max: 6.5, minMg: 185.8, maxMg: 251.3, points: 12 },
  { min: 3.8, max: 4.8, minMg: 146.9, maxMg: 185.8, points: 8 },
  { min: 3.0, max: 3.8, minMg: 116.0, maxMg: 146.9, points: 4 },
];

const HDL_BANDS = [
  { min: 2.2, max: Infinity, minMg: 85.1, maxMg: Infinity, points: -4 },
  { min: 1.4, max: 2.2, minMg: 54.1, maxMg: 85.1, points: -2 },
];

const TAG_BANDS = [
  { min: 4.5, max: Infinity, minMg: 398.6, maxMg: Infinity, points: -6 },
  { min: 3.5, max: 4.5, minMg: 310.5, maxMg: 398.6, points: -4 },
  { min: 2.0, max: 3.5, minMg: 177.2, maxMg: 310.5, points: -2 },
];

/* ── Below-lowest-band fallback ──────────────────────────────
   When a lipid value sits below the lowest band's minimum the
   scorer returns the band record below so the UI can still render
   a "X ≤ threshold" matched line without duplicating the lowest-
   band cut-off. */

const LDL_BELOW = { min: -Infinity, max: 3.0, minMg: -Infinity, maxMg: 116.0, points: 0 };
const HDL_BELOW = { min: -Infinity, max: 1.4, minMg: -Infinity, maxMg: 54.1, points: 0 };
const TAG_BELOW = { min: -Infinity, max: 2.0, minMg: -Infinity, maxMg: 177.2, points: 0 };

/** Pick the band whose `(min, max]` window contains `value`. */
function _matchBand(value, bands) {
  if (value === null || value === undefined || Number.isNaN(value)) return null;
  for (const band of bands) {
    if (value > band.min && value <= band.max) return band;
  }
  return null;
}

/* ── Interpretation buckets ──────────────────────────────────
   Score → diagnostic category. Boundaries follow the
   published table:
     < 6   Unlikely FH
     6–8   Possible FH (follow-up required)
     ≥ 9   Probable FH
──────────────────────────────────────────────────────────────*/

const BUCKETS = [
  { min: -Infinity, max: 6, label: 'Unlikely FH', id: 'unlikely' },
  { min: 6, max: 9, label: 'Possible FH (follow-up required)', id: 'possible' },
  { min: 9, max: Infinity, label: 'Probable FH', id: 'probable' },
];

/** Bucket whose `[min, max)` window contains `score`. */
function bucketForScore(score) {
  for (const b of BUCKETS) {
    if (score >= b.min && score < b.max) return b;
  }
  // Fallback — should be unreachable given the unbounded outer buckets.
  return BUCKETS[0];
}

/* ── Per-component scorers ───────────────────────────────────
   Each takes the relevant raw input and returns:
     {
       points:   number,         contribution to the total
       missing:  boolean,        true iff the input was null/blank
       band:     object | null,  matched band (lipid components only;
                                 may be the LDL_BELOW / HDL_BELOW /
                                 TAG_BELOW fallback)
       matched:  string,         human-readable description (used by
                                 BMI + family history; empty string for
                                 lipid components — the UI generates a
                                 unit-aware description from `band`).
     }
   Components that find no matching band still return points: 0 with
   matched: '' so the caller can sum blindly.
──────────────────────────────────────────────────────────────*/

function scoreLdl(ldl) {
  if (ldl === null || ldl === undefined) {
    return { points: 0, band: null, matched: '', missing: true };
  }
  const band = _matchBand(ldl, LDL_BANDS) || LDL_BELOW;
  return { points: band.points, band, matched: '', missing: false };
}

function scoreHdl(hdl) {
  if (hdl === null || hdl === undefined) {
    return { points: 0, band: null, matched: '', missing: true };
  }
  const band = _matchBand(hdl, HDL_BANDS) || HDL_BELOW;
  return { points: band.points, band, matched: '', missing: false };
}

function scoreTag(tag) {
  if (tag === null || tag === undefined) {
    return { points: 0, band: null, matched: '', missing: true };
  }
  const band = _matchBand(tag, TAG_BANDS) || TAG_BELOW;
  return { points: band.points, band, matched: '', missing: false };
}

function scoreBmi(bmiZ) {
  if (bmiZ === null || bmiZ === undefined) return { points: 0, matched: '', missing: true };
  if (bmiZ > 1.645) {
    return {
      points: -2,
      matched: 'BMI Z-score > 1.645 (> 95th percentile)',
      missing: false,
    };
  }
  return { points: 0, matched: 'BMI Z-score ≤ 1.645', missing: false };
}

/* ── Family history (winner-take-all) ────────────────────────
   Per the published footnote, ONLY ONE row of the family-history
   category may contribute — pick the maximum.

   Sub-questions (4-level, 0 = No, 1 = first-degree, 2 = second-
   degree, 3 = first AND second-degree):
     fh_high_cholesterol
     fh_premature_cad
     fh_acrus_senilis        arcus cornealis <45 years
     fh_xant                 tendinous xanthoma <45 years —
                             grouped with arcus under one row in
                             the published table, so the candidate
                             builder evaluates the two together at
                             each tier.
   Sub-questions (binary, 0 = No, 1 = Yes):
     fh_genetic              first-degree relative with
                             genetically-confirmed FH.

   Mapping table (input → points, label):
     genetic-FH first-degree                              → 4
     arcus / xanthoma first-degree                        → 4
     high-cholesterol or premature-CAD first-degree       → 2
     arcus / xanthoma second-degree                       → 2
     high-cholesterol or premature-CAD second-degree      → 1
──────────────────────────────────────────────────────────────*/

const FH_HC_FIRST = 'First-degree relative with high cholesterol';
const FH_HC_SECOND = 'Second-degree relative with high cholesterol';
const FH_CAD_FIRST = 'First-degree relative with premature coronary artery disease';
const FH_CAD_SECOND = 'Second-degree relative with premature coronary artery disease';
const FH_XANT_FIRST = 'First-degree relative with arcus cornealis <45 years / tendon xanthomas';
const FH_XANT_SECOND = 'Second-degree relative with arcus cornealis <45 years / tendon xanthomas';
const FH_GEN_FIRST = 'First-degree relative with genetically-confirmed FH';

/** Convenience: does a 0–3 family-history code include "first-degree"? */
function _hasFirstDegree(code) {
  return code === 1 || code === 3;
}
/** Convenience: does a 0–3 family-history code include "second-degree"? */
function _hasSecondDegree(code) {
  return code === 2 || code === 3;
}

function scoreFamilyHistory(raw) {
  const highChol = raw.fh_high_cholesterol; // 0–3
  const cad = raw.fh_premature_cad; // 0–3
  const arcus = raw.fh_acrus_senilis; // 0–3
  const xant = raw.fh_xant; // 0–3
  const gen = raw.fh_genetic; // 0/1

  // Treat null-only inputs as "missing" so the UI can warn that the
  // family-history row could not contribute. Any non-null input is
  // considered "answered" even if the value is 0/No.
  const allNull =
    (highChol === null || highChol === undefined) &&
    (cad === null || cad === undefined) &&
    (xant === null || xant === undefined) &&
    (arcus === null || arcus === undefined) &&
    (gen === null || gen === undefined);
  if (allNull) {
    return { points: 0, matched: '', missing: true, candidates: [] };
  }

  // Build the full list of candidate rows the user qualifies for,
  // then pick the highest-scoring one. We surface the full list in
  // the result panel so users can see *why* a particular row was
  // selected even when several apply.
  const candidates = [];

  if (gen === 1) {
    candidates.push({ points: 4, label: FH_GEN_FIRST });
  }
  // Arcus and xanthoma share the same scoring row in the published
  // table; both are 4-level codes here. A first-degree finding from
  // either earns the 4-point row; a second-degree finding from
  // either earns the 2-point row.
  if (_hasFirstDegree(arcus) || _hasFirstDegree(xant)) {
    candidates.push({ points: 4, label: FH_XANT_FIRST });
  }
  if (_hasSecondDegree(arcus) || _hasSecondDegree(xant)) {
    candidates.push({ points: 2, label: FH_XANT_SECOND });
  }
  if (_hasFirstDegree(highChol)) {
    candidates.push({ points: 2, label: FH_HC_FIRST });
  }
  if (_hasFirstDegree(cad)) {
    candidates.push({ points: 2, label: FH_CAD_FIRST });
  }
  if (_hasSecondDegree(highChol)) {
    candidates.push({ points: 1, label: FH_HC_SECOND });
  }
  if (_hasSecondDegree(cad)) {
    candidates.push({ points: 1, label: FH_CAD_SECOND });
  }

  if (candidates.length === 0) {
    return { points: 0, matched: 'No family-history criteria met', missing: false, candidates: [] };
  }

  // Highest-scoring candidate wins.
  let best = candidates[0];
  for (const c of candidates) {
    if (c.points > best.points) best = c;
  }
  return { points: best.points, matched: best.label, missing: false, candidates };
}

/* ── Entry point ─────────────────────────────────────────────
   Compute the full breakdown for a raw sample (model-native
   units, already produced by formSampleToRawSample). The
   returned object is shaped for direct consumption by the
   FH-PeDS Prediction panel:

     {
       components: [{ id, label, ...result }, ...],
       total: number,
       bucket: { id, label, ... },
       missing: number,           // count of components with no input
     }
──────────────────────────────────────────────────────────────*/

const COMPONENT_ORDER = [
  { id: 'ldl', label: 'LDL-C' },
  { id: 'hdl', label: 'HDL-C' },
  { id: 'tag', label: 'TAG' },
  { id: 'bmi', label: 'Body Mass Index' },
  { id: 'family_history', label: 'Family History' },
];

function calculateFHPEDS(raw) {
  const results = {
    ldl: scoreLdl(raw.ldl_cholesterol),
    hdl: scoreHdl(raw.hdl_cholesterol),
    tag: scoreTag(raw.tag),
    bmi: scoreBmi(raw.bmi_z_score),
    family_history: scoreFamilyHistory(raw),
  };

  const components = COMPONENT_ORDER.map(({ id, label }) => ({
    id,
    label,
    ...results[id],
  }));
  const total = components.reduce((sum, c) => sum + c.points, 0);
  const missing = components.filter((c) => c.missing).length;
  const bucket = bucketForScore(total);
  return { components, total, bucket, missing };
}

/* ── Browser globals ─────────────────────────────────────────
   Top-level `const` in a classic <script> does not attach to
   `window`, so re-expose explicitly. Mirrors model.js. */

if (typeof window !== 'undefined') {
  window.FH_PEDS = {
    calculateFHPEDS,
    bucketForScore,
    BUCKETS,
    COMPONENT_ORDER,
  };
}
