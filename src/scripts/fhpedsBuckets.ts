/**
 * FH-PeDS interpretation buckets — display copy.
 *
 * The numeric ranges and bucket IDs are duplicated from
 * public/js/fh_peds_score.js (`BUCKETS` constant). The JS module owns the
 * canonical thresholds because the same data is consumed by the
 * runtime calculator in the browser; this TypeScript module exposes the
 * same buckets for build-time consumers (Astro components rendering the
 * static interpretation key in the Prediction panel).
 *
 * Keeping the two definitions in sync is checked manually for now; both
 * are short, public, and unlikely to drift.
 */

export interface FhpedsBucketDisplay {
  id: 'unlikely' | 'possible' | 'probable';
  /** Score-range label, e.g. "6 ≤ FH-PeDS < 9". */
  range: string;
  /** Human-readable label, e.g. "Possible FH (follow-up required)". */
  label: string;
}

// Ordered top-to-bottom as they appear in the Interpretation key:
// most-severe first so the user reads from "worst case" downwards.
// The underlying scoring buckets in public/js/fh_peds_score.js are
// ordered ascending by score; the display order is purely a UI concern.
export const FHPEDS_BUCKETS: readonly FhpedsBucketDisplay[] = [
  { id: 'probable', range: 'FH-PeDS ≥ 9', label: 'Probable FH' },
  { id: 'possible', range: '6 ≤ FH-PeDS < 9', label: 'Possible FH (Follow-up required)' },
  { id: 'unlikely', range: 'FH-PeDS < 6', label: 'Unlikely FH' },
];
