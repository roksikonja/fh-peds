/**
 * Canonical example-patient prefills used by the "Patient ➕ / ➖"
 * example chips on both calculator pages.
 *
 * The ML-FH-PeDS page (/) reuses the values verbatim — they correspond
 * to the inference_samples.json fixture and reproduce probabilities of
 * roughly 98% (Patient X) and ~40% (Patient Y) once the form is
 * populated.
 *
 * The FH-PeDS page (/fh-peds) reuses Patient X verbatim but tweaks
 * Patient Y so it lands firmly in the "Unlikely FH" bucket (FH-PeDS
 * score < 6). Without that tweak Patient Y would score 9 points on
 * the published FH-PeDS table — exactly at the "Probable FH"
 * boundary — which would make both example chips show the same
 * verdict and defeat the purpose of having a positive + negative pair.
 *
 * Each patient is a flat string→string map keyed by form-field
 * `name` attributes. Fields the host form does not expose are simply
 * ignored by the prefill helper (it walks the form's own inputs
 * rather than the patient object).
 */

export type SamplePatient = Readonly<Record<string, string>>;

/* ── ML-FH-PeDS canonical patients ───────────────────────────
   Values are taken from the inference_samples.json fixture and
   expressed in the form's default units (mmol/L for cholesterols
   & TAG, mg/L for Lp(a), z-score for BMI). */

const ML_PATIENT_X: SamplePatient = {
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
};

const ML_PATIENT_Y: SamplePatient = {
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
};

export const ML_SAMPLE_PATIENTS: Readonly<Record<string, SamplePatient>> = {
  patientX: ML_PATIENT_X,
  patientY: ML_PATIENT_Y,
};

/* ── FH-PeDS patients ───────────────────────────────────────
   Three example patients, one for each interpretation bucket so
   users can quickly see the full output range:

   Patient X (Probable FH, score 14)
     Reuses the ML Patient X verbatim. LDL 6.2 mmol/L lands in
     the 12-point band; first-degree high cholesterol or CAD
     contributes 2 points.

   Patient Z (Possible FH, score 7)
     Constructed specifically for this page so the middle bucket
     has a representative example. Plausible borderline case:
       LDL 4.0 mmol/L → 3.8 < 4.0 ≤ 4.8         → +8 pts
       HDL 1.5 mmol/L → 1.4 < 1.5 ≤ 2.2         → −2 pts
       Family history → 2nd-degree high chol    → +1 pt
       ────────────────────────────────────────
       Total                                       7 pts → Possible FH

   Patient Y (Unlikely FH, score 5)
     Reuses the ML Patient Y demographics + family history but
     drops LDL-C from 4.3 → 3.4 mmol/L so the score lands
     firmly below the Possible threshold:
       LDL 3.4 mmol/L → 3.0 < 3.4 ≤ 3.8         → +4 pts
       Family history → 2nd-degree high chol    → +1 pt
       ────────────────────────────────────────
       Total                                       5 pts → Unlikely FH

   All three carry an explicit fh_genetic answer ("No") so the
   field is visibly populated. */

const FHPEDS_PATIENT_X: SamplePatient = {
  ...ML_PATIENT_X,
  fh_genetic: '0',
};

const FHPEDS_PATIENT_Z: SamplePatient = {
  age: '9.4',
  gender: '0',
  ldl_cholesterol: '4.0',
  ldl_cholesterol_unit: 'mmol/L',
  fh_high_cholesterol: '2',
  fh_premature_cad: '0',
  fh_xant: '0',
  fh_acrus_senilis: '0',
  hdl_cholesterol: '1.5',
  hdl_cholesterol_unit: 'mmol/L',
  tag: '1.0',
  tag_unit: 'mmol/L',
  bmi: '0.4',
  bmi_unit: 'z-score',
  fh_genetic: '0',
};

const FHPEDS_PATIENT_Y: SamplePatient = {
  ...ML_PATIENT_Y,
  ldl_cholesterol: '3.4',
  fh_genetic: '0',
};

export const FHPEDS_SAMPLE_PATIENTS: Readonly<Record<string, SamplePatient>> = {
  patientX: FHPEDS_PATIENT_X,
  patientZ: FHPEDS_PATIENT_Z,
  patientY: FHPEDS_PATIENT_Y,
};
