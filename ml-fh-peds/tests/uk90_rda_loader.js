/**
 * uk90_rda_loader.js
 *
 * Minimal R .rda (RDX2 / XDR) parser specialised for sitar::uk90.
 *
 * Why hand-rolled? The .rda format is well-documented and we only need to
 * extract one specific structure (a data.frame with named numeric columns
 * and one factor column). Pulling in a 3rd-party npm dependency just to
 * read 45 kB of reference data felt heavier than the parser itself.
 *
 * The file is fetched once from the sitar GitHub repository (Tim Cole's
 * own R package — the canonical source of UK90 LMS values) and cached
 * locally under tests/.cache/ (gitignored).
 *
 * Public API
 * ----------
 *   loadUK90BMI(opts) → { male: [{years, L, M, S}, …], female: [...] }
 *
 * Format reference
 * ----------------
 *   • R serialization: https://github.com/wch/r-source/blob/trunk/src/main/serialize.c
 *   • XDR (RFC 1014): big-endian, 4-byte aligned ints/floats
 *   • RDX2 header: 'RDX2\nX\n' (ASCII format marker) followed by version
 *     and writer/reader R versions, then the serialized SEXP tree.
 *
 * Only the SEXPTYPEs actually present in uk90.rda are implemented:
 *   NILVALUE_SXP (254), SYMSXP (1), LISTSXP (2), CHARSXP (9),
 *   LGLSXP (10), INTSXP (13), REALSXP (14), STRSXP (16), VECSXP (19),
 *   REFSXP (255), with ATTR/TAG flags.
 */

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import https from 'node:https';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Pinned to sitar v1.5.0 (released 2025-07-29). Using the immutable commit
// SHA rather than the tag so the reference cannot drift if upstream ever
// rewrites tags. The expected SHA-256 below provides an additional integrity
// check at load time.
const SITAR_COMMIT = 'b7174bb39020d723a87b7a8652e47cbf94a80ec6'; // tag v1.5.0
const SITAR_URL = `https://raw.githubusercontent.com/statist7/sitar/${SITAR_COMMIT}/data/uk90.rda`;
const UK90_SHA256 = 'fb84a44ca748b90c91ab244f8cb1a8e3c5253c4e8ee64f64f2549dc41738215f';

const CACHE_DIR = path.join(__dirname, '.cache');
const CACHE_FILE = path.join(CACHE_DIR, 'uk90.rda');

/* ── Network: fetch & cache uk90.rda ───────────────────────────────────── */

function downloadToCache() {
  return new Promise((resolve, reject) => {
    if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR, { recursive: true });

    const tmp = CACHE_FILE + '.tmp';
    const out = fs.createWriteStream(tmp);

    const req = https.get(SITAR_URL, (res) => {
      if (res.statusCode !== 200) {
        res.resume();
        reject(new Error(`Download failed: HTTP ${res.statusCode} from ${SITAR_URL}`));
        return;
      }
      res.pipe(out);
      out.on('finish', () =>
        out.close(() => {
          fs.renameSync(tmp, CACHE_FILE);
          resolve();
        })
      );
    });
    req.on('error', reject);
    out.on('error', reject);
  });
}

function sha256OfFile(file) {
  const h = crypto.createHash('sha256');
  h.update(fs.readFileSync(file));
  return h.digest('hex');
}

async function ensureCached() {
  // If the cache exists but its hash is wrong (e.g. cache key collision in CI,
  // partial download, or someone hand-edited it), discard and re-fetch.
  if (fs.existsSync(CACHE_FILE)) {
    if (sha256OfFile(CACHE_FILE) === UK90_SHA256) return;
    console.log(`  · cached uk90.rda hash mismatch — re-fetching`);
    fs.unlinkSync(CACHE_FILE);
  }
  console.log(`  · fetching ${SITAR_URL} → ${path.relative(process.cwd(), CACHE_FILE)}`);
  await downloadToCache();

  const got = sha256OfFile(CACHE_FILE);
  if (got !== UK90_SHA256) {
    fs.unlinkSync(CACHE_FILE);
    throw new Error(
      `Downloaded uk90.rda SHA-256 mismatch.\n` +
        `  expected: ${UK90_SHA256}\n` +
        `  got:      ${got}\n` +
        `  URL:      ${SITAR_URL}`
    );
  }
}

/* ── Minimal RDX2 / XDR parser ─────────────────────────────────────────── */

// SEXPTYPE constants we may encounter
const NILSXP = 0;
const SYMSXP = 1;
const LISTSXP = 2;
const CHARSXP = 9;
const LGLSXP = 10;
const INTSXP = 13;
const REALSXP = 14;
const STRSXP = 16;
const VECSXP = 19;
const NILVALUE_SXP = 254;
const REFSXP = 255;

class Reader {
  constructor(buf) {
    this.buf = buf;
    this.pos = 0;
    this.refs = []; // R's reference table (1-indexed)
  }
  int32() {
    const v = this.buf.readInt32BE(this.pos);
    this.pos += 4;
    return v;
  }
  float64() {
    const v = this.buf.readDoubleBE(this.pos);
    this.pos += 8;
    return v;
  }
  bytes(n) {
    const v = this.buf.subarray(this.pos, this.pos + n);
    this.pos += n;
    return v;
  }
  string(n) {
    return this.bytes(n).toString('latin1');
  }
}

/**
 * Decode flags word into { type, hasAttr, hasTag, isObject, levels, refIdx }.
 * See serialize.c: UnpackFlags().
 */
function unpackFlags(flags) {
  const type = flags & 0xff;
  const levels = flags >>> 12;
  const isObject = (flags & (1 << 8)) !== 0;
  const hasAttr = (flags & (1 << 9)) !== 0;
  const hasTag = (flags & (1 << 10)) !== 0;
  // For REFSXP the reference index is encoded in the upper bits.
  return { type, levels, isObject, hasAttr, hasTag };
}

function readHeader(r) {
  // Format marker: "RDX2\nX\n" (RDX2 binary subformat is wrong here — the
  // file uses XDR binary, but with an ASCII format marker at the very top.)
  const marker = r.string(7); // "RDX2\nX\n" or "RDB2\nB\n"
  if (!marker.startsWith('RDX2') && !marker.startsWith('RDB2')) {
    throw new Error(`Not an RDX2/RDB2 file (marker=${JSON.stringify(marker)})`);
  }
  // Version triplet
  r.int32(); // serialization version (2)
  r.int32(); // writer R version
  r.int32(); // min reader R version
}

function readItem(r) {
  const flags = r.int32();
  const f = unpackFlags(flags);

  switch (f.type) {
    case NILVALUE_SXP:
    case NILSXP:
      return null;

    case REFSXP: {
      // Reference index encoded in upper bits of flags (R serialize.c)
      let idx = flags >>> 8;
      if (idx === 0) idx = r.int32();
      const ref = r.refs[idx - 1];
      if (ref === undefined) throw new Error(`Unknown ref index ${idx}`);
      return ref;
    }

    case SYMSXP: {
      // SYMSXP body: a CHARSXP (printname). Symbol is added to ref table.
      const name = readItem(r);
      const sym = { _sym: typeof name === 'string' ? name : name.value };
      r.refs.push(sym);
      return sym;
    }

    case LISTSXP: {
      // Pairlist node: optional attr, optional tag, head item, tail (recursive).
      let attr = null,
        tag = null;
      if (f.hasAttr) attr = readItem(r);
      if (f.hasTag) tag = readItem(r);
      const head = readItem(r);
      const tail = readItem(r);
      // Flatten into an array of { tag, value } so callers can walk it.
      const node = { tag, value: head, attr };
      const list = [node];
      if (Array.isArray(tail)) list.push(...tail);
      else if (tail !== null) list.push(tail);
      return list;
    }

    case CHARSXP: {
      const len = r.int32();
      if (len === -1) return null; // NA_STRING
      return r.string(len);
    }

    case LGLSXP:
    case INTSXP: {
      const len = r.int32();
      const arr = new Int32Array(len);
      for (let i = 0; i < len; i++) arr[i] = r.int32();
      const node = { _type: f.type === LGLSXP ? 'lgl' : 'int', value: arr };
      if (f.hasAttr) node.attr = readItem(r);
      return node;
    }

    case REALSXP: {
      const len = r.int32();
      const arr = new Float64Array(len);
      for (let i = 0; i < len; i++) arr[i] = r.float64();
      const node = { _type: 'real', value: arr };
      if (f.hasAttr) node.attr = readItem(r);
      return node;
    }

    case STRSXP: {
      const len = r.int32();
      const arr = new Array(len);
      for (let i = 0; i < len; i++) arr[i] = readItem(r); // each is CHARSXP
      const node = { _type: 'str', value: arr };
      if (f.hasAttr) node.attr = readItem(r);
      return node;
    }

    case VECSXP: {
      const len = r.int32();
      const arr = new Array(len);
      for (let i = 0; i < len; i++) arr[i] = readItem(r);
      const node = { _type: 'vec', value: arr };
      if (f.hasAttr) node.attr = readItem(r);
      return node;
    }

    default:
      throw new Error(`Unsupported SEXPTYPE ${f.type} at byte ${r.pos}`);
  }
}

/**
 * Top-level rda body: a pairlist mapping symbol names to values.
 * Returns { name: value, … }.
 */
function readRdaBody(r) {
  const out = {};
  // Top-level structure: an integer count, then pairs of (sym, value)? Actually
  // R's saveRDS-style format uses a pairlist with TAG=symbol. We can rely on
  // the SYMSXP path: each top-level entry is a LISTSXP node whose tag is the
  // variable name symbol and whose value is the SEXP.
  const root = readItem(r);
  // root is an array of {tag, value, attr} nodes (our flattened LISTSXP).
  if (!Array.isArray(root)) {
    throw new Error('Expected top-level pairlist');
  }
  for (const node of root) {
    const name = node.tag && node.tag._sym ? node.tag._sym : '<unnamed>';
    out[name] = node.value;
  }
  return out;
}

/* ── Higher-level: extract uk90 BMI rows ───────────────────────────────── */

/**
 * Convert a VECSXP data.frame node into an array of row objects, using the
 * 'names' attribute to label columns. Factors are decoded to their level
 * strings.
 */
function vecsxpToDataFrame(vec) {
  if (!vec || vec._type !== 'vec') throw new Error('Not a VECSXP');

  // Find column names from the attr pairlist
  const attr = vec.attr || [];
  let colNames = null;
  for (const a of attr) {
    if (a.tag && a.tag._sym === 'names') {
      const namesNode = a.value;
      if (namesNode._type === 'str') {
        colNames = namesNode.value.map((v) => (typeof v === 'string' ? v : (v && v.value) || v));
      }
    }
  }
  if (!colNames) colNames = vec.value.map((_, i) => `V${i + 1}`);

  // Resolve each column to a plain JS array
  const cols = vec.value.map((col) => {
    if (col == null) return [];
    if (col._type === 'real') return Array.from(col.value);

    if (col._type === 'int') {
      // Might be a factor: check attr for 'levels' (STRSXP) and class 'factor'
      let levels = null;
      let isFactor = false;
      if (col.attr) {
        for (const a of col.attr) {
          if (!a.tag) continue;
          if (a.tag._sym === 'levels' && a.value && a.value._type === 'str') {
            levels = a.value.value.map((v) => (typeof v === 'string' ? v : v && v.value));
          }
          if (a.tag._sym === 'class' && a.value && a.value._type === 'str') {
            const classes = a.value.value.map((v) => (typeof v === 'string' ? v : v && v.value));
            if (classes.includes('factor')) isFactor = true;
          }
        }
      }
      if (isFactor && levels) {
        return Array.from(col.value).map((i) =>
          i >= 1 && i <= levels.length ? levels[i - 1] : null
        );
      }
      return Array.from(col.value);
    }

    if (col._type === 'str') {
      return col.value.map((v) => (typeof v === 'string' ? v : v && v.value));
    }

    return Array.from(col.value || []);
  });

  // Pivot to row-objects
  const nRows = cols[0] ? cols[0].length : 0;
  const rows = new Array(nRows);
  for (let r = 0; r < nRows; r++) {
    const row = {};
    for (let c = 0; c < colNames.length; c++) {
      row[colNames[c]] = cols[c][r];
    }
    rows[r] = row;
  }
  return rows;
}

async function loadUK90BMI() {
  await ensureCached();
  const compressed = fs.readFileSync(CACHE_FILE);
  const raw = zlib.gunzipSync(compressed);

  const r = new Reader(raw);
  readHeader(r);
  const body = readRdaBody(r);

  if (!body.uk90) {
    throw new Error(`uk90 variable not found; got: ${Object.keys(body).join(', ')}`);
  }

  const rows = vecsxpToDataFrame(body.uk90);

  // Keep only rows where L.bmi is defined (non-NaN). R's NA double is encoded
  // as a specific bit-pattern in REALSXP that surfaces in JS as NaN.
  const out = { male: [], female: [] };
  for (const row of rows) {
    if (row['L.bmi'] == null || Number.isNaN(row['L.bmi'])) continue;
    const entry = {
      years: Number(row['years']),
      L: Number(row['L.bmi']),
      M: Number(row['M.bmi']),
      S: Number(row['S.bmi']),
    };
    // sex factor: '1' = Male, '2' = Female (sitar uk90 convention)
    if (row['sex'] === '1') out.male.push(entry);
    else if (row['sex'] === '2') out.female.push(entry);
  }

  out.male.sort((a, b) => a.years - b.years);
  out.female.sort((a, b) => a.years - b.years);
  return out;
}

/* ── Linear interpolation helper ───────────────────────────────────────── */

/**
 * Linearly interpolate a column ('L'|'M'|'S') at the given age within the
 * sorted reference array. Out-of-range ages return null.
 */
function interpRef(refRows, age, col) {
  if (refRows.length === 0) return null;
  if (age < refRows[0].years || age > refRows[refRows.length - 1].years) return null;

  // Binary search for the bracket
  let lo = 0,
    hi = refRows.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (refRows[mid].years <= age) lo = mid;
    else hi = mid;
  }
  const a = refRows[lo],
    b = refRows[hi];
  if (a.years === b.years) return a[col];
  const t = (age - a.years) / (b.years - a.years);
  return a[col] + t * (b[col] - a[col]);
}

export { loadUK90BMI, interpRef, SITAR_URL, SITAR_COMMIT, UK90_SHA256 };
