"""bmi_zscore_to_js.py

Generate public/js/bmi_zscore_table.js from the canonical UK90 reference.

Source of truth
---------------
This script reads ``uk90.rda`` from Tim Cole's `sitar` R package -- the
authoritative source of UK90 LMS parameters. The file is downloaded once
from GitHub (pinned by commit + SHA-256), cached under ``data/bmi/.cache/``
(gitignored), and parsed in pure Python (minimal RDX2/XDR reader, no R
or `pyreadr` dependency).

Output
------
``public/js/bmi_zscore_table.js`` contains:

* ``BMI_LMS_MALE``   (gender = 1)
* ``BMI_LMS_FEMALE`` (gender = 0)

Each is an object with three ``Float32Array``s -- ``L``, ``M``, ``S`` --
plus a ``years`` ``Float32Array`` giving the age (in years) at each knot.
Knots are sitar's native (non-uniform) age points restricted to
``[0, 18]`` years, which is the clinically relevant range for the
calculator.

A ``bmiToZScore(bmi, age, gender)`` helper is emitted alongside; it
performs binary search on ``years`` and linearly interpolates ``L``,
``M``, ``S`` between adjacent knots before applying the Cole & Green
(1992) Z-score formula::

    Z = ((BMI / M) ** L - 1) / (L * S)

Ages outside ``[years[0], years[-1]]`` are clamped to the endpoints.

This script is run **manually**, once per UK90 release (i.e. essentially
never -- UK90 has been stable since 1990). The emitted JS file is
committed to the repo and is what ships to the browser and is loaded by
the Node test suite. No part of the build or test pipeline invokes this
script automatically.

References
----------
* Cole TJ, Green PJ. Smoothing reference centile curves: the LMS method
  and penalized likelihood. Stat Med. 1992;11(10):1305-19.
  doi:10.1002/sim.4780111005
* Cole TJ, Freeman JV, Preece MA. British 1990 growth reference centiles
  for weight, height, body mass index and head circumference fitted by
  maximum penalized likelihood. Stat Med. 1998;17(4):407-29.
  doi:10.1002/(SICI)1097-0258(19980228)17:4<407::AID-SIM742>3.0.CO;2-L
* sitar R package: https://github.com/statist7/sitar (Tim Cole)

Usage
-----
    cd data/bmi
    python bmi_zscore_to_js.py            # uses system Python with stdlib only
"""

from __future__ import annotations

import gzip
import hashlib
import struct
import sys
import urllib.request
from pathlib import Path
from typing import Any


# ── sitar pin (kept in sync with ml-fh-peds/tests/uk90_rda_loader.js) ────────

SITAR_COMMIT = "b7174bb39020d723a87b7a8652e47cbf94a80ec6"  # tag v1.5.0
SITAR_URL = (
    f"https://raw.githubusercontent.com/statist7/sitar/{SITAR_COMMIT}/data/uk90.rda"
)
UK90_SHA256 = "fb84a44ca748b90c91ab244f8cb1a8e3c5253c4e8ee64f64f2549dc41738215f"

# ── Paths ────────────────────────────────────────────────────────────────────

HERE = Path(__file__).parent  # data/bmi/
REPO_ROOT = HERE.parent.parent
CACHE_DIR = HERE / ".cache"
CACHE_FILE = CACHE_DIR / "uk90.rda"
OUT_JS = REPO_ROOT / "public" / "js" / "bmi_zscore_table.js"

# ── Age range to emit ────────────────────────────────────────────────────────

AGE_MIN = 0.0
AGE_MAX = 18.0

# ── Fetch & verify uk90.rda ──────────────────────────────────────────────────


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_cached() -> Path:
    if CACHE_FILE.exists():
        if sha256_of(CACHE_FILE) == UK90_SHA256:
            return CACHE_FILE
        print(f"  · cached {CACHE_FILE.name} hash mismatch -- re-fetching")
        CACHE_FILE.unlink()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  · fetching {SITAR_URL}")
    with urllib.request.urlopen(SITAR_URL) as resp:  # noqa: S310 -- pinned https URL
        data = resp.read()
    CACHE_FILE.write_bytes(data)

    got = sha256_of(CACHE_FILE)
    if got != UK90_SHA256:
        CACHE_FILE.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded uk90.rda SHA-256 mismatch.\n"
            f"  expected: {UK90_SHA256}\n"
            f"  got:      {got}\n"
            f"  URL:      {SITAR_URL}"
        )
    return CACHE_FILE


# ── Minimal RDX2 / XDR parser (Python port of uk90_rda_loader.js) ────────────
#
# References
#   R serialization: src/main/serialize.c in r-source
#   XDR (RFC 1014): big-endian, 4-byte aligned

NILSXP = 0
SYMSXP = 1
LISTSXP = 2
CHARSXP = 9
LGLSXP = 10
INTSXP = 13
REALSXP = 14
STRSXP = 16
VECSXP = 19
NILVALUE_SXP = 254
REFSXP = 255


class _Reader:
    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.pos = 0
        self.refs: list[Any] = []

    def int32(self) -> int:
        (v,) = struct.unpack_from(">i", self.buf, self.pos)
        self.pos += 4
        return v

    def float64(self) -> float:
        (v,) = struct.unpack_from(">d", self.buf, self.pos)
        self.pos += 8
        return v

    def bytes_(self, n: int) -> bytes:
        v = self.buf[self.pos : self.pos + n]
        self.pos += n
        return v

    def string(self, n: int) -> str:
        return self.bytes_(n).decode("latin-1")


def _unpack_flags(flags: int) -> dict:
    return {
        "type": flags & 0xFF,
        "levels": (flags >> 12) & 0xFFFFF,
        "is_object": bool(flags & (1 << 8)),
        "has_attr": bool(flags & (1 << 9)),
        "has_tag": bool(flags & (1 << 10)),
    }


def _read_header(r: _Reader) -> None:
    marker = r.string(7)  # "RDX2\nX\n" or "RDB2\nB\n"
    if not (marker.startswith("RDX2") or marker.startswith("RDB2")):
        raise ValueError(f"Not an RDX2/RDB2 file (marker={marker!r})")
    r.int32()  # serialization version
    r.int32()  # writer R version
    r.int32()  # min reader R version


def _read_item(r: _Reader) -> Any:  # noqa: PLR0911, PLR0912 -- big switch on SEXPTYPE
    flags = r.int32()
    f = _unpack_flags(flags)
    t = f["type"]

    if t in (NILVALUE_SXP, NILSXP):
        return None

    if t == REFSXP:
        idx = flags >> 8
        if idx == 0:
            idx = r.int32()
        ref = r.refs[idx - 1]
        if ref is None:
            raise ValueError(f"Unknown ref index {idx}")
        return ref

    if t == SYMSXP:
        name = _read_item(r)
        if isinstance(name, dict) and "value" in name:
            sym_name = name["value"]
        else:
            sym_name = name
        sym = {"_sym": sym_name}
        r.refs.append(sym)
        return sym

    if t == LISTSXP:
        attr = _read_item(r) if f["has_attr"] else None
        tag = _read_item(r) if f["has_tag"] else None
        head = _read_item(r)
        tail = _read_item(r)
        node = {"tag": tag, "value": head, "attr": attr}
        out = [node]
        if isinstance(tail, list):
            out.extend(tail)
        elif tail is not None:
            out.append(tail)
        return out

    if t == CHARSXP:
        n = r.int32()
        if n == -1:
            return None  # NA_STRING
        return r.string(n)

    if t in (LGLSXP, INTSXP):
        n = r.int32()
        vals = [r.int32() for _ in range(n)]
        node = {"_type": "lgl" if t == LGLSXP else "int", "value": vals}
        if f["has_attr"]:
            node["attr"] = _read_item(r)
        return node

    if t == REALSXP:
        n = r.int32()
        vals = [r.float64() for _ in range(n)]
        node = {"_type": "real", "value": vals}
        if f["has_attr"]:
            node["attr"] = _read_item(r)
        return node

    if t == STRSXP:
        n = r.int32()
        vals = [_read_item(r) for _ in range(n)]  # each is CHARSXP
        node = {"_type": "str", "value": vals}
        if f["has_attr"]:
            node["attr"] = _read_item(r)
        return node

    if t == VECSXP:
        n = r.int32()
        vals = [_read_item(r) for _ in range(n)]
        node = {"_type": "vec", "value": vals}
        if f["has_attr"]:
            node["attr"] = _read_item(r)
        return node

    raise ValueError(f"Unsupported SEXPTYPE {t} at byte {r.pos}")


def _read_rda_body(r: _Reader) -> dict:
    root = _read_item(r)
    if not isinstance(root, list):
        raise ValueError("Expected top-level pairlist")
    out: dict = {}
    for node in root:
        tag = node.get("tag")
        name = tag["_sym"] if isinstance(tag, dict) and "_sym" in tag else "<unnamed>"
        out[name] = node["value"]
    return out


def _vecsxp_to_dataframe(vec: dict) -> list[dict]:
    """Convert an R data.frame VECSXP into a list of row dicts, decoding factors."""
    if not vec or vec.get("_type") != "vec":
        raise ValueError("Not a VECSXP")

    # Column names from attr 'names'.
    col_names: list[str] | None = None
    for a in vec.get("attr") or []:
        tag = a.get("tag")
        if tag and tag.get("_sym") == "names":
            names_node = a["value"]
            if names_node and names_node.get("_type") == "str":
                col_names = list(names_node["value"])
                break
    if col_names is None:
        col_names = [f"V{i + 1}" for i in range(len(vec["value"]))]

    cols: list[list] = []
    for col in vec["value"]:
        if col is None:
            cols.append([])
            continue
        ct = col.get("_type")
        if ct == "real":
            cols.append(list(col["value"]))
        elif ct == "int":
            # Factor? Look for levels (STRSXP) and class 'factor'.
            levels = None
            is_factor = False
            for a in col.get("attr") or []:
                tag = a.get("tag")
                if not tag:
                    continue
                if tag.get("_sym") == "levels" and a["value"].get("_type") == "str":
                    levels = list(a["value"]["value"])
                if tag.get("_sym") == "class" and a["value"].get("_type") == "str":
                    if "factor" in a["value"]["value"]:
                        is_factor = True
            if is_factor and levels:
                cols.append(
                    [levels[i - 1] if 1 <= i <= len(levels) else None for i in col["value"]]
                )
            else:
                cols.append(list(col["value"]))
        elif ct == "str":
            cols.append(list(col["value"]))
        else:
            cols.append(list(col.get("value") or []))

    n_rows = len(cols[0]) if cols else 0
    return [{col_names[c]: cols[c][r] for c in range(len(col_names))} for r in range(n_rows)]


# ── High-level: extract UK90 BMI LMS rows ────────────────────────────────────


def load_uk90_bmi() -> dict[str, list[dict]]:
    """Return {'male': [{years, L, M, S}, ...], 'female': [...]}.

    Rows where L.bmi is NaN (i.e. the sitar table covers other measurements
    too -- BMI is only populated for a subset of age knots) are dropped.
    """
    path = ensure_cached()
    raw = gzip.decompress(path.read_bytes())
    r = _Reader(raw)
    _read_header(r)
    body = _read_rda_body(r)

    if "uk90" not in body:
        raise ValueError(f"uk90 variable not found; got: {list(body)}")

    rows = _vecsxp_to_dataframe(body["uk90"])

    out: dict[str, list[dict]] = {"male": [], "female": []}
    for row in rows:
        l_bmi = row.get("L.bmi")
        # R encodes NA-double as a specific NaN bit pattern; Python sees it as nan.
        if l_bmi is None or (isinstance(l_bmi, float) and l_bmi != l_bmi):  # noqa: PLR0124 -- NaN check
            continue
        entry = {
            "years": float(row["years"]),
            "L": float(row["L.bmi"]),
            "M": float(row["M.bmi"]),
            "S": float(row["S.bmi"]),
        }
        sex = row.get("sex")
        # sitar uk90 convention: '1' = male, '2' = female.
        if sex == "1":
            out["male"].append(entry)
        elif sex == "2":
            out["female"].append(entry)

    out["male"].sort(key=lambda e: e["years"])
    out["female"].sort(key=lambda e: e["years"])
    return out


# ── JS emission ──────────────────────────────────────────────────────────────

_JS_TEMPLATE = """\
// AUTO-GENERATED by data/bmi/bmi_zscore_to_js.py -- do not edit by hand.
//
// UK90 LMS parameters for BMI Z-score (SDS) calculation.
//
// Source: sitar R package (Tim Cole), data/uk90.rda
//   https://github.com/statist7/sitar (pinned to commit {commit})
//   SHA-256: {sha}
//
// References
// ----------
// Cole TJ, Green PJ. Stat Med. 1992;11(10):1305-19.
//   doi:10.1002/sim.4780111005
// Cole TJ, Freeman JV, Preece MA. Stat Med. 1998;17(4):407-29.
//   doi:10.1002/(SICI)1097-0258(19980228)17:4<407::AID-SIM742>3.0.CO;2-L
//
// Two LMS parameter sets, one per sex:
//   BMI_LMS_MALE   -- gender = 1 (male)
//   BMI_LMS_FEMALE -- gender = 0 (female)
//
// Each is an object with four Float32Arrays:
//   years -- age at each knot (years), strictly increasing
//   L, M, S -- LMS parameters at the same knots
//
// Knots are sitar's native (non-uniform) age points restricted to [0, 18] y
// ({n_male_knots} male, {n_female_knots} female knots).
//
// Z-score formula (Cole & Green 1992):
//   Z = ((BMI / M) ** L - 1) / (L * S)
//
// Use bmiToZScore(bmi, age, gender) for interpolated lookup.

{arrays}

/**
 * Convert raw BMI to age- and sex-adjusted Z-score (SDS) using the
 * UK90 LMS reference parameters.
 *
 * Linearly interpolates L, M, S between adjacent age knots, then applies
 * the Cole & Green formula:  Z = ((BMI / M)^L - 1) / (L * S)
 *
 * Ages outside [years[0], years[-1]] are clamped to the endpoints.
 *
 * @param {{number}} bmi     Body Mass Index in kg/m^2
 * @param {{number}} age     Age in years
 * @param {{number}} gender  0 = Female, 1 = Male
 * @returns {{number}}       BMI Z-score (SDS)
 */
function bmiToZScore(bmi, age, gender) {{
  const lms = gender === 1 ? BMI_LMS_MALE : BMI_LMS_FEMALE;
  const years = lms.years;
  const n = years.length;

  // Clamp to the supported age range.
  const a = Math.min(Math.max(age, years[0]), years[n - 1]);

  // Binary search for the bracket [years[lo], years[hi]] containing `a`.
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {{
    const mid = (lo + hi) >> 1;
    if (years[mid] <= a) lo = mid;
    else hi = mid;
  }}

  const span = years[hi] - years[lo];
  const t = span > 0 ? (a - years[lo]) / span : 0;

  const L = lms.L[lo] * (1 - t) + lms.L[hi] * t;
  const M = lms.M[lo] * (1 - t) + lms.M[hi] * t;
  const S = lms.S[lo] * (1 - t) + lms.S[hi] * t;

  return ((bmi / M) ** L - 1) / (L * S);
}}
"""


def _fmt_array(values: list[float], *, decimals: int) -> str:
    """Compact JS array literal with values rounded to ``decimals`` places.

    sitar stores L, M, S to 4-6 significant figures; rounding to 6 dp keeps
    every distinguishable digit while shrinking the emitted file.
    """
    return "[" + ",".join(f"{v:.{decimals}f}".rstrip("0").rstrip(".") or "0" for v in values) + "]"


def _make_const(name: str, knots: list[dict]) -> str:
    years = [k["years"] for k in knots]
    L = [k["L"] for k in knots]
    M = [k["M"] for k in knots]
    S = [k["S"] for k in knots]
    return (
        f"const BMI_LMS_{name} = {{\n"
        f"  years: new Float32Array({_fmt_array(years, decimals=6)}),\n"
        f"  L:     new Float32Array({_fmt_array(L, decimals=6)}),\n"
        f"  M:     new Float32Array({_fmt_array(M, decimals=6)}),\n"
        f"  S:     new Float32Array({_fmt_array(S, decimals=6)}),\n"
        f"}};"
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def _restrict_age(knots: list[dict]) -> list[dict]:
    """Keep knots in [AGE_MIN, AGE_MAX], inclusive."""
    return [k for k in knots if AGE_MIN <= k["years"] <= AGE_MAX]


def _sanity_check(knots: list[dict], label: str) -> None:
    if not knots:
        raise RuntimeError(f"{label}: no knots in age range [{AGE_MIN}, {AGE_MAX}]")
    if knots[0]["years"] > 0.5:
        raise RuntimeError(
            f"{label}: first knot at age {knots[0]['years']:.3f} y is suspiciously high"
        )
    if knots[-1]["years"] < AGE_MAX - 1.0:
        raise RuntimeError(
            f"{label}: last knot at age {knots[-1]['years']:.3f} y is far below AGE_MAX"
        )
    for a, b in zip(knots, knots[1:]):  # noqa: B905 -- pairwise; lengths differ by 1 by construction
        if not (b["years"] > a["years"]):
            raise RuntimeError(f"{label}: knots not strictly increasing at {a['years']}")


def main() -> None:
    print(f"Loading sitar::uk90 from {SITAR_URL}")
    ref = load_uk90_bmi()
    print(f"  · raw knots:  male={len(ref['male'])}, female={len(ref['female'])}")

    male = _restrict_age(ref["male"])
    female = _restrict_age(ref["female"])
    _sanity_check(male, "male")
    _sanity_check(female, "female")
    print(
        f"  · age-restricted to [{AGE_MIN}, {AGE_MAX}] y: "
        f"male={len(male)}, female={len(female)}"
    )

    arrays_js = _make_const("MALE", male) + "\n\n" + _make_const("FEMALE", female)
    out = _JS_TEMPLATE.format(
        commit=SITAR_COMMIT,
        sha=UK90_SHA256,
        n_male_knots=len(male),
        n_female_knots=len(female),
        arrays=arrays_js,
    )

    OUT_JS.parent.mkdir(parents=True, exist_ok=True)
    OUT_JS.write_text(out)
    size_kb = OUT_JS.stat().st_size / 1024
    print(f"\nWritten: {OUT_JS}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 -- top-level entry point
        sys.exit(f"ERROR: {e}")
