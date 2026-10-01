#!/usr/bin/env python
"""
Build a MAGCOR (calib-correction) table for a Roman SNANA sim.

Output format mirrors the DES calib_corrections MAGCOR_FILE that snlc_fit.exe
reads via the MAGCOR_FILE key, but trimmed to: MJD, BAND, SCA, and a single
MAGCOR column (all 0.0 for now -- placeholder to be filled in later).

Usage
-----
    python make_magcor_roman.py \
        --genversion /path/to/SIM/ROMAN_SIM_GENVERSION \
        --outfile out_magcor_roman.dat

The --genversion dir is the SNANA sim output folder containing the
*HEAD.FITS.gz and *PHOT.FITS.gz files (and a .LIST file).
"""

import argparse
import glob
import os
import sys

import numpy as np
from astropy.io import fits


# Wavecor (Angstroms) per band-key = the character after '/' in the BAND
# string (e.g. "Z087-Z/p" -> 'p'). Maps to SCA/band combination.
WAVECOR = {
    'a': 3.034743, 'b': 0.0, 'c': -15.645429, 'd': -9.047145, 'e': -9.624221,
    'f': -23.687393, 'g': -35.608254, 'h': -30.761735, 'i': -38.974908,
    'j': 2.792135, 'k': 0.0, 'l': -14.373832, 'm': -8.315911, 'n': -8.845913,
    'o': -21.747907, 'p': -32.661116, 'q': -28.226710, 'r': -35.739461,
    's': 3.496308, 't': 0.0, 'u': -18.000851, 'v': -10.413929, 'w': -11.077687,
    'x': -27.237029, 'y': -40.907726, 'z': -35.352628, 'A': -44.764242,
    'B': 5.299038, 'C': 0.0, 'D': -27.302776, 'E': -15.791303, 'F': -16.798229,
    'G': -41.325814, 'H': -62.099046, 'I': -53.655418, 'J': -67.962877,
    'K': 4.819989, 'L': 0.0, 'M': -24.810511, 'N': -14.354519, 'O': -15.269327,
    'P': -37.537002, 'Q': -56.369242, 'R': -48.717374, 'S': -61.680876,
    'T': 6.273980, 'U': 0.0, 'V': -32.305786, 'W': -18.688902, 'X': -19.880171,
    'Y': -48.884550, 'Z': -73.426607, '0': -63.453454, '1': -80.350666,
}


def wavecor_for_band(band):
    """Return the wavecor for a BAND string, keyed by the char after '/'."""
    key = band.split('/')[-1] if '/' in band else band
    if key not in WAVECOR:
        sys.exit(f"ERROR: no wavecor mapping for band '{band}' (key '{key}')")
    return WAVECOR[key]


# Band-key (char after '/') -> the pair of physical SCAs it maps to.
# From the Roman kcor FILTER list: each key serves two SCAs sharing one
# throughput. When the PHOT table has no DETNUM column, the true SCA is
# ambiguous within the pair, so we default to the first (see sca_for_band).
_SCA_GROUPS = [
    "abcdefghi",   # R062
    "jklmnopqr",   # Z087
    "stuvwxyzA",   # Y106
    "BCDEFGHIJ",   # J129
    "KLMNOPQRS",   # H158
    "TUVWXYZ01",   # F184
]
_SCA_FIRST = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_SCA_SECOND = [10, 11, 12, 14, 13, 15, 18, 17, 16]
SCA_PAIR = {}
for _grp in _SCA_GROUPS:
    for _i, _ch in enumerate(_grp):
        SCA_PAIR[_ch] = (_SCA_FIRST[_i], _SCA_SECOND[_i])


def sca_for_band(band):
    """Derive SCA from the band-key. Returns the FIRST of the SCA pair
    (the pair is ambiguous without a DETNUM column)."""
    key = band.split('/')[-1] if '/' in band else band
    if key not in SCA_PAIR:
        sys.exit(f"ERROR: no SCA mapping for band '{band}' (key '{key}')")
    return SCA_PAIR[key][0]


def find_fits_pairs(genversion):
    """Return list of (head_file, phot_file) pairs from the .LIST file.

    The SNANA .LIST file in a genversion dir names the data files (one per
    line, typically the HEAD files). For each entry we resolve its HEAD and
    matching PHOT file.
    """
    lists = glob.glob(os.path.join(genversion, "*.LIST"))
    if not lists:
        sys.exit(f"ERROR: no .LIST file found in {genversion}")
    if len(lists) > 1:
        sys.exit(f"ERROR: multiple .LIST files in {genversion}: {lists}")
    list_file = lists[0]

    def resolve(path):
        # .LIST may omit the .gz suffix while files on disk are gzipped
        # (or vice versa); glob matches either form in one shot.
        stem = path[:-3] if path.endswith(".gz") else path
        hits = glob.glob(stem) + glob.glob(stem + ".gz")
        return hits[0] if hits else None

    pairs = []
    with open(list_file) as f:
        for line in f:
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            # entry may name the HEAD or PHOT file; derive both
            base = os.path.join(genversion, os.path.basename(entry))
            if "HEAD.FITS" in base:
                head_name = base
                phot_name = base.replace("HEAD.FITS", "PHOT.FITS")
            elif "PHOT.FITS" in base:
                phot_name = base
                head_name = base.replace("PHOT.FITS", "HEAD.FITS")
            else:
                sys.exit(f"ERROR: unrecognized .LIST entry (no HEAD/PHOT): {entry}")
            head = resolve(head_name)
            phot = resolve(phot_name)
            if head is None:
                sys.exit(f"ERROR: HEAD file listed but not found: {head_name}[.gz]")
            if phot is None:
                sys.exit(f"ERROR: PHOT file listed but not found: {phot_name}[.gz]")
            pairs.append((head, phot))

    if not pairs:
        sys.exit(f"ERROR: no usable entries in {list_file}")
    print(f"Read {len(pairs)} file(s) from {os.path.basename(list_file)}")
    return pairs


def pick_col(colnames, candidates):
    """Return the first column name present, matched case-insensitively."""
    upper = {c.upper(): c for c in colnames}
    for cand in candidates:
        if cand.upper() in upper:
            return upper[cand.upper()]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genversion", required=True,
                    help="SNANA sim output dir with *HEAD.FITS / *PHOT.FITS")
    ap.add_argument("--outfile", default="out_magcor_roman.dat")
    args = ap.parse_args()

    pairs = find_fits_pairs(args.genversion)

    rows = []
    sca_found = False

    for head_file, phot_file in pairs:
        with fits.open(head_file) as hdu:
            head = hdu[1].data
            hcols = head.columns.names
            cid_col = pick_col(hcols, ["SNID", "CID"])
            cids = head[cid_col]
            # 1-based [first, last] pointers into the PHOT table
            ptr_min = head["PTROBS_MIN"]
            ptr_max = head["PTROBS_MAX"]

        with fits.open(phot_file) as hdu:
            phot = hdu[1].data
            pcols = phot.columns.names
            mjd_col = pick_col(pcols, ["MJD"])
            band_col = pick_col(pcols, ["BAND", "FLT"])
            sca_col = pick_col(pcols, ["SCA", "DETNUM", "CCDNUM", "IMGNUM", "CCDID"])

            if band_col is None or mjd_col is None:
                sys.exit(f"ERROR: could not find MJD/BAND in {phot_file}\n"
                         f"       available columns: {pcols}")
            if sca_col is not None:
                sca_found = True

            mjd = phot[mjd_col]
            band = phot[band_col]
            sca = phot[sca_col] if sca_col is not None else None

            for cid, i0, i1 in zip(cids, ptr_min, ptr_max):
                # SNANA pointers are 1-based and inclusive
                for j in range(i0 - 1, i1):
                    b = str(band[j]).strip()
                    if b in ("", "-"):        # skip separator rows
                        continue
                    m = float(mjd[j])
                    # Prefer the real DETNUM; otherwise derive SCA from band-key
                    # (first of the ambiguous pair).
                    s = int(sca[j]) if sca is not None else sca_for_band(b)
                    cid_str = str(cid).strip()
                    ws = wavecor_for_band(b)   # keyed by full band string
                    filt = b.split('/')[0]       # filter without SCA suffix: e.g. R062-R

                    # xxx mark delete by RK row_id = f"{cid_str}-{m:.3f}-{filt}"
                    row_id = f"{cid_str}-{m:.3f}-{filt[-1]}"  # RK fix
                    rows.append((row_id, cid_str, m, filt, s, ws))

    if not sca_found:
        print("WARNING: no DETNUM/SCA column in PHOT table; SCA derived from "
              "the band-key (first of each SCA pair -- ambiguous without "
              "DETNUM).", file=sys.stderr)

    # ---- write output in SNANA MAGCOR_FILE style ----
    with open(args.outfile, "w") as f:
        f.write("DOCUMENTATION:\n")
        f.write("  PURPOSE:  Placeholder MAGCOR table for Roman SNANA sim\n")
        f.write("  INTENT:   MAGCOR = 0 for all rows (to be filled in later)\n")
        f.write("  USAGE_KEY:   MAGCOR_FILE\n")
        f.write("  USAGE_CODE:  snlc_fit.exe\n")
        f.write("DOCUMENTATION_END:\n")
        f.write("\n")
        f.write("VARNAMES: ROW CID MJD BAND SCA WAVECOR MAGCOR\n")
        for row_id, cid, m, b, s, ws in rows:
            f.write(f"ROW: {row_id} {cid} {m:.4f} {b} {s} {ws:.6f} 0.0000\n")

    print(f"Wrote {len(rows)} rows to {args.outfile}")


if __name__ == "__main__":
    main()
