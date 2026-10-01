"""Readers for SNANA text tables (FITRES, LCPLOT), ported from the prototype.

Used to pull per-observation metadata that MODELSPEC doesn't carry (e.g. SCA,
pixel position) so it can be joined onto the spectra with
:func:`handshaker.modelspec.attach_columns`.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def load_lcplot(path: str) -> pd.DataFrame:
    """Load a SNANA LCPLOT(text:csv) table into a DataFrame (stripped column names)."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_fitres(path: str) -> pd.DataFrame:
    """Load a SNANA FITRES text table (VARNAMES:/SN: rows) into a DataFrame.

    Numeric columns are converted to numbers; anything non-numeric stays a string.
    """
    names, rows = None, []
    with open(path) as f:
        for line in f:
            if line.startswith("VARNAMES:"):
                names = line.split()[1:]
            elif line.startswith("SN:"):
                rows.append(line.split()[1:])
    df = pd.DataFrame(rows, columns=names)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().all():
            df[col] = converted
    return df


def write_snana_filters(filters, bands, outdir, ice_model=None, thickness_nm=0.0,
                        sca=None, prefix="ROMAN"):
    """Write SNANA 2-column transmission files, optionally with ice and/or an SCA shift.

    For each band writes ``<outdir>/<prefix>_<band>.dat`` (wavelength [A], throughput)
    on the source filter's native grid. ``sca=None`` uses the nominal curve, an int
    uses that SCA's shifted curve; if ``ice_model`` is given and ``thickness_nm>0``
    the throughput is multiplied by the ice ratio T(t)/T(0) at that thickness. Point
    the LC fit's FILTPATH at ``outdir`` to produce the perturbed (ice) sim/fit to
    validate against. Returns the list of written paths.
    """
    os.makedirs(outdir, exist_ok=True)
    written = []
    for band in bands:
        bp = filters.shifted(band, sca) if sca is not None else filters.nominal(band)
        wave = np.asarray(bp.wave, dtype=float)
        thru = np.asarray(bp.thru, dtype=float)
        if ice_model is not None and thickness_nm and thickness_nm > 0:
            ice = ice_model.relative_transmission(thickness_nm, band=band)
            thru = thru * np.interp(wave, ice.wave, ice.thru, left=1.0, right=1.0)
        path = os.path.join(outdir, f"{prefix}_{band}.dat")
        with open(path, "w") as f:
            f.write("# wave trans\n")
            for w, t in zip(wave, thru):
                f.write(f"{w:10.3f}  {t:.6e}\n")
        written.append(path)
    return written
