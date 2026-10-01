"""Read SNANA MODELSPEC output (the ``SPECTRA`` ROOT tree) and run magcor on it.

The LC fit's MODELSPEC dump stores, per observation, the model SED (SPEC_LAM /
SPEC_FLUX arrays) plus its metadata (CID, BAND, detector=SCA, MJD, TREST). This
module turns that tree into per-observation ``(Observation, FixedSED)`` records
and computes, for each, the chromatic magnitude correction

    magcor = mag(SED through the TRUE filter  = SCA-shifted x ice)
           - mag(SED through the REFERENCE filter = a fixed SCA, no ice)

using the *same* model SED the fit used -- the "alter the data" side of the
magcor<->wavecor equivalence test. ``wavecor`` (the effective-wavelength shift,
the "alter the model" side) is reported alongside.

Reading the ROOT file needs ``uproot`` (+ ``awkward``); install into the env with
``pip install uproot awkward``. Field names are auto-detected so this tolerates
the exact SNANA branch naming.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .magcor import compute_magcor
from .observation import Observation
from .sed import FixedSED

log = get_logger("modelspec")

# SNANA single-char band -> ROMAN_FILTERS band directory name. Override via
# load_modelspec(..., band_map=...) if your sim uses different codes.
DEFAULT_BAND_MAP = {"R": "F062", "Z": "F087", "Y": "F106",
                    "J": "F129", "H": "F158", "F": "F184"}

# Toy ice-deposition model (position-independent). Rate from the released ice
# response metadata (thickness_time_conversion: 50 nm / 4.60 days); decon cadence
# from ice_rep_sim (epoch 0 = MJD 61295.0 = 2026-09-12, 20-day period).
DEFAULT_ICE_RATE_NM_PER_DAY = 50.0 / 4.60
DECON_EPOCH0_MJD = 61295.0
DECON_PERIOD_DAYS = 20.0


def mjd_ice_thickness(mjd, rate_nm_per_day=DEFAULT_ICE_RATE_NM_PER_DAY,
                      decon_period_days=DECON_PERIOD_DAYS, epoch0_mjd=DECON_EPOCH0_MJD):
    """Ice thickness [nm] from MJD: steady growth reset every decon period.

    ``thickness = rate * ((mjd - epoch0) mod decon_period)``. A single-rate
    approximation to ``ice_rep_sim`` -- it captures the MJD dependence but not the
    per-(x,y) growth-rate mosaic. Pass ``ice_thickness_nm=lambda r: mjd_ice_thickness(r.mjd)``
    to ``magcor_table`` to get per-observation thicknesses.
    """
    return rate_nm_per_day * ((mjd - epoch0_mjd) % decon_period_days)


_CID_NAMES = ["CCID", "CID", "SNID"]
_BAND_NAMES = ["BAND", "FLT", "FILTER"]
_SCA_NAMES = ["SCA", "DETNUM", "DET"]
_MJD_NAMES = ["MJD"]
_TREST_NAMES = ["TREST", "PHASE", "TOBS"]


def _pick(fields, candidates):
    for c in candidates:
        if c in fields:
            return c
    return None


def load_modelspec(root_path, tree="MODELSPEC", band_map=None):
    """Load the MODELSPEC tree into a DataFrame with a ``sed`` column of FixedSED.

    The ROOT tree is named ``MODELSPEC`` (``sntable_dump`` calls the same table
    ``SPECTRA``). Columns: cid, band (mapped), band_raw, sca, mjd, trest, nlam, sed.
    """
    import uproot
    import awkward as ak

    band_map = DEFAULT_BAND_MAP if band_map is None else band_map
    f = uproot.open(root_path)
    # Match the tree by (base)name, tolerating the ";cycle" suffix and any
    # directory nesting. Don't filter on the class string (it varies by version).
    keys = f.keys(recursive=True)
    key = next((n for n in keys if n.split("/")[-1].split(";")[0] == tree), None)
    if key is None:
        raise ValueError(f"tree {tree!r} not found in {root_path}. Objects present: {keys}")
    ev = f[key].arrays()
    fields = list(ev.fields)

    # The two per-spectrum array branches: wavelength (monotone, >1000 A) and flux.
    arr = [fl for fl in fields
           if hasattr(ev[fl][0], "__len__") and not isinstance(ev[fl][0], str)]

    def _is_wave(a):
        a = ak.to_numpy(a)
        return a.size > 2 and np.all(np.diff(a) >= 0) and a.max() > 1000

    lam_f = next((fl for fl in arr if _is_wave(ev[fl][0])), None)
    flux_f = next((fl for fl in arr if fl != lam_f), None)
    if lam_f is None or flux_f is None:
        raise ValueError(f"could not identify wave/flux array branches among {arr}")

    cid_f = _pick(fields, _CID_NAMES)
    band_f = _pick(fields, _BAND_NAMES)
    sca_f = _pick(fields, _SCA_NAMES)
    mjd_f = _pick(fields, _MJD_NAMES)
    trest_f = _pick(fields, _TREST_NAMES)
    log.info("MODELSPEC fields: lam=%s flux=%s cid=%s band=%s sca=%s mjd=%s",
             lam_f, flux_f, cid_f, band_f, sca_f, mjd_f)

    rows, seds = [], []
    for i in range(len(ev)):
        wave = ak.to_numpy(ev[lam_f][i]).astype(float)
        flux = ak.to_numpy(ev[flux_f][i]).astype(float)
        band_raw = str(ev[band_f][i]).strip() if band_f else None
        band = band_map.get(band_raw, band_raw)
        cid = str(ev[cid_f][i]).strip() if cid_f else None
        mjd = float(ev[mjd_f][i]) if mjd_f else None
        # SNANA writes -9 / -999 as "not set"; a real SCA is >= 1.
        sca_raw = int(ev[sca_f][i]) if sca_f else None
        sca = sca_raw if (sca_raw is not None and sca_raw >= 1) else None
        rows.append({
            "cid": cid, "band": band, "band_raw": band_raw, "sca": sca,
            "mjd": mjd,
            "trest": float(ev[trest_f][i]) if trest_f else None,
            "nlam": len(wave),
        })
        seds.append(FixedSED(wave, flux, name=f"{cid}_{mjd}"))

    df = pd.DataFrame(rows)
    df["sed"] = seds
    log.info("loaded %d MODELSPEC spectra from %s (tree %s)", len(df), root_path, key)
    if df["sca"].isna().all():
        log.warning("SPECTRA has no valid SCA (%s is null/-9); join it from the LC-fit "
                    "table with attach_columns() before magcor_table()", sca_f)
    return df


def attach_columns(spec_df, table, columns, cid_cols=("cid", "CID"),
                   mjd_cols=("mjd", "MJD"), mjd_round=3):
    """Bring per-observation columns (sca, x, y, ice_thickness_nm, ...) into a
    load_modelspec DataFrame from another per-epoch table, matched on CID + MJD.

    Parameters
    ----------
    spec_df : DataFrame from load_modelspec.
    table : DataFrame with the extra columns (e.g. a FITRES/LCPLOT export).
    columns : dict {name_in_table: name_to_use} or list of names (same on both).
    cid_cols, mjd_cols : (spec_col, table_col) key names on each side.
    mjd_round : decimals to round MJD to when matching (guards float mismatch).
    """
    colmap = {c: c for c in columns} if not isinstance(columns, dict) else dict(columns)
    left = spec_df.drop(columns=[v for v in colmap.values() if v in spec_df.columns]).copy()
    left["_k"] = (left[cid_cols[0]].astype(str).str.strip() + "@"
                  + left[mjd_cols[0]].round(mjd_round).astype(str))
    right = table.copy()
    right["_k"] = (right[cid_cols[1]].astype(str).str.strip() + "@"
                   + right[mjd_cols[1]].round(mjd_round).astype(str))
    right = right[["_k", *colmap]].rename(columns=colmap).drop_duplicates("_k")
    out = left.merge(right, on="_k", how="left").drop(columns="_k")
    n_missing = out[list(colmap.values())[0]].isna().sum() if colmap else 0
    if n_missing:
        log.warning("attach_columns: %d/%d rows had no match in the table", n_missing, len(out))
    return out


def magcor_table(df, filters, ice_model, reference_sca=2, calib_star=None,
                 ice_thickness_nm=0.0, grid=None):
    """Compute magcor for each MODELSPEC row against a fixed reference SCA.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`load_modelspec` (needs cid, band, sca, mjd, sed).
    filters, ice_model : the true/reference bandpass and ice sources.
    reference_sca : int
        Fixed reference SCA the correction is measured against (default 2).
    calib_star : SEDModel or None
        If given, magcors are the SN-minus-star chromatic differential.
    ice_thickness_nm : float or callable(row)->float
        Per-observation ice thickness. SPECTRA has no (x,y)/thickness, so supply
        it here -- a constant, or a callable taking the DataFrame row (e.g. to
        look it up from a position table or the deposition model).

    Returns one row per spectrum with band/sca/mjd, magcor_shift/ice/total, and
    wavecor (effective-wavelength shift), ready to compare against SNANA's wavecor.
    """
    rows = []
    for r in df.itertuples():
        thk = ice_thickness_nm(r) if callable(ice_thickness_nm) else ice_thickness_nm
        obs = Observation(cid=r.cid, band=r.band, sca=r.sca, mjd=r.mjd, z=0.0,
                          ice_thickness_nm=thk)
        res = compute_magcor(obs, r.sed, filters, ice_model, calib_star=calib_star,
                             grid=grid, reference_sca=reference_sca)
        rows.append({
            "cid": r.cid, "band": r.band, "sca": r.sca, "mjd": r.mjd, "trest": r.trest,
            "ice_thickness_nm": res["ice_thickness"],
            # exact magcor from our process
            "magcor_shift": res["magcor_shift"],
            "magcor_ice": res["magcor_ice"],
            "magcor_total": res["magcor_total"],
            # wavecor-corrected mag (what a pure wavelength shift predicts) + residual
            "magcorwave_shift": res["magcorwave_shift"],
            "magcorwave_ice": res["magcorwave_ice"],
            "magcorwave_total": res["magcorwave_total"],
            "resid_total": res["resid_total"],
            "resid_shift": res["resid_shift"],
            "resid_ice": res["resid_ice"],
            "wavecor": res["wavecor"],
            "lam_eff_ref": res["lam_eff_ref"], "lam_eff_obs": res["lam_eff_obs"],
        })
    out = pd.DataFrame(rows)
    log.info("magcor over %d MODELSPEC rows (reference SCA %s)", len(out), reference_sca)
    return out
