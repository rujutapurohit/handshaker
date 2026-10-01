"""Magnitude corrections (magcor) - the core of Handshaker.

For each observation we build several versions of the bandpass and compare
synthetic magnitudes of the SN SED through them:

    reference : nominal(band)                       - no shift, no ice
    shift     : shifted(band, sca)                  - SCA/chromatic shift only
    ice       : nominal(band) * ice(thickness)      - ice absorption only
    total     : shifted(band, sca) * ice(thickness) - both together

All curves are resampled onto the canonical wavelength grid (see grid.py) once,
so the effect bandpasses above are literally elementwise products and each
synthetic magnitude is a single trapz -- no per-effect interpolation.

The raw correction for an effect is ``mag(perturbed) - mag(reference)`` on the SN
SED (this matches the prototype's ``handshake``). Because the photometric system
is tied to calibration stars observed through the *same* perturbed bandpass, the
correction actually applied is the **chromatic differential** between the SN and
the average calibration star (per DES):

    magcor = [mag_SN(perturbed) - mag_SN(nominal)]
           - [mag_star(perturbed) - mag_star(nominal)]

With no calibration-star SED this reduces to the SN-only shift (a correction to an
idealized flat reference).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .grid import DEFAULT_GRID
from .logging_utils import get_logger
from .observation import Observation
from .synphot import synmag_on_grid, effective_wavelength_on_grid

log = get_logger("magcor")

# The perturbation applied to the bandpass for each named effect.
EFFECTS = ("shift", "ice", "total")


def wavecor_table(filters, bands, scas=range(1, 19), reference_sca=2, sed=None, grid=None):
    """Per-(band, SCA) effective-wavelength shift vs a fixed reference SCA.

    ``wavecor(band, sca) = lam_eff[shifted(band, sca)] - lam_eff[shifted(band, ref)]``.

    This is a *filter property* (one number per band/SCA), unlike the per-observation
    ``wavecor`` from ``compute_magcor``. Weighting: ``sed=None`` -> flat (the
    throughput-weighted mean wavelength, int(T*lam)/int(T)); pass a FixedSED-like
    object (``.wave``, ``.flux``) to SED-weight instead. Use this to compare against
    a reference wavecor table (e.g. the SNANA/kcor per-SCA values).
    """
    grid = grid or DEFAULT_GRID
    flux = grid.regrid(sed.wave, sed.flux) if sed is not None else None
    rows = []
    for band in bands:
        ref = filters.shifted(band, reference_sca).regrid(grid)
        lam_ref = effective_wavelength_on_grid(grid, ref, flux)
        for sca in scas:
            cur = filters.shifted(band, sca).regrid(grid)
            lam = effective_wavelength_on_grid(grid, cur, flux)
            rows.append({"band": band, "sca": sca,
                         "lam_eff": lam, "wavecor": lam - lam_ref})
    return pd.DataFrame(rows)


def _delta_mag(grid, flux_on_grid, ref_thru, pert_thru):
    """``mag(perturbed) - mag(reference)`` on a grid-aligned SED, nan if undefined."""
    m_ref = synmag_on_grid(grid, flux_on_grid, ref_thru)
    m_pert = synmag_on_grid(grid, flux_on_grid, pert_thru)
    if not (np.isfinite(m_ref) and np.isfinite(m_pert)):
        return np.nan
    return m_pert - m_ref


def _shift_bandpass(grid, thru, dlam):
    """Rigidly translate a grid-sampled bandpass redward by ``dlam`` Angstrom.

    ``T_shifted(lambda) = T(lambda - dlam)``. Represents the passband change as a
    pure wavelength shift, so mag(SED; shifted) - mag(SED; ref) is the magnitude a
    wavecor (Delta-lambda) *predicts* -- to compare against the exact magcor.
    """
    return np.interp(grid.wave - dlam, grid.wave, thru, left=0.0, right=0.0)


def compute_magcor(obs: Observation, sn_sed, filters, ice_model, calib_star=None,
                   effects=EFFECTS, grid=None, reference_sca=None):
    """Compute per-effect and total magcor for a single observation.

    Parameters
    ----------
    obs : Observation
    sn_sed : SEDModel
        The supernova SED (evaluated at ``obs.epoch``, redshifted to ``obs.z``).
    filters : FilterSource
    ice_model : IceModel
    calib_star : SEDModel or None
        Average calibration-star SED (z=0). If given, magcors are the SN-minus-star
        chromatic differential; if None, they are the SN-only shift.
    grid : WaveGrid, optional
        Canonical wavelength grid (defaults to grid.DEFAULT_GRID).
    reference_sca : int or None
        The bandpass everything is measured against. None -> the nominal
        (un-shifted) curve. An int -> that fixed SCA's shifted curve, i.e. the
        photometric-system reference the zeropoints are tied to (e.g. SCA 2).

    Returns
    -------
    dict
        ``ice_thickness``, and for each effect ``magcor_<effect>`` plus the
        ``magcor_<effect>_sn`` / ``magcor_<effect>_star`` breakdown; ``magcor_total``
        is aliased to the ``total`` effect. Effective wavelengths of the reference
        and the observation's (shifted) band are included for diagnostics, and
        ``wavecor`` is their difference (the "alter the model" side of the test).
    """
    grid = grid or DEFAULT_GRID
    band = obs.band

    # Every wavelength-dependent quantity onto the shared grid, once.
    if reference_sca is not None:
        reference = filters.shifted(band, reference_sca).regrid(grid)   # fixed reference SCA
    else:
        reference = filters.nominal(band).regrid(grid)                  # nominal reference
    shifted = filters.shifted(band, obs.sca).regrid(grid)               # this obs's SCA
    thickness = float(np.asarray(ice_model.thickness_nm(obs)))
    # band routed through so the polyfit ice model can fit per band; others ignore it.
    ice_factor = ice_model.relative_transmission(thickness, band=band).regrid(grid)  # PERT -> 1 outside

    # Effect bandpasses are now just elementwise products on the grid, all vs the
    # reference: shift only, ice only (on the reference filter), and both (the
    # "true" filter for this observation = its SCA's curve x ice).
    perturbed = {
        "shift": shifted,
        "ice": reference * ice_factor,
        "total": shifted * ice_factor,
    }

    # SEDs onto the grid (SN in observed frame; calibration star is Galactic, z=0).
    sn_flux = grid.regrid(*sn_sed.observed_frame(epoch=obs.epoch, z=obs.z, x1=obs.x1, c=obs.c))
    star_flux = grid.regrid(*calib_star.observed_frame(z=0.0)) if calib_star is not None else None

    out = {"ice_thickness": thickness}
    lam_ref = effective_wavelength_on_grid(grid, reference, sn_flux)
    out["lam_eff_ref"] = lam_ref
    for effect in effects:
        pert_thru = perturbed[effect]
        # (1) exact magcor from our process: mag(true filter) - mag(reference).
        d_sn = _delta_mag(grid, sn_flux, reference, pert_thru)
        d_star = _delta_mag(grid, star_flux, reference, pert_thru) if star_flux is not None else 0.0
        out[f"magcor_{effect}_sn"] = d_sn
        out[f"magcor_{effect}_star"] = d_star
        out[f"magcor_{effect}"] = d_sn - d_star

        # (2) wavelength-shift view: the effect's Delta-lambda_eff, and the magnitude
        # a *pure* rigid shift of the reference passband by that Delta-lambda predicts
        # (the "wavecor-corrected mag").
        dlam = effective_wavelength_on_grid(grid, pert_thru, sn_flux) - lam_ref
        magcorwave = _delta_mag(grid, sn_flux, reference, _shift_bandpass(grid, reference, dlam))
        out[f"wavecor_{effect}"] = dlam
        out[f"magcorwave_{effect}"] = magcorwave
        # (3) residual = the part of magcor that is NOT a wavelength shift
        #     (~0 for the filter shift; nonzero for ice, a passband-shape change).
        out[f"resid_{effect}"] = out[f"magcor_{effect}"] - magcorwave

    out.setdefault("magcor_total", np.nan)
    out["wavecor"] = out.get("wavecor_total", np.nan)              # Delta-lambda of total effect
    out["magcorwave_total"] = out.get("magcorwave_total", np.nan)  # wavecor-corrected mag
    out["lam_eff_obs"] = lam_ref + out["wavecor"]
    return out


def bandpasses_on_grid(band, filters, ice_model, sca=None, thickness_nm=0.0, grid=None):
    """All bandpass pieces for a band, sampled on the shared grid, as a DataFrame.

    Columns: wave, nominal, shifted, ice_factor, and the elementwise products
    nominal*ice and shifted*ice. Everything is aligned to grid.wave, so it's a
    one-liner to plot or diff the curves when debugging.
    """
    import pandas as pd
    grid = grid or DEFAULT_GRID
    nominal = filters.nominal(band).regrid(grid)
    shifted = filters.shifted(band, sca).regrid(grid) if sca is not None else nominal
    ice = ice_model.relative_transmission(thickness_nm).regrid(grid)
    return pd.DataFrame({
        "wave": grid.wave,
        "nominal": nominal,
        "shifted": shifted,
        "ice_factor": ice,
        "nominal_x_ice": nominal * ice,
        "shifted_x_ice": shifted * ice,
    })
