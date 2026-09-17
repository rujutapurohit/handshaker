"""Magnitude corrections (magcor) - the core of Handshaker.

For each observation we build several versions of the bandpass and compare
synthetic magnitudes of the SN SED through them:

    reference : nominal(band)                       - no shift, no ice
    shift     : shifted(band, sca)                  - SCA/chromatic shift only
    ice       : nominal(band) * ice(MJD)            - ice absorption only
    total     : shifted(band, sca) * ice(MJD)       - both together

The raw correction for an effect is ``mag(perturbed) - mag(reference)`` on the SN
SED (this matches the prototype's ``handshake``). Because the photometric system
is tied to calibration stars observed through the *same* perturbed bandpass, the
correction actually applied is the **chromatic differential** between the SN and
the average calibration star (per DES):

    magcor = [mag_SN(perturbed) - mag_SN(nominal)]
           - [mag_star(perturbed) - mag_star(nominal)]

With no calibration-star SED this reduces to the SN-only shift (a correction to an
idealized flat reference), i.e. the prototype's ``handshake`` result.
"""

from __future__ import annotations

import numpy as np

from .logging_utils import get_logger
from .observation import Observation
from .synphot import synphot_mag, effective_wavelength

log = get_logger("magcor")

# The perturbation applied to the bandpass for each named effect.
EFFECTS = ("shift", "ice", "total")


def _perturbed_bandpass(effect, band, sca, mjd, filters, ice_model):
    """Return ``(wave, thru)`` for the named effect's perturbed bandpass."""
    if effect == "shift":
        return filters.shifted(band, sca)
    if effect == "ice":
        wave, thru = filters.nominal(band)
        return wave, ice_model.apply(wave, thru, mjd)
    if effect == "total":
        wave, thru = filters.shifted(band, sca)
        return wave, ice_model.apply(wave, thru, mjd)
    raise ValueError(f"unknown effect {effect!r}")


def _delta_mag(sed_wave, sed_flux, ref_band, pert_band):
    """``mag(perturbed) - mag(reference)`` for one SED, or nan if either is undefined."""
    m_ref = synphot_mag(sed_wave, sed_flux, ref_band[0], ref_band[1])
    m_pert = synphot_mag(sed_wave, sed_flux, pert_band[0], pert_band[1])
    if not (np.isfinite(m_ref) and np.isfinite(m_pert)):
        return np.nan
    return m_pert - m_ref


def compute_magcor(obs: Observation, sn_sed, filters, ice_model, calib_star=None,
                   effects=EFFECTS):
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

    Returns
    -------
    dict
        ``ice_thickness``, and for each effect ``magcor_<effect>`` plus the
        ``magcor_<effect>_sn`` / ``magcor_<effect>_star`` breakdown; ``magcor_total``
        is aliased to the ``total`` effect. Effective wavelengths of the reference
        and shifted band are included for diagnostics.
    """
    band = obs.band
    ref_band = filters.nominal(band)

    # SN SED in the observed frame; calibration star is Galactic (z=0).
    sn_wave, sn_flux = sn_sed.observed_frame(epoch=obs.epoch, z=obs.z, x1=obs.x1, c=obs.c)
    if calib_star is not None:
        star_wave, star_flux = calib_star.observed_frame(z=0.0)

    out = {"ice_thickness": float(np.asarray(ice_model.thickness(obs.mjd)))}

    for effect in effects:
        pert_band = _perturbed_bandpass(effect, band, obs.sca, obs.mjd, filters, ice_model)
        d_sn = _delta_mag(sn_wave, sn_flux, ref_band, pert_band)
        if calib_star is not None:
            d_star = _delta_mag(star_wave, star_flux, ref_band, pert_band)
        else:
            d_star = 0.0
        out[f"magcor_{effect}_sn"] = d_sn
        out[f"magcor_{effect}_star"] = d_star
        out[f"magcor_{effect}"] = d_sn - d_star

    out["magcor_total"] = out.get("magcor_total", np.nan)
    out["lam_eff_nominal"] = effective_wavelength(ref_band[0], ref_band[1], sn_wave, sn_flux)
    shift_band = filters.shifted(band, obs.sca)
    out["lam_eff_shifted"] = effective_wavelength(shift_band[0], shift_band[1], sn_wave, sn_flux)
    out["wavecor"] = out["lam_eff_shifted"] - out["lam_eff_nominal"]
    return out
