"""Mode 2: polyfit -- validate the polynomial approximation to the ICE throughput.

The ice model multiplies throughput by a wavelength-dependent factor
``T(t)/T(t=0)`` (the released ``spectral_responses`` model). The science question
is whether that factor can be replaced, within a band, by a low-order polynomial
without biasing the photometry. For a (band, ice thickness) we take the true ice
ratio over the band's wavelength support and fit it with an Nth-order polynomial:

    ROW  SCA  ICE_THICK  BAND  WAVE  RATIO_TRUE  RATIO_POLY  RESIDUAL

``RATIO_TRUE`` is the true ice ratio, ``RATIO_POLY`` the polynomial approximation,
``RESIDUAL = RATIO_TRUE - RATIO_POLY``. Plots come straight from the table and the
residual RMS says how good order N is. (This is distinct from the per-SCA filter
*shift* -- the field-dependent chromatic effect -- which lives in ``magcor``.)

The fit is in reduced wavelength l = (wave - wmid)/whalf over the band support for
conditioning; ``coeffs`` are in that reduced basis (numpy.polyfit order).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .grid import DEFAULT_GRID
from .logging_utils import get_logger

log = get_logger("polyfit")


def polyfit_ice_band(band, thickness_nm, filters, ice_model, order=5, sca=None,
                     support_frac=0.01, grid=None):
    """Fit the true ice ratio ``T(t)/T(0)`` in one band with an Nth-order polynomial.

    Parameters
    ----------
    band : str
        Band whose wavelength support the ice ratio is fit over.
    thickness_nm : float
        Ice thickness [nm] at which to evaluate the true ice model.
    filters : FilterSource
        Used only to get the band's nominal curve (defines the support).
    ice_model : IceModel
        Supplies the true ice ratio via ``relative_transmission(thickness_nm)``.
    order : int
        Polynomial order.
    sca : int, optional
        SCA of the ice response (for the table's SCA column).
    support_frac : float
        Fit over wavelengths where the nominal throughput exceeds this fraction
        of its peak.
    """
    grid = grid or DEFAULT_GRID
    # Both curves on the shared grid; the band's support is where nominal has flux.
    nominal_thru = filters.nominal(band).regrid(grid)             # FULL -> 0 outside
    ice_ratio = ice_model.relative_transmission(thickness_nm).regrid(grid)  # PERT -> 1
    mask = nominal_thru > support_frac * nominal_thru.max()
    wave = grid.wave[mask]
    ratio_true = ice_ratio[mask]

    wmid = 0.5 * (wave.min() + wave.max())
    whalf = 0.5 * (wave.max() - wave.min()) or 1.0
    l = (wave - wmid) / whalf
    coeffs = np.polyfit(l, ratio_true, order)
    ratio_poly = np.polyval(coeffs, l)
    residual = ratio_true - ratio_poly
    residual_rms = float(np.sqrt(np.mean(residual ** 2)))
    max_abs_residual = float(np.max(np.abs(residual)))

    table = pd.DataFrame({
        "ROW": np.arange(len(wave)),
        "SCA": sca,
        "ICE_THICK": float(thickness_nm),
        "BAND": band,
        "WAVE": wave,
        "RATIO_TRUE": ratio_true,
        "RATIO_POLY": ratio_poly,
        "RESIDUAL": residual,
    })
    log.info("polyfit ice %s d=%.0fnm: order=%d residual_rms=%.2e max|res|=%.2e",
             band, thickness_nm, order, residual_rms, max_abs_residual)
    return {
        "table": table,
        "coeffs": coeffs,
        "residual_rms": residual_rms,
        "max_abs_residual": max_abs_residual,
        "support": (float(wave.min()), float(wave.max())),
    }
