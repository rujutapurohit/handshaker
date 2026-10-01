"""Spectral energy distributions for Handshaker.

Two kinds of SED are needed (per the spec's "Magcor accounts for ... SED
difference between SN and average calibration star"):

* the **supernova** SED - here the SALT3 model, with (x1, c) fit from a spectrum;
* the **calibration star** SED - a static reference spectrum.

Everything is expressed through a small :class:`SEDModel` interface so the pipeline
never hard-codes SALT3: tests and the runnable example use :func:`synthetic_sed`
(a data-free blackbody :class:`FixedSED`), while production uses
:class:`SALT3SEDModel`, which reads the SALT3 templates from ``$SNDATA_ROOT``.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares

from .logging_utils import get_logger

log = get_logger("sed")

# SALT2/SALT3 color law is defined in reduced wavelength l = (wave - B)/(V - B).
SALT2_B_WAVELENGTH = 4302.57
SALT2_V_WAVELENGTH = 5428.55

class SEDModel:
    """Abstract SED provider.

    Subclasses implement :meth:`rest_frame`; the base class supplies the
    cosmological redshifting used by every consumer.
    """

    def rest_frame(self, epoch=0.0, x1=0.0, c=0.0):
        """Return rest-frame ``(wave [A], flux [erg/s/cm^2/A])`` at ``epoch`` days."""
        raise NotImplementedError

    def observed_frame(self, epoch=0.0, z=0.0, x1=0.0, c=0.0):
        """Redshift the rest-frame SED to ``z`` with surface-brightness dimming."""
        wave, flux = self.rest_frame(epoch=epoch, x1=x1, c=c)
        return wave * (1.0 + z), flux / (1.0 + z)


class FixedSED(SEDModel):
    """A fixed spectrum (e.g. a calibration star) that ignores epoch/x1/c."""

    def __init__(self, wave, flux, name="fixed"):
        self.wave = np.asarray(wave, dtype=float)
        self.flux = np.asarray(flux, dtype=float)
        self.name = name

    def rest_frame(self, epoch=0.0, x1=0.0, c=0.0):
        return self.wave, self.flux


def _planck(wave_angstrom, temperature_k):
    """Planck spectral radiance in per-Angstrom units (arbitrary normalization)."""
    wave_cm = np.asarray(wave_angstrom, dtype=float) * 1e-8
    h, c_light, kB = 6.62607015e-27, 2.99792458e10, 1.380649e-16  # cgs
    x = h * c_light / (wave_cm * kB * temperature_k)
    return 1.0 / (wave_cm ** 5) / np.expm1(x)


def synthetic_sed(temperature_k=9000.0, wave=None, name="synthetic"):
    """A data-free blackbody spectrum as a :class:`FixedSED`.

    Used by the runnable example and the test suite so Handshaker exercises the
    full shift/ice/synphot machinery without any external data. It is static in
    epoch/x1/c (redshift still applies via ``observed_frame``); use
    :class:`SALT3SEDModel` for a real, epoch/x1/c-dependent SN SED.
    """
    wave = np.arange(3000.0, 25000.0, 10.0) if wave is None else np.asarray(wave, float)
    return FixedSED(wave, _planck(wave, temperature_k), name=name)


def default_salt3_model_dir():
    """``$SNDATA_ROOT/models/SALT3/SALT3.P22-NIR``"""
    root = os.environ.get("SNDATA_ROOT")
    if not root:
        return None
    return os.path.join(root, "models", "SALT3", "SALT3.P22-NIR")


def _color_law_poly(l, coeffs):
    """SALT2/SALT3 color-law polynomial P(l): P(0)=0, P(1)=-1 by construction."""
    val = -l
    for i, p in enumerate(coeffs):
        val = val + p * (l - l ** (i + 2))
    return val


def _color_law_poly_deriv(l, coeffs):
    d = -1.0
    for i, p in enumerate(coeffs):
        d = d + p * (1 - (i + 2) * l ** (i + 1))
    return d


class SALT3SEDModel(SEDModel):
    """SALT3 SN SED read from the SALT3 model directory.

    ``model_dir`` defaults to ``$SNDATA_ROOT/models/SALT3/SALT3.P22-NIR``. The
    templates and color law are loaded lazily on first use and cached, so
    constructing the model is cheap and never touches disk.
    """

    def __init__(self, model_dir=None):
        self.model_dir = model_dir or default_salt3_model_dir()
        if not self.model_dir:
            raise ValueError(
                "No SALT3 model_dir given and $SNDATA_ROOT is not set. "
                "Pass model_dir=... or use synthetic_sed() for testing."
            )
        self._cl = None  # (wave_lo, wave_hi, coeffs)

    # -- template access --------------------------------------------------- #
    @lru_cache(maxsize=None)
    def _load_template(self, component):
        path = os.path.join(self.model_dir, f"salt3_template_{component}.dat.gz")
        if not os.path.exists(path):
            path = os.path.join(self.model_dir, f"salt3_template_{component}.dat")
        phase, wave, flux = np.loadtxt(path, unpack=True)
        phases = np.unique(phase)
        waves = np.unique(wave)
        return phases, waves, flux.reshape(len(phases), len(waves))

    def components_at_epoch(self, epoch):
        """Rest-frame ``(wave, M0, M1)`` interpolated to ``epoch`` (days from peak)."""
        phases0, waves0, grid0 = self._load_template(0)
        phases1, waves1, grid1 = self._load_template(1)
        m0 = RectBivariateSpline(phases0, waves0, grid0, kx=1, ky=1)(epoch, waves0)[0]
        m1 = RectBivariateSpline(phases1, waves1, grid1, kx=1, ky=1)(epoch, waves1)[0]
        return waves0, m0, m1

    # -- color law --------------------------------------------------------- #
    def _color_law_params(self):
        if self._cl is None:
            info_path = os.path.join(self.model_dir, "SALT3.INFO")
            with open(info_path) as f:
                for line in f:
                    if line.strip().startswith("COLORCOR_PARAMS:"):
                        vals = line.split(":", 1)[1].split()
                        lo, hi, n = float(vals[0]), float(vals[1]), int(vals[2])
                        coeffs = np.array([float(v) for v in vals[3:3 + n]])
                        self._cl = (lo, hi, coeffs)
                        break
            if self._cl is None:
                raise ValueError(f"COLORCOR_PARAMS not found in {info_path}")
        return self._cl

    def color_law(self, wave):
        """SALT2/SALT3 color law CL(wave); flux factor is ``10**(-0.4*c*CL)``.

        Linearly extrapolated outside the model's wavelength range using the
        polynomial's value and derivative at the boundary.
        """
        lo, hi, coeffs = self._color_law_params()
        wave = np.atleast_1d(np.asarray(wave, dtype=float))
        span = SALT2_V_WAVELENGTH - SALT2_B_WAVELENGTH
        l = (wave - SALT2_B_WAVELENGTH) / span
        l_lo = (lo - SALT2_B_WAVELENGTH) / span
        l_hi = (hi - SALT2_B_WAVELENGTH) / span
        p_lo, p_hi = _color_law_poly(l_lo, coeffs), _color_law_poly(l_hi, coeffs)
        dp_lo, dp_hi = _color_law_poly_deriv(l_lo, coeffs), _color_law_poly_deriv(l_hi, coeffs)
        return np.where(
            l < l_lo, p_lo + dp_lo * (l - l_lo),
            np.where(l > l_hi, p_hi + dp_hi * (l - l_hi), _color_law_poly(l, coeffs)))

    # -- SED --------------------------------------------------------------- #
    def rest_frame(self, epoch=0.0, x1=0.0, c=0.0):
        wave, m0, m1 = self.components_at_epoch(epoch)
        flux = 1.0 * (m0 + x1 * m1) * 10.0 ** (-0.4 * c * self.color_law(wave))
        return wave, flux

    # -- fitting ----------------------------------------------------------- #
    def fit_x1_color(self, spectrum_wave, spectrum_flux, epoch, x0_guess=1.0):
        """Fit (x0, x1, c) to a rest-frame flux-calibrated spectrum at ``epoch``."""
        model_wave, m0, m1 = self.components_at_epoch(epoch)
        m0_on = np.interp(spectrum_wave, model_wave, m0)
        m1_on = np.interp(spectrum_wave, model_wave, m1)
        cl_on = self.color_law(spectrum_wave)

        def residuals(params):
            x0, x1, c = params
            model = x0 * (m0_on + x1 * m1_on) * 10.0 ** (-0.4 * c * cl_on)
            return model - spectrum_flux

        result = least_squares(residuals, x0=[x0_guess, 0.0, 0.0])
        x0, x1, c = result.x
        return {"x0": x0, "x1": x1, "c": c, "success": result.success, "cost": result.cost}
