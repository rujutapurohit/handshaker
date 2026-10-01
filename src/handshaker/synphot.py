"""Synthetic photometry: integrate an SED through a bandpass.

from the handshaker notebook. Only *differences* between two synthetic
magnitudes are ever used downstream (magcor is a difference of two synmags), so
any fixed zeropoint convention works as long as it is applied consistently.
"""

from __future__ import annotations

import numpy as np

# numpy>=2.0 renamed trapz -> trapezoid; fall back for older installs.
try:  # pragma: no cover - trivial shim
    _trapz = np.trapezoid
except AttributeError:  # pragma: no cover
    _trapz = np.trapz


def band_flux(wave, flux, band_wave, band_thru):
    """Photon-weighted band flux ``\\int f(l) T(l) l dl``.

    Parameters
    ----------
    wave, flux : array
        SED wavelength grid [A] and flux density [erg/s/cm^2/A].
    band_wave, band_thru : array
        Bandpass wavelength grid [A] and throughput (0-1).

    The throughput is interpolated onto the (finer) SED grid, zero outside the
    band's support, so the SED grid controls the integration resolution.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    thru_on_sed = np.interp(wave, band_wave, band_thru, left=0.0, right=0.0)
    return _trapz(flux * thru_on_sed * wave, wave)


def synphot_mag(wave, flux, band_wave, band_thru):
    """Synthetic magnitude ``-2.5 log10(band_flux)`` (arbitrary but consistent ZP).

    Returns ``+inf`` if the band flux is non-positive (no overlap / all-zero SED),
    so callers can detect and skip undefined magnitudes instead of hitting a
    log-of-nonpositive warning.
    """
    bf = band_flux(wave, flux, band_wave, band_thru)
    if bf <= 0:
        return np.inf
    return -2.5 * np.log10(bf)


def synmag_on_grid(grid, flux, thru):
    """Synthetic magnitude of a grid-aligned SED through a grid-aligned bandpass.

    ``flux`` and ``thru`` are both already sampled on ``grid.wave`` (same length),
    so this is a single trapz with no interpolation. Returns +inf on non-positive
    band flux.
    """
    bf = grid.integrate(np.asarray(flux) * np.asarray(thru) * grid.wave)
    if bf <= 0:
        return np.inf
    return -2.5 * np.log10(bf)


def effective_wavelength_on_grid(grid, thru, flux=None):
    """Effective wavelength of a grid-aligned bandpass (optionally SED-weighted)."""
    w = grid.wave
    weight = thru * w if flux is None else np.asarray(flux) * thru * w
    den = grid.integrate(weight)
    if den <= 0:
        return np.nan
    return grid.integrate(weight * w) / den


def effective_wavelength(band_wave, band_thru, wave=None, flux=None):
    """Throughput- (and optionally SED-) weighted mean wavelength of a bandpass.

    With no SED, returns the flat-spectrum pivot-style mean ``\\int T l dl / \\int T dl``.
    With an SED, weights by ``f(l) T(l) l`` to give the source's effective wavelength.
    """
    band_wave = np.asarray(band_wave, dtype=float)
    band_thru = np.asarray(band_thru, dtype=float)
    if wave is None or flux is None:
        num = _trapz(band_thru * band_wave, band_wave)
        den = _trapz(band_thru, band_wave)
    else:
        wave = np.asarray(wave, dtype=float)
        thru_on_sed = np.interp(wave, band_wave, band_thru, left=0.0, right=0.0)
        weight = np.asarray(flux, dtype=float) * thru_on_sed * wave
        num = _trapz(weight * wave, wave)
        den = _trapz(weight, wave)
    if den <= 0:
        return np.nan
    return num / den
