"""Water-ice throughput model for Roman WFI.

Ice accreting on the cold optics multiplies the nominal throughput by a
wavelength-dependent, thickness-dependent factor T(t)/T(t=0). handshaker treats
that factor as a :class:`~handshaker.filters.Transmission` PERTURBATION composed
onto the nominal bandpass.

The "released" model reads the interim transmission-vs-thickness response
delivered by the calibration group (``spectral_responses_SCA<n>.ecsv``): columns
``T_ratio_d<thickness>nm`` giving T(t)/T(t=0) at a grid of ice thicknesses. For a
requested thickness we linearly interpolate between the two bracketing columns,
per wavelength. Wavelengths in the ecsv are in nm and are converted to Angstrom
to match the filter bandpasses.

Ice *thickness* per observation is taken from ``obs.ice_thickness_nm`` when
supplied. The toy deposition-vs-position/MJD model in :mod:`ice_rep_sim` (needs
the growth-rate mosaic ``.npz``) can compute a thickness from (SCA, pixel, MJD)
instead; that path is exposed via :meth:`ReleasedIceModel.thickness_from_position`
but is not required for the default per-observation-thickness workflow, so the
model runs without the mosaic file.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

import numpy as np
from astropy.table import Table

from .filters import Transmission, TransmissionKind
from .logging_utils import get_logger

log = get_logger("ice")

NM_TO_ANGSTROM = 10.0
_BUNDLED_RESPONSE = "data/spectral_responses_SCA10.ecsv"


def _bundled_response_path() -> str:
    """Absolute path to the ice response ecsv shipped inside the package."""
    from importlib.resources import files
    return str(files("handshaker").joinpath(_BUNDLED_RESPONSE))


@lru_cache(maxsize=None)
def _load_response(path: str):
    """Load a released response ecsv -> (wave_A, thicknesses_nm, ratio[nwave, nthick])."""
    tab = Table.read(path)
    wave_a = np.asarray(tab["wavelength"], dtype=float) * NM_TO_ANGSTROM
    cols = []
    for name in tab.colnames:
        if "_d" in name and name.endswith("nm"):
            thick = float(name[name.find("_d") + 2:name.find("nm")])
            cols.append((thick, name))
    cols.sort()
    thicknesses = np.array([t for t, _ in cols])
    ratio = np.column_stack([np.asarray(tab[name], dtype=float) for _, name in cols])
    log.debug("loaded ice response %s: %d wavelengths, thicknesses %s nm",
              path, len(wave_a), thicknesses.tolist())
    return wave_a, thicknesses, ratio


class IceModel:
    """Common interface: a thickness per observation and a throughput factor."""

    def thickness_nm(self, obs) -> float:
        raise NotImplementedError

    def relative_transmission(self, thickness_nm: float, band=None) -> Transmission:
        """T(t)/T(0) as a PERTURBATION :class:`Transmission` (wave in Angstrom).

        ``band`` is used only by models that fit per band (e.g. the polyfit model);
        the true/none models ignore it.
        """
        raise NotImplementedError

    def apply(self, wave, thru, obs):
        """Return ``thru`` multiplied by the ice factor at this observation's thickness."""
        raise NotImplementedError


class NoIceModel(IceModel):
    """Ice disabled: zero thickness, throughput unchanged."""

    name = "none"

    def thickness_nm(self, obs) -> float:
        return 0.0

    def relative_transmission(self, thickness_nm: float, band=None) -> Transmission:
        wave = np.array([1000.0, 30000.0])
        return Transmission(wave, np.ones_like(wave), TransmissionKind.PERTURBATION, name="ice_none")

    def apply(self, wave, thru, obs):
        return np.asarray(thru, dtype=float)


class ReleasedIceModel(IceModel):
    """Ice throughput from a released transmission-vs-thickness response ecsv."""

    name = "released"

    def __init__(self, response_path: str | None = None):
        self.response_path = response_path or _bundled_response_path()
        # Fail fast with a clear message if the response file is missing/unreadable.
        _load_response(self.response_path)
        m = re.search(r"SCA(\d+)", os.path.basename(self.response_path))
        self.sca = int(m.group(1)) if m else None

    def thickness_nm(self, obs) -> float:
        if getattr(obs, "ice_thickness_nm", None) is not None:
            return float(obs.ice_thickness_nm)
        return 0.0

    def thickness_from_position(self, sca, pix_row_col, mjd, rate_mosaic_path=None) -> float:
        """Toy deposition thickness from (SCA, pixel, MJD) via :mod:`ice_rep_sim`.

        Requires the growth-rate mosaic ``.npz``. Not used by the default workflow
        (which takes thickness straight from the observation); provided so the
        rep-sim path is available once the mosaic is on disk.
        """
        from . import ice_rep_sim
        if rate_mosaic_path is not None:
            ice_rep_sim.rate_mosaic_filename = rate_mosaic_path
        return float(ice_rep_sim.ice_thickness(sca, pix_row_col, mjd))

    def relative_transmission(self, thickness_nm: float, band=None) -> Transmission:
        wave_a, thicknesses, ratio = _load_response(self.response_path)
        t = float(thickness_nm)
        lo, hi = thicknesses.min(), thicknesses.max()
        if t < lo or t > hi:
            log.warning("ice thickness %.1f nm outside response grid [%.0f, %.0f]; clamping", t, lo, hi)
            t = min(max(t, lo), hi)
        idx = int(np.clip(np.searchsorted(thicknesses, t, side="right") - 1, 0, len(thicknesses) - 2))
        t0, t1 = thicknesses[idx], thicknesses[idx + 1]
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        r = (1.0 - w) * ratio[:, idx] + w * ratio[:, idx + 1]
        return Transmission(wave_a, r, TransmissionKind.PERTURBATION, name=f"ice_{t:.0f}nm")

    def apply(self, wave, thru, obs):
        thk = self.thickness_nm(obs)
        thru = np.asarray(thru, dtype=float)
        if thk <= 0:
            return thru
        pert = self.relative_transmission(thk)
        factor = np.interp(wave, pert.wave, pert.thru, left=1.0, right=1.0)
        return thru * factor


class PolyfitIceModel(IceModel):
    """Polynomial approximation to the released ice ratio, fit *per band*.

    This is the "POLYFIT" ice model of the spec: because ice is a passband-shape
    change (not a wavelength shift), it can't be captured by a wavecor, but an
    Nth-order polynomial across a band *can*. For each (band, thickness) it fits the
    true ice ratio T(t)/T(0) over the band's support and returns the polynomial
    ratio. Feed it to magcor exactly like the true model; the difference in the
    resulting ice magcor is the polynomial-approximation error (mag_poly - mag_true).
    """

    name = "polyfit"

    def __init__(self, true_model, filters, order=5, support_frac=0.01, grid=None):
        from .grid import DEFAULT_GRID
        self.true = true_model
        self.filters = filters
        self.order = int(order)
        self.support_frac = float(support_frac)
        self.grid = grid or DEFAULT_GRID
        self.sca = getattr(true_model, "sca", None)

    def thickness_nm(self, obs) -> float:
        return self.true.thickness_nm(obs)

    def relative_transmission(self, thickness_nm: float, band=None) -> Transmission:
        true = self.true.relative_transmission(thickness_nm)
        if band is None:                       # no band context -> fall back to the true ratio
            return true
        g = self.grid
        nom = self.filters.nominal(band).regrid(g)
        wave = g.wave[nom > self.support_frac * nom.max()]
        true_on = np.interp(wave, true.wave, true.thru, left=1.0, right=1.0)
        wmid = 0.5 * (wave.min() + wave.max())
        whalf = 0.5 * (wave.max() - wave.min()) or 1.0
        l = (wave - wmid) / whalf
        poly = np.polyval(np.polyfit(l, true_on, self.order), l)
        return Transmission(wave, poly, TransmissionKind.PERTURBATION,
                            name=f"ice_poly{self.order}_{band}_{float(thickness_nm):.0f}nm")

    def apply(self, wave, thru, obs):
        thk = self.thickness_nm(obs)
        thru = np.asarray(thru, dtype=float)
        if thk <= 0:
            return thru
        pert = self.relative_transmission(thk, band=getattr(obs, "band", None))
        return thru * np.interp(wave, pert.wave, pert.thru, left=1.0, right=1.0)


class SocIceModel(ReleasedIceModel):
    """Official SOC ice response (not available yet).

    Same mechanism as the released model but reads a SOC-delivered response ecsv;
    set ``ice.response_path`` once it exists. Errors clearly until then.
    """

    name = "soc"

    def __init__(self, response_path=None):
        if not response_path:
            raise NotImplementedError(
                "SOC ice model not available yet; set ice.response_path when it is.")
        super().__init__(response_path=response_path)


_MODELS = {
    "thushara": "released",  # the released ice model (Thushara's)
    "released": "released",
    "true": "released",
    "soc": "soc",            # official SOC ice response (not available yet)
    "polyfit": "polyfit",    # polynomial approximation to the true ice ratio, per band
    "poly": "polyfit",
    "none": "none",
    "off": "none",
}


def get_ice_model(flag, response_path=None, filters=None, order=5) -> IceModel:
    """Build an :class:`IceModel` from a config source/flag.

    'thushara'/'released' -> the released model; 'soc' -> the SOC response (needs
    ``response_path``); 'polyfit' -> the per-band polynomial approx (needs
    ``filters``); 'none' -> disabled.
    """
    key = _MODELS.get(str(flag).lower())
    if key is None:
        raise ValueError(f"unknown ice source {flag!r}; choose 'thushara', 'soc', 'polyfit', or 'none'")
    if key == "none":
        log.info("ice model disabled (flag=%s)", flag)
        return NoIceModel()
    if key == "soc":
        log.info("using SOC ice model from %s", response_path)
        return SocIceModel(response_path=response_path)
    true = ReleasedIceModel(response_path=response_path)
    if key == "polyfit":
        if filters is None:
            raise ValueError("polyfit ice model needs filters= (for each band's support)")
        log.info("using polyfit ice model (order %d) over %s", order, true.response_path)
        return PolyfitIceModel(true, filters, order=order)
    log.info("using released (thushara) ice model from %s", true.response_path)
    return true
