"""Filter transmission sources.

The source of filter transmissions is kept separate behind a common interface::

    get_filter("roman_filters", root=...) -> RomanDirFilterSource  (current nominal set)
    get_filter("soc")                      -> SOCFilterSource        (not available yet)

(``synthetic`` also exists as a data-free fixture for tests / the no-data demo.)
Every source exposes the same three-method interface:

    nominal(band)          -> (wave, thru)   reference / un-shifted bandpass
    shifted(band, sca)     -> (wave, thru)   SCA-specific "chromatic" bandpass
    available_bands()      -> list[str]

so :mod:`handshaker.magcor` can compute ``mag(shifted) - mag(nominal)`` without
knowing the provenance of the curves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.special import erf

from .logging_utils import get_logger

log = get_logger("filters")


# Transmission value type: carries an explicit "what is this curve" flag
class TransmissionKind(str, Enum):
    """Whether a returned curve is an absolute bandpass or a multiplicative factor."""

    FULL = "full"                  # absolute throughput T(lambda), ~[0, 1]
    PERTURBATION = "perturbation"  # relative factor to multiply onto a FULL curve


@dataclass(frozen=True)
class Transmission:
    """A transmission curve tagged as a FULL bandpass or a PERTURBATION.

    Unpacks and indexes like a ``(wave, thru)`` tuple, so existing callers that do
    ``wave, thru = source.nominal(band)`` keep working, while ``.kind`` /
    ``.is_full`` / ``.is_perturbation`` expose the explicit flag.
    """

    wave: np.ndarray
    thru: np.ndarray
    kind: TransmissionKind = TransmissionKind.FULL
    name: str = ""

    def __iter__(self):
        yield self.wave
        yield self.thru

    def __getitem__(self, i):
        return (self.wave, self.thru)[i]

    @property
    def is_full(self):
        return self.kind == TransmissionKind.FULL

    @property
    def is_perturbation(self):
        return self.kind == TransmissionKind.PERTURBATION

    def _default_fill(self):
        # Outside its support a bandpass has zero throughput; a perturbation has
        # unit (no-op) factor. So combining regridded curves stays correct.
        return 1.0 if self.is_perturbation else 0.0

    def regrid(self, grid, fill=None):
        """Return this curve's values resampled onto ``grid`` (a WaveGrid)."""
        return grid.regrid(self.wave, self.thru, self._default_fill() if fill is None else fill)

    def on_grid(self, grid, fill=None):
        """Return a new Transmission on ``grid.wave`` (same kind/name)."""
        return Transmission(grid.wave, self.regrid(grid, fill), self.kind, self.name)


def compose(base, *perturbations):
    """Apply PERTURBATION curves onto a FULL ``base``, returning a FULL Transmission.

    Each perturbation is interpolated onto ``base.wave`` (taken as 1 outside its
    support) and multiplied in. Raises if ``base`` isn't FULL or a factor isn't a
    PERTURBATION, so full/relative curves can never be silently mixed up.
    """
    if not isinstance(base, Transmission) or not base.is_full:
        raise TypeError("compose() base must be a FULL Transmission")
    thru = np.asarray(base.thru, dtype=float).copy()
    names = [base.name] if base.name else []
    for p in perturbations:
        if not isinstance(p, Transmission) or not p.is_perturbation:
            raise TypeError("compose() extra args must be PERTURBATION Transmissions")
        thru = thru * np.interp(base.wave, p.wave, p.thru, left=1.0, right=1.0)
        if p.name:
            names.append(p.name)
    return Transmission(base.wave, thru, TransmissionKind.FULL, name="*".join(names))


# Base class
class FilterSource:
    """Abstract provider of nominal and SCA-shifted bandpasses.

    Both accessors return a FULL :class:`Transmission` (an absolute bandpass).
    :meth:`nominal` is the canonical reference curve; perturbations (e.g. the ice
    model) are composed onto it with :func:`compose`.
    """

    name = "base"

    def nominal(self, band):
        """Canonical reference bandpass for ``band`` as a FULL :class:`Transmission`."""
        raise NotImplementedError

    def shifted(self, band, sca):
        """SCA-specific chromatic bandpass as a FULL :class:`Transmission`."""
        raise NotImplementedError

    def available_bands(self):
        raise NotImplementedError

    def reference(self, band):
        """Bandpass used as the polyfit reference (defaults to the nominal one)."""
        return self.nominal(band)


# ROMAN_FILTERS directory layout  ( thushara / soc in the future?)
def _load_two_column(path):
    """Load a 2-column wavelength/throughput file, skipping any non-numeric lines.

    Nominal Roman files are inconsistent about whether headers start with '#', (this is very annoying sometimes)
    so we skip any line that isn't exactly two floats.
    """
    wave, thru = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                w, t = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            wave.append(w)
            thru.append(t)
    if not wave:
        raise ValueError(f"no 2-column numeric data found in {path}")
    return np.array(wave), np.array(thru)


class RomanDirFilterSource(FilterSource):
    """Bandpasses laid out as ``<root>/<band>/ROMAN_<band>[_shifted_<sca>].dat``.

    This is the layout used in the prototype notebook
    (``ROMAN_FILTERS/{band}/ROMAN_{band}.dat`` for nominal, and
    ``ROMAN_{band}_shifted_{sca}.dat`` for the per-SCA curves). 
    """

    def __init__(self, root, name="roman_dir", n_sca=18):
        self.root = root
        self.name = name
        self.n_sca = n_sca

    def _nominal_path(self, band):
        in_band = os.path.join(self.root, band, f"ROMAN_{band}.dat")
        if os.path.exists(in_band):
            return in_band
        return os.path.join(self.root, f"ROMAN_{band}.dat")  # flat fallback

    def _shifted_path(self, band, sca):
        return os.path.join(self.root, band, f"ROMAN_{band}_shifted_{sca}.dat")

    def nominal(self, band):
        path = self._nominal_path(band)
        log.debug("nominal transmission for %s: %s", band, path)
        wave, thru = _load_two_column(path)
        return Transmission(wave, thru, TransmissionKind.FULL, name=f"{band}/nominal")

    def shifted(self, band, sca):
        if sca is None:
            return self.nominal(band)
        path = self._shifted_path(band, sca)
        log.debug("shifted transmission for %s SCA %s: %s", band, sca, path)
        wave, thru = _load_two_column(path)
        return Transmission(wave, thru, TransmissionKind.FULL, name=f"{band}/sca{sca}")

    def available_bands(self):
        if not os.path.isdir(self.root):
            return []
        return sorted(
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d)))


class SOCFilterSource(FilterSource):
    """Official SOC transmissions -- not available yet (empty placeholder).

    Populate later: either give it a ``root`` of SOC filter files laid out like
    ROMAN_FILTERS (then it can mirror :class:`RomanDirFilterSource`), or implement
    the real SOC format here. Until then it errors clearly.
    """

    name = "soc"

    def __init__(self, root=None):
        self.root = root

    def nominal(self, band):
        raise NotImplementedError("SOC filters not available yet (filters.source: soc)")

    def shifted(self, band, sca):
        raise NotImplementedError("SOC filters not available yet (filters.source: soc)")

    def available_bands(self):
        return []


# Approximate (center [A], full width [A]) per Roman band, for the synthetic set.
_ROMAN_BANDS = {
    "R062": (6200.0, 2800.0),
    "Z087": (8700.0, 2200.0),
    "Y106": (10600.0, 2800.0),
    "J129": (12900.0, 3400.0),
    "H158": (15800.0, 4100.0),
    "F184": (18400.0, 3200.0),
    "K213": (21300.0, 3500.0),
    "W146": (14600.0, 7700.0),
}


class SyntheticFilterSource(FilterSource):
    """Data-free bandpasses: soft-edged top hats built from :data:`_ROMAN_BANDS`.

    Lets handshaker run end-to-end (example config, tests) with no external filter
    files. ``shifted(band, sca)`` shifts the passband by ``shift_per_sca * sca``
    Angstrom so the SCA/chromatic effect is nonzero and testable.
    """

    def __init__(self, shift_per_sca=6.0, edge_softness=50.0, step=10.0,
                 bands=None, name="synthetic"):
        self.shift_per_sca = float(shift_per_sca)
        self.edge_softness = float(edge_softness)
        self.step = float(step)
        self.bands = dict(bands) if bands else dict(_ROMAN_BANDS)
        self.name = name

    def _tophat(self, center, width, shift=0.0):
        lo, hi = center - width, center + width
        wave = np.arange(lo, hi + self.step, self.step)
        c = center + shift
        half = width / 2.0
        thru = 0.5 * (erf((wave - (c - half)) / self.edge_softness)
                      - erf((wave - (c + half)) / self.edge_softness))
        return wave, thru

    def nominal(self, band):
        center, width = self.bands[band]
        wave, thru = self._tophat(center, width)
        return Transmission(wave, thru, TransmissionKind.FULL, name=f"{band}/nominal")

    def shifted(self, band, sca):
        center, width = self.bands[band]
        shift = 0.0 if sca is None else self.shift_per_sca * sca
        wave, thru = self._tophat(center, width, shift=shift)
        return Transmission(wave, thru, TransmissionKind.FULL, name=f"{band}/sca{sca}")

    def available_bands(self):
        return sorted(self.bands)


_SOURCES = {
    "roman_filters": RomanDirFilterSource,   # the current nominal ROMAN_FILTERS set
    "soc": SOCFilterSource,                   # official SOC transmissions (not available yet)
    "synthetic": SyntheticFilterSource,       # data-free; tests/no-data demo only
}


def get_filter(source, **kwargs):
    """Construct a :class:`FilterSource` by name: ``roman_filters`` or ``soc``.

    ``root=`` gives the ROMAN_FILTERS directory for ``roman_filters``. (``synthetic``
    is a data-free fixture for tests/the no-data demo, not a real source.)
    """
    key = source.lower()
    if key not in _SOURCES:
        raise ValueError(f"unknown filter source {source!r}; choose 'roman_filters' or 'soc'")
    cls = _SOURCES[key]
    if cls is RomanDirFilterSource and "name" not in kwargs:
        kwargs["name"] = key
    log.info("using filter source '%s' (%s)", key, cls.__name__)
    return cls(**kwargs)
