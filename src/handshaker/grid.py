"""The canonical wavelength grid.

Every wavelength-dependent quantity (filter transmission, ice factor, SED flux)
is resampled onto ONE fixed grid at load time. After that:

* combining transmissions is elementwise: ``total = nominal * ice * shifted``
  (no bin-by-bin interpolation, no "which grid is this on?" bookkeeping);
* a synthetic magnitude is a single ``trapz`` over the shared grid;
* everything is the same length and lines up, so you can ``np.column_stack`` the
  curves and eyeball/plot them directly when debugging.

Interpolation happens exactly once per curve, here, at the boundary where raw data
enters the package. The grid range/step is configurable (``grid:`` in the config).
"""

from __future__ import annotations

import numpy as np

# numpy>=2.0 renamed trapz -> trapezoid; fall back for older installs.
try:  # pragma: no cover - trivial shim
    _trapz = np.trapezoid
except AttributeError:  # pragma: no cover
    _trapz = np.trapz


class WaveGrid:
    """A fixed wavelength grid [Angstrom] plus resample/integrate helpers."""

    def __init__(self, lo: float = 3000.0, hi: float = 25500.0, step: float = 10.0):
        self.lo, self.hi, self.step = float(lo), float(hi), float(step)
        # +half-step so `hi` is included regardless of floating-point edge effects.
        self.wave = np.arange(self.lo, self.hi + 0.5 * self.step, self.step)

    def __len__(self):
        return self.wave.size

    def __repr__(self):
        return f"WaveGrid({self.lo:.0f}-{self.hi:.0f} A, step {self.step:.0f} A, n={len(self)})"

    def regrid(self, wave, values, fill=0.0):
        """Resample ``values(wave)`` onto this grid.

        ``fill`` is the value outside the source support: 0 for an absolute
        bandpass (no throughput), 1 for a multiplicative perturbation (no effect).
        A 2-tuple sets (left, right) separately.
        """
        left, right = (fill, fill) if np.isscalar(fill) else (fill[0], fill[1])
        return np.interp(self.wave, np.asarray(wave, dtype=float),
                         np.asarray(values, dtype=float), left=left, right=right)

    def integrate(self, values):
        """Trapezoidal integral of ``values`` over the grid."""
        return _trapz(np.asarray(values, dtype=float), self.wave)

    @classmethod
    def from_config(cls, spec) -> "WaveGrid":
        """Build from a config mapping ``{min, max, step}`` (any missing key -> default)."""
        spec = dict(spec or {})
        return cls(lo=spec.get("min", 3000.0), hi=spec.get("max", 25500.0),
                   step=spec.get("step", 10.0))


DEFAULT_GRID = WaveGrid()
