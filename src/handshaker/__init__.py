"""handshaker: photometric corrections (magcor) for Roman WFI calibration effects,
and validation of polynomial approximations to the wavelength-dependent ice
throughput.

Importable as a module and executable as a standalone script (``handshaker``).
"""

from __future__ import annotations

__version__ = "0.1.0"

from .observation import Observation
from .grid import WaveGrid, DEFAULT_GRID
from .filters import get_filter, Transmission, TransmissionKind, compose
from .ice import get_ice_model, ReleasedIceModel, PolyfitIceModel, NoIceModel
from .sed import synthetic_sed, SALT3SEDModel, FixedSED
from .magcor import compute_magcor, bandpasses_on_grid, wavecor_table, EFFECTS
from .modelspec import load_modelspec, magcor_table, attach_columns, mjd_ice_thickness
from .io import load_lcplot, load_fitres, write_snana_filters
from .validate import empirical_magcor, compare_to_empirical
from .pipeline import run, run_config

__all__ = [
    "__version__",
    "Observation",
    "WaveGrid",
    "DEFAULT_GRID",
    "get_filter",
    "Transmission",
    "TransmissionKind",
    "compose",
    "get_ice_model",
    "ReleasedIceModel",
    "PolyfitIceModel",
    "NoIceModel",
    "synthetic_sed",
    "SALT3SEDModel",
    "FixedSED",
    "compute_magcor",
    "bandpasses_on_grid",
    "wavecor_table",
    "load_modelspec",
    "magcor_table",
    "attach_columns",
    "mjd_ice_thickness",
    "load_lcplot",
    "load_fitres",
    "write_snana_filters",
    "empirical_magcor",
    "compare_to_empirical",
    "EFFECTS",
    "run",
    "run_config",
]
