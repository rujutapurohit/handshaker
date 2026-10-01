"""The per-observation input record.

One :class:`Observation` is the unit handshaker corrects: a single WFI exposure
of a source in a band, with the SN parameters needed to build its SED and the
metadata needed to place it on the focal plane. Observations come from the config
(inline list) or a CSV (see :func:`handshaker.pipeline.load_observations`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Observation:
    """A single WFI observation.

    Attributes
    ----------
    band : str
        Filter/band name (e.g. "J129").
    mjd : float
        Observation MJD (used for the ice thickness/time).
    sca : int, optional
        1-indexed detector (WFI SCA). Cross-checked against (ra, dec) when a real
        focal-plane model is available.
    ra, dec : float, optional
        Sky position (deg) used to place the observation on the focal plane.
    z : float
        Redshift of the source (SN).
    epoch : float
        Rest-frame phase (days from peak). Derived from (mjd, t0, z) if not given.
    x1, c : float
        SALT3 shape/color parameters of the SN.
    ice_thickness_nm : float, optional
        Ice layer thickness at this observation, in nm. If None, the ice model
        may compute it from (sca, position, mjd); if still unknown, treated as 0.
    cid : Any, optional
        Candidate/object id, carried through to the output table.
    t0 : float, optional
        Time of peak (MJD), used to derive ``epoch`` when it isn't supplied.
    meta : dict
        Any extra columns from the input table, carried through untouched.
    """

    band: Optional[str] = None
    mjd: Optional[float] = None
    sca: Optional[int] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    z: float = 0.0
    epoch: float = 0.0
    x1: float = 0.0
    c: float = 0.0
    ice_thickness_nm: Optional[float] = None
    cid: Any = None
    t0: Optional[float] = None
    meta: dict = field(default_factory=dict)
