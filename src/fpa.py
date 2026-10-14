"""Focal-plane assembly (FPA) / SCA handling.

New in this version of handshaker: it should operate at the pixel level. Given
(RA, DEC) it should determine the focal-plane position and the implied SCA, then
check the *supplied* SCA against it (warn/abort on disagreement) rather than
silently trusting the input.

Only the ``NullFocalPlane`` (no geometry, trusts the supplied SCA) is implemented
today. A real WCS/geometry-backed ``FocalPlane`` subclass slots in behind the same
interface once the Roman FPA model is wired in.
"""

from __future__ import annotations

import warnings

from .logging_utils import get_logger

log = get_logger("fpa")


class FocalPlane:
    """Abstract focal-plane model: (ra, dec) -> (sca, pixel)."""

    def sca_for_position(self, ra, dec):
        """Return the SCA implied by a sky position, or None if unknown."""
        raise NotImplementedError

    def pixel_for_position(self, ra, dec):
        """Return the (row, col) pixel implied by a sky position, or None."""
        raise NotImplementedError


class NullFocalPlane(FocalPlane):
    """No geometry: trusts whatever SCA the observation provides.

    A stand-in until the Roman FPA projection is available. It never contradicts
    the supplied SCA, so no warnings are ever raised.
    """

    name = "null"

    def sca_for_position(self, ra, dec):
        return None

    def pixel_for_position(self, ra, dec):
        return None


def validate_observation_sca(obs, focal_plane, on_mismatch: str = "warn"):
    """Check ``obs.sca`` against the SCA implied by ``obs`` position; return the SCA to use.

    With ``NullFocalPlane`` (or missing ra/dec) there is nothing to check and the
    supplied SCA is returned unchanged. When a real focal plane implies a different
    SCA, this warns (default), aborts, or ignores per ``on_mismatch``.
    """
    if obs.ra is None or obs.dec is None:
        return obs.sca
    implied = focal_plane.sca_for_position(obs.ra, obs.dec)
    if implied is None or obs.sca is None or implied == obs.sca:
        return obs.sca
    msg = (f"SCA mismatch for obs at (RA,DEC)=({obs.ra},{obs.dec}): "
           f"supplied SCA={obs.sca}, position implies SCA={implied}")
    if on_mismatch == "abort":
        raise ValueError(msg)
    if on_mismatch == "warn":
        warnings.warn(msg)
        log.warning(msg)
    return implied
