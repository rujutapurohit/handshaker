"""Regression guard: handshaker magcor must track SNANA to a few tenths of a mmag.

Skips cleanly where SNANA (``snlc_sim.exe`` + the asset files) is absent, so it's
a no-op off-cluster and a real check on Midway / CI with SNANA. Point it at the
assets with ``$SNDATA_ROOT`` + ``$HANDSHAKER_HOME`` (or the ``HANDSHAKER_SNANA_*``
overrides); see ``handshaker.snana_crosscheck.SnanaAssets.from_env``.
"""

import numpy as np
import pytest

from handshaker.snana_crosscheck import (
    ALARM_MMAG, SnanaAssets, SNParams, crosscheck, random_sn_params, snana_available,
)

_assets = None
try:
    _assets = SnanaAssets.from_env()
except Exception:
    _assets = None

pytestmark = pytest.mark.skipif(
    not snana_available(_assets),
    reason="SNANA (snlc_sim.exe + assets) not available; cross-check skipped",
)


def test_crosscheck_fixed_point():
    """A known-good point (x1=0,c=0,z=0.5,Trest~+6.7) must pass the alarm threshold."""
    res = crosscheck(SNParams(x1=0.0, c=0.0, z=0.5, trest=6.667), _assets)
    assert res.handshaker, "no bands compared"
    assert res.passed, f"magcor residual too large:\n{res.report()}"
    assert res.maxabs < ALARM_MMAG


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_crosscheck_random(seed):
    """Random SNe across the parameter space must all stay under the threshold."""
    res = crosscheck(random_sn_params(np.random.default_rng(seed)), _assets)
    assert res.passed, f"ALARM:\n{res.report()}"
