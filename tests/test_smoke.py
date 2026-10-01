"""No-data smoke tests: exercise the full pipeline with the synthetic sources.

These run anywhere (no SALT3, no ROMAN_FILTERS, no Midway data) because they use
the synthetic filter set, a blackbody SED, and the bundled ice response ecsv.
"""

from __future__ import annotations

import numpy as np
import pytest

from handshaker import get_filter, get_ice_model, synthetic_sed, compute_magcor, Observation
from handshaker.config import Config
from handshaker.pipeline import run_config


def _obs(**kw):
    base = dict(cid="SN", band="J129", mjd=62200.0, sca=3, ra=10.0, dec=-44.0,
                z=0.5, epoch=0.0, x1=0.5, c=0.05, ice_thickness_nm=0.0)
    base.update(kw)
    return Observation(**base)


def test_synthetic_shift_is_nonzero():
    filters = get_filter("synthetic", shift_per_sca=6.0)
    ice = get_ice_model("none")
    sn = synthetic_sed()
    out = compute_magcor(_obs(), sn, filters, ice)
    assert np.isfinite(out["magcor_shift"])
    assert out["magcor_shift"] != 0.0
    assert out["magcor_ice"] == 0.0            # ice disabled


def test_ice_grows_with_thickness():
    filters = get_filter("synthetic")
    ice = get_ice_model("released")
    sn = synthetic_sed()
    m0 = compute_magcor(_obs(ice_thickness_nm=0.0), sn, filters, ice)["magcor_ice"]
    m1 = compute_magcor(_obs(ice_thickness_nm=90.0), sn, filters, ice)["magcor_ice"]
    m2 = compute_magcor(_obs(ice_thickness_nm=240.0), sn, filters, ice)["magcor_ice"]
    assert m0 == 0.0
    assert abs(m1) > 0.0 and abs(m2) > 0.0


def test_ice_thickness_reported():
    filters = get_filter("synthetic")
    ice = get_ice_model("released")
    out = compute_magcor(_obs(ice_thickness_nm=150.0), synthetic_sed(), filters, ice)
    assert out["ice_thickness"] == pytest.approx(150.0)


def test_magcor_pipeline_runs(tmp_path):
    cfg = Config.from_dict({
        "mode": "magcor",
        "output_dir": str(tmp_path / "magcor"),
        "filters": {"source": "synthetic", "shift_per_sca": 6.0},
        "sn_sed": {"model": "synthetic"},
        "calib_star": {"model": "synthetic", "temperature_k": 9500},
        "ice": {"flag": "released"},
        "effects": ["shift", "ice", "total"],
        "observations": [
            {"cid": "SN1", "band": "J129", "mjd": 62200.0, "sca": 3, "z": 0.5,
             "epoch": 0.0, "x1": 0.5, "c": 0.05, "ice_thickness_nm": 90.0},
        ],
    })
    df = run_config(cfg)
    assert len(df) == 1
    assert {"magcor_shift", "magcor_ice", "magcor_total", "ice_thickness"} <= set(df.columns)


def test_reference_sca():
    filters = get_filter("synthetic", shift_per_sca=6.0)
    ice = get_ice_model("none")
    sed = synthetic_sed()
    obs = _obs(band="J129", sca=10)
    # shift magcor vs the same SCA must be identically zero
    r_self = compute_magcor(obs, sed, filters, ice, reference_sca=10)
    assert r_self["magcor_shift"] == 0.0
    # vs a different reference SCA it must be nonzero, and differ from the nominal reference
    r_ref = compute_magcor(obs, sed, filters, ice, reference_sca=2)
    r_nom = compute_magcor(obs, sed, filters, ice)
    assert r_ref["magcor_shift"] != 0.0
    assert r_ref["magcor_shift"] != r_nom["magcor_shift"]


def test_magcor_vs_wavecor_equivalence():
    # A pure filter shift IS a wavelength shift -> magcor ~= wavecor-corrected mag.
    # Ice is a passband-shape change -> magcor != wavecor-corrected mag.
    filters = get_filter("synthetic", shift_per_sca=6.0)
    ice = get_ice_model("released")
    sed = synthetic_sed()
    r = compute_magcor(_obs(band="J129", sca=10, ice_thickness_nm=150.0),
                       sed, filters, ice, reference_sca=2)
    assert abs(r["resid_shift"]) < 1e-3          # shift reproduced by the wavelength shift
    assert abs(r["resid_ice"]) > 5 * abs(r["resid_shift"])  # ice is not a wavelength shift


def test_polyfit_ice_beats_wavecor():
    # The polyfit ice model reproduces the true ice magcor far better than the
    # wavelength-shift (wavecor) approximation does.
    filters = get_filter("synthetic", shift_per_sca=6.0)
    sed = synthetic_sed()
    ice_true = get_ice_model("released")
    ice_poly = get_ice_model("polyfit", filters=filters, order=5)
    o = _obs(band="H158", sca=10, ice_thickness_nm=240.0)
    rt = compute_magcor(o, sed, filters, ice_true, reference_sca=2)
    rp = compute_magcor(o, sed, filters, ice_poly, reference_sca=2)
    err_poly = abs(rp["magcor_ice"] - rt["magcor_ice"])
    err_wave = abs(rt["magcorwave_ice"] - rt["magcor_ice"])
    assert err_poly < 1e-3           # polynomial ice ~ true ice
    assert err_wave > 20 * err_poly  # wavecor is much worse for ice


def test_magcor_table_over_fake_modelspec():
    import pandas as pd
    from handshaker import magcor_table
    from handshaker.sed import FixedSED
    filters = get_filter("synthetic")
    ice = get_ice_model("released")
    s = synthetic_sed()
    df = pd.DataFrame({
        "cid": ["SN1", "SN2"], "band": ["J129", "F184"], "sca": [10, 4],
        "mjd": [62000.0, 62100.0], "trest": [0.0, 5.0],
        "sed": [FixedSED(s.wave, s.flux), FixedSED(s.wave, s.flux)],
    })
    tbl = magcor_table(df, filters, ice, reference_sca=2, ice_thickness_nm=90.0)
    assert len(tbl) == 2
    assert {"magcor_shift", "magcor_ice", "magcor_total", "wavecor"} <= set(tbl.columns)


def test_validation_harness_closed_loop():
    # Fabricate a "perfect" sim: pert flux = ref flux * 10**(-0.4*predicted magcor).
    # The empirical magcor must then equal the prediction -> residual 0, slope 1, corr 1.
    import numpy as np, pandas as pd
    from handshaker import magcor_table, empirical_magcor, compare_to_empirical
    from handshaker.sed import FixedSED
    filters = get_filter("synthetic", shift_per_sca=6.0)
    ice = get_ice_model("released")
    s = synthetic_sed()
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "cid": [f"SN{i}" for i in range(30)],
        "band": rng.choice(["J129", "H158", "F184"], 30),
        "sca": rng.integers(1, 19, 30), "mjd": rng.uniform(62000, 62500, 30),
        "trest": 0.0,
    })
    df["sed"] = [FixedSED(s.wave, s.flux) for _ in range(len(df))]
    pred = magcor_table(df, filters, ice, reference_sca=2, ice_thickness_nm=120.0)

    ref = pd.DataFrame({"CID": pred.cid, "MJD": pred.mjd, "BAND": pred.band, "FLUXCAL": 100.0})
    pert = ref.copy()
    pert["FLUXCAL"] = 100.0 * 10 ** (-0.4 * pred.magcor_total.to_numpy())
    emp = empirical_magcor(ref, pert)
    merged, stats = compare_to_empirical(emp, pred, pred_col="magcor_total")
    assert stats["n"] >= 25
    assert abs(stats["residual_std"]) < 1e-6      # exact recovery
    assert abs(stats["slope"] - 1.0) < 1e-3
    assert stats["correlation"] > 0.999


def test_polyfit_pipeline_runs(tmp_path):
    # polyfit validates the polynomial approx to the ICE ratio per (band, thickness).
    cfg = Config.from_dict({
        "mode": "polyfit",
        "output_dir": str(tmp_path / "polyfit"),
        "filters": {"source": "synthetic", "shift_per_sca": 6.0},
        "ice": {"flag": "released"},
        "polyfit": {"order": 5, "bands": ["J129"], "thicknesses": [90, 240]},
    })
    df = run_config(cfg)
    assert len(df) == 2                     # 1 band x 2 thicknesses
    assert {"residual_rms", "ice_thick_nm", "max_abs_residual"} <= set(df.columns)
