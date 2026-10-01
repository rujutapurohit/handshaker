"""Orchestration: config in -> corrections table out.

Ties the pieces together for both modes described in the spec:

* ``mode = magcor``  -> per-observation, per-effect magcor + ``magcor_total`` and
  ice thickness, written to ``magcor.csv``.
* ``mode = polyfit`` -> per band/SCA wavecor and an Nth-order polynomial for
  ``T(l)/T_ref(l)``, with ratio/residual tables written per band.

The config file is copied into the output directory first, for reproducibility.
"""

from __future__ import annotations

import os

import pandas as pd

from .config import Config
from .filters import get_filter
from .grid import WaveGrid
from .fpa import NullFocalPlane, validate_observation_sca
from .ice import get_ice_model
from .logging_utils import get_logger
from .magcor import EFFECTS, compute_magcor
from .observation import Observation
from .polyfit import polyfit_ice_band
from .sed import SALT3SEDModel, FixedSED, synthetic_sed

log = get_logger("pipeline")

# Builders: config sub-dicts -> objects
def build_filter_source(spec):
    spec = dict(spec or {"source": "synthetic"})
    source = spec.pop("source", "synthetic")
    return get_filter(source, **spec)

def build_sn_sed(spec):
    spec = dict(spec or {"model": "synthetic"})
    model = spec.pop("model", "synthetic").lower()
    if model in ("salt3", "salt"):
        return SALT3SEDModel(model_dir=spec.get("model_dir"))
    if model in ("synthetic", "blackbody", "bb"):
        return synthetic_sed(temperature_k=spec.get("temperature_k", 9000.0))
    raise ValueError(f"unknown sn_sed model {model!r}")

def build_calib_star(spec):
    if not spec:
        return None
    spec = dict(spec)
    model = spec.pop("model", "synthetic").lower()
    if model in ("synthetic", "blackbody", "bb"):
        return synthetic_sed(temperature_k=spec.get("temperature_k", 9500.0), name="calib_star")
    if model in ("file", "ascii"):
        import numpy as np
        wave, flux = np.loadtxt(spec["path"], unpack=True, usecols=(0, 1))
        return FixedSED(wave, flux, name=os.path.basename(spec["path"]))
    raise ValueError(f"unknown calib_star model {model!r}")

def build_ice_model(spec, filters=None):
    spec = dict(spec or {"source": "none"})
    source = spec.pop("source", None) or spec.pop("flag", "none")   # 'source' preferred; 'flag' legacy
    return get_ice_model(source, response_path=spec.get("response_path"),
                         filters=filters, order=spec.get("order", 5))

def build_focal_plane(spec):
    # Only the null (unverified) focal plane is available until Roman FPA geometry
    # is wired in; a real WCS-backed FocalPlane subclass slots in here.
    return NullFocalPlane()

# Observations
_OBS_FIELDS = {"band", "mjd", "sca", "ra", "dec", "z", "epoch", "x1", "c",
               "ice_thickness_nm", "cid", "t0"}

def _row_to_observation(row: dict) -> Observation:
    kwargs = {k: row[k] for k in _OBS_FIELDS if k in row and pd.notna(row[k])}
    meta = {k: v for k, v in row.items() if k not in _OBS_FIELDS}
    if "sca" in kwargs and kwargs["sca"] is not None:
        kwargs["sca"] = int(kwargs["sca"])
    return Observation(meta=meta, **kwargs)


def _dataset_path(spec):
    """Filesystem path if ``observations`` points to a dataset file, else None."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict) and "path" in spec:
        return spec["path"]
    return None


def load_observations(spec) -> list:
    """Load observations from a dataset path (.root MODELSPEC or .csv) or an inline list."""
    path = _dataset_path(spec)
    if isinstance(path, str) and path.endswith(".root"):
        from .modelspec import load_modelspec
        df = load_modelspec(path)
        rows = df[["cid", "band", "sca", "mjd", "trest"]].to_dict("records")
    elif isinstance(path, str):
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        rows = df.to_dict("records")
    elif isinstance(spec, list):
        rows = spec
    else:
        raise ValueError("observations must be a dataset path (.root/.csv) or an inline list")
    obs = [r if isinstance(r, Observation) else _row_to_observation(dict(r)) for r in rows]
    log.info("loaded %d observations", len(obs))
    return obs

def _resolve_epoch(obs: Observation) -> Observation:
    """Derive rest-frame epoch from (mjd, t0, z) when epoch wasn't given."""
    if obs.epoch == 0.0 and obs.t0 is not None:
        obs.epoch = (obs.mjd - obs.t0) / (1.0 + obs.z)
    return obs


# Modes
def run_magcor(cfg: Config, observations, filters, sn_sed, ice_model,
               calib_star=None, focal_plane=None, effects=EFFECTS, grid=None,
               reference_sca=None):
    """Mode = magcor: per-observation, per-effect corrections table."""
    focal_plane = focal_plane or NullFocalPlane()
    rows = []
    for obs in observations:
        _resolve_epoch(obs)
        obs.sca = validate_observation_sca(obs, focal_plane)
        result = compute_magcor(obs, sn_sed, filters, ice_model,
                                calib_star=calib_star, effects=effects, grid=grid,
                                reference_sca=reference_sca)
        row = {
            "cid": obs.cid, "band": obs.band, "mjd": obs.mjd, "sca": obs.sca,
            "z": obs.z, "epoch": obs.epoch, "x1": obs.x1, "c": obs.c,
            "ice_thickness_nm": obs.ice_thickness_nm,
        }
        row.update(result)
        rows.append(row)
    df = pd.DataFrame(rows)
    out = cfg.ensure_output_dir()
    path = os.path.join(out, "magcor.csv")
    df.to_csv(path, index=False)
    log.info("wrote %d magcor rows to %s", len(df), path)
    return df


def run_polyfit(cfg: Config, observations, filters, ice_model, grid=None):
    """Mode = polyfit: fit the true ICE ratio T(t)/T(0) per (band, thickness).

    Iterates over ``polyfit.bands`` x ``polyfit.thicknesses`` (nm). Bands default
    to the observations' bands, then to the filter source's available bands.
    """
    order = cfg.polyfit.get("order", 5)
    support_frac = cfg.polyfit.get("support_frac", 0.01)
    thicknesses = cfg.polyfit.get("thicknesses") or [30.0, 90.0, 150.0, 240.0, 300.0]
    bands = cfg.polyfit.get("bands")
    if not bands:
        bands = sorted({o.band for o in observations if o.band}) or filters.available_bands()
    if not bands:
        raise ValueError("polyfit mode needs bands (set polyfit.bands, give observations, "
                         "or use a filter source that lists available_bands)")
    sca = getattr(ice_model, "sca", None)
    out = cfg.ensure_output_dir()

    summary = []
    for band in bands:
        for thk in thicknesses:
            try:
                res = polyfit_ice_band(band, thk, filters, ice_model, order=order,
                                       sca=sca, support_frac=support_frac, grid=grid)
            except FileNotFoundError as e:
                log.warning("skipping %s: missing filter file (%s)", band, e)
                break  # band's nominal curve is missing; skip all its thicknesses
            tbl_path = os.path.join(out, f"polyfit_{band}_d{int(thk)}nm.csv")
            res["table"].to_csv(tbl_path, index=False)
            summary.append({
                "band": band, "ice_thick_nm": thk, "sca": sca, "order": order,
                "residual_rms": res["residual_rms"], "max_abs_residual": res["max_abs_residual"],
                "support_lo": res["support"][0], "support_hi": res["support"][1],
                **{f"c{i}": v for i, v in enumerate(res["coeffs"][::-1])},  # c0 + c1*l + ...
                "table_file": os.path.basename(tbl_path),
            })
    df = pd.DataFrame(summary)
    path = os.path.join(out, "polyfit_summary.csv")
    df.to_csv(path, index=False)
    log.info("wrote polyfit summary (%d band/thickness rows) to %s", len(df), path)
    return df


# Run confid etc 

def run_magcor_modelspec(cfg, filters, ice_model, grid, dataset_path) -> pd.DataFrame:
    """magcor over a SNANA MODELSPEC .root dataset (per-observation model SEDs)."""
    from .modelspec import load_modelspec, magcor_table, mjd_ice_thickness
    spec = load_modelspec(dataset_path)
    thk = (cfg.ice or {}).get("thickness", "mjd") if isinstance(cfg.ice, dict) else "mjd"
    thickness = (lambda r: mjd_ice_thickness(r.mjd)) if str(thk) == "mjd" else float(thk)
    tbl = magcor_table(spec, filters, ice_model, reference_sca=cfg.reference_sca,
                       ice_thickness_nm=thickness, grid=grid)
    out = cfg.ensure_output_dir()
    path = os.path.join(out, "magcor.csv")
    tbl.to_csv(path, index=False)
    log.info("wrote %d magcor rows to %s", len(tbl), path)
    return tbl


def run_config(cfg: Config) -> pd.DataFrame:
    """Run a fully-parsed :class:`Config` and return the output table."""
    cfg.copy_to_output()
    grid = WaveGrid.from_config(cfg.grid)
    log.info("wavelength grid: %s", grid)
    filters = build_filter_source(cfg.filters)
    ice_model = build_ice_model(cfg.ice, filters=filters)
    dataset = _dataset_path(cfg.observations)

    if cfg.mode == "magcor":
        # A .root dataset carries a per-observation model SED (from the LC fit);
        # use it directly. Otherwise fall back to an sn_sed model + obs table.
        if isinstance(dataset, str) and dataset.endswith(".root"):
            return run_magcor_modelspec(cfg, filters, ice_model, grid, dataset)
        observations = load_observations(cfg.observations)
        effects = tuple(cfg.effects) if cfg.effects else EFFECTS
        sn_sed = build_sn_sed(cfg.sn_sed)
        calib_star = build_calib_star(cfg.calib_star)
        focal_plane = build_focal_plane(cfg.focal_plane)
        return run_magcor(cfg, observations, filters, sn_sed, ice_model,
                          calib_star=calib_star, focal_plane=focal_plane, effects=effects,
                          grid=grid, reference_sca=cfg.reference_sca)
    if cfg.mode == "polyfit":
        observations = load_observations(cfg.observations) if dataset or isinstance(cfg.observations, list) else []
        return run_polyfit(cfg, observations, filters, ice_model, grid=grid)
    raise ValueError(f"unknown mode {cfg.mode!r}; choose 'magcor' or 'polyfit'")


def run(config_path: str) -> pd.DataFrame:
    """Load a config file and run it end-to-end."""
    cfg = Config.from_file(config_path)
    return run_config(cfg)
