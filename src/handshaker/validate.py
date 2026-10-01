"""Validate handshaker magcor against the empirical delta-mag from two SNANA sims.

The controlled experiment: fit the SAME light curves twice, differing ONLY by the
chromatic effect --

    REF  : nominal filter at the reference SCA, no ice
    PERT : the true per-SCA shifted filter x ice

SNANA's own synthetic photometry then gives, per epoch, an empirical correction

    empirical_magcor = -2.5 log10(FLUXCAL_pert / FLUXCAL_ref)

which handshaker should reproduce with ``magcor_total``. Agreement (residual mean
~ 0, slope ~ 1, correlation ~ 1) means handshaker predicts what SNANA actually
does -- i.e. it works. Overlaying the true / polyfit / wavecor predictions shows
which correction method tracks the empirical result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .logging_utils import get_logger

log = get_logger("validate")


def _key(df, cid, mjd, band, mjd_round):
    return (df[cid].astype(str).str.strip() + "@"
            + df[mjd].round(mjd_round).astype(str) + "@"
            + df[band].astype(str).str.strip())


def empirical_magcor(ref, pert, cid="CID", mjd="MJD", band="BAND", flux="FLUXCAL",
                     dataflag=None, mjd_round=3):
    """Per-epoch empirical magcor = -2.5 log10(flux_pert / flux_ref), matched on CID+MJD+BAND.

    ``ref``/``pert`` are the two sims' LCPLOT/FITRES tables (e.g. from
    ``handshaker.io.load_lcplot``). Rows with non-positive flux are dropped. If
    ``dataflag`` is given, only rows with that flag (e.g. DATAFLAG==1, real data)
    are used, which also avoids data/model-grid key collisions.
    """
    r, p = ref.copy(), pert.copy()
    if dataflag is not None:
        r = r[r[dataflag] == 1]
        p = p[p[dataflag] == 1]
    for d in (r, p):
        d["_k"] = _key(d, cid, mjd, band, mjd_round)
    m = r[["_k", cid, mjd, band, flux]].merge(
        p[["_k", flux]], on="_k", suffixes=("_ref", "_pert"))
    m = m[(m[f"{flux}_ref"] > 0) & (m[f"{flux}_pert"] > 0)].copy()
    m["empirical_magcor"] = -2.5 * np.log10(m[f"{flux}_pert"] / m[f"{flux}_ref"])
    m = m.rename(columns={cid: "cid", mjd: "mjd", band: "band"})
    log.info("empirical magcor on %d matched epochs", len(m))
    return m[["_k", "cid", "mjd", "band", "empirical_magcor"]]


def compare_to_empirical(empirical, predicted, pred_col="magcor_total",
                         cid="cid", mjd="mjd", band="band", mjd_round=3):
    """Merge handshaker predictions onto the empirical magcor and score the agreement.

    ``empirical`` from :func:`empirical_magcor`; ``predicted`` a handshaker
    ``magcor_table`` (needs cid/mjd/band + ``pred_col``). Returns (merged_df, stats)
    where stats has the residual mean/std, Pearson correlation, and best-fit slope
    of empirical vs predicted.
    """
    p = predicted.copy()
    p["_k"] = _key(p, cid, mjd, band, mjd_round)
    m = empirical.merge(p[["_k", pred_col]], on="_k", how="inner")
    m = m.rename(columns={pred_col: "predicted_magcor"})
    m["residual"] = m["empirical_magcor"] - m["predicted_magcor"]
    e, pr = m["empirical_magcor"].to_numpy(), m["predicted_magcor"].to_numpy()
    ok = np.isfinite(e) & np.isfinite(pr)
    slope = float(np.polyfit(pr[ok], e[ok], 1)[0]) if ok.sum() > 1 else np.nan
    corr = float(np.corrcoef(e[ok], pr[ok])[0, 1]) if ok.sum() > 1 else np.nan
    stats = {"n": int(ok.sum()),
             "residual_mean": float(np.mean(m["residual"][ok])),
             "residual_std": float(np.std(m["residual"][ok])),
             "slope": slope, "correlation": corr}
    log.info("validation: n=%d residual=%.4f+/-%.4f slope=%.3f corr=%.3f",
             stats["n"], stats["residual_mean"], stats["residual_std"], slope, corr)
    return m, stats


def plot_validation(merged, stats=None, path="validation.png"):
    """Scatter empirical vs predicted magcor with the 1:1 line and a residual panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mmag = 1e3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax1.scatter(mmag * merged.predicted_magcor, mmag * merged.empirical_magcor, s=18, alpha=0.6)
    lo = mmag * np.nanmin([merged.predicted_magcor.min(), merged.empirical_magcor.min()])
    hi = mmag * np.nanmax([merged.predicted_magcor.max(), merged.empirical_magcor.max()])
    ax1.plot([lo, hi], [lo, hi], "--", color="grey", label="1:1")
    ax1.set(xlabel="predicted magcor [mmag]", ylabel="empirical magcor (SNANA) [mmag]",
            title="handshaker vs SNANA")
    ax1.legend(); ax1.grid(alpha=0.3)
    if stats:
        ax1.text(0.05, 0.95, f"slope={stats['slope']:.2f}\ncorr={stats['correlation']:.2f}\n"
                 f"resid={1e3*stats['residual_mean']:.1f}±{1e3*stats['residual_std']:.1f} mmag",
                 transform=ax1.transAxes, va="top", fontsize=9)
    ax2.hist(mmag * merged.residual.dropna(), bins=30, alpha=0.8)
    ax2.axvline(0, color="k", lw=0.6)
    ax2.set(xlabel="residual empirical − predicted [mmag]", ylabel="count", title="residual")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    log.info("wrote %s", path)
    return fig
