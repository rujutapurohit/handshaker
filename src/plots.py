"""Plot handshaker polyfit output.

Reads a polyfit output directory (``polyfit_summary.csv`` + the per-(band,
thickness) ``polyfit_<band>_d<thk>nm.csv`` ratio tables written by
``run_polyfit``) and makes two kinds of figure, saved under ``<output_dir>/plots``:

* ``polyfit_<band>.png`` -- for each band, the true ice ratio T(t)/T(0) (solid)
  and its polynomial approximation (dashed) vs wavelength, one color per ice
  thickness, with the residual (true - poly) underneath.
* ``polyfit_summary.png`` -- RMS and max |residual| vs ice thickness, one line
  per band: how good order N is, and where it starts to break down.

Run:  python -m handshaker.plots <output_dir>
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # backend left as-is so notebooks keep inline display

from .logging_utils import get_logger

log = get_logger("plots")


def plot_polyfit_output(output_dir: str, plots_dir: str | None = None) -> str:
    """Make all polyfit plots for an output directory; return the plots dir."""
    plots_dir = plots_dir or os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    summary = pd.read_csv(os.path.join(output_dir, "polyfit_summary.csv"))
    for band, g in summary.groupby("band"):
        tables = {}
        for row in g.itertuples():
            tables[row.ice_thick_nm] = pd.read_csv(os.path.join(output_dir, row.table_file))
        _plot_band(band, tables, plots_dir)
    _plot_summary(summary, plots_dir)
    log.info("wrote %d plots to %s", len(summary["band"].unique()) + 1, plots_dir)
    return plots_dir


def _plot_band(band, tables, plots_dir):
    thicks = sorted(tables)
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(thicks), 2)))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 7),
                                   gridspec_kw={"height_ratios": [2, 1]})
    for c, t in zip(colors, thicks):
        df = tables[t]
        ax1.plot(df.WAVE, df.RATIO_TRUE, color=c, lw=1.6, label=f"{t:.0f} nm")
        ax1.plot(df.WAVE, df.RATIO_POLY, color=c, lw=1.0, ls="--")
        ax2.plot(df.WAVE, df.RESIDUAL, color=c, lw=1.0)
    ax1.set_ylabel("T(t) / T(0)")
    ax1.set_title(f"{band}: ice throughput ratio  (solid = true, dashed = polynomial)")
    ax1.legend(title="ice thickness", fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_ylabel("residual\n(true - poly)")
    ax2.set_xlabel("wavelength [Å]")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plots_dir, f"polyfit_{band}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_summary(summary, plots_dir):
    order = int(summary["order"].iloc[0]) if "order" in summary else None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for band, g in summary.groupby("band"):
        g = g.sort_values("ice_thick_nm")
        axes[0].plot(g.ice_thick_nm, g.residual_rms, marker="o", label=band)
        axes[1].plot(g.ice_thick_nm, g.max_abs_residual, marker="o", label=band)
    for ax, title in zip(axes, ["RMS residual", "max |residual|"]):
        ax.set_xlabel("ice thickness [nm]")
        ax.set_ylabel(title)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_title(f"polynomial fit accuracy vs thickness (order {order})")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "polyfit_summary.png"), dpi=130)
    plt.close(fig)


def plot_magcor(tbl, path="magcor_diagnostics.png"):
    """4-panel diagnostic from a magcor table (handshaker.magcor_table output).

    (1) magcor_total vs wavecor -- the magcor<->wavecor equivalence test;
    (2) magcor_shift vs SCA -- the field-dependent (per-SCA) filter effect;
    (3) magcor_ice vs ice thickness -- the ice effect;
    (4) distributions of the three magcor components [mmag].
    Saves to ``path`` and returns the figure.
    """
    mmag = 1e3
    bands = sorted(tbl["band"].dropna().unique())
    colors = {b: plt.cm.tab10(i % 10) for i, b in enumerate(bands)}
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    has_wave = "magcorwave_total" in tbl.columns
    for b in bands:
        s = tbl[tbl.band == b]
        if has_wave:
            ax[0, 0].scatter(mmag * s.magcorwave_total, mmag * s.magcor_total,
                             s=25, color=colors[b], alpha=0.7, label=b)
        ax[0, 1].scatter(s.sca, mmag * s.magcor_shift, s=25, color=colors[b], alpha=0.7, label=b)
        ax[1, 0].scatter(s.ice_thickness_nm, mmag * s.magcor_ice, s=25, color=colors[b], alpha=0.7, label=b)

    if has_wave:  # magcor (exact) vs wavecor-corrected mag; equivalence => the 1:1 line
        lo = mmag * min(tbl.magcorwave_total.min(), tbl.magcor_total.min())
        hi = mmag * max(tbl.magcorwave_total.max(), tbl.magcor_total.max())
        ax[0, 0].plot([lo, hi], [lo, hi], "--", color="grey", lw=1, label="1:1")
    ax[0, 0].set(xlabel="wavecor-corrected mag [mmag]", ylabel="magcor_total (exact) [mmag]",
                 title="magcor vs wavecor-corrected mag")
    ax[0, 1].set(xlabel="SCA", ylabel="magcor_shift [mmag]", title="filter shift vs SCA (field dependence)")
    ax[1, 0].set(xlabel="ice thickness [nm]", ylabel="magcor_ice [mmag]", title="ice magcor vs thickness")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.axhline(0, lw=0.5, color="k"); a.grid(alpha=0.3); a.legend(fontsize=8, ncol=2)

    # residuals: magcor - wavecor-corrected mag (the non-wavelength-shift part)
    if "resid_total" in tbl.columns:
        for col, c in [("resid_shift", "C0"), ("resid_ice", "C1"), ("resid_total", "C2")]:
            if col in tbl.columns:
                ax[1, 1].hist(mmag * tbl[col].dropna(), bins=20, alpha=0.5, color=c, label=col)
        ax[1, 1].set(xlabel="magcor − wavecor-corrected mag [mmag]", ylabel="count",
                     title="residual: non-wavelength-shift part")
    else:
        for col, c in [("magcor_shift", "C0"), ("magcor_ice", "C1"), ("magcor_total", "C2")]:
            ax[1, 1].hist(mmag * tbl[col].dropna(), bins=20, alpha=0.5, color=c, label=col)
        ax[1, 1].set(xlabel="magcor [mmag]", ylabel="count", title="magcor distributions")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        log.info("wrote magcor diagnostics to %s", path)
    return fig


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")  # headless for the CLI (login node / no display)
    p = argparse.ArgumentParser(prog="handshaker-plot", description="Plot handshaker polyfit output.")
    p.add_argument("output_dir", help="polyfit output dir (contains polyfit_summary.csv)")
    p.add_argument("--plots-dir", default=None, help="where to write PNGs (default <output_dir>/plots)")
    args = p.parse_args(argv)
    dest = plot_polyfit_output(args.output_dir, args.plots_dir)
    print(f"wrote plots to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
