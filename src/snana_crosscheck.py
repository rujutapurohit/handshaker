"""Permanent handshaker <-> SNANA magcor cross-check (regression guard).

The idea (validated 2026-10-13, residual rms ~0.07 mmag over a 4D SN grid):
SNANA generates a SALT3 SN and photometers it through two kcors that differ ONLY
by the ice-perturbed filters; handshaker folds the SAME SED through the SAME
filters. The per-band magcor

    magcor(band) = mag_reference(band) - mag_ice(band)

must agree. The difference is entirely the SN-SED interpolation (SNANA vs
handshaker SALT3 grids) -- the K-star/primary part cancels exactly -- so the
alarm threshold sits just above that floor: a few tenths of a mmag.

Run random spot-checks any time with::

    handshaker-snana-check -n 8            # CLI
    from handshaker.snana_crosscheck import run_random_checks
    run_random_checks(8)                    # Python

Needs SNANA (``snlc_sim.exe`` on PATH) and the asset files (built once; see
``snana/magcor_test_README.md``). Where SNANA is absent the check short-circuits
(``snana_available()`` is False) so unit tests skip cleanly off-cluster.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np

from .logging_utils import get_logger

log = get_logger("snana_crosscheck")

# AB reference f_lambda prefactor: 3631 Jy * c[A/s]
_AB_PREFAC = 3631e-23 * 2.99792458e18

#: default alarm threshold on the worst per-band residual (handshaker - SNANA)
ALARM_MMAG = 0.3
#: SALT3 obs-frame coverage doesn't reach ~2.1 um, so K213 is excluded by default
DEFAULT_BANDS = ("R062-R", "Z087-Z", "Y106-Y", "J129-J", "H158-H", "F184-F")


def _short(flt: str) -> str:
    """SNANA band name -> handshaker short code, e.g. 'R062-R' -> 'R062'."""
    return flt.split("-")[0]


@dataclass
class SnanaAssets:
    """Paths/handles the cross-check needs. Build with :meth:`from_env`.

    The kcors must be the AB-mag-system pair (nominal vs ice filters); the K-star
    system trips SNANA's crazy-flux guard (SN mag ~9.5). ``ref``/``ice`` filter
    dirs are the exact dirs those kcors point at, so handshaker folds identical
    passbands.
    """

    sndata_root: str
    nom_filter_dir: str
    ice_filter_dir: str
    ref_kcor: str
    ice_kcor: str
    simlib: str
    sim_template: str
    salt3_model_dir: str | None = None
    sim_exe: str = "snlc_sim.exe"
    mjd_obs: float = 61010.0
    bands: tuple = DEFAULT_BANDS

    @classmethod
    def from_env(cls, home: str | None = None) -> "SnanaAssets":
        """Defaults from ``$SNDATA_ROOT`` and the handshaker checkout.

        ``home`` (or ``$HANDSHAKER_HOME``) is the dir holding ``FILT_ice150`` and
        ``tests/NEW_TEST`` (the SNANA assets). Individual paths may be overridden
        with ``HANDSHAKER_SNANA_*`` env vars.
        """
        snd = os.environ.get("SNDATA_ROOT", "")
        home = home or os.environ.get("HANDSHAKER_HOME", "")
        new = os.path.join(home, "tests", "NEW_TEST")
        env = os.environ.get
        return cls(
            sndata_root=snd,
            nom_filter_dir=env("HANDSHAKER_SNANA_NOMDIR", f"{snd}/filters/ROMAN/20210614"),
            ice_filter_dir=env("HANDSHAKER_SNANA_ICEDIR", f"{home}/FILT_ice150"),
            ref_kcor=env("HANDSHAKER_SNANA_REFKCOR", f"{new}/calib_kgiant_primAB_nom.fits"),
            ice_kcor=env("HANDSHAKER_SNANA_ICEKCOR", f"{new}/calib_kgiant_primAB_ice.fits"),
            simlib=env("HANDSHAKER_SNANA_SIMLIB", f"{new}/magcor_test.SIMLIB"),
            sim_template=env("HANDSHAKER_SNANA_TEMPLATE", f"{new}/simgen.input"),
            salt3_model_dir=env("HANDSHAKER_SALT3_DIR") or None,
        )

    def missing(self) -> list[str]:
        """List of required files/dirs that don't exist (empty => ready)."""
        need = [self.nom_filter_dir, self.ice_filter_dir, self.ref_kcor,
                self.ice_kcor, self.simlib, self.sim_template]
        return [p for p in need if not os.path.exists(p)]


def snana_available(assets: SnanaAssets | None = None) -> bool:
    """True if ``snlc_sim.exe`` is on PATH and all asset files exist."""
    try:
        assets = assets or SnanaAssets.from_env()
    except Exception:
        return False
    if not assets.sndata_root:
        return False
    if shutil.which(assets.sim_exe) is None:
        return False
    return not assets.missing()


@dataclass
class SNParams:
    x1: float = 0.0
    c: float = 0.0
    z: float = 0.5
    trest: float = 0.0


# population-ish ranges for random spot-checks (representative, not the real LF)
_RANGES = dict(x1=(-2.0, 2.0), c=(-0.1, 0.1), z=(0.2, 0.9), trest=(-5.0, 25.0))


def random_sn_params(rng=None) -> SNParams:
    """A random SN within representative x1/c/z/Trest ranges."""
    rng = rng or np.random.default_rng()
    return SNParams(
        x1=float(rng.uniform(*_RANGES["x1"])),
        c=float(rng.uniform(*_RANGES["c"])),
        z=float(rng.uniform(*_RANGES["z"])),
        trest=float(rng.uniform(*_RANGES["trest"])),
    )


# ------------------------------------------------------------------ SNANA side
def _fill_template(template_text: str, gv: str, kcor: str, p: SNParams, mjd_obs: float) -> str:
    peakmjd = mjd_obs - p.trest * (1.0 + p.z)
    repl = {
        "__GENVERSION__": gv, "__KCOR__": kcor, "__PEAKMJD__": f"{peakmjd:.4f}",
        "__Z__": repr(p.z), "__X1__": repr(p.x1), "__C__": repr(p.c),
    }
    s = template_text
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def _read_sim_magobs(sndata_root: str, gv: str) -> dict:
    """Parse ``SIM_MAGOBS`` per band from a sim's TEXT light curve (`*_SN*.DAT`)."""
    hits = glob.glob(f"{sndata_root}/SIM/{gv}/{gv}_SN*.DAT")
    if not hits:
        raise FileNotFoundError(f"no sim output for GENVERSION {gv}")
    names, rows = None, []
    with open(hits[0]) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] in ("VARLIST:", "VARNAMES:"):
                names = t[1:]
            elif t[0] == "OBS:" and names:
                rows.append(t[1:])
    i = {n: k for k, n in enumerate(names)}
    fc = i.get("FLT", i.get("BAND"))
    mc = i["SIM_MAGOBS"]
    return {r[fc]: float(r[mc]) for r in rows if float(r[mc]) < 90}


def snana_magcor(p: SNParams, assets: SnanaAssets, workdir: str | None = None) -> dict:
    """magcor(band) [mmag] from two SNANA sims differing only by the kcor."""
    tmpl = open(assets.sim_template).read()
    made = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="hs_xcheck_")
    try:
        # the sim reads KCOR_FILE / SIMLIB_FILE relative to the run dir
        for src in (assets.ref_kcor, assets.ice_kcor, assets.simlib):
            dst = os.path.join(workdir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy(src, dst)
        out = {}
        for gv, kcor in (("XREF", assets.ref_kcor), ("XICE", assets.ice_kcor)):
            inp = os.path.join(workdir, f"sim_{gv}.input")
            with open(inp, "w") as f:
                f.write(_fill_template(tmpl, gv, os.path.basename(kcor), p, assets.mjd_obs))
            subprocess.run([assets.sim_exe, os.path.basename(inp)], cwd=workdir,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            out[gv] = _read_sim_magobs(assets.sndata_root, gv)
        ref, ice = out["XREF"], out["XICE"]
        return {b: (ref[b] - ice[b]) * 1e3 for b in assets.bands if b in ref and b in ice}
    finally:
        if made:
            shutil.rmtree(workdir, ignore_errors=True)


# ------------------------------------------------------------- handshaker side
def handshaker_magcor(p: SNParams, assets: SnanaAssets, grid=None) -> dict:
    """magcor(band) [mmag] from handshaker: AB synphot of the SALT3 SED through
    the reference vs ice passbands (same files the kcors use)."""
    from .grid import WaveGrid
    from .filters import RomanDirFilterSource
    from .sed import SALT3SEDModel

    grid = grid or WaveGrid()
    wl = grid.wave
    flam_ab = _AB_PREFAC / wl**2
    nom = RomanDirFilterSource(assets.nom_filter_dir)
    ice = RomanDirFilterSource(assets.ice_filter_dir)
    salt = SALT3SEDModel(model_dir=assets.salt3_model_dir)

    wr, fr = salt.rest_frame(epoch=p.trest, x1=p.x1, c=p.c)
    flam_sn = grid.regrid(wr * (1.0 + p.z), fr, fill=0.0)

    def mag(src, short):
        t = src.nominal(short).on_grid(grid).thru
        num = grid.integrate(flam_sn * t * wl)
        if num <= 0:
            return np.nan
        return -2.5 * np.log10(num / grid.integrate(flam_ab * t * wl))

    out = {}
    for b in assets.bands:
        s = _short(b)
        out[b] = (mag(nom, s) - mag(ice, s)) * 1e3
    return out


# ------------------------------------------------------------------- compare
@dataclass
class CrosscheckResult:
    params: SNParams
    snana: dict
    handshaker: dict
    residuals: dict            # handshaker - SNANA, per band [mmag]
    rms: float
    maxabs: float
    threshold: float
    passed: bool

    def report(self) -> str:
        head = (f"x1={self.params.x1:+.2f} c={self.params.c:+.3f} "
                f"z={self.params.z:.3f} Trest={self.params.trest:+.1f}")
        lines = [f"{'band':8}{'hs':>9}{'SNANA':>9}{'resid':>9}"]
        for b in self.residuals:
            lines.append(f"{b:8}{self.handshaker[b]:9.3f}{self.snana[b]:9.3f}"
                         f"{self.residuals[b]:9.3f}")
        flag = "OK" if self.passed else "*** ALARM ***"
        lines.append(f"rms={self.rms:.3f}  max|resid|={self.maxabs:.3f} mmag  "
                     f"(thresh {self.threshold:.3f})  {flag}")
        return head + "\n" + "\n".join(lines)


def crosscheck(p: SNParams, assets: SnanaAssets | None = None,
               threshold: float = ALARM_MMAG, grid=None) -> CrosscheckResult:
    """Run one SN through both codes and compare. ``passed`` is False (alarm) if
    any band's |handshaker - SNANA| exceeds ``threshold`` [mmag]."""
    assets = assets or SnanaAssets.from_env()
    sn = snana_magcor(p, assets)
    hs = handshaker_magcor(p, assets, grid=grid)
    bands = [b for b in sn if b in hs and np.isfinite(hs[b])]
    resid = {b: hs[b] - sn[b] for b in bands}
    vals = np.array(list(resid.values())) if resid else np.array([np.nan])
    rms = float(np.sqrt(np.mean(vals**2)))
    maxabs = float(np.max(np.abs(vals)))
    return CrosscheckResult(p, {b: sn[b] for b in bands}, {b: hs[b] for b in bands},
                            resid, rms, maxabs, threshold, bool(maxabs <= threshold))


def run_random_checks(n: int = 5, assets: SnanaAssets | None = None,
                      threshold: float = ALARM_MMAG, seed: int | None = None) -> list:
    """Run ``n`` random SN cross-checks; returns the list of results and logs each."""
    assets = assets or SnanaAssets.from_env()
    rng = np.random.default_rng(seed)
    results = []
    for k in range(n):
        res = crosscheck(random_sn_params(rng), assets, threshold=threshold)
        log.info("[%d/%d] %s", k + 1, n, "PASS" if res.passed else "ALARM")
        results.append(res)
    return results


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Random handshaker<->SNANA magcor spot-checks")
    ap.add_argument("-n", type=int, default=5, help="number of random SNe")
    ap.add_argument("--threshold", type=float, default=ALARM_MMAG,
                    help="alarm threshold on worst band residual [mmag]")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--home", default=None, help="handshaker home (holds FILT_ice150, tests/NEW_TEST)")
    args = ap.parse_args(argv)

    assets = SnanaAssets.from_env(home=args.home)
    if not snana_available(assets):
        miss = (assets.missing() if assets.sndata_root else ["SNDATA_ROOT unset"])
        print("SNANA not available for cross-check; missing:", miss)
        return 2
    rng = np.random.default_rng(args.seed)
    results, n_alarm = [], 0
    for k in range(args.n):
        res = crosscheck(random_sn_params(rng), assets, threshold=args.threshold)
        print(f"[{k + 1}/{args.n}]")
        print(res.report())
        print("-" * 52, flush=True)          # per-check, so a long run isn't silent
        n_alarm += not res.passed
        results.append(res)
    worst = max((r.maxabs for r in results), default=float("nan"))
    print(f"SUMMARY: {len(results)} checks, {n_alarm} ALARM, worst max|resid|={worst:.3f} mmag")
    return 1 if n_alarm else 0


if __name__ == "__main__":
    raise SystemExit(main())
