"""Command-line magcor over a SNANA MODELSPEC .root file.

    handshaker-magcor --modelspec CODETEST_HANDSHAKER.root \
        --filters-root /project2/.../ROMAN_FILTERS \
        --reference-sca 2 --ice released --thickness-from-mjd -o magcor.csv

Reads the MODELSPEC tree, computes the per-observation magcor (true filter vs the
reference SCA, plus the wavecor-corrected mag and residual), and writes a CSV.
This is the MODELSPEC path; ``handshaker <config.yaml>`` is the config-driven one.
"""

from __future__ import annotations

import argparse
import sys

from .filters import get_filter
from .ice import get_ice_model
from .logging_utils import configure, get_logger
from .modelspec import load_modelspec, magcor_table, mjd_ice_thickness

log = get_logger("magcor_cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="handshaker-magcor",
                                description="Per-observation magcor over a SNANA MODELSPEC .root file.")
    p.add_argument("--modelspec", required=True, help="SNANA .root file with the MODELSPEC tree")
    p.add_argument("--tree", default="MODELSPEC", help="tree name (default MODELSPEC)")
    p.add_argument("--filters-root", required=True, help="ROMAN_FILTERS directory")
    p.add_argument("--filter-source", default="roman_filters", help="filter source (roman_filters | soc)")
    p.add_argument("--reference-sca", type=int, default=2, help="fixed reference SCA (default 2)")
    p.add_argument("--ice", default="released", choices=["released", "polyfit", "none"],
                   help="ice model (default released)")
    p.add_argument("--order", type=int, default=5, help="polynomial order for --ice polyfit")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--thickness", type=float, help="fixed ice thickness [nm] for every observation")
    g.add_argument("--thickness-from-mjd", action="store_true",
                   help="ice thickness from MJD (steady growth + decon cadence)")
    p.add_argument("-o", "--output", default="magcor.csv", help="output CSV (default magcor.csv)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    configure("DEBUG" if args.verbose else "INFO")

    filters = get_filter(args.filter_source, root=args.filters_root)
    ice = (get_ice_model("polyfit", filters=filters, order=args.order)
           if args.ice == "polyfit" else get_ice_model(args.ice))
    spec = load_modelspec(args.modelspec, tree=args.tree)

    if args.thickness_from_mjd:
        thickness = lambda r: mjd_ice_thickness(r.mjd)
    elif args.thickness is not None:
        thickness = args.thickness
    else:
        thickness = 0.0

    tbl = magcor_table(spec, filters, ice, reference_sca=args.reference_sca,
                       ice_thickness_nm=thickness)
    tbl.to_csv(args.output, index=False)
    log.info("wrote %d rows to %s", len(tbl), args.output)
    print(tbl.head(min(len(tbl), 10)).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
