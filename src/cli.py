"""Command-line entry point: ``handshaker <config.yaml>``."""

from __future__ import annotations

import argparse
import sys

from .config import Config
from .logging_utils import configure, get_logger
from .pipeline import run_config

log = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="handshaker",
        description="Chromatic magnitude corrections (magcor) for Roman SN photometry.",
    )
    p.add_argument("config", help="Path to the YAML config file.")
    p.add_argument("--mode", choices=["magcor", "polyfit"], help="Override config mode.")
    p.add_argument("-o", "--output-dir", help="Override config output directory.")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    configure("DEBUG" if args.verbose else "INFO")

    cfg = Config.from_file(args.config)
    if args.mode:
        cfg.mode = args.mode
    if args.output_dir:
        cfg.output_dir = args.output_dir

    df = run_config(cfg)
    log.info("done: %d rows in mode '%s' -> %s", len(df), cfg.mode, cfg.output_dir)
    # Brief human-readable summary to stdout.
    print(df.head(min(len(df), 10)).to_string(index=False))
    return 0


if __name__ == "__main__": 
    sys.exit(main())
