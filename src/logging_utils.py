"""Tiny logging helper shared across the package.

All handshaker loggers live under the ``handshaker.`` namespace so a single
``configure()`` call (from the CLI) controls verbosity for the whole run.
"""

from __future__ import annotations

import logging

_ROOT = "handshaker"


def get_logger(name: str) -> logging.Logger:
    """Return the ``handshaker.<name>`` logger."""
    return logging.getLogger(f"{_ROOT}.{name}")


def configure(level: str = "INFO") -> None:
    """Configure root-level logging for a CLI run."""
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
