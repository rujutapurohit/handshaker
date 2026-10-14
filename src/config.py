"""Configuration loading.

Per the spec, a run "starts with a config file which should be copied into the
output directory for easy reproducibility". :class:`Config` loads that YAML file,
exposes the global inputs (filter source, SN/calibration SED, ice flag, output
directory, mode), and copies the original file verbatim into the output directory.

See ``configs/example.yaml`` for the full schema.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from .logging_utils import get_logger

log = get_logger("config")


@dataclass
class Config:
    """Parsed Handshaker configuration.

    Sub-sections (``filters``, ``sn_sed``, ``calib_star``, ``ice``, ``polyfit``,
    ``focal_plane``) are kept as plain dicts; :mod:`handshaker.pipeline` turns them
    into objects. This keeps config parsing decoupled from the physics classes.
    """

    mode: str = "magcor"                     # "magcor" or "polyfit"
    output_dir: str = "handshaker_out"
    filters: dict = field(default_factory=lambda: {"source": "synthetic"})
    sn_sed: dict = field(default_factory=lambda: {"model": "synthetic"})
    calib_star: Optional[dict] = None
    ice: dict = field(default_factory=lambda: {"flag": "none"})
    effects: Optional[list] = None
    polyfit: dict = field(default_factory=dict)
    focal_plane: Optional[dict] = None
    observations: Any = field(default_factory=list)  # inline list or {"path": csv}
    reference_filters: Optional[dict] = None
    reference_sca: Optional[int] = None               # fixed reference SCA (e.g. 2); None -> nominal
    grid: dict = field(default_factory=dict)          # canonical wave grid {min,max,step}
    raw: dict = field(default_factory=dict)
    source_path: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict, source_path: Optional[str] = None) -> "Config":
        known = {f for f in cls.__dataclass_fields__ if f not in ("raw", "source_path")}
        kwargs = {k: v for k, v in d.items() if k in known}
        unknown = set(d) - known
        if unknown:
            log.warning("ignoring unknown config keys: %s", sorted(unknown))
        return cls(raw=d, source_path=source_path, **kwargs)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        log.info("loaded config from %s", path)
        return cls.from_dict(d, source_path=os.path.abspath(path))

    def ensure_output_dir(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir

    def copy_to_output(self) -> Optional[str]:
        """Copy the original config file into the output dir (reproducibility)."""
        self.ensure_output_dir()
        dest = os.path.join(self.output_dir, "config.used.yaml")
        if self.source_path and os.path.exists(self.source_path):
            shutil.copyfile(self.source_path, dest)
        else:
            # No source file (built in-memory): dump the raw dict instead.
            with open(dest, "w") as f:
                yaml.safe_dump(self.raw or self._as_dict(), f, sort_keys=False)
        log.info("copied config to %s", dest)
        return dest

    def _as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "output_dir": self.output_dir,
            "filters": self.filters,
            "sn_sed": self.sn_sed,
            "calib_star": self.calib_star,
            "ice": self.ice,
            "effects": self.effects,
            "polyfit": self.polyfit,
            "focal_plane": self.focal_plane,
            "observations": self.observations,
            "reference_filters": self.reference_filters,
            "reference_sca": self.reference_sca,
            "grid": self.grid,
        }
