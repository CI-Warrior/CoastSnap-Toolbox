"""Where a CoastSnap database lives on disk (replaces CSPloadPaths/CSPsetPaths).

The MATLAB toolbox hardcodes Windows paths in ``CSPloadPaths.m``. Here the
base folder comes from, in order:

1. an explicit argument (``load_paths("D:/CoastSnap")`` or ``--base`` on the CLI),
2. the ``COASTSNAP_BASE`` environment variable,
3. ``coastsnap.toml`` in the working directory, or
   ``~/.config/coastsnap/config.toml``.

A config file may also override individual folders::

    [paths]
    base = "D:/CoastSnap"
    tide = "E:/shared/Tide Data"   # optional; relative paths are under base
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CONFIG_FILES = (Path("coastsnap.toml"), Path.home() / ".config" / "coastsnap" / "config.toml")

# Folder names used by CSPloadPaths, relative to the base folder.
_DEFAULTS = {
    "database": "Database",
    "images": "Images",
    "shorelines": "Shorelines",
    "tide": "Tide Data",
    "transects": os.path.join("Shorelines", "Transect Files"),
}


@dataclass(frozen=True)
class Paths:
    """The folders of a CoastSnap database (the variables CSPloadPaths sets)."""

    base: Path
    database: Path
    images: Path
    shorelines: Path
    tide: Path
    transects: Path

    @classmethod
    def from_base(cls, base, **overrides) -> "Paths":
        base = Path(base).expanduser()
        dirs = {k: base / v for k, v in _DEFAULTS.items()}
        for k, v in overrides.items():
            if k not in _DEFAULTS:
                raise KeyError(f"unknown path {k!r}; expected one of {sorted(_DEFAULTS)}")
            v = Path(v).expanduser()
            dirs[k] = v if v.is_absolute() else base / v
        return cls(base=base, **dirs)

    @property
    def db_file(self) -> Path:
        """``Database/CoastSnapDB.xlsx``."""
        return self.database / "CoastSnapDB.xlsx"

    def site_images(self, site: str, kind: str = "Processed") -> Path:
        """``Images/<site>/<kind>`` where kind is Raw, Processed, Registered or Rectified."""
        return self.images / site / kind

    def site_shorelines(self, site: str) -> Path:
        return self.shorelines / site


def _read_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_paths(base=None, config_file=None) -> Paths:
    """Resolve the database folders; see the module docstring for the order."""
    if base is not None:
        return Paths.from_base(base)
    env = os.environ.get("COASTSNAP_BASE")
    if env:
        return Paths.from_base(env)
    candidates = [Path(config_file)] if config_file else list(CONFIG_FILES)
    for f in candidates:
        if f.is_file():
            section = dict(_read_toml(f).get("paths", {}))
            if "base" not in section:
                raise ValueError(f"{f} has no [paths] base entry")
            return Paths.from_base(section.pop("base"), **section)
    raise FileNotFoundError(
        "CoastSnap database folder not set: pass a base folder, set COASTSNAP_BASE, "
        "or create coastsnap.toml with a [paths] base entry"
    )
