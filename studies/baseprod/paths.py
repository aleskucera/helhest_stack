"""Filesystem layout for the E2 BASEPROD run. Every path is derived from BASEPROD_ROOT.

    $BASEPROD_ROOT/
      data/drone_map/dsm/bardenas_dsm.tif      the truth raster
      data/calib/                              calibration archive (unused by the scored path)
      data/design/<traverse>/                  the two DESIGN traverses (CSVs + RS_DEPTH_16bit)
      data/heldout/<traverse>/                 one HELD-OUT traverse at a time, deleted after use
      out/                                     every artifact this study writes
      studies/                                 this code, deployed

The design/held-out split is a directory split, not a flag: `traverse_dir` looks in
`data/design` first and `data/heldout` second, and `is_held_out` reports which it found.
The runner refuses to touch `data/heldout` before FREEZE.json exists.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("BASEPROD_ROOT", "/local/kuceral4/baseprod"))
DATA = ROOT / "data"
DSM = DATA / "drone_map" / "dsm" / "bardenas_dsm.tif"
DESIGN_DIR = DATA / "design"
HELDOUT_DIR = DATA / "heldout"
OUT = ROOT / "out"

# The two traverses opened during the design phase (BASEPROD_AUDIT.md Task 2/4). Everything
# else in traverse_manifest.json is held out.
DESIGN_TRAVERSES = ("2023-07-23_13-05-11", "2023-07-22_14-18-23")
# short keys used throughout the spike and the rehearsals, kept so v3 artifacts stay legible
KEY = {"2023-07-23_13-05-11": "t1", "2023-07-22_14-18-23": "t2"}
NAME = {v: k for k, v in KEY.items()}

CONDITIONS = ("hindsight", "foresight")


def traverse_dir(name: str) -> Path:
    """Directory holding one traverse's CSVs and RS_DEPTH_16bit/."""
    name = NAME.get(name, name)
    for base in (DESIGN_DIR, HELDOUT_DIR):
        p = base / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"traverse {name} not found under {DESIGN_DIR} or {HELDOUT_DIR}")


def is_held_out(name: str) -> bool:
    return NAME.get(name, name) not in DESIGN_TRAVERSES


def window_dir(name: str, condition: str) -> Path:
    return OUT / "windows" / condition / KEY.get(name, name)


def freeze_path() -> Path:
    return OUT / "FREEZE.json"


def frozen() -> bool:
    return freeze_path().exists()
