"""Run the engine's standalone parity scripts under pytest.

Each script in this directory prints OK / REVIEW / FAIL rather than asserting, so it is only
checked when someone runs it by hand -- which is how the golden fixture sat broken for a week and
the momentum check crashed unnoticed. Each runs in its own process (the scripts own `wp.init`
and module state) and fails here on a non-zero exit or any REVIEW / FAIL line.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = [
    "cylinder",
    "cylinder_contact",
    "delay",
    "envelope",
    "golden",
    "gpu_check",
    "gradients",
    "integrator",
    "momentum",
    "step",
    "terrain",
    "traction",
    "verify_device",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_engine_selftest(script: str) -> None:
    run = subprocess.run(
        [sys.executable, "-m", f"tests.engine.{script}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    flagged = [ln for ln in run.stdout.splitlines() if "REVIEW" in ln or "FAIL" in ln]
    assert run.returncode == 0 and not flagged, (
        "\n".join(flagged) + run.stdout[-2000:] + run.stderr[-2000:]
    )
