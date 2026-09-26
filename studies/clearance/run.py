"""drive_sim with the study worlds registered and plan_* values overridden from the command line.

  python studies/clearance/run.py --world cornerL24 --set plan_clear_c0=0.2 --history 4 --out x.npz

Every argument except --set goes to drive_sim unchanged; --set KEY=VALUE (repeatable) overrides a
plan_* value on top of the robot's params file.
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "closed_loop"))

import worlds  # noqa: E402,F401  (registers the study worlds)
import drive_sim  # noqa: E402


def _pop_overrides(argv: list[str]) -> tuple[list[str], dict[str, float]]:
    rest, sets = [], {}
    it = iter(argv)
    for arg in it:
        if arg == "--set":
            key, value = next(it).split("=", 1)
            sets[key] = float(value)
        else:
            rest.append(arg)
    return rest, sets


def main() -> None:
    sys.argv[1:], overrides = _pop_overrides(sys.argv[1:])
    plan_params = drive_sim._plan_params

    def with_overrides(a):
        return {**plan_params(a), **overrides}

    drive_sim._plan_params = with_overrides
    drive_sim.main()


if __name__ == "__main__":
    main()
