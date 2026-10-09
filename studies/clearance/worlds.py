"""Study-only worlds for the clearance law, registered into `helhest.worlds` at import.

They are deliberately not in helhest.worlds itself: the corridor family was removed from the stress
set. Importing this module adds them for a run of drive_sim (see run.py).

- corridor24 / corridor26: 2.4 m and 2.6 m clear, walls 1 m tall, 16 m long; the robot starts
  0.5 m off the centre line (a real corridor read ~2.4 m clear).
- cornerL24: a 2.4 m corridor with a 90 deg left bend, 8 m east then 11 m north.
"""

from __future__ import annotations

import numpy as np

from helhest import worlds as W

_WALL_T = 0.2


def _builder(name: str, xlim: tuple[float, float], ylim: tuple[float, float]):
    def build(cell: float = 0.06) -> W.Heightmap:
        XX, YY = W._grid(xlim, ylim, cell)
        H = np.zeros_like(XX)
        for b in W.OBSTACLES[name]:
            W._box(H, XX, YY, b.cx, b.cy, b.hx, b.hy)
        return W.Heightmap(H, (xlim[0], ylim[0]), cell)

    return build


def register() -> None:
    for name, clear in (("corridor24", 2.4), ("corridor26", 2.6)):
        yw = 0.5 * clear + 0.5 * _WALL_T
        W.OBSTACLES[name] = (W.Box(6.0, yw, 8.0, 0.1), W.Box(6.0, -yw, 8.0, 0.1))
        W.WORLDS[name] = (_builder(name, (-3.0, 17.0), (-4.0, 4.0)), (0.0, -0.5, 0.0), (12.0, 0.0))
    W.OBSTACLES["cornerL24"] = (
        W.Box(3.1, -1.3, 5.1, 0.1),  # south wall of the east leg
        W.Box(8.1, 5.3, 0.1, 6.7),  # outer wall
        W.Box(1.8, 1.3, 3.8, 0.1),  # north wall of the east leg
        W.Box(5.5, 6.6, 0.1, 5.4),  # west wall of the north leg
    )
    W.WORLDS["cornerL24"] = (
        _builder("cornerL24", (-3.0, 10.0), (-3.0, 13.0)),
        (0.0, 0.0, 0.0),
        (6.8, 10.0),
    )


register()
