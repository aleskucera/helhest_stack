"""drive_sim with the worlds this study adds registered: places built to make the reference stage
draw something odd. Same arguments as drive_sim; `--world` takes these names too.

    python sim_worlds.py --world single_on_line --frames 900 --ref-record out.npz

  uturn          open ground, the goal straight behind the start: pivot or loop?
  sidegoal       open ground, the goal 90 deg to the left
  single_on_line one 1 m block exactly on the line: left or right, and does it flip?
  wall_on_line   a 6 m wall across the line, centred: the same choice, bigger detour
  chicane        alternating half-walls, 3 m apart along x
  door_offaxis   a long wall with a 2.8 m door 4 m off the line, the goal past it on the other side
  cul_de_sac     a U open toward the robot with the goal behind its back wall
  goal_at_wall   the goal 0.6 m in front of a wall face
  facing_wall    the start 1.8 m from a wall it faces, the goal behind the wall
  forest         0.5 m trunks scattered over 30 m (seeded), the goal across it
  narrow_s       a 2.8 m wide S-shaped corridor
  trunks         five trunks beside a 45 m straight (run_trunks.py on dasenka)
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "clearance"))
sys.path.insert(0, str(HERE.parents[0] / "closed_loop"))
import worlds as _cw  # noqa: E402  studies/clearance/worlds.py: corridors, cornerL24, _builder

from helhest import worlds as W  # noqa: E402

_cw.register()
B = W.Box


def add(name: str, boxes, xlim, ylim, start, goal) -> None:
    W.OBSTACLES[name] = tuple(boxes)
    W.WORLDS[name] = (_cw._builder(name, xlim, ylim), start, goal)


add("uturn", [B(30.0, 30.0, 0.1, 0.1)], (-14.0, 8.0), (-8.0, 8.0), (0.0, 0.0, 0.0), (-8.0, 0.0))
add("sidegoal", [B(30.0, 30.0, 0.1, 0.1)], (-6.0, 10.0), (-6.0, 14.0), (0.0, 0.0, 0.0), (0.0, 9.0))
add(
    "single_on_line",
    [B(6.0, 0.0, 0.5, 0.5)],
    (-3.0, 17.0),
    (-6.0, 6.0),
    (0.0, 0.0, 0.0),
    (13.0, 0.0),
)
add(
    "wall_on_line", [B(6.0, 0.0, 0.1, 3.0)], (-3.0, 17.0), (-8.0, 8.0), (0.0, 0.0, 0.0), (13.0, 0.0)
)
add(
    "chicane",
    [
        B(4.0, 1.6, 0.1, 2.4),
        B(7.0, -1.6, 0.1, 2.4),
        B(10.0, 1.6, 0.1, 2.4),
        B(13.0, -1.6, 0.1, 2.4),
    ],
    (-3.0, 19.0),
    (-5.0, 5.0),
    (0.0, 0.0, 0.0),
    (16.5, 0.0),
)
add(
    "door_offaxis",
    [B(6.0, -4.4, 0.1, 5.0), B(6.0, 8.4, 0.1, 3.0)],  # door between y 0.6 and 5.4 -> centred 3.0
    (-3.0, 15.0),
    (-10.0, 12.0),
    (0.0, 0.0, 0.0),
    (12.0, -2.0),
)
add(
    "cul_de_sac",
    [B(8.0, 0.0, 0.1, 2.5), B(5.5, 2.5, 2.5, 0.1), B(5.5, -2.5, 2.5, 0.1)],
    (-3.0, 15.0),
    (-8.0, 8.0),
    (0.0, 0.0, 0.0),
    (11.0, 0.0),
)
add(
    "goal_at_wall", [B(10.0, 0.0, 0.1, 3.0)], (-3.0, 14.0), (-6.0, 6.0), (0.0, 0.0, 0.0), (9.3, 0.0)
)
add(
    "facing_wall",
    [B(1.9, 0.0, 0.1, 3.5)],
    (-6.0, 10.0),
    (-7.0, 7.0),
    (0.0, 0.0, 0.0),
    (5.0, 0.0),
)
_rng = np.random.default_rng(7)
_trees, _tries = [], 0
while len(_trees) < 26 and _tries < 5000:
    _tries += 1
    x, y = _rng.uniform(3.0, 27.0), _rng.uniform(-6.0, 6.0)
    if all(np.hypot(x - a, y - b) > 2.8 for a, b in _trees):
        _trees.append((x, y))
add(
    "forest",
    [B(x, y, 0.25, 0.25) for x, y in _trees],
    (-3.0, 33.0),
    (-9.0, 9.0),
    (0.0, 0.0, 0.0),
    (30.0, 0.0),
)
add(
    "narrow_s",
    [
        B(4.05, -1.5, 5.05, 0.1),  # leg 1 (east, y -1.4..1.4): south wall
        B(2.55, 1.5, 3.55, 0.1),  # leg 1 north wall, open past x 6.1
        B(9.1, 1.85, 0.1, 3.35),  # leg 1 end / up-leg east wall
        B(6.1, 4.8, 0.1, 3.3),  # up-leg (x 6.2..9.0) west wall
        B(11.05, 8.1, 4.95, 0.1),  # leg 2 (east, y 5.2..8.0) north wall
        B(12.55, 5.1, 3.45, 0.1),  # leg 2 south wall
    ],
    (-3.0, 17.0),
    (-4.0, 11.0),
    (0.0, 0.0, 0.0),
    (14.0, 6.6),
)
add(
    "trunks",
    [
        B(x, y, 0.25, 0.25)
        for x, y in ((10.0, 0.85), (17.0, -1.0), (24.0, 0.8), (31.0, -1.2), (38.0, 0.9))
    ],
    (-3.0, 50.0),
    (-5.0, 5.0),
    (0.0, 0.0, 0.0),
    (45.0, 0.0),
)

import drive_sim  # noqa: E402

if __name__ == "__main__":
    drive_sim.main()
