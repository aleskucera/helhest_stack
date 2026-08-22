"""The solid obstacles in `worlds.OBSTACLES` must describe the same worlds the
heightmap builders draw.

The builders rasterise obstacles into cells, which is what the planner consumes.
A physics simulator wants solids instead: extruding a rasterised wall produces
one sliver quad per cell across the height discontinuity -- 0.06 m wide and
1.0 m tall -- so a wheel finds a few badly-conditioned contacts where a box face
would give a dense manifold.

Both descriptions therefore exist, and they can drift. This rasterises the
solids and diffs them against the builder output so an edit to one that is not
mirrored in the other fails here rather than silently changing the simulated
world.
"""

import numpy as np
import pytest

from helhest.worlds import OBSTACLES, WORLDS, rasterise

# The (xlim, ylim) each builder uses. Kept here rather than exported, so that a
# builder changing its extent also fails this test.
LIMITS = {
    "gap": ((-2.0, 14.0), (-5.0, 5.0)),
    "slalom": ((-2.0, 19.0), (-5.0, 5.0)),
    "pillars": ((-2.0, 16.0), (-5.0, 5.0)),
    "pocket": ((-2.0, 16.0), (-5.0, 5.0)),
    "ridge": ((-2.0, 14.0), (-5.0, 5.0)),
}

# ridge is the one approximation. Its notch is a vertical cut in x, while a box
# ends perpendicular to its own axis, so the two end faces are off by the ridge
# angle (16.7 deg). Everything else is exact.
TOLERANCE = {"ridge": 0.02}

CELL = 0.06


@pytest.mark.parametrize("name", sorted(LIMITS))
def test_obstacles_match_heightmaps(name):
    reference = WORLDS[name][0]().H >= 0.99
    stamped = rasterise(OBSTACLES[name], *LIMITS[name], CELL) >= 0.99

    assert stamped.shape == reference.shape, "grid extent changed"

    wall_cells = int(reference.sum())
    mismatched = int((stamped != reference).sum())
    allowed = TOLERANCE.get(name, 0.0) * wall_cells
    assert mismatched <= allowed, (
        f"{name}: {mismatched} of {wall_cells} wall cells differ "
        f"({100 * mismatched / wall_cells:.2f}%), allowed {100 * TOLERANCE.get(name, 0.0):.0f}%"
    )


def test_bumpy_has_no_solid_obstacles():
    """bumpy is summed Gaussian mounds -- continuous terrain, not boxes.

    A heightmap represents it correctly and a box cannot, so it carries no
    solids and a simulator should keep using its heightmap.
    """
    assert OBSTACLES["bumpy"] == ()
    assert WORLDS["bumpy"][0]().H.max() > 0.0, "bumpy should still have relief"


def test_every_world_has_an_obstacle_entry():
    assert set(OBSTACLES) == set(WORLDS)


def test_boxes_are_well_formed():
    for name, boxes in OBSTACLES.items():
        for box in boxes:
            assert box.hx > 0 and box.hy > 0, f"{name}: non-positive half extent"
            assert box.h > 0, f"{name}: non-positive height"
            assert np.isfinite([box.cx, box.cy, box.hx, box.hy, box.h, box.yaw]).all()
