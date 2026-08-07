"""Cylinder vs sphere wheel envelope (RobotParams.wheel_width), against hand geometry.

Run:  python -m tests.engine.cylinder

Three worlds, each with a closed-form answer:

  transverse ridge  -- a ridge across the path. Sphere and cylinder envelopes must be IDENTICAL:
                       along travel a cylinder presents the same circle, and for terrain that
                       varies only along x the disk's best offset is dy = 0, which the rectangle
                       also contains.
  lateral ridge     -- a ridge beside the wheel track, 0.30 m out from the left wheel centre. The
                       sphere reaches 0.35 m sideways and climbs it; the 0.2 m-wide cylinder does
                       not reach it at all and must stay level.
  lateral ridge, yaw 90 deg -- the same ridge, approached head-on. Now it IS along travel, so the
                       cylinder must climb it exactly like the sphere. This is what proves the
                       stack is indexed by heading rather than all bins holding one table.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams

CELL = 0.05
CELLS = 201
ORIGIN = (-5.0, -5.0)
# the real wheel, ruler-measured: 0.10 m wide, i.e. a 0.05 m half-width against the sphere's 0.35
WHEEL_WIDTH = 0.10
RIDGE_HEIGHT = 0.3
RIDGE_GAP = 0.30  # lateral gap from the left wheel centre to the near edge of the ridge


def _device() -> str:
    return "cuda" if wp.get_cuda_device_count() > 0 else "cpu"


def _ridge(along_x: bool, edge: float, width: float = 0.2) -> np.ndarray:
    """A flat-topped ridge of `width`, its near edge at `edge`, running along x (or along y)."""
    axis = ORIGIN[0] + CELL * (np.arange(CELLS) + 0.5)
    X, Y = np.meshgrid(axis, axis)
    coord = Y if along_x else X
    inside = (coord >= edge) & (coord <= edge + width)
    return np.ascontiguousarray(np.where(inside, RIDGE_HEIGHT, 0.0), np.float32)


def _run(wheel_width: float | None, heights: np.ndarray, pose: tuple[float, float, float]):
    """Settle the robot on `heights` at `pose` and return its envelope and settled state."""
    rp = RobotParams(wheel_width=wheel_width)
    device = _device()
    sim = ForwardSimulator(
        rp, SolverParams(), GridParams(CELLS, CELLS, CELL, *ORIGIN), 1, 1, device=device
    )
    sim.set_uniform_friction(0.6)
    sim.set_terrain(wp.array(heights, dtype=wp.float32, device=device))
    sim.rollout(np.zeros((1, 1, 3), np.float32), pose)
    derived = sim.derived.numpy()[1, 0]
    return {
        "envelope": sim.envelope_stack.numpy(),
        "z": float(derived[0]),
        "pitch": float(derived[1]),
        "roll": float(derived[2]),
    }


def _pose_from_lift(rp: RobotParams, lift: float) -> tuple[float, float, float]:
    """Settled (z, pitch, roll) with only the LEFT wheel lifted by `lift`, solved by hand.

    The settle puts every wheel centre one radius above the envelope, so with the other two on
    flat ground:  z = R + lift/2,  sin(pitch) = -lift / (2 rear_offset),
    sin(roll) = lift / (2 half_track cos(pitch)).
    """
    z = rp.wheel_radius + lift / 2.0
    pitch = math.asin(-lift / (2.0 * rp.rear_offset))
    roll = math.asin(lift / (2.0 * rp.half_track * math.cos(pitch)))
    return z, pitch, roll


def _continuous_lift(rp: RobotParams, ridge_edge: float) -> float:
    """Spherical-cap lift under the left wheel, in CONTINUOUS geometry (no grid).

    Rolling swings the wheel centre inboard to half_track cos(roll), which widens the gap to the
    ridge and reduces the lift, so the two are iterated to a fixed point. The dilation is
    discrete, though: it can only reach the ridge at whole-cell offsets, so the engine's lift is
    the cap at a distance rounded UP to a cell multiple. At this gap dcap/dgap = -1.7 m/m, so a
    half cell of rounding is ~0.04 m of lift -- do not expect a tight match to this number.
    """
    radius, lift, roll = rp.wheel_radius, 0.0, 0.0
    for _ in range(64):
        gap = ridge_edge - rp.half_track * math.cos(roll)
        cap = math.sqrt(radius**2 - gap**2) - radius if gap < radius else -radius
        lift = max(RIDGE_HEIGHT + cap, 0.0)
        roll = _pose_from_lift(rp, lift)[2]
    return lift


def _sample(field: np.ndarray, x: float, y: float) -> float:
    """Bilinear sample of a grid field, matching the engine's cell-centre convention."""
    fx = (x - ORIGIN[0]) / CELL - 0.5
    fy = (y - ORIGIN[1]) / CELL - 0.5
    ix, iy = int(math.floor(fx)), int(math.floor(fy))
    tx, ty = fx - ix, fy - iy
    return float(
        (1 - tx) * (1 - ty) * field[iy, ix]
        + tx * (1 - ty) * field[iy, ix + 1]
        + (1 - tx) * ty * field[iy + 1, ix]
        + tx * ty * field[iy + 1, ix + 1]
    )


def selftest_transverse_ridge() -> None:
    """Across the path the two envelopes must agree exactly -- the cylinder changes nothing."""
    heights = _ridge(along_x=False, edge=1.0)
    sphere = _run(None, heights, (0.0, 0.0, 0.0))
    cylinder = _run(WHEEL_WIDTH, heights, (0.0, 0.0, 0.0))
    d_env = float(np.abs(sphere["envelope"][0] - cylinder["envelope"][0]).max())
    d_state = max(abs(sphere[k] - cylinder[k]) for k in ("z", "pitch", "roll"))
    print(f"transverse ridge  max|d envelope|={d_env:.3e}  max|d state|={d_state:.3e}")
    # not bit-exact: the disk kernel computes its cap on device in float32 while the cylinder
    # table is built on the host in float64 and cast, which costs an ULP (~3e-8 m)
    assert d_env < 1e-6, "along travel the cylinder envelope must equal the sphere's"
    assert d_state < 1e-6, "same envelope must give the same settled pose"
    print("transverse ridge  OK (identical along travel)")


def selftest_lateral_ridge() -> None:
    """Beside the track the sphere climbs a ridge it should straddle; the cylinder must not."""
    rp = RobotParams()
    ridge_edge = rp.half_track + RIDGE_GAP
    heights = _ridge(along_x=True, edge=ridge_edge)
    sphere = _run(None, heights, (0.0, 0.0, 0.0))
    cylinder = _run(WHEEL_WIDTH, heights, (0.0, 0.0, 0.0))

    for name, r in (("sphere", sphere), ("cylinder", cylinder)):
        print(
            f"  {name:9s} roll={np.degrees(r['roll']):7.3f} deg  pitch={np.degrees(r['pitch']):7.3f}"
            f" deg  z={r['z']:.4f} m"
        )
    # the lift the sphere actually found, read off its envelope at the settled wheel centre
    pitch_s, roll_s = sphere["pitch"], sphere["roll"]
    wheel_x = rp.half_track * math.sin(pitch_s) * math.sin(roll_s)
    wheel_y = rp.half_track * math.cos(roll_s)
    lift = _sample(sphere["envelope"][0], wheel_x, wheel_y)
    z_h, pitch_h, roll_h = _pose_from_lift(rp, lift)
    continuous = _continuous_lift(rp, ridge_edge)
    print(
        f"  measured lift {lift:.4f} m (continuous cap geometry: {continuous:.4f} m, "
        f"{abs(lift - continuous) / continuous:.0%} low from whole-cell dilation offsets)"
    )
    print(
        f"  hand pose for that lift: roll={math.degrees(roll_h):.3f} deg "
        f"pitch={math.degrees(pitch_h):.3f} deg z={z_h:.4f} m"
    )
    rel_roll = abs(roll_s - roll_h) / roll_h
    rel_pitch = abs(pitch_s - pitch_h) / abs(pitch_h)
    assert rel_roll < 0.01, "settled roll does not match the one-wheel-lifted geometry"
    assert rel_pitch < 0.01, "settled pitch does not match the one-wheel-lifted geometry"
    assert abs(sphere["z"] - z_h) < 1e-3, "settled z does not match the one-wheel-lifted geometry"
    assert abs(lift - continuous) < 0.4 * continuous, "lift is far from the spherical cap"
    assert roll_s > np.radians(5.0), "the sphere should be climbing this ridge"
    assert abs(cylinder["roll"]) < 1e-6, "the cylinder reached a ridge 0.30 m off its track"
    assert abs(cylinder["pitch"]) < 1e-6, "the cylinder was lifted by a ridge beside the track"
    # ...but a ridge INSIDE the tread must still lift it. The cylinder's underside is a straight
    # line across its width, so an obstacle at zero along-travel offset lifts by its FULL height
    # with no cap falloff -- the envelope steps 0 -> 0.30 m across a single cell at the tread
    # edge, where the sphere's cap tapered smoothly. The settled lift here (0.175 m of a 0.30 m
    # ridge) is that step, bilinearly sampled under the wheel centre.
    near = _run(WHEEL_WIDTH, _ridge(along_x=True, edge=rp.half_track + 0.03), (0.0, 0.0, 0.0))
    print(
        f"  ridge 0.03 m out (inside the {WHEEL_WIDTH / 2:.2f} m half-width): cylinder rolls "
        f"{np.degrees(near['roll']):.2f} deg"
    )
    assert near["roll"] > np.radians(10.0), "the cylinder ignored an obstacle under its own tread"
    print(
        f"lateral ridge  OK (sphere tilts {np.degrees(sphere['roll']):.2f} deg, cylinder "
        f"{np.degrees(cylinder['roll']):.2e} deg)"
    )


def selftest_yaw_binning() -> None:
    """Head-on, the same ridge IS along travel -- so the cylinder must climb it like the sphere.

    Exercises bin 8 of 32 (90 deg); if every bin held the yaw = 0 table this world would report
    the cylinder driving through the ridge.
    """
    rp = RobotParams()
    ridge_edge = rp.half_track + RIDGE_GAP
    heights = _ridge(along_x=True, edge=ridge_edge)
    # heading +y, close enough that the front pair is within a wheel radius of the ridge
    pose = (0.0, 0.45, math.pi / 2.0)
    sphere = _run(None, heights, pose)
    cylinder = _run(WHEEL_WIDTH, heights, pose)
    d_state = max(abs(sphere[k] - cylinder[k]) for k in ("z", "pitch", "roll"))
    print(
        f"yaw 90 deg  sphere pitch={np.degrees(sphere['pitch']):.3f} deg  cylinder "
        f"pitch={np.degrees(cylinder['pitch']):.3f} deg  max|d state|={d_state:.3e}"
    )
    assert abs(sphere["pitch"]) > np.radians(1.0), "the head-on ridge should tilt the robot"
    assert d_state < 1e-6, "head-on, the cylinder must match the sphere (wrong yaw bin?)"
    print("yaw binning  OK (bin 8 = 90 deg is the along-travel element)")


if __name__ == "__main__":
    wp.init()
    selftest_transverse_ridge()
    print()
    selftest_lateral_ridge()
    print()
    selftest_yaw_binning()
