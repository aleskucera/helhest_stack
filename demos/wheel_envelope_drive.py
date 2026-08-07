"""Drive the sphere and cylinder wheel envelopes side by side, live.

Two simulators, one keyboard: identical wheel commands go to a BLUE robot using the old spherical
wheel envelope (`wheel_width=None`, reaching a full 0.35 m sideways) and a RED robot using the
measured 0.10 m cylinder. They start at the same pose and diverge only where the envelopes differ
-- beside the wheel track. Straight over an obstacle they stay locked together, which is itself
the point: the cylinder changes nothing along the direction of travel.

The scene is built to make that visible: a slalom of rocks placed at increasing lateral offsets
from the wheel track, a rock the wheels straddle, and two ridges (one across the path, one along
it). Drive between the rocks and watch the blue robot climb what the red one drives past.

The title bar carries the live numbers -- the pose/tilt gap between the two models, and the new
certificates (tip-over margin, friction saturation, torque stall) for each.

What to expect: the two models barely separate in POSITION (planar motion comes from the wheel
speeds and the grip solve, which the envelope only touches indirectly) but disagree sharply in
TILT -- and tilt is what the planner's feasibility gates and tilt costs actually read. Watch roll,
not the gap.

Controls:
    W / S     forward / back            mouse-drag   orbit camera
    A / D     turn left / right         scroll       zoom
    R         reset both to the start   T            re-sync RED onto BLUE's pose
    Esc / Q   quit

`T` is the useful one for A/B: it removes accumulated divergence so you compare the models'
INSTANTANEOUS response at the same pose instead of two histories.

Needs `glfw` + `PyOpenGL` and a display:  uv pip install glfw PyOpenGL
    python demos/wheel_envelope_drive.py
    python demos/wheel_envelope_drive.py --device cpu --wheel-width 0.2
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import warp as wp

from helhest import heightmap as hm_mod
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.viz.render import _draw
from helhest.viz.render import _init_gl
from helhest.viz.render import build_robot
from helhest.viz.render import build_terrain

CELL = 0.05
XLIM, YLIM = (-3.0, 13.0), (-6.0, 6.0)
START_POSE = (-1.5, 0.0, 0.0)
DT = 0.1
BASE_SPEED, TURN_SPEED = 3.0, 2.0
SPHERE_RGB, CYLINDER_RGB = (0.27, 0.51, 0.71), (0.86, 0.08, 0.24)
PW, PH = 1280, 820


def _scene() -> hm_mod.Heightmap:
    """Rocks at increasing lateral offsets, one straddleable rock, and two ridges."""
    xs = np.arange(XLIM[0], XLIM[1], CELL)
    ys = np.arange(YLIM[0], YLIM[1], CELL)
    X, Y = np.meshgrid(xs, ys)
    H = np.zeros_like(X)

    # narrow rocks (sigma 0.09): at these lateral offsets they sit entirely outside the 0.05 m
    # cylinder half-width while still well inside the sphere's 0.35 m reach
    def rock(cx: float, cy: float, height: float, sigma: float = 0.09) -> np.ndarray:
        return height * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma**2))

    # slalom: lateral offset from the LEFT wheel track (y = +0.365) grows past the sphere's reach
    for i, offset in enumerate((0.14, 0.22, 0.28, 0.33)):
        H = np.maximum(H, rock(1.0 + 1.6 * i, 0.365 + offset, 0.30))
    # the same series on the right track, so a straight run is bracketed
    for i, offset in enumerate((0.33, 0.28, 0.22, 0.14)):
        H = np.maximum(H, rock(1.8 + 1.6 * i, -0.365 - offset, 0.30))
    # a rock ON the centreline: both models straddle it (it is between the wheels)
    H = np.maximum(H, rock(8.5, 0.0, 0.35))
    # ridge ACROSS the path -- both models must climb it identically
    H = np.maximum(H, np.where((X > 10.2) & (X < 10.6), 0.22, 0.0))
    # ridge ALONG the path, just outside the left track
    H = np.maximum(H, np.where((Y > 0.72) & (Y < 1.0) & (X > 2.0) & (X < 9.0), 0.30, 0.0))
    return hm_mod.Heightmap(H, (XLIM[0], YLIM[0]), CELL)


class _Model:
    """One simulator plus the host-side pose it is driven from."""

    def __init__(self, name: str, wheel_width: float | None, scene, grid, device: str, rgb):
        self.name, self.rgb = name, rgb
        self.sim = ForwardSimulator(
            RobotParams(wheel_width=wheel_width),
            SolverParams(dt=DT),
            grid,
            batch_size=1,
            n_steps=1,
            device=device,
        )
        self.sim.set_uniform_friction(0.6)
        self.sim.set_terrain(
            wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
        )
        self.reset()

    def reset(self, pose: tuple[float, float, float] = START_POSE) -> None:
        self.pose = np.asarray(pose, np.float64)
        self.trail: list[np.ndarray] = []
        self.step((0.0, 0.0))  # settle in place so z/pitch/roll are valid before the first draw

    def step(self, drive: tuple[float, float]) -> None:
        """Advance one dt under (left, right) wheel speed; the rear wheel follows the mean."""
        left, right = drive
        omega = np.array([[[left, right, 0.5 * (left + right)]]], np.float32)
        controlled, derived, clearance, residual = self.sim.rollout(omega, tuple(self.pose))
        self.pose = controlled[1, 0].astype(np.float64)
        self.derived = derived[1, 0]
        self.clearance, self.residual = float(clearance[0, 0]), float(residual[0, 0])
        self.stability = float(self.sim.stability.numpy()[0, 0])
        self.saturation = float(self.sim.saturation.numpy()[0, 0])
        self.stall = float(self.sim.stall.numpy()[0, 0])
        self.trail.append(np.array([self.pose[0], self.pose[1], self.derived[0] + 0.05]))
        if len(self.trail) > 4000:
            self.trail.pop(0)

    @property
    def state(self) -> SimpleNamespace:
        """Draw pose: R = Rz(yaw) Ry(pitch) Rx(roll), the engine's convention."""
        yaw, pitch, roll = self.pose[2], self.derived[1], self.derived[2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        R = np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            np.float32,
        )
        return SimpleNamespace(x=self.pose[0], y=self.pose[1], z=self.derived[0], R=R)


def _tint(robot: tuple, rgb: tuple[float, float, float]) -> tuple:
    """Recolour the robot mesh, keeping the wheels dark so the body reads as the model's colour."""
    V, N, C, _ = robot
    tinted = (0.45 * C + 0.55 * np.asarray(rgb, np.float32)).astype(np.float32)
    return V, N, tinted


def _draw_model(mesh: tuple, model: _Model) -> None:
    from OpenGL import GL as gl

    st = model.state
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = st.R
    gl.glPushMatrix()
    gl.glTranslatef(st.x, st.y, st.z)
    gl.glMultMatrixf(np.ascontiguousarray(M.T))
    _draw(*mesh)
    gl.glPopMatrix()

    if len(model.trail) > 1:
        gl.glDisable(gl.GL_LIGHTING)
        gl.glColor3f(*model.rgb)
        gl.glLineWidth(2.5)
        gl.glBegin(gl.GL_LINE_STRIP)
        for p in model.trail:
            gl.glVertex3f(*p)
        gl.glEnd()
        gl.glEnable(gl.GL_LIGHTING)


def _camera_callbacks(cam: list[float], mouse: dict):
    import glfw

    def on_button(win, button, action, _mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse["down"] = action == glfw.PRESS
            mouse["x"], mouse["y"] = glfw.get_cursor_pos(win)

    def on_cursor(_win, x, y):
        if mouse["down"]:
            cam[0] -= 0.008 * (x - mouse["x"])
            cam[1] = float(np.clip(cam[1] + 0.008 * (y - mouse["y"]), 0.05, 1.45))
            mouse["x"], mouse["y"] = x, y

    def on_scroll(_win, _dx, dy):
        cam[2] = float(np.clip(cam[2] * (0.9 if dy > 0 else 1.1), 2.0, 60.0))

    return on_button, on_cursor, on_scroll


def run(device: str, wheel_width: float) -> None:
    import glfw
    from OpenGL import GL as gl
    from OpenGL import GLU as glu

    scene = _scene()
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)
    sphere = _Model("sphere", None, scene, grid, device, SPHERE_RGB)
    cylinder = _Model("cylinder", wheel_width, scene, grid, device, CYLINDER_RGB)

    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    win = glfw.create_window(PW, PH, "wheel envelope: sphere vs cylinder", None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError("could not create a window (no display?)")
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    _init_gl()

    terrain = build_terrain(scene)
    base_robot = build_robot()
    meshes = {m.name: _tint(base_robot, m.rgb) for m in (sphere, cylinder)}
    cam = [-2.4, 0.55, 9.0]
    mouse = {"down": False, "x": 0.0, "y": 0.0}
    bcb, ccb, scb = _camera_callbacks(cam, mouse)
    glfw.set_mouse_button_callback(win, bcb)
    glfw.set_cursor_pos_callback(win, ccb)
    glfw.set_scroll_callback(win, scb)

    key_was_down = {"R": False, "T": False}
    while not glfw.window_should_close(win):
        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            break
        if glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS:
            break

        left = right = 0.0
        if glfw.get_key(win, glfw.KEY_W) == glfw.PRESS:
            left += BASE_SPEED
            right += BASE_SPEED
        if glfw.get_key(win, glfw.KEY_S) == glfw.PRESS:
            left -= BASE_SPEED
            right -= BASE_SPEED
        if glfw.get_key(win, glfw.KEY_A) == glfw.PRESS:
            left -= TURN_SPEED
            right += TURN_SPEED
        if glfw.get_key(win, glfw.KEY_D) == glfw.PRESS:
            left += TURN_SPEED
            right -= TURN_SPEED

        reset_now = glfw.get_key(win, glfw.KEY_R) == glfw.PRESS
        if reset_now and not key_was_down["R"]:
            sphere.reset()
            cylinder.reset()
        key_was_down["R"] = reset_now
        sync_now = glfw.get_key(win, glfw.KEY_T) == glfw.PRESS
        if sync_now and not key_was_down["T"]:
            cylinder.reset(tuple(sphere.pose))
        key_was_down["T"] = sync_now

        sphere.step((left, right))
        cylinder.step((left, right))

        d_pos = float(np.hypot(*(sphere.pose[:2] - cylinder.pose[:2])))
        d_roll = np.degrees(sphere.derived[2] - cylinder.derived[2])
        d_pitch = np.degrees(sphere.derived[1] - cylinder.derived[1])
        glfw.set_window_title(
            win,
            f"BLUE sphere  roll {np.degrees(sphere.derived[2]):+5.1f} pitch "
            f"{np.degrees(sphere.derived[1]):+5.1f} sat {sphere.saturation:4.2f} "
            f"stab {sphere.stability:4.2f} stall {sphere.stall:4.2f}   |   "
            f"RED cylinder  roll {np.degrees(cylinder.derived[2]):+5.1f} pitch "
            f"{np.degrees(cylinder.derived[1]):+5.1f} sat {cylinder.saturation:4.2f} "
            f"stab {cylinder.stability:4.2f} stall {cylinder.stall:4.2f}   |   "
            f"gap {d_pos:4.2f} m  d_roll {d_roll:+5.1f} deg  d_pitch {d_pitch:+5.1f} deg",
        )

        w, h = glfw.get_framebuffer_size(win)
        az, el, dist = cam
        target = np.array([sphere.pose[0], sphere.pose[1], sphere.derived[0]])
        eye = target + dist * np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        gl.glViewport(0, 0, w, h)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(50.0, w / max(h, 1), 0.1, 200.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        glu.gluLookAt(*eye, *target, 0, 0, 1)

        _draw(*terrain[:3], terrain[3])
        for model in (sphere, cylinder):
            _draw_model(meshes[model.name], model)
        glfw.swap_buffers(win)

    glfw.terminate()


def run_headless(device: str, wheel_width: float, steps: int = 90) -> None:
    """Same two models, a scripted straight run through the slalom, printed instead of drawn.

    Useful without a display, and as a smoke test of the demo itself.
    """
    scene = _scene()
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)
    sphere = _Model("sphere", None, scene, grid, device, SPHERE_RGB)
    cylinder = _Model("cylinder", wheel_width, scene, grid, device, CYLINDER_RGB)
    print(
        f"{'x [m]':>7} | {'sphere roll':>11} {'pitch':>7} {'sat':>5} | "
        f"{'cyl roll':>9} {'pitch':>7} {'sat':>5} | {'gap [m]':>8}"
    )
    worst_roll, worst_gap = 0.0, 0.0
    for i in range(steps):
        sphere.step((BASE_SPEED, BASE_SPEED))
        cylinder.step((BASE_SPEED, BASE_SPEED))
        gap = float(np.hypot(*(sphere.pose[:2] - cylinder.pose[:2])))
        d_roll = abs(np.degrees(sphere.derived[2] - cylinder.derived[2]))
        worst_roll, worst_gap = max(worst_roll, d_roll), max(worst_gap, gap)
        if i % 6 == 0:
            print(
                f"{sphere.pose[0]:7.2f} | {np.degrees(sphere.derived[2]):11.2f} "
                f"{np.degrees(sphere.derived[1]):7.2f} {sphere.saturation:5.2f} | "
                f"{np.degrees(cylinder.derived[2]):9.2f} {np.degrees(cylinder.derived[1]):7.2f} "
                f"{cylinder.saturation:5.2f} | {gap:8.3f}"
            )
    print(f"worst tilt disagreement {worst_roll:.2f} deg, worst position gap {worst_gap:.3f} m")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--wheel-width",
        type=float,
        default=0.10,
        help="cylinder wheel width [m] (measured: 0.10)",
    )
    ap.add_argument("--headless", action="store_true", help="scripted run, printed, no window")
    args = ap.parse_args()
    wp.init()
    if args.headless:
        run_headless(args.device, args.wheel_width)
    else:
        run(args.device, args.wheel_width)


if __name__ == "__main__":
    main()
