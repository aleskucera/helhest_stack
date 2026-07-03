"""Motor actuator lag comparison — OpenGL live view.

Single window, two sequential animations in the same scene:

  Stage 1  tau = 0.0 s  (instantaneous) — blue robot draws its trail
  Stage 2  tau = 0.5 s  (lagged)        — red robot draws its trail
           on top of the already-drawn tau=0 trail

The animation loops automatically.  Both trajectories are pre-computed once
with the Warp ForwardSimulator.

Controls:
  Mouse-drag  orbit camera
  Scroll      zoom
  ESC / Q     quit

    python demos/motor_lag_comparison_opengl.py
    python demos/motor_lag_comparison_opengl.py --device cpu
    python demos/motor_lag_comparison_opengl.py --speed 0.3   # 30 % of real-time
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import numpy as np
import warp as wp

from helhest import dynamics
from helhest import friction as friction_mod
from helhest import heightmap as hm_mod
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.viz.render import _draw
from helhest.viz.render import _init_gl
from helhest.viz.render import build_robot
from helhest.viz.render import build_terrain

# ── scenario ──────────────────────────────────────────────────────────────────
DT = 0.1  # [s]
PHASES = [
    (30, [2.5, 2.5, 2.5]),  # straight forward
    (30, [2.5, -1.0, 0.75]),  # asymmetric turn left
    (30, [2.5, 2.5, 2.5]),  # straight forward again
]
PHASE_NAMES = ["Phase 1: straight", "Phase 2: turn", "Phase 3: straight"]
TAUS = [0.0, 0.5]
LABELS = ["tau = 0.0 s  (instantaneous)", "tau = 0.5 s  (lagged)"]
# trail / robot body colours (steelblue, crimson)
COLORS_RGB = [(0.27, 0.51, 0.71), (0.86, 0.08, 0.24)]
XLIM = (-2.0, 14.0)
YLIM = (-8.0, 8.0)

PW, PH = 1100, 820  # window pixel size


def _build_setpoints() -> np.ndarray:
    return np.vstack([np.tile(np.asarray(cmd, np.float64), (n, 1)) for n, cmd in PHASES])


def _run_warp(
    tau: float,
    setpoints: np.ndarray,
    scene: hm_mod.Heightmap,
    mu: hm_mod.Heightmap,
    grid: GridParams,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (controlled [T+1,3], derived [T+1,3])."""
    T = len(setpoints)
    solver = dynamics.planning_solver(dt=DT)
    solver.tau_motor = tau
    sim = ForwardSimulator(
        dynamics.robot_params(), solver, grid, batch_size=1, n_steps=T, device=device
    )
    sim.set_terrain(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
    )
    sim.set_friction(mu)
    controlled, derived, *_ = sim.rollout(
        np.ascontiguousarray(setpoints[:, None, :], np.float32), (0.0, 0.0, 0.0), np.zeros(3)
    )
    return controlled[:, 0, :], derived[:, 0, :]


def _pose_to_st(pose3: np.ndarray, der3: np.ndarray) -> SimpleNamespace:
    x, y, yaw = float(pose3[0]), float(pose3[1]), float(pose3[2])
    z, pitch, roll = float(der3[0]), float(der3[1]), float(der3[2])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], np.float32)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], np.float32)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], np.float32)
    R = (Rz @ Ry @ Rx).astype(np.float32)
    return SimpleNamespace(x=x, y=y, yaw=yaw, valid=True, place={"z": z, "R": R})


def _view(cam: list[float], st: SimpleNamespace, w: int, h: int) -> None:
    from OpenGL import GL as gl
    from OpenGL import GLU as glu

    az, el, dist = cam
    tgt = np.array([st.x, st.y, st.place["z"]])
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    eye = tgt + dist * d
    gl.glViewport(0, 0, w, h)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(50.0, w / max(h, 1), 0.1, 100.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    glu.gluLookAt(*eye, *tgt, 0, 0, 1)


def _draw_robot(robot: tuple, st: SimpleNamespace) -> None:
    from OpenGL import GL as gl

    V, N, C, _ = robot
    R4 = np.eye(4, dtype=np.float32)
    R4[:3, :3] = st.place["R"]
    gl.glPushMatrix()
    gl.glTranslatef(st.x, st.y, st.place["z"])
    gl.glMultMatrixf(np.ascontiguousarray(R4.T))
    _draw(V, N, C)
    gl.glPopMatrix()


def _draw_trail(trail: list, color_rgb: tuple[float, float, float]) -> None:
    from OpenGL import GL as gl

    if len(trail) < 2:
        return
    gl.glDisable(gl.GL_LIGHTING)
    gl.glColor3f(*color_rgb)
    gl.glLineWidth(3.0)
    gl.glBegin(gl.GL_LINE_STRIP)
    for p in trail:
        gl.glVertex3f(*p)
    gl.glEnd()
    gl.glEnable(gl.GL_LIGHTING)


def _draw_phase_markers(phase_x: list[float]) -> None:
    from OpenGL import GL as gl

    gl.glDisable(gl.GL_LIGHTING)
    gl.glColor3f(0.55, 0.55, 0.55)
    gl.glLineWidth(1.2)
    for px in phase_x:
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(px, YLIM[0], 0.0)
        gl.glVertex3f(px, YLIM[1], 0.0)
        gl.glEnd()
    gl.glEnable(gl.GL_LIGHTING)


def _cbs(cam: list[float], ms: dict) -> tuple:
    import glfw

    def on_button(w_, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            ms["down"] = action == glfw.PRESS
            ms["x"], ms["y"] = glfw.get_cursor_pos(w_)

    def on_cursor(w_, x, y):
        if ms["down"]:
            cam[0] -= (x - ms["x"]) * 0.01
            cam[1] = float(np.clip(cam[1] + (y - ms["y"]) * 0.01, 0.05, 1.5))
            ms["x"], ms["y"] = x, y

    def on_scroll(w_, dx, dy):
        cam[2] = float(np.clip(cam[2] - dy * 0.8, 2.0, 60.0))

    return on_button, on_cursor, on_scroll


def _current_phase(t: int) -> str:
    step = 0
    for i, (n, _) in enumerate(PHASES):
        if t < step + n:
            return PHASE_NAMES[i]
        step += n
    return PHASE_NAMES[-1]


def _make_robot_colored(base_robot: tuple, color_rgb: tuple[float, float, float]) -> tuple:
    """Return a robot mesh tuple with a custom body colour."""
    V, N, C, red = base_robot
    custom_C = np.tile(np.array(color_rgb, np.float32), (len(V), 1))
    return V, N, custom_C, red


def run(device: str = "cuda:0", speed: float = 1.0) -> None:
    import glfw
    from OpenGL import GL as gl

    wp.init()

    # ── pre-compute ──────────────────────────────────────────────────────────
    setpoints = _build_setpoints()
    T = len(setpoints)
    scene = hm_mod.flat(xlim=XLIM, ylim=YLIM)
    mu = friction_mod.uniform(0.8, xlim=XLIM, ylim=YLIM)
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)

    print("Pre-computing Warp trajectories …")
    trajectories: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for tau in TAUS:
        trajectories[tau] = _run_warp(tau, setpoints, scene, mu, grid, device)
        print(f"  tau={tau} done")

    # phase-boundary x positions on tau=0 trajectory
    ctrl0 = trajectories[0.0][0]
    phase_step_ends = np.cumsum([0] + [n for n, _ in PHASES])
    phase_x = [float(ctrl0[idx, 0]) for idx in phase_step_ends[1:-1]]

    # ── meshes ───────────────────────────────────────────────────────────────
    terrain = build_terrain(scene)
    base_robot = build_robot()
    robots = [_make_robot_colored(base_robot, c) for c in COLORS_RGB]

    # ── GLFW ─────────────────────────────────────────────────────────────────
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")

    win = glfw.create_window(PW, PH, LABELS[0], None, None)
    if win is None:
        raise RuntimeError("Failed to create GLFW window")
    glfw.set_window_pos(win, 100, 80)
    glfw.make_context_current(win)
    _init_gl()

    cam = [-2.1, 0.85, 12.0]
    ms = {"down": False, "x": 0.0, "y": 0.0}
    bcb, ccb, scb = _cbs(cam, ms)
    glfw.set_mouse_button_callback(win, bcb)
    glfw.set_cursor_pos_callback(win, ccb)
    glfw.set_scroll_callback(win, scb)

    # ── animation state ──────────────────────────────────────────────────────
    # stage 0: tau=0 animating  |  stage 1: tau=0.5 animating over frozen tau=0 trail
    stage = 0
    t = 0
    trails = [[], []]  # trails[0] = tau=0,  trails[1] = tau=0.5
    frame_dt = DT / max(speed, 1e-3)
    t_next = time.monotonic()

    while not glfw.window_should_close(win):
        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            break
        if glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS:
            break

        now = time.monotonic()
        if now >= t_next:
            t += 1
            t_next = now + frame_dt
            if t > T:
                # transition or loop
                if stage == 0:
                    stage = 1
                    t = 0
                else:
                    stage = 0
                    t = 0
                    trails[0].clear()
                    trails[1].clear()

        tau = TAUS[stage]
        ctrl, der = trajectories[tau]
        t_idx = min(t, T)
        st = _pose_to_st(ctrl[t_idx], der[t_idx])
        trails[stage].append((st.x, st.y, st.place["z"] + 0.03))

        phase_label = _current_phase(t_idx)
        glfw.set_window_title(
            win,
            f"{LABELS[stage]}  —  {phase_label}  [step {t_idx}/{T}]",
        )

        w, h = glfw.get_framebuffer_size(win)
        _view(cam, st, w, h)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        _draw(*terrain[:3], terrain[3])
        _draw_phase_markers(phase_x)

        # always draw the tau=0 trail (frozen once stage 1 starts)
        _draw_trail(trails[0], COLORS_RGB[0])
        # during stage 1 also draw the growing tau=0.5 trail
        if stage == 1:
            _draw_trail(trails[1], COLORS_RGB[1])

        _draw_robot(robots[stage], st)

        glfw.swap_buffers(win)

    glfw.terminate()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0", help="Warp device (default: cuda:0)")
    ap.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback speed multiplier (default 1.0; 0.3 = 30%% of real-time)",
    )
    args = ap.parse_args()
    run(device=args.device, speed=args.speed)


if __name__ == "__main__":
    main()
