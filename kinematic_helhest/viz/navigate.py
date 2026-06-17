"""Integrated 3D viewer: robot-centered rolling-map MPPI driving to a goal.

The autonomous counterpart to follow.py. Each frame the rolling-map `Navigator`
(planning/navigate.py) ingests a robot-centered local window (synthetic perception
crop of the world), replans an MPPI horizon to the goal in the robot frame, and the
robot executes the first control. The magenta line is the current plan (transformed
local->world); the robot goes red on high-center / wall wedge.

For this synthetic demo the full world is known, so the terrain is rendered once
(stationary) and the robot drives across it. On the real robot (Phase 4) there is no
global map: the viewer will instead render each cycle's local window placed at the
robot's odom pose -- same `render.py`, terrain mesh from the live elevation.

Keys:  ESC/Q quit ; mouse orbit, scroll zoom.
Run:        python -m kinematic_helhest.viz.navigate [--device cuda] [--gx 4 --gy 1.5]
Shot test:  python -m kinematic_helhest.viz.navigate --shot /tmp/navigate.png
"""
import argparse
import time

import numpy as np

from .. import friction
from .. import heightmap as hmmod
from ..planning.navigate import NavConfig
from ..planning.navigate import Navigator
from ..planning.synthetic_perception import crop_window
from ..planning.synthetic_perception import to_local
from .drive import WarpDriver
from .render import DT
from .render import WIN_H
from .render import WIN_W
from .render import _init_gl
from .render import _render
from .render import build_robot
from .render import build_terrain


def _plan_to_world(plan_local, st):
    """Rotate+translate a local plan path [K,2] into world coords by pose st."""
    c, s = np.cos(st.yaw), np.sin(st.yaw)
    wx = st.x + c * plan_local[:, 0] - s * plan_local[:, 1]
    wy = st.y + s * plan_local[:, 0] + c * plan_local[:, 1]
    return np.stack([wx, wy], axis=1)


def _draw_plan(plan_world, world_hm, goal):
    """Magenta plan line + red goal pole, clamped to terrain height (from follow.py)."""
    from OpenGL import GL as gl
    gl.glDisable(gl.GL_LIGHTING)
    z = np.minimum(world_hm.sample(plan_world[:, 0], plan_world[:, 1]), 0.55) + 0.06
    gl.glColor3f(1.0, 0.0, 1.0); gl.glLineWidth(5.0)
    gl.glBegin(gl.GL_LINE_STRIP)
    for (x, y), zz in zip(plan_world, z):
        gl.glVertex3f(float(x), float(y), float(zz))
    gl.glEnd()
    gz = float(world_hm.sample(np.array([goal[0]]), np.array([goal[1]]))[0])
    gl.glColor3f(0.95, 0.1, 0.1); gl.glLineWidth(5.0)
    gl.glBegin(gl.GL_LINES)
    gl.glVertex3f(float(goal[0]), float(goal[1]), gz)
    gl.glVertex3f(float(goal[0]), float(goal[1]), gz + 1.2)
    gl.glEnd()
    gl.glEnable(gl.GL_LIGHTING)


def run(shot=None, device="cpu", goal=(4.0, 1.5), cfg=None, replan_every=4):
    import glfw
    from OpenGL import GL as gl

    cfg = cfg or NavConfig()
    world = hmmod.demo_terrain()
    mu = friction.uniform(0.8, xlim=(-3.0, 10.0), ylim=(-4.0, 4.0), cell=0.06)
    drv = WarpDriver(world, mu, init_pose=(0.0, 0.0, 0.0), device=device,
                     resid_tol=cfg.resid_tol, clear_margin=cfg.clear_margin)
    nav = Navigator(cfg, device=device)
    goal = np.asarray(goal, np.float64)

    if not glfw.init():
        raise RuntimeError("glfw init failed")
    if shot:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(WIN_W, WIN_H, "Helhest — rolling MPPI navigate", None, None)
    glfw.make_context_current(win)
    _init_gl()
    terrain, robot = build_terrain(world), build_robot()
    cam = [-2.2, 0.5, 6.0]
    mouse = {"down": False, "x": 0.0, "y": 0.0}

    def on_mouse_button(w, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse["down"] = action == glfw.PRESS
            mouse["x"], mouse["y"] = glfw.get_cursor_pos(w)

    def on_cursor(w, x, y):
        if mouse["down"]:
            cam[0] -= (x - mouse["x"]) * 0.01
            cam[1] = float(np.clip(cam[1] + (y - mouse["y"]) * 0.01, -1.4, 1.4))
            mouse["x"], mouse["y"] = x, y

    def on_scroll(w, dx, dy):
        cam[2] = float(np.clip(cam[2] - dy * 0.5, 1.5, 30.0))

    glfw.set_mouse_button_callback(win, on_mouse_button)
    glfw.set_cursor_pos_callback(win, on_cursor)
    glfw.set_scroll_callback(win, on_scroll)

    U = np.full((cfg.T, 2), 1.5, np.float32)
    plan_world = np.zeros((1, 2))
    trail, last_status, frame = [], 0.0, 0
    while not glfw.window_should_close(win):
        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS or \
           glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS:
            break
        st = drv.render_state()
        # Receding horizon: the MPPI replan is the expensive step (~100-200 ms), so
        # run it only every `replan_every` frames and execute the optimized control
        # sequence open-loop in between. Keeps the viewer interactive.
        if frame % replan_every == 0:
            local_map = crop_window(world, (st.x, st.y), st.yaw, cfg.half_extent, cfg.res, device)
            U, plan_local = nav.replan(local_map, to_local(goal, (st.x, st.y, st.yaw)), U)
            plan_world = _plan_to_world(plan_local, st)
        wL, wR = float(U[0, 0]), float(U[0, 1])
        drv.step(np.array([wL, wR, 0.5 * (wL + wR)], np.float32))  # WarpDriver wants (wL,wR,rear)
        U = np.roll(U, -1, axis=0); U[-1] = U[-2]

        trail.append([st.x, st.y, st.place["z"] + 0.02]); trail = trail[-3000:]
        _render(st, cam, terrain, robot, trail)
        _draw_plan(plan_world, world, goal)
        frame += 1

        if shot:
            if frame >= 12:
                gl.glReadBuffer(gl.GL_BACK)
                buf = gl.glReadPixels(0, 0, WIN_W, WIN_H, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                img = np.frombuffer(buf, np.uint8).reshape(WIN_H, WIN_W, 3)[::-1]
                import matplotlib.pyplot as plt
                plt.imsave(shot, img)
                print(f"saved {shot}  robot=({st.x:.2f},{st.y:.2f}) valid={st.valid}")
                break
            continue

        glfw.swap_buffers(win)
        now = time.perf_counter()
        if now - last_status > 0.4:
            dist = float(np.hypot(st.x - goal[0], st.y - goal[1]))
            print(f"\rpos=({st.x:+5.2f},{st.y:+5.2f}) goal_dist={dist:4.2f} "
                  f"valid={st.valid}   ", end="", flush=True)
            last_status = now
        time.sleep(max(0.0, DT - (time.perf_counter() - now)))

    glfw.terminate()
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", default=None, help="render a few auto frames, save PNG, exit")
    ap.add_argument("--device", default="cpu", help="warp device: cpu or cuda")
    ap.add_argument("--gx", type=float, default=4.0)
    ap.add_argument("--gy", type=float, default=1.5)
    ap.add_argument("--replan-every", type=int, default=4,
                    help="run the MPPI replan every N frames (higher = faster, less reactive)")
    args = ap.parse_args()
    run(shot=args.shot, device=args.device, goal=(args.gx, args.gy),
        replan_every=args.replan_every)


if __name__ == "__main__":
    main()
