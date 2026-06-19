"""3D view of the robust / CVaR evaluation: the wheel-slip fans draped on the terrain.

Draws the nominal-best arc and the CVaR-best arc near demo_terrain's wall, each as its K
slip-scenario rollouts lifted onto the 3D surface. Per-scenario colour: RED if that scenario
high-centers (infeasible settle), GREEN if it stays clear. The nominal-best fan (hugs the
wall) lights up red; the CVaR-best fan (wider) stays green -- the disturbance fan diving into
the wall vs sweeping over it, in 3D. Orbit with the mouse to see it against the wall.

Run:        python -m kinematic_helhest.viz.robust3d [--device cuda]
Shot test:  python -m kinematic_helhest.viz.robust3d --shot /tmp/robust3d.png
"""
import argparse

import numpy as np

from .. import heightmap as hmmod
from ..planning.robust import _arc_candidates
from ..planning.robust import evaluate
from .render import WIN_H
from .render import WIN_W
from .render import _draw
from .render import _init_gl
from .render import build_terrain

_RED, _GREEN, _WHITE = (0.88, 0.12, 0.12), (0.15, 0.75, 0.25), (0.98, 0.98, 0.98)


def _polyline(scene, xy, color, width, dz):
    from OpenGL import GL as gl
    z = np.minimum(scene.sample(xy[:, 0], xy[:, 1]), 0.7) + dz  # drape on the surface, clamp at the wall
    gl.glColor3f(*color); gl.glLineWidth(width)
    gl.glBegin(gl.GL_LINE_STRIP)
    for (x, y), zz in zip(xy, z):
        gl.glVertex3f(float(x), float(y), float(zz))
    gl.glEnd()


def _draw_scene(scene, terrain, cam, target, paths, badstep, i_nom, i_cvar, goal):
    from OpenGL import GL as gl
    from OpenGL import GLU as glu

    az, el, dist = cam
    tgt = np.asarray(target, np.float64)
    eye = tgt + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    gl.glViewport(0, 0, WIN_W, WIN_H)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glMatrixMode(gl.GL_PROJECTION); gl.glLoadIdentity()
    glu.gluPerspective(50.0, WIN_W / WIN_H, 0.1, 100.0)
    gl.glMatrixMode(gl.GL_MODELVIEW); gl.glLoadIdentity()
    glu.gluLookAt(*eye, *tgt, 0, 0, 1)

    _draw(*terrain[:3], terrain[3])  # ground mesh

    gl.glDisable(gl.GL_LIGHTING)
    for idx in (i_nom, i_cvar):  # both arcs: each scenario red if it high-centers, else green
        fan, bad = paths[:, idx], badstep[:, idx]
        for k in range(fan.shape[1]):
            _polyline(scene, fan[:, k], _RED if bad[:, k].any() else _GREEN, 1.5, 0.04)
        _polyline(scene, fan[:, 0], _WHITE, 4.0, 0.07)  # the nominal centerline, thick
    gz = float(scene.sample(np.array([goal[0]]), np.array([goal[1]]))[0])
    gl.glColor3f(0.95, 0.1, 0.1); gl.glLineWidth(5.0)
    gl.glBegin(gl.GL_LINES)
    gl.glVertex3f(goal[0], goal[1], gz); gl.glVertex3f(goal[0], goal[1], gz + 1.2)
    gl.glEnd()
    gl.glEnable(gl.GL_LIGHTING)


def run(shot=None, device="cuda", beta=0.5, slip_lo=0.6, B=48, K=16):
    import glfw

    scene = hmmod.demo_terrain()
    mu = hmmod.Heightmap(np.full((scene.ny, scene.nx), 0.8, np.float32), (scene.x0, scene.y0), scene.cell)
    start, goal = np.array([0.0, 0.0, 0.0], np.float32), np.array([4.0, 1.6])
    turns = np.linspace(0.0, 1.8, B)
    cand = _arc_candidates(turns, T=70)
    rng = np.random.default_rng(0)
    slips = np.ones((K, 2), np.float32)
    slips[1:] = rng.uniform(slip_lo, 1.0, (K - 1, 2))
    nominal, cvar, paths, _, badstep = evaluate(scene, mu, start, goal, cand, slips, beta=beta, device=device)
    i_nom, i_cvar = int(np.argmin(nominal)), int(np.argmin(cvar))
    print(f"nominal-best arc#{i_nom} (red fan), CVaR-best arc#{i_cvar} (green fan)")

    if not glfw.init():
        raise RuntimeError("glfw init failed")
    if shot:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(WIN_W, WIN_H, "Helhest — robust CVaR slip fans (orbit: drag, zoom: scroll)", None, None)
    glfw.make_context_current(win)
    _init_gl()
    from OpenGL import GL as gl
    terrain = build_terrain(scene)
    target = (2.4, 0.6, 0.25)            # look at the wall region
    cam = [-2.1, 0.55, 7.5]              # az, el, dist
    mouse = {"down": False, "x": 0.0, "y": 0.0}

    def on_button(w, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse["down"] = action == glfw.PRESS
            mouse["x"], mouse["y"] = glfw.get_cursor_pos(w)

    def on_cursor(w, x, y):
        if mouse["down"]:
            cam[0] -= (x - mouse["x"]) * 0.01
            cam[1] = float(np.clip(cam[1] + (y - mouse["y"]) * 0.01, 0.05, 1.5))
            mouse["x"], mouse["y"] = x, y

    def on_scroll(w, dx, dy):
        cam[2] = float(np.clip(cam[2] - dy * 0.5, 2.0, 30.0))

    glfw.set_mouse_button_callback(win, on_button)
    glfw.set_cursor_pos_callback(win, on_cursor)
    glfw.set_scroll_callback(win, on_scroll)

    frame = 0
    while not glfw.window_should_close(win):
        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS or glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS:
            break
        _draw_scene(scene, terrain, cam, target, paths, badstep, i_nom, i_cvar, goal)
        if shot:
            frame += 1
            if frame >= 3:
                gl.glReadBuffer(gl.GL_BACK)
                buf = gl.glReadPixels(0, 0, WIN_W, WIN_H, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                img = np.frombuffer(buf, np.uint8).reshape(WIN_H, WIN_W, 3)[::-1]
                import matplotlib.pyplot as plt
                plt.imsave(shot, img)
                print(f"saved {shot}")
                break
            continue
        glfw.swap_buffers(win)
    glfw.terminate()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shot", default=None, help="render offscreen, save a PNG, exit")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--slip-lo", type=float, default=0.6)
    args = ap.parse_args()
    run(shot=args.shot, device=args.device, beta=args.beta, slip_lo=args.slip_lo)


if __name__ == "__main__":
    main()
