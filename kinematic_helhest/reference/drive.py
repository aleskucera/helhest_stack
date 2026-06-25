"""Real-time interactive driver for the NUMPY reference physics (glfw + OpenGL).

Drive with I/J/K/L over a demo heightmap to *feel* the kinematic behaviour —
skid-steer turning, climbing/tilting on terrain, and high-center rejection (the
robot turns red when its belly would penetrate). Every frame calls the numpy
`reference.state.step`; this is the oracle's viewer. The Warp-engine equivalent
(the runtime path) is `viz/drive.py`. Rendering/input come from `viz.render`.

Keys:  I forward   K back   J turn-left   L turn-right   ESC/Q quit
Mouse: drag to orbit, scroll to zoom.

Run:        python -m kinematic_helhest.reference.drive
Shot test:  python -m kinematic_helhest.reference.drive --shot /tmp/drive.png
"""
import argparse
import time

import numpy as np

from .. import friction
from .. import heightmap as hmmod
from ..model import WHEEL_RADIUS
from ..viz.render import DT
from ..viz.render import WIN_H
from ..viz.render import WIN_W
from ..viz.render import _commands
from ..viz.render import _init_gl
from ..viz.render import _render
from ..viz.render import build_robot
from ..viz.render import build_terrain
from . import state as stmod


def run(shot=None):
    import glfw
    from OpenGL import GL as gl

    hm = hmmod.demo_terrain()
    surf = hmmod.wheel_envelope(hm, WHEEL_RADIUS)
    mu = friction.uniform(0.8, xlim=(-3.0, 10.0), ylim=(-4.0, 4.0))
    st = stmod.make_state(0.0, 0.0, 0.0, surf, hm)

    if not glfw.init():
        raise RuntimeError("glfw init failed")
    if shot:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(WIN_W, WIN_H, "Helhest kinematic (numpy) — I/J/K/L drive", None, None)
    glfw.make_context_current(win)
    _init_gl()

    terrain = build_terrain(hm)
    robot = build_robot()
    cam = [-2.2, 0.5, 6.0]  # azimuth, elevation, distance

    mouse = {"down": False, "x": 0.0, "y": 0.0}

    def on_mouse_button(w, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse["down"] = action == glfw.PRESS
            mouse["x"], mouse["y"] = glfw.get_cursor_pos(w)

    def on_cursor(w, x, y):
        if mouse["down"]:
            cam[0] -= (x - mouse["x"]) * 0.01
            cam[1] = np.clip(cam[1] + (y - mouse["y"]) * 0.01, -1.4, 1.4)
            mouse["x"], mouse["y"] = x, y

    def on_scroll(w, dx, dy):
        cam[2] = float(np.clip(cam[2] - dy * 0.5, 1.5, 30.0))

    glfw.set_mouse_button_callback(win, on_mouse_button)
    glfw.set_cursor_pos_callback(win, on_cursor)
    glfw.set_scroll_callback(win, on_scroll)

    trail = []
    last_status = 0.0
    frame = 0
    while not glfw.window_should_close(win):
        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS or \
           glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS:
            break
        cmd = np.array([3.0, 3.0, 3.0]) if shot else _commands(
            lambda k: glfw.get_key(win, k))
        st = stmod.step(st, cmd, surf, hm, DT, mu_field=mu)
        trail.append([st.x, st.y, st.place["z"] + 0.02])
        trail = trail[-3000:]

        _render(st, cam, terrain, robot, trail)

        if shot:
            frame += 1
            if frame >= 45:
                gl.glReadBuffer(gl.GL_BACK)
                buf = gl.glReadPixels(0, 0, WIN_W, WIN_H, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                img = np.frombuffer(buf, np.uint8).reshape(WIN_H, WIN_W, 3)[::-1]
                import matplotlib.pyplot as plt
                plt.imsave(shot, img)
                print(f"saved {shot}  (pose=({st.x:.2f},{st.y:.2f}) z={st.place['z']:.2f} "
                      f"pitch={np.rad2deg(st.place['pitch']):+.1f} valid={st.valid})")
                break
            continue

        glfw.swap_buffers(win)
        now = time.perf_counter()
        if now - last_status > 0.4:
            print(f"\rpos=({st.x:+5.2f},{st.y:+5.2f}) yaw={np.rad2deg(st.yaw):+6.1f}  "
                  f"z={st.place['z']:.2f} pitch={np.rad2deg(st.place['pitch']):+5.1f} "
                  f"roll={np.rad2deg(st.place['roll']):+5.1f}  a={st.alpha:.2f} "
                  f"valid={st.valid}   ", end="", flush=True)
            last_status = now
        time.sleep(max(0.0, DT - (time.perf_counter() - now)))

    glfw.terminate()
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", default=None, help="render ~45 auto-drive frames, save PNG, exit")
    args = ap.parse_args()
    run(shot=args.shot)


if __name__ == "__main__":
    main()
