"""Interactive 3D view of the SPLIT-architecture navigation: global routing map + local live-scan planner.

The terrain mesh is the GLOBAL routing map (the accumulated belief, coarse cost-to-go solved on it) --
unknown ground is flat/dark, walls rise as observed. The MPPI plans on only the last few LIVE scans
(drift-free), shown as bright GREEN points at their TRUE positions. With --drift, synthetic SLAM drift
smears the global map (and a magenta vector shows the accumulated error): the green live walls then sit
OFFSET from the smeared mesh, yet the robot stays safe because it plans on the green, not the mesh.
Cyan box = fine planning window; orange box = larger coarse routing window; red pole = goal.
Mouse-drag orbits, scroll zooms, ESC/Q quits.

  python -m kinematic_helhest.viz.navigate_partial_live --world pocket
  python -m kinematic_helhest.viz.navigate_partial_live --world pocket --drift 0.03
  python -m kinematic_helhest.viz.navigate_partial_live --world pocket --shot /tmp/nav3d.png
"""
import argparse
from collections import deque

import numpy as np
import warp as wp

from .. import dynamics
from .. import worlds as W
from ..control.mppi_gpu import MppiGpu
from ..control.terminal import dock_control
from ..driver import WarpDriver
from ..engine import GridParams
from ..engine import Simulator
from ..eval import _LATTICE_W
from ..heightmap import Heightmap
from ..navigate_partial import _crop_window
from ..navigate_partial import _drift_scan
from ..perception.lidar import lidar_scan
from ..perception.lidar import MultiScanMap
from ..planning.costtogo import CostToGo
from .render import _init_gl
from .render import _render
from .render import build_robot
from .render import build_terrain
from .render import WIN_H
from .render import WIN_W


def run(world="pocket", K=8, dock_radius=1.5, lat_coarsen=4, win_m=9.0, route_m=16.0, local_scans=5,
        drift=0.0, fov_deg=180.0, max_range=7.0, device="cuda", shot=None, max_frames=2000):
    import glfw
    from OpenGL import GL as gl

    wp.init()
    builder, start, goal = W.WORLDS[world]
    scene = builder(); mu = W.matching_friction(scene); goal = np.asarray(goal, np.float64)
    cell = scene.cell
    ww = wh = int(round(win_m / cell))

    drv = WarpDriver(scene, mu, init_pose=tuple(start), device=device)  # reality
    win_grid = GridParams(ww, wh, cell, 0.0, 0.0)
    plan_sim = Simulator(dynamics.robot_params(), dynamics.planning_solver(), win_grid, 4096, 70, device)
    plan_sim.set_uniform_friction(0.8)
    planner = MppiGpu(plan_sim, 0.5, 4.0, _LATTICE_W, 0.05, 1e-2, 0, sigma_knot=1.0, n_knots=4,
                      n_scenarios=K, n_theta=24)
    planner.reset_nominal(1.5)
    # decoupled coarse routing window (see navigate_partial): cost-to-go on a larger bounded grid
    rww = rwh = int(round(max(route_m, win_m) / cell))
    kr = max(1, int(lat_coarsen))
    rcny, rcnx, rccell = rwh // kr, rww // kr, cell * kr
    route_grid = GridParams(rcnx, rcny, rccell, 0.0, 0.0)
    ctg = CostToGo(route_grid, dynamics.robot_params(), dynamics.planning_solver(), n_theta=24, device=device)
    planner.cw.vcap = ctg._vcap  # arm the saturation fallback (explore toward an out-of-window goal)
    sgrid = GridParams(rcnx, rcny, rccell, (ww // 2 - rww // 2) * cell,
                       (wh // 2 - rwh // 2) * cell).build()  # routing field placed in the planning frame
    mm = MultiScanMap(scene.ny, scene.nx)  # GLOBAL routing map (drift-prone)
    scan_buf = deque(maxlen=local_scans) if local_scans >= 1 else None  # local drift-free buffer
    rng = np.random.default_rng(0)
    drift_x = drift_y = drift_yaw = 0.0  # accumulated SE(2) global-map drift (m, m, rad)

    if not glfw.init():
        raise RuntimeError("glfw init failed")
    if shot:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(WIN_W, WIN_H, f"Helhest - {world} (partial-map 3D)", None, None)
    glfw.make_context_current(win)
    _init_gl()
    cam = [-2.1, 0.85, 16.0]  # az, el, dist (orbit around the robot)
    mouse = {"down": False, "x": 0.0, "y": 0.0}

    def on_button(w_, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse["down"] = action == glfw.PRESS
            mouse["x"], mouse["y"] = glfw.get_cursor_pos(w_)

    def on_cursor(w_, x, y):
        if mouse["down"]:
            cam[0] -= (x - mouse["x"]) * 0.01
            cam[1] = float(np.clip(cam[1] + (y - mouse["y"]) * 0.01, 0.05, 1.5))
            mouse["x"], mouse["y"] = x, y

    def on_scroll(w_, dx, dy):
        cam[2] = float(np.clip(cam[2] - dy * 0.8, 3.0, 60.0))

    glfw.set_mouse_button_callback(win, on_button)
    glfw.set_cursor_pos_callback(win, on_cursor)
    glfw.set_scroll_callback(win, on_scroll)

    robot = build_robot()
    trail = []
    scan_pts = None  # last live-scan wall cells (true positions); persists so a still frame shows them
    _twr, _twc = np.nonzero(scene.H > 0.5)
    true_wall = np.column_stack([scene.x0 + _twc * cell, scene.y0 + _twr * cell])  # true wall centers
    f = 0
    while not glfw.window_should_close(win):
        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS or glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS:
            break
        st = drv.render_state()
        rx, ry, yaw = st.x, st.y, st.yaw
        trail.append([rx, ry, st.place["z"] + 0.05]); trail = trail[-6000:]
        d = float(np.hypot(rx - goal[0], ry - goal[1]))

        if d >= 0.3:  # perceive + plan + drive until reached, then just keep rendering
            obs, known = lidar_scan(scene.H, scene.x0, scene.y0, cell, (rx, ry, yaw),
                                    fov_deg=fov_deg, max_range=max_range)
            # GLOBAL routing map: accumulate, optionally smeared by random-walk SLAM drift
            if drift > 0.0:
                drift_x += float(rng.normal(0.0, drift))
                drift_y += float(rng.normal(0.0, drift))
                drift_yaw += float(rng.normal(0.0, 0.1 * drift))  # coupled rotational drift (rad/step)
                gobs, gkn = _drift_scan(obs, known, scene.x0, scene.y0, cell, rx, ry,
                                        drift_x, drift_y, drift_yaw)
            else:
                gobs, gkn = obs, known
            mm.integrate(gobs, gkn)
            # LOCAL map for the fine MPPI window: last N live scans (drift-free) or the global map
            if scan_buf is not None:
                scan_buf.append((obs, known))
                local = MultiScanMap(scene.ny, scene.nx)
                for o, kk in scan_buf:
                    local.integrate(o, kk)
            else:
                local = mm
            # FINE planning window from the LOCAL map
            elev, kn, wx0, wy0 = _crop_window(local, scene, rx, ry, ww, wh, cell)
            elev = np.where(kn, elev, 0.0).astype(np.float32)
            goal_l = (goal[0] - wx0, goal[1] - wy0)
            state_l = np.array([rx - wx0, ry - wy0, yaw], np.float32)
            plan_sim.set_terrain(wp.array(np.ascontiguousarray(elev), dtype=wp.float32, device=device))
            # COARSE routing window from the GLOBAL map
            relev, rkn, rwx0, rwy0 = _crop_window(mm, scene, rx, ry, rww, rwh, cell)
            relev = np.where(rkn, relev, 0.0).astype(np.float32)
            goal_r = (goal[0] - rwx0, goal[1] - rwy0)
            Hc = relev[:rcny * kr, :rcnx * kr].reshape(rcny, kr, rcnx, kr).max(axis=(1, 3)) if kr > 1 else relev
            V = ctg.compute(wp.array(np.ascontiguousarray(Hc), dtype=wp.float32, device=device), goal_r)
            planner.set_lattice(V, sgrid)
            if dock_radius > 0.0 and d < dock_radius:
                cmd = dock_control(state_l, goal_l)
            else:
                planner.replan(state_l, goal_l, 3)
                u = planner.nominal()
                cmd = np.array([u[0, 0], u[0, 1], 0.5 * (u[0, 0] + u[0, 1])], np.float32)
            drv.step(cmd)
            wc = known & (obs > 0.5)  # the live walls the LOCAL planner sees this frame (true coords)
            ri, ci = np.nonzero(wc)
            if ri.size:
                scan_pts = np.column_stack([scene.x0 + ci * cell, scene.y0 + ri * cell, obs[ri, ci]])

        # rebuild the terrain mesh from the BELIEF (unknown -> flat, dark; walls rise as seen)
        belief = np.where(mm.known, mm.elev, 0.0).astype(np.float32)
        Vt, Nt, Ct, idxt = build_terrain(Heightmap(belief, (scene.x0, scene.y0), cell))
        Ct[(~mm.known).ravel()] = (0.09, 0.09, 0.16)  # unknown -> dark
        _render(st, cam, (Vt, Nt, Ct, idxt), robot, trail)

        gl.glDisable(gl.GL_LIGHTING)
        # the LIVE scan the local planner uses (true wall cells) -> bright yellow; under drift these sit
        # OFFSET from the smeared white global-mesh walls, showing the robot tracks true geometry.
        if scan_pts is not None:
            gl.glColor3f(1.0, 0.95, 0.1); gl.glPointSize(5.0)
            gl.glBegin(gl.GL_POINTS)
            for px, py, pz in scan_pts:
                gl.glVertex3f(float(px), float(py), float(pz) + 0.05)
            gl.glEnd()

        def _box(side, z):  # robot-centered square outline on the ground
            gl.glBegin(gl.GL_LINE_LOOP)
            for dx, dy in [(-side / 2, -side / 2), (side / 2, -side / 2),
                           (side / 2, side / 2), (-side / 2, side / 2)]:
                gl.glVertex3f(rx + dx, ry + dy, z)
            gl.glEnd()
        gl.glColor3f(0.1, 0.9, 0.95); gl.glLineWidth(2.0); _box(win_m, 0.06)            # fine planning
        gl.glColor3f(1.0, 0.6, 0.1); gl.glLineWidth(2.0); _box(max(route_m, win_m), 0.05)  # coarse routing
        if drift > 0.0 and len(true_wall):  # RED ghost of the TRUE walls -> the white belief rotates off it
            gl.glColor3f(0.95, 0.15, 0.15); gl.glPointSize(3.0)
            gl.glBegin(gl.GL_POINTS)
            for px, py in true_wall:
                gl.glVertex3f(float(px), float(py), 0.04)
            gl.glEnd()
        if drift > 0.0:  # accumulated global-map drift vector (magenta)
            gl.glColor3f(1.0, 0.2, 1.0); gl.glLineWidth(3.0)
            gl.glBegin(gl.GL_LINES)
            gl.glVertex3f(rx, ry, 0.1); gl.glVertex3f(rx + drift_x, ry + drift_y, 0.1)
            gl.glEnd()
        gl.glColor3f(0.95, 0.1, 0.1); gl.glLineWidth(5.0)  # goal pole
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(goal[0], goal[1], 0.0); gl.glVertex3f(goal[0], goal[1], 1.6)
        gl.glEnd()
        gl.glEnable(gl.GL_LIGHTING)
        glfw.swap_buffers(win)

        f += 1
        if f >= max_frames or (shot and d < 0.3):  # shot: stop once reached, capture the explored map
            if shot:
                gl.glReadBuffer(gl.GL_BACK)
                buf = gl.glReadPixels(0, 0, WIN_W, WIN_H, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                img = np.frombuffer(buf, np.uint8).reshape(WIN_H, WIN_W, 3)[::-1]
                import matplotlib.pyplot as plt
                plt.imsave(shot, img)
                print(f"saved {shot}")
            break
    glfw.terminate()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="pocket", choices=list(W.WORLDS))
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--dock-radius", type=float, default=1.5)
    ap.add_argument("--lat-coarsen", type=int, default=4)
    ap.add_argument("--win-m", type=float, default=9.0, help="fine planning window side (m)")
    ap.add_argument("--route-m", type=float, default=16.0, help="coarse routing window side (m)")
    ap.add_argument("--local-scans", type=int, default=5,
                    help="local MPPI map = last N live scans (1 = single scan; 0 = the global map)")
    ap.add_argument("--drift", type=float, default=0.0,
                    help="synthetic per-step SLAM drift std (m) on the GLOBAL map only")
    ap.add_argument("--fov-deg", type=float, default=180.0)
    ap.add_argument("--max-range", type=float, default=7.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shot", default=None)
    args = ap.parse_args()
    run(world=args.world, K=args.K, dock_radius=args.dock_radius, lat_coarsen=args.lat_coarsen,
        win_m=args.win_m, route_m=args.route_m, local_scans=args.local_scans, drift=args.drift,
        fov_deg=args.fov_deg, max_range=args.max_range, device=args.device, shot=args.shot)


if __name__ == "__main__":
    main()
