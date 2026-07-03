"""Full-pipeline sim, Stage 1: simulate a noisy 3D lidar + noisy wheel-odometry and
run ICP localization in the loop, so the estimated pose is what the pipeline would
actually use on the robot (not ground truth).

Reality: a scripted SE(2) trajectory through a field of box pillars (the pillars give
ICP the vertical structure it needs to fix x/y/yaw — a bare ground plane is
unobservable in-plane). Each step:
  - cast an Ouster OSDome scan from the TRUE pose against the boxes (datasheet range
    noise + dropout), express it in the base frame (what the sensor "measured");
  - integrate a NOISY wheel-odometry delta (proportional noise + a small yaw bias =
    skid-steer under-rotation) -> a drifting odom pose;
  - Localizer: predict from the odom delta, ICP-register the scan against the rolling
    DeviceMapAccumulator submap, drift-gate -> corrected (estimated) pose;
  - fold the scan into the map at the corrected pose.

Output: true vs. odom-only-dead-reckoning vs. ICP-estimated trajectory, and the
translation-error curve, so you can see ICP holding the estimate to the truth while
raw odom walks off.

  python demos/pipeline_sim.py --shot /tmp/pipeline_sim.png
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp
from helhest.localization import Localizer
from helhest.localization import LocalizerConfig
from helhest.localization.pose_math import invert_pose
from helhest.perception import DeviceMapAccumulator
from helhest.perception import IcpAligner
from helhest.perception import IcpConfig
from helhest.perception.cloud_ops import transform_points
from helhest.perception.sim import GroundSpec
from helhest.perception.sim import make_osdome_lidar
from helhest.perception.sim import osdome_sensor_config

SENSOR_Z = 0.6  # lidar height above the base frame (m)
GROUND = 60.0  # ground-plane half-extent (past max range)
MAP_VOXEL = 0.15  # accumulated-map voxel size (m)
MAP_RADIUS = 25.0  # rolling-map keep radius (m)


def se2_to_mat(x: float, y: float, yaw: float) -> np.ndarray:
    """(x, y, yaw) -> 4x4 SE(3) with a planar (z=0) base pose."""
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4, dtype=np.float64)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    T[0, 3], T[1, 3] = x, y
    return T


def mat_to_se2(T: np.ndarray) -> tuple[float, float, float]:
    return float(T[0, 3]), float(T[1, 3]), float(np.arctan2(T[1, 0], T[0, 0]))


def scripted_trajectory(n: int, dt: float, v: float) -> np.ndarray:
    """A forward S-curve: constant speed, sinusoidal yaw-rate. Returns [n,3] (x,y,yaw)."""
    poses = np.zeros((n, 3))
    x = y = yaw = 0.0
    for k in range(n):
        poses[k] = (x, y, yaw)
        # gentle weave (~15 deg peak heading) so the forward-facing lidar keeps the
        # pillar field in view -- a hard turn into open ground is ICP-unobservable
        wz = 0.15 * np.sin(2.0 * np.pi * k / max(n - 1, 1))  # rad/s
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
        yaw += wz * dt
    return poses


def pillar_world() -> tuple[np.ndarray, np.ndarray]:
    """Box pillars scattered along the corridor (AABB lo/hi corners), 2 m tall."""
    rng = np.random.default_rng(1)
    centers = []
    # dense on both sides, extending WELL past the path end (~x=11) so the
    # forward-facing sensor always has vertical structure ahead to lock onto
    for xc in np.arange(1.0, 19.0, 1.5):
        for side in (-1.0, 1.0):
            yc = side * (1.5 + 0.4 * rng.random()) + 0.3 * rng.standard_normal()
            centers.append((xc + 0.4 * rng.standard_normal(), yc))
    c = np.asarray(centers)
    half = 0.3
    lo = np.column_stack([c[:, 0] - half, c[:, 1] - half, np.zeros(len(c))]).astype(np.float32)
    hi = np.column_stack([c[:, 0] + half, c[:, 1] + half, np.full(len(c), 2.0)]).astype(np.float32)
    return lo, hi


def noisy_odom_delta(
    D_true: np.ndarray, rng: np.random.Generator, trans_noise: float, yaw_bias: float
) -> np.ndarray:
    """Perturb a true base-frame delta: proportional Gaussian noise + a systematic
    yaw under-rotation bias (skid-steer slips when turning)."""
    dx, dy, dyaw = mat_to_se2(D_true)
    step = float(np.hypot(dx, dy))
    dx += rng.normal(0.0, trans_noise * (step + 1e-3))
    dy += rng.normal(0.0, trans_noise * (step + 1e-3))
    dyaw = dyaw * (1.0 - yaw_bias) + rng.normal(0.0, trans_noise * (abs(dyaw) + 1e-3))
    return se2_to_mat(dx, dy, dyaw)


def run(
    device: str = "cuda",
    steps: int = 120,
    dt: float = 0.1,
    speed: float = 1.0,
    columns: int = 512,
    dropout: float = 0.03,
    trans_noise: float = 0.05,
    yaw_bias: float = 0.04,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    sensor = osdome_sensor_config(columns=columns)
    ground = GroundSpec(z=0.0, x_range=(-GROUND, GROUND), y_range=(-GROUND, GROUND))
    lidar = make_osdome_lidar(ground, sensor=sensor, facing="front", dropout=dropout, device=device)
    acc = DeviceMapAccumulator(MAP_VOXEL, MAP_RADIUS, device=device)
    aligner = IcpAligner(IcpConfig(max_iters=30, max_correspondence_dist_m=0.5), device=device)
    localizer = Localizer(aligner, LocalizerConfig())

    box_lo, box_hi = pillar_world()
    true = scripted_trajectory(steps, dt, speed)

    map_wp: wp.array | None = None
    T_odom = se2_to_mat(*true[0])  # odom starts aligned with reality
    est = np.zeros((steps, 3))
    odom = np.zeros((steps, 3))

    for k in range(steps):
        T_true = se2_to_mat(*true[k])
        if k > 0:
            D_true = invert_pose(se2_to_mat(*true[k - 1])) @ T_true
            T_odom = T_odom @ noisy_odom_delta(D_true, rng, trans_noise, yaw_bias)
        odom[k] = mat_to_se2(T_odom)

        # cast a scan from the TRUE sensor pose -> world hit points (+ noise/dropout)
        origin = np.array([true[k, 0], true[k, 1], SENSOR_Z], np.float32)
        pts_wp, valid_wp, _ = lidar.scan(
            origin, float(true[k, 2]), box_lo, box_hi, seed=k + 1, return_device=True
        )
        world_pts = pts_wp.numpy()[valid_wp.numpy().astype(bool)]  # sensor boundary: host once
        # express the measurement in the base frame (what the sensor reports)
        base_pts = (invert_pose(T_true) @ np.c_[world_pts, np.ones(len(world_pts))].T).T[:, :3]
        scan_base = wp.array(np.ascontiguousarray(base_pts, np.float32), dtype=wp.vec3, device=device)

        if not localizer.initialized:
            localizer.bootstrap(T_odom, T_true)  # world == odom at start; seed the map at truth
            T_world_base = T_true
        else:
            pred, _ = localizer.predict(T_odom)
            outcome = localizer.update(scan_base, pred, map_wp, T_odom)
            T_world_base = outcome.pose
        est[k] = mat_to_se2(T_world_base)

        world_corrected = transform_points(scan_base, len(scan_base), T_world_base)
        valid = wp.full(len(scan_base), 1, dtype=wp.int32, device=device)
        map_wp = acc.step(map_wp, None, world_corrected, valid, (T_world_base[0, 3], T_world_base[1, 3]))

    est_err = np.hypot(est[:, 0] - true[:, 0], est[:, 1] - true[:, 1])
    odom_err = np.hypot(odom[:, 0] - true[:, 0], odom[:, 1] - true[:, 1])
    return dict(
        true=true, odom=odom, est=est, box_lo=box_lo, box_hi=box_hi,
        est_err=est_err, odom_err=odom_err,
    )


def box_world(cell: float = 0.06):
    """Pillars lining a straight lane -> both ICP features AND planner obstacles.

    Returns (scene Heightmap for WarpDriver reality, box_lo, box_hi for the OSDome,
    start (x,y,yaw), goal (x,y)). The SAME pillars drive reality (rasterized to a
    heightmap the robot settles on) and the lidar (3D AABBs it ray-casts against).
    """
    from helhest.heightmap import Heightmap

    xlim, ylim = (-2.0, 17.0), (-4.0, 4.0)
    nx = int(round((xlim[1] - xlim[0]) / cell))
    ny = int(round((ylim[1] - ylim[0]) / cell))
    xs = xlim[0] + (np.arange(nx) + 0.5) * cell
    ys = ylim[0] + (np.arange(ny) + 0.5) * cell
    XX, YY = np.meshgrid(xs, ys)  # [ny, nx]
    H = np.zeros((ny, nx), np.float64)

    half, top = 0.3, 2.0
    los, his = [], []
    for xc in np.arange(1.0, 15.5, 1.5):
        for yc in (-1.8, 1.8):
            H[(np.abs(XX - xc) <= half) & (np.abs(YY - yc) <= half)] = top  # obstacle cell
            los.append((xc - half, yc - half, 0.0))
            his.append((xc + half, yc + half, top))
    scene = Heightmap(H, (xlim[0], ylim[0]), cell)
    box_lo = np.asarray(los, np.float32)
    box_hi = np.asarray(his, np.float32)
    return scene, box_lo, box_hi, (0.0, 0.0, 0.0), np.array([14.0, 0.0])


def run_closed_loop(
    device: str = "cuda",
    max_frames: int = 400,
    columns: int = 512,
    dropout: float = 0.03,
    trans_noise: float = 0.08,
    yaw_bias: float = 0.12,
    win_m: float = 8.0,
    lat_coarsen: int = 4,
    K: int = 8,
    n_theta: int = 24,
    B: int = 4096,
    T: int = 70,
    dock_radius: float = 1.2,
    seed: int = 0,
) -> dict:
    """Stage 2: full pipeline closed on the ESTIMATED pose. WarpDriver is reality; the
    planner never sees the true pose or the true map — only the ICP estimate and the
    lidar-built heightmap. Localization error can drive it into a pillar (contacts)."""
    from helhest import dynamics
    from helhest import worlds as W
    from helhest.control.mppi import CostParams
    from helhest.control.mppi import MppiGpu
    from helhest.control.mppi import RobustConfig
    from helhest.control.terminal import dock_control
    from helhest.driver import WarpDriver
    from helhest.engine import ForwardSimulator
    from helhest.engine import GridParams
    from helhest.perception import HeightMapBuilder
    from helhest.planning.costtogo import CostToGo

    rng = np.random.default_rng(seed)
    scene, box_lo, box_hi, start, goal = box_world()
    cell = scene.cell
    mu = W.matching_friction(scene)
    drv = WarpDriver(scene, mu, init_pose=tuple(start), device=device)  # REALITY (ground truth)

    # perception + localization (as Stage 1)
    sensor = osdome_sensor_config(columns=columns)
    ground = GroundSpec(z=0.0, x_range=(-GROUND, GROUND), y_range=(-GROUND, GROUND))
    lidar = make_osdome_lidar(ground, sensor=sensor, facing="front", dropout=dropout, device=device)
    acc = DeviceMapAccumulator(MAP_VOXEL, MAP_RADIUS, device=device)
    aligner = IcpAligner(IcpConfig(max_iters=30, max_correspondence_dist_m=0.5), device=device)
    localizer = Localizer(aligner, LocalizerConfig())

    # planner: fine MPPI window + coarse cost-to-go router (single robot-centered window)
    ww = wh = int(round(win_m / cell))
    win_grid = GridParams(ww, wh, cell, 0.0, 0.0)
    plan_sim = ForwardSimulator(
        dynamics.robot_params(), dynamics.planning_solver(), win_grid, B, T, device
    )
    plan_sim.set_uniform_friction(0.8)
    planner = MppiGpu(plan_sim, CostParams(), robust=RobustConfig(n_slip_samples=K), n_theta=n_theta)
    planner.reset_nominal(1.5)
    kr = max(1, int(lat_coarsen))
    rcny, rcnx, rccell = wh // kr, ww // kr, cell * kr
    ctg = CostToGo(
        GridParams(rcnx, rcny, rccell, 0.0, 0.0),
        dynamics.robot_params(),
        dynamics.planning_solver(),
        n_theta=n_theta,
        device=device,
    )
    planner.cw.lattice_cap = ctg._vcap
    sgrid = GridParams(rcnx, rcny, rccell, 0.0, 0.0).build()  # route window == plan window

    map_wp: wp.array | None = None
    T_odom = se2_to_mat(*start)
    true_tr, est_tr, err = [], [], []
    contacts, reached, f = 0, False, 0

    for f in range(max_frames):
        st = drv.render_state()
        true_tr.append((st.x, st.y))
        d_true = float(np.hypot(st.x - goal[0], st.y - goal[1]))
        if d_true < 0.3:
            reached = True
            break

        # --- sensors + ICP localization (estimated pose is all the planner gets) ---
        T_true = se2_to_mat(st.x, st.y, st.yaw)
        if f > 0:
            D_true = invert_pose(se2_to_mat(*true_tr_prev)) @ T_true
            T_odom = T_odom @ noisy_odom_delta(D_true, rng, trans_noise, yaw_bias)
        true_tr_prev = (st.x, st.y, st.yaw)

        origin = np.array([st.x, st.y, SENSOR_Z], np.float32)
        pts_wp, valid_wp, _ = lidar.scan(origin, float(st.yaw), box_lo, box_hi, seed=f + 1, return_device=True)
        world_pts = pts_wp.numpy()[valid_wp.numpy().astype(bool)]
        base_pts = (invert_pose(T_true) @ np.c_[world_pts, np.ones(len(world_pts))].T).T[:, :3]
        scan_base = wp.array(np.ascontiguousarray(base_pts, np.float32), dtype=wp.vec3, device=device)

        if not localizer.initialized:
            localizer.bootstrap(T_odom, T_true)
            T_wb = T_true
        else:
            pred, _ = localizer.predict(T_odom)
            T_wb = localizer.update(scan_base, pred, map_wp, T_odom).pose
        ex, ey, eyaw = mat_to_se2(T_wb)
        est_tr.append((ex, ey))
        err.append(float(np.hypot(ex - st.x, ey - st.y)))

        world_corrected = transform_points(scan_base, len(scan_base), T_wb)
        valid = wp.full(len(scan_base), 1, dtype=wp.int32, device=device)
        map_wp = acc.step(map_wp, None, world_corrected, valid, (ex, ey))

        # --- lidar-built heightmap window around the ESTIMATED pose ---
        half = win_m / 2.0
        xmin, ymin = ex - half, ey - half
        builder = HeightMapBuilder(cell, (xmin, ex + half, ymin, ey + half), device=device)
        layers = builder.build(map_wp)
        known = layers.count.numpy() > 0
        elev = np.where(known, layers.max.numpy(), 0.0).astype(np.float32)  # unknown -> flat (optimistic)
        elev = elev[:wh, :ww]  # HeightMapBuilder ceils; clip to the planner grid

        # --- plan on the estimate, in the window-local frame ---
        state_l = np.array([ex - xmin, ey - ymin, eyaw], np.float32)
        goal_l = (goal[0] - xmin, goal[1] - ymin)
        d_est = float(np.hypot(ex - goal[0], ey - goal[1]))
        plan_sim.set_terrain(wp.array(np.ascontiguousarray(elev), dtype=wp.float32, device=device))
        Hc = elev[: rcny * kr, : rcnx * kr].reshape(rcny, kr, rcnx, kr).max(axis=(1, 3)) if kr > 1 else elev
        V = ctg.compute(wp.array(np.ascontiguousarray(Hc), dtype=wp.float32, device=device), goal_l)
        planner.set_lattice(V, sgrid)

        if dock_radius > 0.0 and d_est < dock_radius:
            cmd = dock_control(state_l, goal_l)
        else:
            planner.replan(state_l, goal_l, 3)
            u = planner.nominal()
            cmd = np.array([u[0, 0], u[0, 1], 0.5 * (u[0, 0] + u[0, 1])], np.float32)
        drv.step(cmd)  # reality executes (with motor lag)
        if drv.clear < 0.05:
            contacts += 1

    return dict(
        true=np.asarray(true_tr), est=np.asarray(est_tr), err=np.asarray(err),
        box_lo=box_lo, box_hi=box_hi, goal=goal, reached=reached, frames=f + 1, contacts=contacts,
    )


def _viz_closed(res: dict, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (axp, axe) = plt.subplots(1, 2, figsize=(15, 6))
    for lo, hi in zip(res["box_lo"], res["box_hi"]):
        axp.add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], color="#888"))
    axp.plot(res["true"][:, 0], res["true"][:, 1], "-", color="#2ca02c", lw=2.5, label="reality (WarpDriver)")
    if len(res["est"]):
        axp.plot(res["est"][:, 0], res["est"][:, 1], "-", color="#1f77b4", lw=1.6, label="ICP estimate (what the planner sees)")
    axp.plot(*res["goal"], "*", color="red", ms=18, mec="k", label="goal")
    axp.set_aspect("equal")
    axp.legend(loc="upper left")
    axp.set_title(
        f"Closed loop on the ESTIMATE — reached={res['reached']} "
        f"frames={res['frames']} contacts={res['contacts']}"
    )
    axe.plot(res["err"], "-", color="#1f77b4")
    axe.set_xlabel("step")
    axe.set_ylabel("localization error (m)")
    axe.set_title("ICP localization error along the closed-loop drive")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def _viz(res: dict, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (axp, axe) = plt.subplots(1, 2, figsize=(14, 6))
    for lo, hi in zip(res["box_lo"], res["box_hi"]):
        axp.add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], color="#888"))
    axp.plot(res["true"][:, 0], res["true"][:, 1], "-", color="#2ca02c", lw=2.5, label="reality")
    axp.plot(res["odom"][:, 0], res["odom"][:, 1], "--", color="#d62728", lw=1.8, label="odom only (drifts)")
    axp.plot(res["est"][:, 0], res["est"][:, 1], "-", color="#1f77b4", lw=1.8, label="ICP estimate")
    axp.set_aspect("equal")
    axp.legend(loc="upper left")
    axp.set_title("Trajectory: reality vs. dead-reckoned odom vs. ICP-in-the-loop")

    axe.plot(res["odom_err"], "--", color="#d62728", label="odom only")
    axe.plot(res["est_err"], "-", color="#1f77b4", label="ICP estimate")
    axe.set_xlabel("step")
    axe.set_ylabel("translation error (m)")
    axe.legend()
    axe.set_title("Localization error")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--columns", type=int, default=512, help="OSDome azimuth columns (512 fast, 1024 dense)")
    ap.add_argument("--trans-noise", type=float, default=0.08, help="proportional odom noise std")
    ap.add_argument("--yaw-bias", type=float, default=0.12, help="systematic odom under-rotation (skid-steer)")
    ap.add_argument("--max-frames", type=int, default=400, help="closed-loop step budget (Stage 2)")
    ap.add_argument(
        "--localization-only",
        action="store_true",
        help="Stage 1: scripted path, localization only (no planner / closed loop)",
    )
    ap.add_argument("--shot", default=None)
    args = ap.parse_args()
    wp.init()
    if args.localization_only:
        res = run(
            device=args.device, steps=args.steps, columns=args.columns,
            trans_noise=args.trans_noise, yaw_bias=args.yaw_bias,
        )
        print(
            f"final drift  odom={res['odom_err'][-1]:.2f} m   ICP={res['est_err'][-1]:.2f} m   "
            f"(mean ICP {res['est_err'].mean():.2f} m, mean odom {res['odom_err'].mean():.2f} m)"
        )
        if args.shot:
            _viz(res, args.shot)
    else:
        res = run_closed_loop(
            device=args.device, max_frames=args.max_frames, columns=args.columns,
            trans_noise=args.trans_noise, yaw_bias=args.yaw_bias,
        )
        print(
            f"CLOSED LOOP  reached={res['reached']} frames={res['frames']} "
            f"contacts={res['contacts']} mean-loc-err={res['err'].mean():.2f} m "
            f"max={res['err'].max():.2f} m"
        )
        if args.shot:
            _viz_closed(res, args.shot)


if __name__ == "__main__":
    main()
