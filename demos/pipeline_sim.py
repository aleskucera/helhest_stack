"""Full-pipeline sim on synthetic sensors — noisy 3D lidar + noisy wheel/IMU odometry
+ ICP localization, with the option to close the whole planning/control loop on the
ESTIMATED pose (never ground truth).

Odometry model (non-holonomic dead reckoning):
  - translation from the wheels (forward distance + small scale error/noise);
  - heading from EITHER the wheel differential (a skid-steer slips when turning ->
    systematic under-rotation) OR a gyro (true yaw-rate + a small constant bias +
    white noise). On a skid-steer the gyro heading is far better, so it shrinks the
    drift the localizer has to fight.

Stage 1 (`--localization-only`): scripted arcing path through a pillar field, three-way
compare -- wheel-only odom vs. gyro-aided odom vs. ICP-in-the-loop.
Stage 2 (default): WarpDriver reality; plan + control run on the ICP estimate.

  python demos/pipeline_sim.py --localization-only --shot /tmp/loc.png   # Stage 1 (IMU compare)
  python demos/pipeline_sim.py --shot /tmp/closed.png                    # Stage 2 (closed loop)
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


def odom_step(
    D_true: np.ndarray,
    dt: float,
    rng: np.random.Generator,
    source: str,
    trans_noise: float,
    yaw_bias: float,
    gyro_bias: float,
    gyro_noise: float,
    wheel_scale: float,
) -> np.ndarray:
    """One non-holonomic odometry increment from a true base-frame delta.

    Translation (forward distance) comes from the wheels. Heading is either the wheel
    differential (skid-steer under-rotates: `yaw_bias`) or a gyro (true yaw-rate + a
    constant `gyro_bias` + white noise). Returned as a constant-curvature arc chord,
    with NO lateral component (wheels can't measure sideways slip)."""
    dx, dy, dyaw = mat_to_se2(D_true)
    ds = float(np.hypot(dx, dy)) * np.sign(dx if abs(dx) > 1e-9 else 1.0)
    ds_odom = ds * (1.0 + wheel_scale) + rng.normal(0.0, trans_noise * (abs(ds) + 1e-3))
    if source == "gyro":
        dyaw_odom = dyaw + gyro_bias * dt + rng.normal(0.0, gyro_noise)
    else:  # wheel differential -> skid under-rotation
        dyaw_odom = dyaw * (1.0 - yaw_bias) + rng.normal(0.0, trans_noise * (abs(dyaw) + 1e-3))
    a = 0.5 * dyaw_odom  # arc chord along the average heading
    return se2_to_mat(ds_odom * np.cos(a), ds_odom * np.sin(a), dyaw_odom)


def scripted_trajectory(n: int, dt: float, v: float) -> np.ndarray:
    """A steady LEFT arc (constant yaw-rate): heading turns ~80 deg over the run, so
    the skid-steer wheel-odom under-rotation drifts clearly while the gyro tracks it."""
    poses = np.zeros((n, 3))
    x = y = yaw = 0.0
    for k in range(n):
        poses[k] = (x, y, yaw)
        wz = 0.12  # rad/s, steady left turn (arc radius = v / wz ~ 8 m)
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
        yaw += wz * dt
    return poses


def pillar_field() -> tuple[np.ndarray, np.ndarray]:
    """A 2D grid of box pillars covering the arc region, so the forward lidar always
    has vertical structure to lock onto as the heading sweeps through the turn."""
    rng = np.random.default_rng(1)
    xs = np.arange(-1.0, 12.0, 2.5)
    ys = np.arange(-2.0, 10.0, 2.5)
    los, his, half, top = [], [], 0.3, 2.0
    for xc in xs:
        for yc in ys:
            cx = xc + 0.5 * rng.standard_normal()
            cy = yc + 0.5 * rng.standard_normal()
            los.append((cx - half, cy - half, 0.0))
            his.append((cx + half, cy + half, top))
    return np.asarray(los, np.float32), np.asarray(his, np.float32)


def _scan_base(lidar, x, y, yaw, box_lo, box_hi, seed, device):
    """Cast from the true sensor pose; return the valid returns in the BASE frame."""
    origin = np.array([x, y, SENSOR_Z], np.float32)
    pts_wp, valid_wp, _ = lidar.scan(origin, float(yaw), box_lo, box_hi, seed=seed, return_device=True)
    world_pts = pts_wp.numpy()[valid_wp.numpy().astype(bool)]  # sensor boundary: host once
    base = (invert_pose(se2_to_mat(x, y, yaw)) @ np.c_[world_pts, np.ones(len(world_pts))].T).T[:, :3]
    return wp.array(np.ascontiguousarray(base, np.float32), dtype=wp.vec3, device=device)


def run(
    device: str = "cuda",
    steps: int = 120,
    dt: float = 0.1,
    speed: float = 1.0,
    columns: int = 512,
    dropout: float = 0.03,
    trans_noise: float = 0.05,
    yaw_bias: float = 0.12,
    gyro_bias_dps: float = 0.3,
    gyro_noise: float = 0.001,
    wheel_scale: float = 0.01,
    seed: int = 0,
) -> dict:
    """Stage 1: localization only. Dead-reckons wheel-odom AND gyro-odom from the same
    true path, uses the (better) gyro-aided odom as the ICP prior."""
    rng = np.random.default_rng(seed)
    gyro_bias = np.deg2rad(gyro_bias_dps)
    sensor = osdome_sensor_config(columns=columns)
    ground = GroundSpec(z=0.0, x_range=(-GROUND, GROUND), y_range=(-GROUND, GROUND))
    lidar = make_osdome_lidar(ground, sensor=sensor, facing="front", dropout=dropout, device=device)
    acc = DeviceMapAccumulator(MAP_VOXEL, MAP_RADIUS, device=device)
    aligner = IcpAligner(IcpConfig(max_iters=30, max_correspondence_dist_m=0.5), device=device)
    localizer = Localizer(aligner, LocalizerConfig())

    box_lo, box_hi = pillar_field()
    true = scripted_trajectory(steps, dt, speed)

    map_wp: wp.array | None = None
    T_wheel = se2_to_mat(*true[0])  # wheel-only dead reckoning
    T_gyro = se2_to_mat(*true[0])  # gyro-aided dead reckoning (the ICP prior)
    est = np.zeros((steps, 3))
    wheel = np.zeros((steps, 3))
    gyro = np.zeros((steps, 3))

    def _odom(D, src):
        return odom_step(D, dt, rng, src, trans_noise, yaw_bias, gyro_bias, gyro_noise, wheel_scale)

    for k in range(steps):
        T_true = se2_to_mat(*true[k])
        if k > 0:
            D_true = invert_pose(se2_to_mat(*true[k - 1])) @ T_true
            T_wheel = T_wheel @ _odom(D_true, "wheel")
            T_gyro = T_gyro @ _odom(D_true, "gyro")
        wheel[k] = mat_to_se2(T_wheel)
        gyro[k] = mat_to_se2(T_gyro)

        scan_base = _scan_base(lidar, true[k, 0], true[k, 1], true[k, 2], box_lo, box_hi, k + 1, device)
        if not localizer.initialized:
            localizer.bootstrap(T_gyro, T_true)
            T_wb = T_true
        else:
            pred, _ = localizer.predict(T_gyro)
            T_wb = localizer.update(scan_base, pred, map_wp, T_gyro).pose
        est[k] = mat_to_se2(T_wb)

        world_corrected = transform_points(scan_base, len(scan_base), T_wb)
        valid = wp.full(len(scan_base), 1, dtype=wp.int32, device=device)
        map_wp = acc.step(map_wp, None, world_corrected, valid, (T_wb[0, 3], T_wb[1, 3]))

    def _err(a):
        return np.hypot(a[:, 0] - true[:, 0], a[:, 1] - true[:, 1])

    return dict(
        true=true, wheel=wheel, gyro=gyro, est=est, box_lo=box_lo, box_hi=box_hi,
        wheel_err=_err(wheel), gyro_err=_err(gyro), est_err=_err(est),
    )


def _viz(res: dict, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (axp, axe) = plt.subplots(1, 2, figsize=(14, 6))
    for lo, hi in zip(res["box_lo"], res["box_hi"]):
        axp.add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], color="#bbb"))
    axp.plot(res["true"][:, 0], res["true"][:, 1], "-", color="#2ca02c", lw=2.8, label="reality")
    axp.plot(res["wheel"][:, 0], res["wheel"][:, 1], "--", color="#d62728", lw=1.8, label="wheel-only odom")
    axp.plot(res["gyro"][:, 0], res["gyro"][:, 1], "--", color="#ff7f0e", lw=1.8, label="gyro-aided odom")
    axp.plot(res["est"][:, 0], res["est"][:, 1], "-", color="#1f77b4", lw=1.8, label="ICP estimate")
    axp.set_aspect("equal")
    axp.legend(loc="best")
    axp.set_title("Odometry sources vs. ICP on an arcing path")

    axe.plot(res["wheel_err"], "--", color="#d62728", label="wheel-only odom")
    axe.plot(res["gyro_err"], "--", color="#ff7f0e", label="gyro-aided odom")
    axe.plot(res["est_err"], "-", color="#1f77b4", label="ICP estimate")
    axe.set_xlabel("step")
    axe.set_ylabel("translation error (m)")
    axe.legend()
    axe.set_title("Localization error: gyro cuts the skid-steer heading drift")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def box_world(cell: float = 0.06):
    """Pillars lining a straight lane -> both ICP features AND planner obstacles.

    Returns (scene Heightmap for WarpDriver reality, box_lo, box_hi for the OSDome,
    start (x,y,yaw), goal (x,y)). The SAME pillars drive reality (rasterized to a
    heightmap the robot settles on) and the lidar (3D AABBs it ray-casts against)."""
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
            H[(np.abs(XX - xc) <= half) & (np.abs(YY - yc) <= half)] = top
            los.append((xc - half, yc - half, 0.0))
            his.append((xc + half, yc + half, top))
    scene = Heightmap(H, (xlim[0], ylim[0]), cell)
    return scene, np.asarray(los, np.float32), np.asarray(his, np.float32), (0.0, 0.0, 0.0), np.array([14.0, 0.0])


def run_closed_loop(
    device: str = "cuda",
    max_frames: int = 400,
    dt: float = 0.1,
    columns: int = 512,
    dropout: float = 0.03,
    heading: str = "gyro",
    trans_noise: float = 0.08,
    yaw_bias: float = 0.12,
    gyro_bias_dps: float = 0.3,
    gyro_noise: float = 0.001,
    wheel_scale: float = 0.01,
    win_m: float = 8.0,
    lat_coarsen: int = 4,
    K: int = 8,
    n_theta: int = 24,
    B: int = 4096,
    T: int = 70,
    dock_radius: float = 1.2,
    seed: int = 0,
) -> dict:
    """Stage 2: full pipeline closed on the ESTIMATED pose (odom heading from `heading`)."""
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
    gyro_bias = np.deg2rad(gyro_bias_dps)
    scene, box_lo, box_hi, start, goal = box_world()
    cell = scene.cell
    mu = W.matching_friction(scene)
    drv = WarpDriver(scene, mu, init_pose=tuple(start), device=device)  # REALITY

    sensor = osdome_sensor_config(columns=columns)
    ground = GroundSpec(z=0.0, x_range=(-GROUND, GROUND), y_range=(-GROUND, GROUND))
    lidar = make_osdome_lidar(ground, sensor=sensor, facing="front", dropout=dropout, device=device)
    acc = DeviceMapAccumulator(MAP_VOXEL, MAP_RADIUS, device=device)
    aligner = IcpAligner(IcpConfig(max_iters=30, max_correspondence_dist_m=0.5), device=device)
    localizer = Localizer(aligner, LocalizerConfig())

    ww = wh = int(round(win_m / cell))
    win_grid = GridParams(ww, wh, cell, 0.0, 0.0)
    plan_sim = ForwardSimulator(dynamics.robot_params(), dynamics.planning_solver(), win_grid, B, T, device)
    plan_sim.set_uniform_friction(0.8)
    planner = MppiGpu(plan_sim, CostParams(), robust=RobustConfig(n_slip_samples=K), n_theta=n_theta)
    planner.reset_nominal(1.5)
    kr = max(1, int(lat_coarsen))
    rcny, rcnx, rccell = wh // kr, ww // kr, cell * kr
    ctg = CostToGo(
        GridParams(rcnx, rcny, rccell, 0.0, 0.0),
        dynamics.robot_params(), dynamics.planning_solver(), n_theta=n_theta, device=device,
    )
    planner.cw.lattice_cap = ctg._vcap
    sgrid = GridParams(rcnx, rcny, rccell, 0.0, 0.0).build()

    map_wp: wp.array | None = None
    T_odom = se2_to_mat(*start)
    true_tr, est_tr, err = [], [], []
    contacts, reached, f, prev = 0, False, 0, start

    for f in range(max_frames):
        st = drv.render_state()
        true_tr.append((st.x, st.y))
        if float(np.hypot(st.x - goal[0], st.y - goal[1])) < 0.3:
            reached = True
            break

        T_true = se2_to_mat(st.x, st.y, st.yaw)
        if f > 0:
            D_true = invert_pose(se2_to_mat(*prev)) @ T_true
            T_odom = T_odom @ odom_step(
                D_true, dt, rng, heading, trans_noise, yaw_bias, gyro_bias, gyro_noise, wheel_scale
            )
        prev = (st.x, st.y, st.yaw)

        scan_base = _scan_base(lidar, st.x, st.y, st.yaw, box_lo, box_hi, f + 1, device)
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

        half = win_m / 2.0
        xmin, ymin = ex - half, ey - half
        builder = HeightMapBuilder(cell, (xmin, ex + half, ymin, ey + half), device=device)
        layers = builder.build(map_wp)
        known = layers.count.numpy() > 0
        elev = np.where(known, layers.max.numpy(), 0.0).astype(np.float32)[:wh, :ww]

        state_l = np.array([ex - xmin, ey - ymin, eyaw], np.float32)
        goal_l = (goal[0] - xmin, goal[1] - ymin)
        plan_sim.set_terrain(wp.array(np.ascontiguousarray(elev), dtype=wp.float32, device=device))
        Hc = elev[: rcny * kr, : rcnx * kr].reshape(rcny, kr, rcnx, kr).max(axis=(1, 3)) if kr > 1 else elev
        V = ctg.compute(wp.array(np.ascontiguousarray(Hc), dtype=wp.float32, device=device), goal_l)
        planner.set_lattice(V, sgrid)

        if dock_radius > 0.0 and float(np.hypot(ex - goal[0], ey - goal[1])) < dock_radius:
            cmd = dock_control(state_l, goal_l)
        else:
            planner.replan(state_l, goal_l, 3)
            u = planner.nominal()
            cmd = np.array([u[0, 0], u[0, 1], 0.5 * (u[0, 0] + u[0, 1])], np.float32)
        drv.step(cmd)
        if drv.clear < 0.05:
            contacts += 1

    return dict(
        true=np.asarray(true_tr), est=np.asarray(est_tr), err=np.asarray(err),
        box_lo=box_lo, box_hi=box_hi, goal=goal, reached=reached, frames=f + 1,
        contacts=contacts, heading=heading,
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
        axp.plot(res["est"][:, 0], res["est"][:, 1], "-", color="#1f77b4", lw=1.6, label="ICP estimate (planner input)")
    axp.plot(*res["goal"], "*", color="red", ms=18, mec="k", label="goal")
    axp.set_aspect("equal")
    axp.legend(loc="upper left")
    axp.set_title(
        f"Closed loop on the ESTIMATE ({res['heading']} odom) — reached={res['reached']} "
        f"frames={res['frames']} contacts={res['contacts']}"
    )
    axe.plot(res["err"], "-", color="#1f77b4")
    axe.set_xlabel("step")
    axe.set_ylabel("localization error (m)")
    axe.set_title("ICP localization error along the closed-loop drive")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--columns", type=int, default=512, help="OSDome azimuth columns")
    ap.add_argument("--max-frames", type=int, default=400, help="closed-loop step budget (Stage 2)")
    ap.add_argument("--heading", choices=["gyro", "wheel"], default="gyro", help="closed-loop odom heading source")
    ap.add_argument("--yaw-bias", type=float, default=0.12, help="wheel-diff skid under-rotation")
    ap.add_argument("--gyro-bias-dps", type=float, default=0.3, help="gyro constant bias (deg/s)")
    ap.add_argument("--localization-only", action="store_true", help="Stage 1: scripted path, IMU compare")
    ap.add_argument("--shot", default=None)
    args = ap.parse_args()
    wp.init()
    if args.localization_only:
        res = run(
            device=args.device, steps=args.steps, columns=args.columns,
            yaw_bias=args.yaw_bias, gyro_bias_dps=args.gyro_bias_dps,
        )
        print(
            f"final drift  wheel={res['wheel_err'][-1]:.2f} m  gyro={res['gyro_err'][-1]:.2f} m  "
            f"ICP={res['est_err'][-1]:.2f} m   (mean wheel {res['wheel_err'].mean():.2f}, "
            f"gyro {res['gyro_err'].mean():.2f}, ICP {res['est_err'].mean():.2f})"
        )
        if args.shot:
            _viz(res, args.shot)
    else:
        res = run_closed_loop(
            device=args.device, max_frames=args.max_frames, columns=args.columns,
            heading=args.heading, yaw_bias=args.yaw_bias, gyro_bias_dps=args.gyro_bias_dps,
        )
        print(
            f"CLOSED LOOP ({res['heading']} odom)  reached={res['reached']} frames={res['frames']} "
            f"contacts={res['contacts']} mean-loc-err={res['err'].mean():.2f} m max={res['err'].max():.2f} m"
        )
        if args.shot:
            _viz_closed(res, args.shot)


if __name__ == "__main__":
    main()
