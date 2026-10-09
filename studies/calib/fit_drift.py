"""Fit Odin's own height-drift rate `q_z` from a bag, instead of inheriting one.

`elevation_belief.DriftRates` ships the Oxford Spires dead-reckoning calibration and its own
docstring says to recalibrate -- and it matters by two orders of magnitude, because Odin's pose
is on-device SLAM rather than dead reckoning. `q_z` is what the belief's motion update adds to
every cell's height variance per second, so it sets how much a SEAM between old and new data
costs a planner's margins.

A cell's stored height is measured, then re-measured later. On static ground the two differ by
measurement noise plus whatever pose drift accrued in between, and if that drift is a random
walk on the pose then

    E[(h2 - h1)^2] = 2 * var_meas + q_z * dt

so a straight line through the observed variance against dt has slope q_z and intercept
2*var_meas. This is q_z measured on THIS robot rather than inherited.
"""

import sys
import numpy as np

sys.path.insert(0, "studies")
from calib import bagio
from mcap_ros2.reader import read_ros2_messages

CELL = 0.2
MIN_PTS = 4  # enough returns in a cell for its mean height to mean something
MAX_ROUGH = 0.02  # [m] and flat enough that a shifted sample pattern is not mistaken for drift

path = bagio.bag_path(sys.argv[1] if len(sys.argv) > 1 else "out_odin0")
odom = bagio.read_odometry(path)
lo = odom.xyz[:, :2].min(0) - 30.0
hi = odom.xyz[:, :2].max(0) + 30.0
nx, ny = np.ceil((hi - lo) / CELL).astype(int) + 1
last_t = np.full((ny, nx), np.nan)
last_h = np.full((ny, nx), np.nan)
dts, dhs, rngs = [], [], []
t0, k = None, 0

for msg in read_ros2_messages(path, topics=[bagio.CLOUD_TOPIC]):
    c = msg.ros_msg
    t = c.header.stamp.sec + c.header.stamp.nanosec * 1e-9
    if t0 is None:
        t0 = t
    i = int(np.argmin(np.abs(odom.t - t)))
    if abs(odom.t[i] - t) > 0.1:
        continue
    pts = bagio._unpack_cloud(c)
    if pts.size == 0:
        continue
    on = (
        (pts[:, 0] > bagio.SELF_X[0])
        & (pts[:, 0] < bagio.SELF_X[1])
        & (pts[:, 1] > bagio.SELF_Y[0])
        & (pts[:, 1] < bagio.SELF_Y[1])
    )
    pts = pts[~on]
    if pts.size == 0:
        continue
    rng_pt = np.hypot(pts[:, 0], pts[:, 1])
    roll, pitch, yaw = odom.rpy[i]
    w = pts.astype(np.float64) @ bagio._rpy_to_mat(roll, pitch, yaw).T + odom.xyz[i]
    cc = ((w[:, 0] - lo[0]) / CELL).astype(np.int64)
    rr = ((w[:, 1] - lo[1]) / CELL).astype(np.int64)
    ok = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
    flat = np.ravel_multi_index((rr[ok], cc[ok]), (ny, nx))
    z, rg = w[ok, 2], rng_pt[ok]
    order = np.argsort(flat, kind="stable")
    flat, z, rg = flat[order], z[order], rg[order]
    edges = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1], True])
    counts = np.diff(edges)
    sums = np.add.reduceat(z, edges[:-1])
    sqs = np.add.reduceat(z * z, edges[:-1])
    rsum = np.add.reduceat(rg, edges[:-1])
    ids = flat[edges[:-1]]
    mean = sums / counts
    rough = np.sqrt(np.maximum(sqs / counts - mean**2, 0.0))
    good = (counts >= MIN_PTS) & (rough < MAX_ROUGH)
    ids, mean, mrng = ids[good], mean[good], (rsum / counts)[good]
    gr, gc = np.unravel_index(ids, (ny, nx))
    prev_t, prev_h = last_t[gr, gc], last_h[gr, gc]
    seen = np.isfinite(prev_t)
    if seen.any():
        dts.append((t - t0) - prev_t[seen])
        dhs.append(mean[seen] - prev_h[seen])
        rngs.append(mrng[seen])
    last_t[gr, gc] = t - t0
    last_h[gr, gc] = mean
    k += 1

dt = np.concatenate(dts)
dh = np.concatenate(dhs)
rg = np.concatenate(rngs)
near = rg < 6.0  # inside the usable horizon, where meas noise is smallest
dt, dh = dt[near], dh[near]
print(f"{k} frames, {len(dt)} re-measurement pairs within 6 m\n")
bins = list(
    zip(
        *(lambda e: (e[:-1], e[1:]))(
            [0.3, 0.7, 1.2, 2.0, 3.5, 6.0, 10.0, 18.0, 32.0, 60.0, 110.0, 200.0]
        )
    )
)
xs, ys, ns = [], [], []
print(f"{'dt window':>14s} {'pairs':>7s} {'robust sd(dh)':>14s} {'variance':>11s}")
for a, b in bins:
    m = (dt >= a) & (dt < b)
    if m.sum() < 50:
        continue
    # robust: the MAD, so a person walking through does not become the drift rate
    sd = 1.4826 * np.median(np.abs(dh[m] - np.median(dh[m])))
    xs.append(np.median(dt[m]))
    ys.append(sd * sd)
    ns.append(int(m.sum()))
    print(f"{a:>6.1f}-{b:<7.1f} {m.sum():>7d} {sd:>13.4f}m {sd*sd:>10.3e}")
x, y = np.array(xs), np.array(ys)
w = np.sqrt(np.array(ns, float))  # weight a bin by how many pairs stand behind it
A = np.c_[x, np.ones(len(x))]
slope, icpt = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
flat_slope, flat_icpt = np.linalg.lstsq(A, y, rcond=None)[0]
r2 = 1.0 - ((y - (slope * x + icpt)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print("\nfit  E[(dh)^2] = 2*var_meas + q_z*dt")
print(f"  q_z       = {slope:.3e}  [m^2/s]   (unweighted {flat_slope:.3e}; default 7.430e-03)")
print(f"  var_meas  = {icpt/2:.3e}  [m^2] -> sd {np.sqrt(max(icpt,0)/2):.4f} m")
print(f"  R^2       = {r2:.3f}   <- soft; see the table below for why it does not matter")
print(f"  ratio to the inherited default: {slope/7.430e-3:.4f}x\n")
print("every bin against what the inherited rate would predict:")
print(f"  {'dt':>7s} {'observed':>10s} {'predicted':>11s} {'ratio':>7s}")
for xi, yi in zip(x, y):
    pred = np.sqrt(2 * (icpt / 2) + 7.430e-3 * xi)
    print(f"  {xi:>6.1f}s {np.sqrt(yi):>9.4f}m {pred:>10.4f}m {pred/np.sqrt(yi):>6.1f}x")
